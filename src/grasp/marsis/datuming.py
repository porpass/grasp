# SPDX-License-Identifier: BSD-3-Clause
from ..datuming.utils import remove_rwot_offset as _remove_rwot_offset
import numpy as np
from numpy.typing import NDArray
from typing import Any

from .modes import SUBSYSTEM_MODES, SYSTEM_DELAY, iter_channel_filter

def calculate_rwot(rx_trig_sa_prog: NDArray[np.floating],
                   dcg_configuration: NDArray[Any],
                   channel: int,
                   fs: float = 1.4e6,
                   is_raw: bool = False) -> NDArray[np.floating]:
    """Compute receive window opening time for a MARSIS channel.

    Converts the receiver trigger position to seconds and adds
    the frequency-dependent system delay. For the second channel
    (channel 1), an additional 450 µs offset is removed.

    Args:
        rx_trig_sa_prog: Receiver trigger position in samples,
            shape ``(n_col,)``.
        dcg_configuration: DCG configuration values (0–3),
            shape ``(n_col,)``. Maps to system delay via
            ``SYSTEM_DELAY``.
        channel: Channel index (0 or 1).
        fs: Sampling frequency in Hz.
        is_raw: If True, data is flash memory / super frame

    Returns:
        Receive window opening time in seconds, shape ``(n_col,)``.
    """
    delay = SYSTEM_DELAY[dcg_configuration]
    # True Sampling Rate is 2.8 MHz or 2x the recorded FS (THIS WILL MATTER FOR FLASH MEMORY DATA!!!)
    if is_raw:
        rwot = rx_trig_sa_prog / (fs) + delay
    else:
        rwot = rx_trig_sa_prog / (2*fs) + delay
    if channel == 1:
        rwot -= 450e-6
    return rwot

def remove_rwot_offset(science_dict: dict[str, Any],
                       mode: str,
                       altitude: NDArray[np.floating],
                       fs: float = 1.4e6,
                       n_center: int | None = None,
                       verbose: bool = False) -> dict[str, Any]:
    """Align range windows for MARSIS data.

    Computes the receive window opening time per channel from
    ``RX_TRIG_SA_PROG`` and ``DCG_CONFIGURATION``, then removes
    the RWOT offset from each channel/filter echo array.

    Args:
        science_dict: Science data dictionary keyed by PDS field
            names. Modified in-place.
        mode: MARSIS operative mode (e.g., ``"SS3"``).
        altitude: Spacecraft altitude in meters, shape ``(n_col,)``.
        fs: Sampling frequency in Hz.
        n_center: Sample index to center the surface return on.
            If None, defaults to ``n_samp // 2``.
        verbose: If True, print progress messages.

    Returns:
        The updated science data dictionary with aligned arrays.

    Raises:
        ValueError: If the operative mode is not supported.
        KeyError: If expected science keys are missing.
    """
    dt = 1.0 / fs
    mode = mode.upper()
    if mode not in SUBSYSTEM_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    mode_info = SUBSYSTEM_MODES[mode]

    # Precompute per-channel RWOT so the inner loop stays flat.
    rwot_by_chan: dict[str, NDArray] = {}
    for c_idx, c_str in enumerate(mode_info.chan_str):
        rx_trig_sa_prog = science_dict['RX_TRIG_SA_PROGR'][c_idx, :]
        dcg_configuration = science_dict['DCG_CONFIGURATION'][c_idx, :]
        rwot_by_chan[c_str] = calculate_rwot(
            rx_trig_sa_prog, dcg_configuration, c_idx, fs=fs)

    if verbose:
        print("Aligning data windows...")

    for f_str, c_str, k in iter_channel_filter(mode):
        if k not in science_dict:
            raise KeyError(
                f"Key '{k}' not found in science data. You may need "
                f"to preprocess your MARSIS data first.")

        if verbose:
            print(f"\tChannel {c_str} Filter {f_str}")

        sci_data = science_dict[k]
        n_samp = sci_data.shape[0]
        nc = n_center if n_center is not None else n_samp // 2

        science_dict[k], datum = _remove_rwot_offset(
            sci_data, rwot_by_chan[c_str], altitude, dt, n_center=nc)

    return science_dict, datum

