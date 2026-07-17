# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Any

from ..processing.emi.suppression import suppress_emi as _suppress_emi
from ..common.utils import assert_gt_0

from .modes import SUBSYSTEM_MODES, iter_channel_filter

def suppress_emi(science_dict: dict[str, Any],
                    mode: str,
                    *,
                    nfft: int = 512,
                    dt: float = 1/1.4e6,
                    bw: float = 1.0e6,
                    method: str = "adaptive",
                    statistic: str = "mad",
                    k: float = 1.5,
                    window_size: int = 129,
                    replace: str = "INTERP",
                    interp_pad: int = 2,
                    mask: NDArray[np.bool_] | None = None,
                    verbose: bool = False,
                    ) -> dict[str, Any]:
    """Suppress electromagnetic interference in MARSIS EDR data.

    Transforms each channel/filter combination to the frequency
    domain, applies EMI detection and suppression, and transforms
    back. The detected EMI mask is stored alongside the corrected
    data under a separate key.

    Args:
        science_dict: Science data dictionary keyed by PDS field
            names. Modified in-place.
        mode: MARSIS operative mode (e.g., ``"SS3"``).
        nfft: FFT length.
        dt: Sample spacing in seconds.
        bw: Bandwidth in Hz.
        method: EMI detection method (e.g., ``"adaptive"``).
        k: Detection threshold multiplier.
        window_size: Sliding window size for adaptive detection.
        replace: Replacement strategy — ``"INTERP"``, ``"ZERO"``,
            or ``"BASELINE"``.
        interp_pad: Number of bins to pad when interpolating across
            flagged samples.
        mask: Optional pre-computed EMI mask. If None, the mask is
            computed from the data.
        verbose: If True, print progress messages.

    Returns:
        The updated science data dictionary with EMI-suppressed
        arrays and per-channel/filter EMI masks stored under
        ``"EMI_{filt}_{chan}_DIP"`` keys.

    Raises:
        ValueError: If the operative mode is not supported.
        KeyError: If expected science keys are missing.
    """
    mode = mode.upper()
    if mode not in SUBSYSTEM_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    method = method.upper()
    replace = replace.upper()
    assert_gt_0(nfft)
    assert_gt_0(dt)
    assert_gt_0(bw)

    if verbose:
        print("Peforming emi suppression...")
    for f_str, c_str, kk in iter_channel_filter(mode):
        e = f"EMI_{f_str}_{c_str}_DIP"

        if kk not in science_dict:
            raise KeyError(
                f"Key '{kk}' not found in science data. You may need "
                f"to preprocess your MARSIS data first.")

        if verbose:
            print(f"\tChannel {c_str} Filter {f_str}")
        data = science_dict[kk]
        science_dict[kk], emi_mask = _suppress_emi(data, nfft=nfft, dt=dt, bw=bw, f_cen=0.0, method=method,
                                                   statistic=statistic, k=k, window_size=window_size, replace=replace,
                                                   interp_pad=interp_pad, input_time=True, output_time=True)
        science_dict[e] = emi_mask
    return science_dict