# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Any

from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from ..processing.range_compression.chirps import create_complex_baseband_chirp
from ..processing.windows import form_window_bandlimited
from ..common.utils import assert_gt_0

from ..processing.utils import (
    check_monotonically_increasing,
    select_iband,
    to_complex_baseband,
)

from .modes import SUBSYSTEM_MODES, iter_channel_filter


def _make_chirp(ct: str = "ideal",
                nfft: int = 512,
                dt: float = 1 / 1.4e6,
                tau: float = 250e-6,
                bw: float = 1e6,
                chirp_direction: int = 1,
                ) -> NDArray[np.complexfloating]:
    """Create a reference chirp for range compression.

    Args:
        ct: Chirp type. Only ``"ideal"`` is supported.
        nfft: FFT length.
        dt: Sample spacing in seconds.
        tau: Chirp duration in seconds.
        bw: Bandwidth in Hz.
        chirp_direction: ``+1`` for up-chirp, ``-1`` for down-chirp.

    Returns:
        Complex baseband chirp of length ``nfft``.
    """
    ct = ct.lower()
    if ct == "ideal":
        chirp, _, _ = create_complex_baseband_chirp(
            nfft, dt, tau, bw, chirp_direction=chirp_direction)
    else:
        raise NotImplementedError(
            "IDEAL is the only supported chirp type for MARSIS")
    return chirp


def _make_filter(chirp: NDArray[np.complexfloating],
                 f: NDArray[np.floating],
                 bw: float = 1e6,
                 ft: str = "MATCHED",
                 ) -> NDArray[np.complexfloating]:
    """Create a frequency-domain filter from a reference chirp.

    Args:
        chirp: Reference chirp, length ``nfft``.
        f: Frequency vector in Hz, length ``nfft``.
        bw: Bandwidth in Hz.
        ft: Filter type — ``"MATCHED"`` or ``"INVERSE"``.

    Returns:
        Frequency-domain filter, length ``nfft``.

    Raises:
        ValueError: If filter type is not recognized.
    """
    ft = ft.upper()
    out_band = np.abs(f) > bw / 2
    if ft == "MATCHED":
        filt = np.conj(fft(chirp))
    elif ft == "INVERSE":
        filt = 1.0 / fft(chirp)
    else:
        raise ValueError(f"Invalid filter type: {ft}")
    filt[out_band] = 0.0 + 0.0j
    return filt



def range_compression(science_dict: dict[str, Any],
                      mode: str,
                      *,
                      filter_type: str = "MATCHED",
                      chirp_type: str = "IDEAL",
                      nfft: int = 512,
                      dt: float = 1 / 1.4e6,
                      tau: float = 250e-6,
                      bw: float = 1e6,
                      f_cen: float = 0.7e6,
                      window_type: str = "hanning",
                      alpha: float = 1.0,
                      verbose: bool = False,
                      ) -> dict[str, Any]:
    """Apply range compression to MARSIS EDR data.

    Builds a matched or inverse filter from a complex baseband chirp
    and applies it to each channel/filter combination in the science
    dictionary. Data are shifted to complex baseband before filtering.

    Args:
        science_dict: Science data dictionary keyed by PDS field
            names. Modified in-place.
        mode: MARSIS operative mode (e.g., ``"SS3"``).
        filter_type: ``"MATCHED"`` or ``"INVERSE"``.
        chirp_type: Chirp type. Only ``"IDEAL"`` is supported.
        nfft: FFT length.
        dt: Sample spacing in seconds.
        tau: Chirp duration in seconds.
        bw: Bandwidth in Hz.
        f_cen: Center frequency of the preprocessed data (0.7 MHz)
        window_type: Spectral window type.
        alpha: Tukey window taper fraction.
        verbose: If True, print progress messages.

    Returns:
        The updated science data dictionary with range-compressed
        arrays.

    Raises:
        ValueError: If the operative mode or filter type is not
            supported.
        KeyError: If expected science keys are missing.
    """
    mode = mode.upper()
    if mode not in SUBSYSTEM_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    ft = filter_type.upper()
    ct = chirp_type.upper()
    wt = window_type.lower()
    assert_gt_0(nfft)
    assert_gt_0(dt)
    assert_gt_0(tau)

    # Build chirp and filter
    chirp = _make_chirp(ct=ct, nfft=nfft, dt=dt, tau=tau, bw=bw)
    f = fftfreq(nfft, d=dt)
    h_f = _make_filter(chirp, f=f, bw=bw, ft=ft)

    # Build window
    f_nat = fftshift(f)
    wnd = form_window_bandlimited(f_nat, bw, window_type=wt, f_cen=0.0, alpha=alpha)
    wnd = ifftshift(wnd)

    # Apply window to filter
    h_f *= wnd

    if verbose:
        print("Performing range compression...")

    for f_str, c_str, k in iter_channel_filter(mode):
        if k not in science_dict:
            raise KeyError(
                f"Key '{k}' not found in science data. You may need "
                f"to preprocess your MARSIS data first.")

        if verbose:
            print(f"\tChannel {c_str} Filter {f_str}")
        sci_data = science_dict[k]
        # Shift data to Complex Baseband
        sci_data, _ = to_complex_baseband(sci_data, nfft=nfft, dt=dt, f_cen=f_cen, bw=bw,
                                                shift_direction=-1, output_time=False)
        sci_data *= h_f[:, None]
        science_dict[k] = ifft(sci_data, axis=0, workers=-1)
    return science_dict