# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft, rfftfreq, fftfreq, rfft, irfft

from ...common.utils import assert_gt_0

from .adaptive import adaptive_spectral_notch

#####################################################################################################################
#
# Instrument Level Wrapper
#
#####################################################################################################################
def suppress_emi(data: NDArray,
                 *,
                 nfft: int,
                 dt: float,
                 bw: float,
                 f_cen: float = 0.0,
                 method: str = "ADAPTIVE",
                 statistic: str = "MAD",
                 k: float = 6,
                 window_size: int = 129,
                 replace: str = "INTERP",
                 interp_pad: int = 2,
                 mask: NDArray[np.bool_] | None = None,
                 input_time: bool = True,
                 output_time: bool = True
                 ) -> tuple[NDArray, NDArray[np.bool_]]:
    """Apply EMI suppression to radar sounder data.

    Constructs the appropriate frequency vector for the given
    instrument, optionally transforms to/from the frequency
    domain, and dispatches to :func:`_suppress_emi`.

    Args:
        data: Echo data array, shape ``(n_samp, n_cols)``.
        nfft: FFT length used to construct the frequency vector.
        dt: Sample spacing in seconds.
        bw: Signal bandwidth in Hz.
        f_cen: Center frequency of the data in Hz.
        method: EMI detection method — ``"ADAPTIVE"``.
        statistic: Local-statistic method used to form the threshold ("MAD" or "STD").
            Defaults to "MAD".
        k: Detection threshold multiplier.
        window_size: Sliding window size for local statistics.
        replace: Replacement strategy — ``"INTERP"``,
            ``"ZERO"``, or ``"BASELINE"``.
        interp_pad: Number of bins to pad when interpolating
            across flagged samples.
        mask: Optional pre-computed EMI mask. If None, the
            mask is computed from the data.
        input_time: If True, the input is in the time domain
            and will be FFT'd before processing.
        output_time: If True, the output is transformed back
            to the time domain before returning.

    Returns:
        A tuple ``(data, emi_mask)`` where:

        - ``data``: EMI-suppressed echo data, same shape as
            input.
        - ``emi_mask``: Boolean mask of flagged bins, same
            shape as input.

    Raises:
        ValueError: If any required parameter is invalid.
    """
    rfft_trig = False
    method = method.upper()
    replace = replace.upper()
    assert_gt_0(nfft)
    assert_gt_0(dt)
    assert_gt_0(bw)
    if np.isrealobj(data):
        rfft_trig = True
        freqs = rfftfreq(nfft, d=dt)
    else:
        freqs = fftfreq(nfft, d=dt)

    if input_time:
        if rfft_trig:
            data = rfft(data, axis=0, n=nfft, workers=-1)
        else:
            data = fft(data, axis=0, n=nfft, workers=-1)

    data, emi_mask = _suppress_emi(data, freqs=freqs, f_cen=f_cen, bw=bw, method=method, statistic=statistic,
                                   k=k, window_size=window_size, replace=replace, interp_pad=interp_pad, mask=mask)

    if output_time:
        if rfft_trig:
            data = irfft(data, axis=0, n=nfft, workers=-1)
        else:
            data = ifft(data, axis=0, n=nfft, workers=-1)
    return data, emi_mask


#####################################################################################################################
#
# Wrapper Function
#
#####################################################################################################################

def _suppress_emi(data: NDArray[np.complexfloating],
                  freqs: NDArray[np.floating],
                  bw: float,
                  *,
                  f_cen: float = 0.0,
                  method: str = "ADAPTIVE",
                  statistic: str = "MAD",
                  k: float = 1.5,
                  window_size: int = 129,
                  replace: str = "INTERP",
                  interp_pad: int = 2,
                  mask: NDArray[np.bool_] | None = None,
                  ) -> tuple[NDArray[np.complexfloating], NDArray[np.bool_]]:
    """Apply electromagnetic interference suppression to radar spectra.

    Dispatches to the selected EMI suppression method.

    Args:
        data: Science data, shape (n_samp, n_cols).
        freqs: Frequency vector (Hz), length n_samp.
        bw: Signal bandwidth (Hz).
        f_cen: Center frequency of the data (Hz)
        method: Suppression method: "ADAPTIVE".
        statistic: Local-statistic method used to form the threshold ("MAD" or "STD").
            Defaults to "MAD".
        k: Threshold multiplier for outlier detection.
        window_size: Window size for local statistics.
        replace: Replacement strategy for ADAPTIVE: "BASELINE",
            "INTERP", or "ZERO".
        interp_pad: Interpolation anchor padding (ADAPTIVE only).
        mask: Optional precomputed EMI mask (ADAPTIVE only).

    Returns:
        Tuple of (suppressed spectra, EMI mask). Mask is True where
        samples were modified.

    Raises:
        ValueError: If method is not supported.
    """
    method = method.upper()
    if method == "ADAPTIVE":
        out, results = adaptive_spectral_notch(data, freqs, bw, f_cen=f_cen, window_size=window_size, k=k,
                                               statistic=statistic, replace=replace, interp_pad=interp_pad, mask=mask)
    else:
        raise ValueError(f"{method} is not a supported EMI suppression method")
    return out, results
