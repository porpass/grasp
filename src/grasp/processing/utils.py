# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import uniform_filter1d

from typing import Any

def to_complex_baseband(data: NDArray[np.floating],
                        nfft: int,
                        dt: float,
                        f_cen: float,
                        bw: float,
                        *,
                        shift_direction: int = -1,
                        output_time: bool = True,
                        ) -> tuple[NDArray[np.complexfloating[Any]], NDArray[np.complexfloating[Any]]]:
    """Shift data to complex baseband and apply bandlimiting.

    Zero-pads the input to length ``nfft``, multiplies by a complex
    exponential to shift the spectrum to baseband, zeros out-of-band
    frequencies beyond ``bw/2``, and optionally transforms back to
    the time domain.

    Args:
        data: 1D arßray of shape ``(n_samp,)`` or 2D array of shape
            ``(n_samp, n_recs)``. Can be real or complex; promoted to
            complex internally. 1D inputs return a 1D ``s_out``; 2D
            inputs return a 2D ``s_out``.
        nfft: FFT length and zero-padded length along the sample axis.
        dt: Sample spacing in seconds.
        f_cen: Center frequency in Hz. The data are shifted by
            ``fs - f_cen`` where ``fs = 1/dt``.
        bw: Bandwidth in Hz. Frequencies with ``|f| > bw/2`` are
            zeroed after the baseband shift.
        shift_direction: ``+1`` to shift by ``+(fs - f_cen)`` or
            ``-1`` to shift by ``-(fs - f_cen)``.
        output_time: If True, return time-domain data. If False,
            return frequency-domain data.

    Returns:
        A tuple ``(s_out, shift)`` where:

        - ``s_out``: complex array matching the input rank. Shape
            ``(nfft,)`` for 1D input or ``(nfft, n_recs)`` for 2D
            input. Time-domain if ``output_time`` is True, otherwise
            frequency-domain. Out-of-band frequencies have been
            zeroed in both cases.
        - ``shift``: 1D complex array, shape ``(nfft,)``. The
            complex exponential applied for the baseband shift.

    Raises:
        ValueError: If ``data`` is not 1D or 2D, ``nfft < n_samp``,
            ``dt`` or ``bw`` are not positive, or
            ``shift_direction`` is not ±1.
    """
    if data.ndim == 1:
        was_1d = True
        data = data[:, np.newaxis]
    elif data.ndim == 2:
        was_1d = False
    else:
        raise ValueError(f"data must be 1D or 2D, got shape {data.shape!r}")

    n_samp, n_recs = data.shape

    if nfft < n_samp:
        raise ValueError(
            f"nfft ({nfft}) must be >= number of samples ({n_samp})."
        )

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    if bw <= 0.0:
        raise ValueError(f"bw must be positive, got {bw}")

    if abs(shift_direction) != 1:
        raise ValueError("shift_direction must be -1 or 1")

    fs = 1 / dt
    sgn = shift_direction
    # Allocate complex output, zero-padded
    s: NDArray[np.complexfloating[Any]] = np.zeros((nfft, n_recs), dtype=complex)
    s[:n_samp, :] = data

    fshift = float(fs - f_cen)
    k = np.arange(nfft, dtype=float)
    shift: NDArray[np.complexfloating[Any]] = np.exp(sgn*2j * np.pi * fshift * k * dt)

    s *= shift[:, np.newaxis]
    S = fft(s, axis=0, n=nfft)

    f = fftfreq(nfft, d=dt)
    h_bw = 0.5 * bw
    idx = np.where(np.abs(f) > h_bw)[0]
    if len(idx) == 0:
        raise ValueError("No valid frequencies found for this domain.")
    S[idx, :] = 0.0

    if output_time:
        S = ifft(S, axis=0, n=nfft)

    if was_1d:
        S = S[:, 0]
    return S, shift



def assert_monotonically_increasing(arr: NDArray) -> None:
    """
    Assert that an array is monotonically increasing.

    Args:
        arr: Input array-like object.

    Raises:
        ValueError: If `arr` is not monotonically increasing.
    """
    a = np.asarray(arr)

    if a.ndim != 1:
        raise ValueError("Input must be a 1D array")

    if a.size < 2:
        return

    if np.any(np.diff(a) < 0):
        raise ValueError("Array must be monotonically increasing (non-decreasing)")

    if not np.all(np.isfinite(a)):
        raise ValueError("Array must contain only finite values")


def check_monotonically_increasing(arr: np.typing.ArrayLike) -> bool:
    """
    Check that an array is monotonically non-decreasing.

    Args:
        arr: Input array-like object.

    Raises:
        ValueError: If `arr` is not monotonically non-decreasing.
    """
    a = np.asarray(arr)

    if a.ndim != 1:
        raise ValueError("Input must be a 1D array")

    if np.all(np.diff(a) > 0):
        return True
    else:
        return False

def select_iband(freqs, f_cen, bw):
    if f_cen == 0.0:
        iband = np.where(np.abs(freqs) <= (bw / 2.0))[0]
    else:
        iband = np.where(np.abs(freqs - f_cen) <= (bw / 2.0))[0]
    if iband.size == 0:
        raise ValueError(f"Bandwidth {bw} selects no bins for provided freqs.")
    return iband


def smooth(data: NDArray[np.floating],
           window: int | None,
           ) -> NDArray[np.floating]:
    """Boxcar-smooth a 1D array along its single axis.

    Equivalent to IDL's ``smooth(data, window, /edge_truncate)``.
    Uses ``scipy.ndimage.uniform_filter1d`` with ``mode='nearest'``,
    so the boundary values replicate rather than wrap.

    Args:
        data: 1D array to smooth.
        window: Boxcar window length in samples. ``None`` or any value
            ``<= 1`` returns ``data`` unchanged (no-op).

    Returns:
        Smoothed array, same shape as ``data``. Float64 when smoothing
        is applied; otherwise returned as-is.
    """
    if window is None or window <= 1:
        return data
    return uniform_filter1d(np.asarray(data, dtype=np.float64),
                            size=int(window),
                            mode="nearest")
