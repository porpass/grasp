# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

from ..utils import select_iband

def threshold_emi(spectra: NDArray[np.complexfloating],
                  freqs: NDArray[np.floating],
                  bw: float,
                  threshold_k: float,
                  value_k: float,
                  f_cen: float = 0.0,
                  window_size: int = 128,
                  ) -> tuple[NDArray[np.complexfloating], NDArray[np.bool_]]:
    """Suppress narrowband EMI in complex spectra using a moving threshold.

    Operates within the fundamental frequency band (|f| <= bw/2).
    For each trace, computes a moving mean and standard deviation
    of the magnitude spectrum, defines a threshold m + threshold_k*s,
    and clips values above that threshold to m + value_k*s. The
    original phase is preserved.

    Args:
        spectra: Complex spectra, shape (n_samp, n_cols).
        freqs: Frequency vector (Hz), length n_samp.
        bw: Bandwidth (Hz). Only |f| <= bw/2 is processed.
        threshold_k: Multiplier on moving std for clipping threshold.
        value_k: Multiplier on moving std for replacement value.
        f_cen: Center frequency of spectra (Hz)
        window_size: Window length for moving statistics.

    Returns:
        Tuple of (suppressed spectra, EMI mask). Mask is True where
        samples were clipped.

    Raises:
        ValueError: If inputs have inconsistent shapes or invalid values.
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 2-D; got shape {spectra.shape}")
    n_samp, n_cols = spectra.shape

    if freqs.shape[0] != n_samp:
        raise ValueError(
            f"freqs length must match spectra.shape[0]; got {freqs.shape[0]} vs {n_samp}"
        )
    if bw <= 0:
        raise ValueError(f"bw must be positive; got {bw}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive; got {window_size}")

    data_emi: NDArray[np.complexfloating] = np.zeros((n_samp, n_cols), dtype=np.complex128)
    emi_mask = np.zeros((n_samp, n_cols), dtype=bool)

    iband = select_iband(freqs, f_cen, bw)

    band_data = spectra[iband, :]
    mag = np.abs(band_data)
    phase = np.angle(band_data)

    for ii in range(n_cols):
        m, s = movingMeanSTD(mag[:, ii], window_size)
        threshold = m + threshold_k * s
        new_vals = m + value_k * s

        mask = mag[:, ii] > threshold
        mag[mask, ii] = new_vals[mask]

        data_emi[iband, ii] = mag[:, ii] * np.exp(1j * phase[:, ii])
        emi_mask[iband[mask], ii] = True

    return data_emi, emi_mask

def movingMeanSTD(data: NDArray, window_size: int) -> tuple[NDArray, NDArray]:
    """Compute the moving mean and standard deviation of a 1-D signal.

    Uses ``scipy.ndimage.uniform_filter1d`` with ``mode="nearest"`` so that
    edges are extended by replicating the boundary values.

    Args:
        data: Input array.
        window_size: Length of the moving window in samples.

    Returns:
        Tuple ``(mean, std)`` of arrays the same shape as ``data``.
    """
    mean = uniform_filter1d(data, size=window_size, mode="nearest")
    mean_sq = uniform_filter1d(data**2, size=window_size, mode="nearest")
    std = np.sqrt(mean_sq - mean**2)
    return mean, std