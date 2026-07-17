# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from scipy.fft import fft, fftfreq, rfft, rfftfreq

def create_filter(filter_type: str,
                  chirp: NDArray[np.floating] | NDArray[np.complexfloating],
                  dt: float,
                  bw: float,
                  *,
                  f_cen: float | None = None,
                  eps: float | None = 1e-12,
                  ) -> tuple[NDArray, NDArray]:
    """
    Create a frequency-domain filter from a reference chirp.

    Supported filters:
        - "MATCHED": H(f) = conj(S(f))
        - "INVERSE": H(f) = 1 / S(f)   (optionally stabilized via `eps`)

    For complex chirps, uses FFT and masks to |f| <= bw/2.
    For real chirps, uses RFFT and masks to [f_cen - bw/2, f_cen + bw/2].

    ``chirp`` may be 1D (one reference shared across the whole
    observation) or 2D of shape ``(n, n_cols)`` (one reference per
    column — used for temperature-calibrated chirps that vary
    along-track). The output ``h_f`` has the same dimensionality.

    Args:
        filter_type: "MATCHED" or "INVERSE" (case-insensitive).
        chirp: Reference chirp (real or complex). Shape ``(n,)``
            or ``(n, n_cols)``. Time-series
        dt: Sample spacing (s). Must be positive.
        bw: Bandwidth (Hz). Must be positive.
        f_cen: Center frequency (Hz) required for real chirps.
        eps: If not None, bins with |S(f)| < eps are set to 0 for INVERSE.

    Returns:
        h_f: Frequency-domain filter (complex). Same dimensionality
            as ``chirp``.
        f: Frequency vector (Hz) aligned with ``h_f`` along axis 0.

    Raises:
        ValueError: For invalid inputs or unsupported filter type.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if bw <= 0.0:
        raise ValueError("bw must be positive")
    if eps is not None and eps <= 0.0:
        raise ValueError("eps must be positive when provided")
    if chirp.ndim not in (1, 2):
        raise ValueError("chirp must be 1D or 2D")

    n = chirp.shape[0]
    fs = 1.0 / dt
    nyq = 0.5 * fs
    h_bw = 0.5 * bw

    if np.issubdtype(chirp.dtype, np.complexfloating):
        s_f = fft(chirp, n=n, axis=0)
        f = fftfreq(n, d=dt)
        mask = (np.abs(f) <= h_bw)
    else:
        if f_cen is None:
            raise ValueError("f_cen is required when using a real-valued chirp as input")
        fc = float(f_cen)
        flo = fc - h_bw
        fhi = fc + h_bw
        if flo < 0.0 or fhi > nyq:
            raise ValueError("Requested band extends outside [0, Nyquist] for a one-sided RFFT grid")

        s_f = rfft(chirp, n=n, axis=0)
        f = rfftfreq(n, d=dt)
        mask = (f >= flo) & (f <= fhi)

    h_f = np.zeros_like(s_f, dtype=np.complex64)

    ft = filter_type.strip().upper()
    if ft == "MATCHED":
        # Boolean indexing along axis 0 works for both 1D and 2D.
        h_f[mask] = np.conj(s_f[mask]).astype(np.complex64, copy=False)

    elif ft == "INVERSE":
        if eps is None:
            h_f[mask] = (1.0 / s_f[mask]).astype(np.complex64, copy=False)
        else:
            # Broadcast the 1D band mask over columns for 2D s_f.
            band_mask = mask if s_f.ndim == 1 else mask[:, None]
            good = band_mask & (np.abs(s_f) >= eps)
            h_f[good] = (1.0 / s_f[good]).astype(np.complex64, copy=False)
    else:
        raise ValueError(f"filter_type {filter_type!r} not recognized")

    return h_f, f