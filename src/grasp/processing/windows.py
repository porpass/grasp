# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from scipy.fft import ifftshift

from .utils import assert_monotonically_increasing


WINDOW_NAMES: frozenset[str] = frozenset({
    "RECTANGLE", "NONE",
    "HANN", "HANNING",
    "HAMMING",
    "BLACKMAN",
    "BLACKMAN-HARRIS", "BLACKMAN_HARRIS", "BH",
    "NUTTALL",
    "BARTLETT",
    "COSINE", "SINE", "RAISED_COSINE",
    "FLATTOP",
    "TUKEY",
})
"""All window-name aliases accepted by :func:`form_window`. Used by
:meth:`grasp.radar_sounder.RadarSounder.validate_processing_parameters`
to surface typos in job-file window selections up front. Also useful
to populate a UI dropdown."""


def broadening_factor(wnd):
    l = len(wnd)
    return l / (np.sum(wnd) ** 2 / np.sum(wnd ** 2))


def form_window(n: int,
                wt: str,
                *,
                alpha: float | None = None) -> NDArray[np.float32]:
    """
    Construct a 1D window function.

    Args:
        n: Number of samples in the window. Must be positive.
        wt: Window type (case-insensitive).
        alpha: Tukey window taper fraction in [0, 1]. Only used for "TUKEY".
            Defaults to 0.5 when None.

    Returns:
        A 1D NumPy array of dtype float32 containing the window.

    Raises:
        ValueError: If `n <= 0`.
        ValueError: If `wt` is not recognized.
        ValueError: If `alpha` is outside [0, 1] for Tukey windows.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    wt = wt.upper()

    if wt in {'RECTANGLE', 'NONE'}:
        w = np.ones(n, dtype=np.float32)

    elif wt in {'HANN', 'HANNING'}:
        w = hann_window(n)

    elif wt == "HAMMING":
        w = hamming_window(n)

    elif wt == "BLACKMAN":
        w = blackman_window(n)

    elif wt in {"BLACKMAN-HARRIS", "BLACKMAN_HARRIS", "BH"}:
        w = blackman_harris_window(n)

    elif wt == "NUTTALL":
        w = nuttall_window(n)

    elif wt == "BARTLETT":
        w = bartlett_window(n)

    elif wt in {"COSINE", "SINE", "RAISED_COSINE"}:
        w = cosine_window(n)

    elif wt == "FLATTOP":
        w = flattop_window(n)

    elif wt == "TUKEY":
        a = 0.5 if alpha is None else float(alpha)
        if not 0.0 <= a <= 1.0:
            raise ValueError("alpha must be in [0, 1] for Tukey window")
        w = tukey_window(n, alpha=a)

    else:
        raise ValueError(f"window_type {wt!r} not recognized")

    return w


def form_window_bandlimited(f: NDArray[np.floating],
                            bw: float,
                            window_type: str,
                            *,
                            f_cen: float = 0.0,
                            alpha: float | None = None,
                            standard_order: bool = False) -> NDArray[np.float32]:
    """
        Create a frequency-domain weighting window for bandlimited spectra.

        Builds a real-valued weighting array aligned with the provided frequency
        vector `f`. Weights are nonzero only inside the in-band region
        `[f_cen - bw/2, f_cen + bw/2]` (inclusive), and equal to the selected
        window function over the in-band bins.

        This function supports:
          - Complex baseband spectra when `f` spans negative to positive frequencies
            in monotonically increasing order (e.g., `fftshift(fftfreq)`), typically
            with `f_cen = 0`.
          - Real baseband (one-sided) spectra when `f` is produced by `rfftfreq`
            (monotonically increasing, `f >= 0`).

        Args:
            f: Frequency vector in Hz. Must be monotonically increasing.
            bw: Bandwidth in Hz. Must be positive.
            f_cen: Center frequency in Hz for the passband.
            window_type: Window name (case-insensitive). Supported values include:
                "RECTANGLE"/"NONE", "HANN"/"HANNING", "HAMMING", "BLACKMAN",
                "BLACKMAN_HARRIS"/"BLACKMAN-HARRIS"/"BH", "NUTTALL", "BARTLETT",
                "COSINE"/"SINE"/"RAISED_COSINE", "TUKEY" (uses `alpha`).
            alpha: Tukey taper fraction in [0, 1]. Only used for "TUKEY".
                Defaults to 0.5 when None.
            standard_order: If True, performs IFFTSHIFT on window before return

        Returns:
            A `np.float32` array of shape `(len(f),)` containing the weights.
            Returns all zeros if no bins fall within the requested band.

        Raises:
            ValueError: If `bw <= 0`.
            ValueError: If `f` is not monotonically increasing.
            ValueError: If the selected in-band bins are not contiguous.
            ValueError: If `window_type` is not recognized.
            ValueError: If one-sided `f` is used and the requested band extends below 0 Hz.
        """
    if bw <= 0.0:
        raise ValueError("bw must be positive")

    assert_monotonically_increasing(f)

    h_bw = 0.5 * bw
    lo = f_cen - h_bw
    hi = f_cen + h_bw

    one_sided = f.size > 0 and f[0] >= 0.0 and f[-1] >= 0.0
    if one_sided and f_cen > 0.0 and lo < 0.0:
        raise ValueError(
            "Requested band extends below 0 Hz on a one-sided frequency grid. "
            "Reduce bw or use a smaller bw relative to f_cen."
        )

    wnd = np.zeros(int(f.shape[0]), dtype=np.float32)

    idx = np.flatnonzero((f >= lo) & (f <= hi))
    if idx.size == 0:
        return wnd

    if idx.size > 1 and np.any(np.diff(idx) != 1):
        raise ValueError("Selected in-band frequency bins are not contiguous; check f ordering")

    wnd[idx] = form_window(int(idx.size), window_type, alpha=alpha)
    if standard_order:
        wnd = ifftshift(wnd)
    return wnd

########################################################################################################################
#
# Window Functions
#
########################################################################################################################

def hann_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric Hann window.

    The Hann window is a raised cosine taper defined as:

        w[n] = 0.5 * (1 - cos(2 pi n / (N - 1))),  for n = 0, ..., N-1

    This symmetric (aperiodic) form goes exactly to zero at both endpoints
    and has zero first derivative at the boundaries, making it well suited
    for time-domain tapering, FIR filtering, and matched filtering
    applications (e.g., radar range compression).

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the Hann
        window coefficients. The returned dtype is a floating-point type
        determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the *symmetric* Hann window
          (sometimes called the "aperiodic" form).
        - For `n == 1`, the function returns an array containing a single
          value of 1.0.
        - This definition matches `numpy.hanning(n)` exactly.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi*idx / (n - 1)))
    return w


def hamming_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric Hamming window.

    The Hamming window is a raised cosine taper defined as:

       w[n] = 0.54 - 0.46 * cos(2pin / (N - 1)),  for n = 0, ..., N-1

    This symmetric (aperiodic) form does not go to zero at the endpoints,
    but is designed to reduce the amplitude of the first sidelobe in the
    frequency domain relative to the Hann window.

    Args:
       n: Number of samples in the window. Must be positive.

    Returns:
       A one-dimensional NumPy array of shape `(n,)` containing the Hamming
       window coefficients. The returned dtype is a floating-point type
       determined by NumPy (typically `float32`).

    Raises:
       ValueError: If `n` is not positive.

    Notes:
       - This implementation corresponds to the *symmetric* Hamming window.
       - The window has nonzero endpoints (approximately 0.08).
       - This definition matches `numpy.hamming(n)` exactly.

    See Also:
       hann_window: Symmetric Hann window (zero endpoints).
       numpy.hamming: NumPy’s built-in Hamming window.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * idx / (n - 1))
    return w


def blackman_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric Blackman window.

    The Blackman window is a three-term cosine taper defined as:

        w[n] = 0.42
             - 0.50 * cos(2pin / (N - 1))
             + 0.08 * cos(4pin / (N - 1)),
        for n = 0, ..., N-1

    This symmetric (aperiodic) form provides stronger sidelobe suppression
    than Hann or Hamming at the cost of a wider mainlobe, and is commonly
    used to reduce spectral leakage and suppress sidelobes in FFT-based
    processing.

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the Blackman
        window coefficients. The returned dtype is a floating-point type
        determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the *symmetric* Blackman window.
        - For `n == 1`, the function returns an array containing a single
          value of 1.0.
        - This definition matches `numpy.blackman(n)`.

    See Also:
        hann_window: Symmetric Hann window.
        hamming_window: Symmetric Hamming window.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    a0 = 0.42
    a1 = 0.50
    a2 = 0.08
    x = 2.0 * np.pi * idx / (n - 1)
    w = a0 - a1 * np.cos(x) + a2 * np.cos(2.0 * x)
    return w


def blackman_harris_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric 4-term Blackman–Harris window.

    The 4-term Blackman–Harris window is defined as:

        w[n] = a0
             - a1 * cos(2pin / (N - 1))
             + a2 * cos(4pin / (N - 1))
             - a3 * cos(6pin / (N - 1)),
        for n = 0, ..., N-1

    where the coefficients are:
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168

    This symmetric (aperiodic) window provides very strong sidelobe
    suppression (≈ −92 dB for the first sidelobe) at the cost of a
    relatively wide mainlobe. It is commonly used in radar and spectral
    analysis when sidelobe contamination must be minimized.

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the
        Blackman–Harris window coefficients. The returned dtype is a
        floating-point type determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the standard *4-term*
          Blackman–Harris window described by Harris (1978).
        - The window tapers smoothly to (near) zero at both endpoints.
        - For `n == 1`, the function returns an array containing a single
          value of 1.0.

    See Also:
        blackman_window: Classic 3-term Blackman window.
        hann_window: Symmetric Hann window.
        hamming_window: Symmetric Hamming window.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    x = 2.0 * np.pi * idx / (n - 1)

    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    w = a0 - a1 * np.cos(x) + a2 * np.cos(2.0 * x) - a3 * np.cos(3.0 * x)
    return w


def nuttall_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric 4-term Nuttall window.

    This implementation uses the common 4-term Nuttall window with continuous
    first derivative, defined as:

        w[n] = a0
             - a1 * cos(2pin / (N - 1))
             + a2 * cos(4pin / (N - 1))
             - a3 * cos(6pin / (N - 1)),
        for n = 0, ..., N-1

    with coefficients:
        a0 = 0.355768
        a1 = 0.487396
        a2 = 0.144232
        a3 = 0.012604

    The Nuttall window provides very strong sidelobe suppression at the cost
    of a wider mainlobe, making it useful when sidelobe contamination must be
    minimized.

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the Nuttall
        window coefficients. The returned dtype is a floating-point type
        determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the standard 4-term Nuttall window
          with continuous first derivative.
        - For `n == 1`, the function returns an array containing a single value
          of 1.0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    x = 2.0 * np.pi * idx / (n - 1)

    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604
    w = a0 - a1 * np.cos(x) + a2 * np.cos(2.0 * x) - a3 * np.cos(3.0 * x)
    return w


def flattop_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric flat-top window.

    The flat-top window is a five-term cosine taper defined as:

        w[n] = a0
             - a1 * cos(2pin / (N - 1))
             + a2 * cos(4pin / (N - 1))
             - a3 * cos(6pin / (N - 1))
             + a4 * cos(8pin / (N - 1)),
        for n = 0, ..., N-1

    with coefficients:
        a0 = 1.00000
        a1 = 1.93000
        a2 = 1.29000
        a3 = 0.38800
        a4 = 0.02800

    The flat-top window is designed to minimize amplitude (scalloping)
    error in the frequency domain, providing very accurate amplitude
    estimates at the cost of a wide mainlobe. The flat-top window is not
    bounded in amplitude and is not suitable for matched filtering or pulse
    compression.

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the
        flat-top window coefficients. The returned dtype is a floating-
        point type determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the standard symmetric
          5-term flat-top window.
        - The mainlobe is significantly wider than Hann, Hamming,
          Blackman, or Blackman–Harris windows.
        - For `n == 1`, the function returns an array containing a
          single value of 1.0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    x = 2.0 * np.pi * idx / (n - 1)

    a0 = 1.00000
    a1 = 1.93000
    a2 = 1.29000
    a3 = 0.38800
    a4 = 0.02800
    w = a0 - a1 * np.cos(x) + a2 * np.cos(2.0 * x) - a3 * np.cos(3.0 * x) + a4 * np.cos(4.0 * x)
    return w


def bartlett_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric Bartlett (triangular) window.

    The Bartlett window is a triangular taper defined as:

        w[n] = 1 - |(2n / (N - 1)) - 1|,  for n = 0, ..., N-1

    This symmetric (aperiodic) window tapers linearly to zero at both
    endpoints and reaches a maximum of 1.0 near the center. It provides
    modest sidelobe reduction relative to a rectangular window, with a
    relatively small implementation cost.

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the Bartlett
        window coefficients. The returned dtype is a floating-point type
        determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the *symmetric* Bartlett window.
        - For even `n`, the peak occurs across the two center samples.
        - For `n == 1`, the function returns an array containing a single
          value of 1.0.
        - This definition matches `numpy.bartlett(n)`.

    See Also:
        hann_window: Raised cosine Hann window.
        numpy.bartlett: NumPy’s built-in Bartlett window.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    w = 1.0 - np.abs((2.0 * idx / (n - 1)) - 1.0)
    return w


def cosine_window(n: int) -> NDArray[np.float32]:
    """
    Generate a symmetric cosine (sine / raised cosine) window.

    The cosine window is defined as:

        w[n] = sin(pin / (N - 1)),  for n = 0, ..., N-1

    This symmetric (aperiodic) window tapers smoothly to zero at both
    endpoints and provides moderate sidelobe suppression with a relatively
    narrow mainlobe compared to Hann or Hamming windows.

    Args:
        n: Number of samples in the window. Must be positive.

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the cosine
        window coefficients. The returned dtype is a floating-point type
        determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.

    Notes:
        - This implementation corresponds to the *symmetric* cosine window.
        - The cosine window is sometimes referred to as a "sine" or
          "raised cosine" window.
        - For `n == 1`, the function returns an array containing a single
          value of 1.0.

    See Also:
        hann_window: Raised cosine Hann window.
        bartlett_window: Triangular Bartlett window.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    w = np.sin(np.pi * idx / (n - 1))
    return w


def tukey_window(n: int, alpha: float = 0.5) -> NDArray[np.float32]:
    """
    Generate a symmetric Tukey (tapered cosine) window.

    The Tukey window consists of a flat central region with cosine tapers
    applied to both ends. The shape is controlled by the parameter `alpha`:

        - alpha = 0.0  → rectangular window
        - alpha = 1.0  → Hann window
        - 0 < alpha < 1 → tapered cosine with adjustable sidelobes

    Args:
        n: Number of samples in the window. Must be positive.
        alpha: Fraction of the window length occupied by the cosine tapers.
            Must be in the range [0, 1].

    Returns:
        A one-dimensional NumPy array of shape `(n,)` containing the Tukey
        window coefficients. The returned dtype is a floating-point type
        determined by NumPy (typically `float64`).

    Raises:
        ValueError: If `n` is not positive.
        ValueError: If `alpha` is outside the range [0, 1].

    Notes:
        - This implementation corresponds to the *symmetric* Tukey window.
        - For `alpha = 0`, the function returns a rectangular window.
        - For `alpha = 1`, the function reduces exactly to a Hann window.
        - For `n == 1`, the function returns an array containing a single
          value of 1.0.

    See Also:
        hann_window: Symmetric Hann window.
        cosine_window: Simple raised cosine window.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in the range [0, 1]")
    if n == 1:
        return np.ones(1, dtype=np.float32)

    if alpha == 0.0:
        return np.ones(n, dtype=np.float32)

    if alpha == 1.0:
        idx = np.arange(n, dtype=np.float32)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / (n - 1)))

    w = np.ones(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)

    edge = alpha * (n - 1) / 2.0

    left = idx < edge
    right = idx > (n - 1) - edge

    w[left] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * idx[left] / (alpha * (n - 1)) - 1.0))
    )

    w[right] = 0.5 * (
        1.0 + np.cos(
            np.pi * (2.0 * idx[right] / (alpha * (n - 1)) - 2.0 / alpha + 1.0)
        )
    )

    return w

########################################################################################################################
#
# Bandlimited Windowing Functions
#
########################################################################################################################


def create_two_sided_bandpass_window(f: NDArray[np.floating],
                                     bw: float,
                                     f_cen: float,
                                     window_type: str,
                                     *,
                                     alpha: float | None = None,
                                     ) -> NDArray[np.float32]:
    """
    Create a frequency-domain weighting window symmetric about 0 Hz with bands at ±f_cen.

    Nonzero weights occur only within:
      [ f_cen - bw/2,  f_cen + bw/2] and
      [-f_cen - bw/2, -f_cen + bw/2].

    `f` must be a monotonically increasing, two-sided frequency grid (e.g.,
    `fftshift(fftfreq)` output). This function is not intended for one-sided
    `rfftfreq` grids.

    The returned window is intended for frequency-domain weighting of real
    baseband spectra (e.g., band-limiting or tapering a matched filter / spectrum).

    Args:
        f: Frequency vector in Hz.
        bw: Bandwidth in Hz defining the in-band region. The in-band masks are
            `f_cen - bw/2 <= f <= f_cen + bw/2` and `-f_cen - bw/2 <= f <= -f_cen + bw/2`.
            Must be positive.
        f_cen: The center frequency of the real baseband band. Should be positive.
        window_type: Window name. The value is case-insensitive and is matched
            after `.upper()` normalization. Supported values include:
            - "RECTANGLE" / "NONE"
            - "HANN" / "HANNING"
            - "HAMMING"
            - "BLACKMAN"
            - "BLACKMAN_HARRIS" / "BLACKMAN-HARRIS" / "BH"
            - "NUTTALL"
            - "BARTLETT"
            - "COSINE" / "SINE" / "RAISED_COSINE"
            - "TUKEY" (uses `alpha`)
        alpha: Tukey window taper fraction in [0, 1]. Only used when
            `window_type` is "TUKEY". If None, a default of 0.5 is used.

    Returns:
        A real-valued NumPy array of shape `(len(f),)` containing the frequency-
        domain weights. Values are zero outside the specified bands. The returned dtype
        is `np.float32`.

        If no frequency bins fall within the specified band (i.e., the mask is
        empty), the function returns an all-zeros array.

    Raises:
        ValueError: If inputs are invalid or the ± bands cannot be formed symmetrically.

    Notes:
        There is actually no reason someone would need this function since RFFT can be used to
        construct a pseudo-baseband spectra.

    """
    if bw <= 0.0:
        raise ValueError("bw must be positive")
    if f_cen == 0.0:
        raise ValueError("f_cen must be nonzero; use create_baseband_window for f_cen=0")
    if f_cen < 0.0:
        raise ValueError("f_cen must be positive for symmetric ±f_cen windows")

    assert_monotonically_increasing(f)

    if f.size == 0 or f[0] >= 0.0 or f[-1] <= 0.0:
        raise ValueError("f must span negative and positive frequencies (two-sided grid required)")

    h_bw = 0.5 * bw
    lo_p, hi_p = f_cen - h_bw, f_cen + h_bw
    lo_n, hi_n = -f_cen - h_bw, -f_cen + h_bw

    pos_idx = np.flatnonzero((f >= lo_p) & (f <= hi_p))
    neg_idx = np.flatnonzero((f >= lo_n) & (f <= hi_n))

    if pos_idx.size == 0 or neg_idx.size == 0:
        return np.zeros(int(f.shape[0]), dtype=np.float32)

    if pos_idx.size != neg_idx.size:
        raise ValueError(
            "Positive and negative band bin counts differ "
            f"(pos={pos_idx.size}, neg={neg_idx.size})."
        )

    if pos_idx.size > 1 and np.any(np.diff(pos_idx) != 1):
        raise ValueError("Positive band bins are not contiguous; check f ordering/grid")
    if neg_idx.size > 1 and np.any(np.diff(neg_idx) != 1):
        raise ValueError("Negative band bins are not contiguous; check f ordering/grid")

    # Verify mirror alignment so that W(-f) == W(f).
    tol = 10 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(f))))
    if not np.allclose(f[neg_idx], -f[pos_idx][::-1], atol=tol, rtol=0.0):
        raise ValueError("positive and negative bands are not mirror-aligned; cannot form a symmetric window")

    w = form_window(int(pos_idx.size), window_type, alpha=alpha)

    wnd = np.zeros(int(f.shape[0]), dtype=np.float32)
    wnd[pos_idx] = w
    wnd[neg_idx] = w[::-1]
    return wnd





