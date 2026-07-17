# SPDX-License-Identifier: BSD-3-Clause
"""Contrast-method ionospheric compensation for radar sounder data.

This module implements the contrast technique described by
Picardi & Sorge (2000) and refined by Cartacci et al. (2013) for
correcting ionospheric dispersion in orbital radar sounder observations.
The algorithm performs a grid search over the second-order dispersion
coefficient a2, deriving the higher-order terms a3 and a4 analytically,
and selects the candidate that minimizes the L1 norm of the reconstructed
time-domain signal within a sliding neighborhood of records.

All spectral operations are expressed in terms of offset frequencies
(f - f0) so that the correction is valid for data processed at either
complex baseband or real baseband.

References:
    Picardi, G., & Sorge, S. (2000, April). Adaptive compensation of
    ionosphere dispersion to improve subsurface detection capabilities
    in low-frequency radar systems. In Eighth International Conference
    on Ground Penetrating Radar (Vol. 4084, pp. 624-629). SPIE.

    Cartacci, M., et al. (2013). Mars ionosphere total electron content
    analysis from MARSIS subsurface data. Icarus, 223(1), 423-437.
"""
import numpy as np
from numpy.typing import NDArray

from scipy.fft import fft, ifft, ifftshift, fftfreq


from ...constants import C, CAMPBELL_DELAY_COEFF, CAMPBELL_TEC_COEFF, PLASMA_FREQ_COEFF
from ..utils import check_monotonically_increasing, select_iband
from ..windows import form_window
from .metrics import METRICS



def contrast_method(ts: NDArray[np.complexfloating],
                    dt: float,
                    bw: float,
                    f_cen: float | NDArray[np.floating] | None,
                    n_take: int,
                    wnd: str = "hanning",
                    alpha: float | None = None,
                    L_eq: float = 80e3,
                    n_grid: int = 801,
                    *,
                    metric: str | None = None,
                    ) -> tuple[
                            NDArray[np.complex64],
                            NDArray[np.float32],
                            NDArray[np.float32],
                            NDArray[np.float32],
                            NDArray[np.floating],
                    ]:
    """
    Apply the Contrast Method ionospheric compensation.

    This implements a grid-search over dispersion parameter ``a2`` for a subset
    of columns, using a stacked metric computed from a local neighborhood of
    records (columns) that share the same center frequency ``f_cen``. The best
    candidate is selected by minimizing a concentration metric based on the L1
    norm of the reconstructed time series.

    The solution (a2/a3/a4) is then applied across the full observation in the
    frequency domain.

    Args:
        ts: Range compressed time series (complex) with shape (n_samp, n_col).
            Axis 0 is fast-time, axis 1 is record/along-track index.
        dt: Sample spacing in seconds.
        bw: Bandwidth in Hz. The algorithm uses in-band bins (``bw/2``).
        f_cen: Physical RF center frequency in Hz used by the dispersion
            polynomial. Either a scalar (applied to all columns) or a 1D array
            of length n_col so the carrier can vary along-track. Values are
            assumed to be truly discrete.
        n_take: Size of the stacking neighborhood. For a solution evaluated at
            column ``col``, the algorithm attempts to use +/-(n_take//2) records
            around ``col``, but truncates the stack to remain within the largest
            contiguous region (around ``col``) that shares the same ``f_cen``.
        wnd: Window name passed to ``form_window`` (e.g., "hanning").
        alpha: Tukey taper fraction in [0, 1]. Only used when
            ``wnd`` is ``"TUKEY"``. Defaults to 0.5 when None.
        L_eq: Equivalent path length scale (meters). Used to compute tau_0.
        n_grid: Number of grid points in the a2 search. Must be odd.
        metric: Autofocus metric name from :data:`METRICS`. Defaults to
            ``"L4"`` (whole-record fourth-power concentration; current
            behavior). Cartacci et al. (2013) originally use L1
            minimization (``"L1"``); ``"PEAK_SNR"`` mirrors the SHARAD
            PDS ``focus_array`` routine.

    Returns:
        A tuple of:

        - ts_out: Corrected time series (complex) with the autofocus phase
            removed, shape (n_samp, n_col), complex64. The bulk ionospheric
            delay is **not** applied here — ``delay_cells`` carries it as
            metadata for the downstream datuming step.
        - a2s: Estimated a2 per column, shape (n_col,), float32.
        - a3s: Estimated a3 per column, shape (n_col,), float32.
        - a4s: Estimated a4 per column, shape (n_col,), float32.
        - delay_cells: Per-column bulk delay (range cells) derived from
            ``a2`` via the Level 1 Cartacci 2013 + Campbell 2014 path.
            Returned for the downstream datuming routine to apply at
            SAR-frame resolution; not applied to ``ts_out``. Shape
            (n_col,), float64.

    Raises:
        ValueError: If input shapes are inconsistent or ``f_cen`` has
            invalid length.

    Notes:
        - The phase model is expressed in terms of offset frequencies
          (f - f_cen). The polynomial evaluation is correct
          regardless of whether the input was processed at complex
          baseband or real baseband — what matters is that ``f_cen``
          names the true RF carrier.

    References:
        Picardi, G., & Sorge, S. (2000, April). Adaptive compensation of
        ionosphere dispersion to improve subsurface detection capabilities
        in low-frequency radar systems. In Eighth International Conference
        on Ground Penetrating Radar (Vol. 4084, pp. 624-629). SPIE.

        Cartacci, M., et al. (2013). Mars ionosphere total electron content
        analysis from MARSIS subsurface data. Icarus, 223(1), 423-437.

    """
    ts = np.asarray(ts)
    if ts.ndim != 2:
        raise ValueError(f"Input time series must be a 2D array")
    if not np.issubdtype(ts.dtype, np.complexfloating):
        raise ValueError("Input time series must be complex.")
    if n_grid < 3 or n_grid % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 3")

    metric_name = (metric or "PEAK_SNR").upper()
    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; valid options: {sorted(METRICS)}"
        )
    metric_fn, metric_dir = METRICS[metric_name]

    n_samp, n_col = ts.shape

    # FFT along fast-time
    spectra = fft(ts, axis=0, workers=-1).astype(np.complex64, copy=False)
    spec_samp = spectra.shape[0]

    freqs = fftfreq(n_samp, d=dt).astype(np.float32)
    # !!!Input data is assumed to be at complex baseband!!!
    iband = select_iband(freqs, 0.0, bw)

    f_diff = freqs[iband]       # (n_band,)
    spec_in = spectra[iband]    # (n_band, n_col)
    n_band = f_diff.size

    window = form_window(n_band, wnd, alpha=alpha)
    if not check_monotonically_increasing(f_diff):
        window = ifftshift(window)

    if np.isscalar(f_cen):
        f_cen_arr = np.full(n_col, float(f_cen), dtype=np.float32)
    else:
        f_cen_arr = np.asarray(f_cen, dtype=np.float32)
    if f_cen_arr.ndim != 1 or f_cen_arr.size != n_col:
        raise ValueError(
            "f_phys must be a scalar or a 1D array of length n_col.")

    half_take = n_take // 2

    #
    # Outputs
    #
    ks = np.zeros(n_col, dtype=np.uint32)
    a2s = np.zeros(n_col, dtype=np.float32)
    a3s = np.zeros(n_col, dtype=np.float32)
    a4s = np.zeros(n_col, dtype=np.float32)
    spec_in_w = np.zeros_like(spec_in)

    tau_0 = np.float32(2 * L_eq / C)

    spec_full = np.zeros((spec_samp, n_grid), dtype=np.complex64)
    metric = np.zeros(n_grid, dtype=np.float32)

    a2_prev = 0.0

    f_in = f_diff
    f2 = (f_in * f_in).astype(np.float32, copy=False)
    f3 = (f2 * f_in).astype(np.float32, copy=False)
    f4 = (f2 * f2).astype(np.float32, copy=False)

    # Main Loop
    for col in range(0, n_col, half_take):
        # Base window bounds
        lo0 = max(0, col - n_take)
        hi0 = min(n_col, col + n_take + 1)

        f_p = np.float32(f_cen_arr[col])

        # Truncate to contiguous same-f_phys region around col
        lo = col
        while lo - 1 >= lo0 and f_cen_arr[lo - 1] == f_p:
            lo -= 1

        hi = col + 1
        while hi < hi0 and f_cen_arr[hi] == f_p:
            hi += 1

        # Window the chunk
        chnk = spec_in[:, lo:hi]
        chnk_w = (chnk * window[:, None]).astype(np.complex64, copy=False)

        # Grid Set up
        if col == 0:
            delta_a2 = np.float32(0.1 * (2 * np.pi / bw ** 2))
            a2_start = 0.0  # Initial Guess
        else:
            delta_a2 = np.float32(0.01 * (2 * np.pi / bw ** 2))
            a2_start = a2_prev  # Initial Guess

        k = np.arange(n_grid, dtype=np.float32)

        a2 = a2_start + (k - (n_grid / 2)) * delta_a2
        a3 = -(a2 / f_p) * (1 - (a2 * f_p) / (np.pi * tau_0))
        a4 = (a2 / f_p ** 2) * (1 - (a2 * f_p) / (0.5 * np.pi * tau_0))

        delta_phi = (a2[None, :] * f2[:, None] +
                     a3[None, :] * f3[:, None] +
                     a4[None, :] * f4[:, None]).astype(np.float32, copy=False)

        phase = np.exp(1j * delta_phi).astype(np.complex64, copy=False)

        metric.fill(0.0)

        # Accumulate the per-grid-point metric across records in the chunk.
        for rec in range(lo, hi):
            trace_w = chnk_w[:, rec - lo]
            spec_full[iband, :] = trace_w[:, None] * phase
            ts_test = ifft(spec_full, axis=0, workers=-1)
            metric += np.asarray(metric_fn(ts_test), dtype=np.float32)
        k_opt = int(np.argmax(metric) if metric_dir == "maximize" else np.argmin(metric))

        fill_hi = min(col + half_take, hi)
        ks[col:fill_hi] = k_opt
        a2s[col:fill_hi] = a2[k_opt]
        a3s[col:fill_hi] = a3[k_opt]
        a4s[col:fill_hi] = a4[k_opt]
        a2_prev = a2[k_opt]
        spec_in_w[:, lo:hi] = chnk_w
    #
    # Once finished, apply the adjustment
    #
    delta_phi_f = (a2s * f2[:, None] +
                   a3s * f3[:, None] +
                   a4s * f4[:, None]).astype(np.float32, copy=False)

    phase_all = np.exp(1j * delta_phi_f).astype(np.complex64, copy=False)
    spec_out = np.zeros_like(spectra)
    spec_out[iband, :] = spec_in_w * phase_all
    ts_out = ifft(spec_out, axis=0, workers=-1).astype(np.complex64, copy=False)

    # Compute (but do not apply) the Level 1 bulk delay (Cartacci 2013
    # TEC formula + Campbell 2014 empirical TEC→delay). For each column,
    # |a2| → TEC → equivalent SHARAD E → delay seconds. Known biased
    # ~10–20% on the dayside. The downstream datuming step is responsible
    # for smoothing at SAR-frame resolution and applying it.
    #
    # TODO(Cartacci 2018): Replace with the simulator-coupled scheme that
    #   recovers the linear term a1 from cross-correlation against a CSIM
    #   reference. Requires processing-step ordering to allow CSIM before
    #   ionospheric correction.
    delay_cells = _contrast_delay_cells(a2s, f_cen_arr, dt)

    return ts_out, a2s, a3s, a4s, delay_cells


def _contrast_delay_cells(a2: NDArray[np.floating],
                          f_phys: NDArray[np.floating],
                          dt: float,
                          ) -> NDArray[np.floating]:
    """Per-column ionospheric delay in range cells from Contrast a2.

    Level 1 path: convert |a2| to TEC via Cartacci 2013 Eq. 7, convert
    TEC to a SHARAD-equivalent E via Campbell 2011's calibration
    (TEC ≈ 0.29·E), then use Campbell 2014's empirical TEC→delay slope.
    """
    a2_abs = np.abs(np.asarray(a2, dtype=np.float64))
    f = np.asarray(f_phys, dtype=np.float64)
    tec = a2_abs * C * f**3 / (2.0 * np.pi * PLASMA_FREQ_COEFF**2)
    e_equiv = tec / CAMPBELL_TEC_COEFF
    return (e_equiv / 1.0e16) * CAMPBELL_DELAY_COEFF / dt


