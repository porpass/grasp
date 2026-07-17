# SPDX-License-Identifier: BSD-3-Clause
"""Campbell-method ionospheric compensation for radar sounder data.

This module implements the autofocus technique described by Campbell et al.
(2011, 2021) for correcting ionospheric phase distortion in orbital radar
sounder observations. The algorithm performs a grid search over candidate
phase corrections, selecting the one that minimizes the L1 norm of the
reconstructed time-domain signal within a sliding neighborhood of records.

The phase model uses a power-law relationship between frequency and
ionospheric phase delay, parameterized by a single scalar (the E-value)
per along-track segment.

All spectral operations are expressed in terms of offset frequencies
(f - f0) so that the correction is valid for data processed at either
complex baseband or real baseband.

References:
    Campbell, B. A. et al. (2011). Autofocus correction of phase distortion
    effects on SHARAD echoes. IEEE Geoscience and Remote Sensing Letters,
    8(5), 939–942.

    Campbell, B. A. et al. (2021). Calibration of Mars Reconnaissance Orbiter
    Shallow Radar (SHARAD) data for subsurface probing and surface
    reflectivity studies. Icarus, 360, 114358.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Literal

from scipy.fft import fft, fftfreq, ifft, ifftshift

from ..utils import check_monotonically_increasing, select_iband
from ..windows import form_window
from ...constants import CAMPBELL_DELAY_COEFF
from .metrics import METRICS



def make_phase_array(freqs: NDArray[np.floating],
                     *,
                     n_phase: int = 400,
                     delta: float = 0.0125,
                     b: float = 1.93,
                     sgn: Literal[-1, 1] = 1,
                     ) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """
    Construct phase adjustment array for the Campbell method.

    Builds a phase-correction lookup table for ionospheric compensation as
    described by Campbell et al. The output ordering follows the ordering
    of `f_diff`: if `f_diff` is monotonically increasing ("natural order"),
    the result is returned in that order; otherwise the function assumes the
    input is FFT-shifted ("standard order") and returns in standard order.

    Args:
        freqs: Emitted frequencies (Hz)
        n_phase: Number of phase states (columns) to generate.
        delta: Step size used to generate E-values (unitless scale factor).
        b: Power-law exponent.
        sgn: Sign convention for the phase exponential. Must be -1 or +1.

    Returns:
        A tuple `(phases, E_values)` where:

        - phases: Complex phase correction array with shape (n_freq, n_phase).
        - E_values: 1D array of length `n_phase`.

    Raises:
        ValueError: If `f_diff` is not 1D, if `n_phase` <= 0, or if `sgn`
            is not ±1.

    Notes:
        This implementation fits a cubic polynomial to the adjustment curve
        for each phase column and removes the constant and linear terms
        (intercept and slope) before forming the complex exponential.

    References:
        Campbell, B. A. et al. (2011). Autofocus correction of phase distortion
        effects on SHARAD echoes. IEEE Geoscience and Remote Sensing Letters,
        8(5), 939–942.

        Campbell, B. A. et al. (2021). Calibration of Mars Reconnaissance Orbiter
        Shallow Radar (SHARAD) data for subsurface probing and surface
        reflectivity studies. Icarus, 360, 114358.
    """
    if n_phase <= 0:
        raise ValueError("n_phase must be positive")
    if sgn not in (-1, 1):
        raise ValueError("sgn must be -1 or +1")

    freqs = np.asarray(freqs)
    if freqs.ndim != 1:
        raise ValueError(f"f_diff must be 1D, got shape {freqs.shape}")

    natural_order = check_monotonically_increasing(freqs)
    if not natural_order:
        freqs = np.fft.fftshift(freqs)

    if np.any(freqs <= 0.0):
        raise ValueError("f_0 + f_diff must be positive for non-integer exponent b")

    n_freq = freqs.size
    freqs2 = freqs.reshape(n_freq, 1)  # (n_freq, 1)

    e_values = (np.arange(n_phase, dtype=np.float32) * np.float32(delta)) * np.float32(1e16)

    # Compute adjustment in float64 for stability, then store float32.
    adj64 = e_values.astype(np.float64) / np.power(freqs2, float(b))  # (n_freq, n_phase), float64
    adj = adj64.astype(np.float32)

    for col in range(n_phase):
        t = adj64[:, col]  # float64 view
        p = np.polyfit(freqs, t, 1)
        adj[:, col] = (t - p[1] - p[0] * freqs).astype(np.float32)

    phases = np.exp((1j * sgn) * adj).astype(np.complex64, copy=False)

    if not natural_order:
        phases = np.fft.ifftshift(phases, axes=0)

    return phases, e_values


def campbell_method(
        ts: NDArray[np.complexfloating],
        dt: float,
        bw: float,
        f_cen: float | NDArray[np.floating] | None,
        n_take: int,
        b: float,
        n_phase: int,
        delta: float,
        *,
        sgn: Literal[-1, 1] = 1,
        wnd: str = "hanning",
        alpha: float | None = None,
        metric: str | None = None,
    ) -> tuple[
        NDArray[np.complexfloating],
        NDArray[np.floating],
        NDArray[np.complexfloating],
        NDArray[np.floating],
    ]:
    """Apply Campbell-method ionospheric compensation to a 2D time series.

    Args:
        ts: Complex time series with shape (n_samp, n_col). Rows are fast-time,
            columns are slow-time/records.
        dt: Sample spacing in seconds.
        bw: Processed bandwidth in Hz. In band selection (``bw/2``).
        f_cen: Physical RF center frequency in Hz used by the dispersion
            power-law. Either a scalar (applied to all columns) or a 1D
            array of length n_col so the carrier can vary along-track.
        n_take: Chunking parameter controlling neighborhood width. Must be >= 2.
        b: Power-law exponent for the Campbell phase model. (SHARAD: 1.93; MARSIS: 2.167)
        n_phase: Number of phase states to test.
        delta: Step size used to generate E-values (unitless scale factor).
        sgn: Sign convention for phase exponential (-1 or +1).
        wnd: Spectral window type (passed to `form_window`).
        alpha: Tukey taper fraction in [0, 1]. Only used when
            ``wnd`` is ``"TUKEY"``. Defaults to 0.5 when None.
        metric: Autofocus metric name from :data:`METRICS`. Defaults to
            ``"PEAK_SNR"`` (mirrors the SHARAD PDS ``focus_array`` routine).
            Other options: ``"L4"`` (whole-record fourth-power
            concentration), ``"L1"`` (minimization), ``"ENTROPY"``
            (minimization).

    Returns:
        A tuple `(ts_out, e_values, iono_phase, delay_cells)` where:

        - ts_out: Corrected complex time series with the autofocus phase
            removed, shape (n_samp, n_col). The bulk ionospheric delay is
            **not** applied here — ``delay_cells`` carries it as metadata
            for the downstream datuming step.
        - e_values: Optimal E value per column, shape (n_col,).
        - iono_phase: Complex phase correction in frequency domain,
            shape (n_samp, n_col).
        - delay_cells: Per-column bulk delay (range cells) computed from
            the SHARAD-empirical Campbell 2014 calibration. Shape (n_col,).
            Returned for the downstream datuming routine to apply at SAR-frame
            resolution; not applied to ``ts_out``.

    Raises:
        ValueError: If `ts` is not 2D complex, if parameters are invalid, or if
            `f_cen` has wrong shape.

    References:
        Campbell, B. A. et al. (2011). Autofocus correction of phase distortion
        effects on SHARAD echoes. IEEE Geoscience and Remote Sensing Letters,
        8(5), 939–942.

        Campbell, B. A. et al. (2021). Calibration of Mars Reconnaissance Orbiter
        Shallow Radar (SHARAD) data for subsurface probing and surface
        reflectivity studies. Icarus, 360, 114358.

    """
    ts_arr = np.asarray(ts)
    if ts_arr.ndim != 2:
        raise ValueError("ts must be a 2D array with shape (n_samp, n_col)")
    if not np.issubdtype(ts_arr.dtype, np.complexfloating):
        raise ValueError("ts must be a complex array")

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if bw <= 0.0:
        raise ValueError("bw must be positive")
    if n_phase <= 0:
        raise ValueError("n_phase must be positive")
    if n_take < 2:
        raise ValueError("n_take must be >= 2")
    if sgn not in (-1, 1):
        raise ValueError("sgn must be -1 or +1")

    metric_name = (metric or "PEAK_SNR").upper()
    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; valid options: {sorted(METRICS)}"
        )
    metric_fn, metric_dir = METRICS[metric_name]

    n_samp, n_col = ts_arr.shape
    half_take = n_take // 2
    if half_take <= 0:
        raise ValueError("n_take is too small; results in zero step size")

    # FFT along fast-time (rows).
    spectra = fft(ts_arr, axis=0, workers=-1).astype(np.complex64, copy=False)

    freqs = fftfreq(n_samp, d=dt).astype(np.float32, copy=False)
    #
    # !!!Assumes the data is at complex baseband!!!
    #
    iband = select_iband(freqs, 0.0, bw)

    f_diff = freqs[iband].astype(np.float32, copy=False)       # (n_band,)
    spec_in = spectra[iband, :]                               # (n_band, n_col)
    n_band = f_diff.size
    window = form_window(n_band, wnd, alpha=alpha)
    if not check_monotonically_increasing(f_diff):
        window = ifftshift(window)
    # Normalize f_phys to 1D float32 array of length n_col.
    if np.isscalar(f_cen):
        f_cen_arr = np.full(n_col, float(f_cen), dtype=np.float32)
    else:
        f_cen_arr = np.asarray(f_cen, dtype=np.float32)
    if f_cen_arr.ndim != 1 or f_cen_arr.size != n_col:
        raise ValueError(
            "f_phys must be a scalar or a 1D array of length n_col")

    # Outputs
    iono_phase = np.zeros((n_samp, n_col), dtype=np.complex64)
    e_values = np.zeros(n_col, dtype=np.float32)
    spec_in_w = np.zeros_like(spec_in)

    # Initial phase array for the first column's f_phys.
    f_cen_prev = float(f_cen_arr[0])
    f_abs = f_cen_prev + f_diff
    phase_array, phase_values = make_phase_array(f_abs, n_phase=n_phase,
                                                 b=b, delta=delta, sgn=sgn)

    # Iterate over columns in steps of half_take.
    for col in range(0, n_col, half_take):
        f_p = float(f_cen_arr[col])
        if f_p != f_cen_prev:
            # Only the physical carrier changed; f_diff (band geometry) is unchanged.
            f_abs = f_p + f_diff
            phase_array, phase_values = make_phase_array(f_abs, n_phase=n_phase, b=b, delta=delta, sgn=sgn)
            f_cen_prev = f_p

        # Neighborhood bounds (clipped).
        lo0 = max(0, col - n_take)
        hi0 = min(n_col, col + n_take + 1)

        # Expand within neighborhood while f_phys is constant.
        lo = col
        while lo - 1 >= lo0 and float(f_cen_arr[lo - 1]) == f_p:
            lo -= 1

        hi = col + 1
        while hi < hi0 and float(f_cen_arr[hi]) == f_p:
            hi += 1

        chnk = spec_in[:, lo:hi]
        chnk_w = (chnk * window[:, None]).astype(np.complex64, copy=False)
        chnk_col = chnk_w.shape[1]

        spec_test = np.zeros((n_samp, chnk_col), dtype=np.complex64)

        # Autofocus metric: scalarize the per-column metric across the
        # chunk and pick the optimum per the metric's direction.
        if metric_dir == "maximize":
            best_tp = -np.inf
            better = lambda new, old: new > old
        else:
            best_tp = np.inf
            better = lambda new, old: new < old
        best_m = 0
        for m in range(n_phase):
            spec_test[iband, :] = chnk_w * phase_array[:, m][:, None]
            ts_test = ifft(spec_test, axis=0, workers=-1)
            tp = float(np.sum(metric_fn(ts_test)))
            if better(tp, best_tp):
                best_tp = tp
                best_m = m
        iono_phase[iband, lo:hi] = phase_array[:, best_m][:, None]
        e_values[lo:hi] = phase_values[best_m]
        spec_in_w[:, lo:hi] = chnk_w

    # Apply the autofocus phase correction.
    spec_out = np.zeros_like(spectra)
    spec_out[iband, :] = spec_in_w
    spec_out *= iono_phase
    ts_out = ifft(spec_out, axis=0, workers=-1).astype(np.complex64, copy=False)

    # Compute (but do not apply) the SHARAD-empirical bulk delay from
    # Campbell 2014/2016. The downstream datuming step is responsible
    # for smoothing this at SAR-frame resolution and applying it to the
    # data.
    delay_cells = _campbell_delay_cells(e_values, dt)

    return ts_out, e_values, iono_phase, delay_cells


def _campbell_delay_cells(e_values: NDArray[np.floating],
                          dt: float,
                          ) -> NDArray[np.floating]:
    """Per-column ionospheric delay in range cells from Campbell E.

    Uses the SHARAD-empirical calibration from Campbell et al. 2014
    (refined 2016): ``delay_seconds = (E / 1e16) * CAMPBELL_DELAY_COEFF``.
    Accurate to <50 m RMS for SHARAD; reused for MARSIS as a Level 1
    approximation pending Cartacci 2018 simulator-coupled refinement.
    """
    return (np.asarray(e_values) / 1.0e16) * CAMPBELL_DELAY_COEFF / dt


