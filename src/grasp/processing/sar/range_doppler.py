# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftshift, ifft
from scipy.ndimage import map_coordinates
from typing import Literal

from ..windows import form_window, broadening_factor
from .utils import determine_aperture_bounds, determine_aperture_resolution, determine_aperture_step, aperture_range


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _coerce_arrays(n_col: int,
                   del_x: float | NDArray[np.floating],
                   Vt: float | NDArray[np.floating],
                   Vr: float | NDArray[np.floating],
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Broadcast scalar or 1-element inputs to full-length arrays.

    Args:
        n_col: Number of azimuth columns.
        del_x: Along-track sample spacing (m).
        Vt: Tangential velocity (m/s).
        Vr: Radial velocity (m/s).

    Returns:
        del_x, Vt, Vr as 1D float64 arrays of length n_col.
    """
    del_x = np.atleast_1d(np.asarray(del_x, dtype=np.float64))
    Vt = np.atleast_1d(np.asarray(Vt, dtype=np.float64))
    Vr = np.atleast_1d(np.asarray(Vr, dtype=np.float64))
    if del_x.size == 1:
        del_x = np.full(n_col, del_x[0])
    if Vt.size == 1:
        Vt = np.full(n_col, Vt[0])
    if Vr.size == 1:
        Vr = np.full(n_col, Vr[0])
    return del_x, Vt, Vr


def _resolve_aperture_length(L_n: int | None,
                             D: float | None,
                             lamb: float,
                             R_st: NDArray[np.floating],
                             del_x_mean: float,
                             n_col: int,
                             ) -> int:
    """Determine the synthetic aperture length in samples.

    If L_n is provided, ensures it is odd and clamped to the data width.
    If L_n is None, derives it from the antenna length D.

    Args:
        L_n: User-specified aperture length, or None.
        D: Real antenna length (m). Required when L_n is None.
        lamb: Radar wavelength (m).
        R_st: Spacecraft-to-target vectors, shape (3, n_azimuth).
        del_x_mean: Mean along-track spacing (m).
        n_col: Number of azimuth columns.

    Returns:
        L_n as an odd integer clamped to the data width.

    Raises:
        ValueError: If L_n is None and D is not provided.
    """
    if L_n is None:
        if D is None:
            raise ValueError(
                "D (antenna length) is required when L_n is not provided"
            )
        R0 = np.linalg.norm(R_st, axis=0).mean()
        L_max = lamb * R0 / D
        L_n = int(L_max / del_x_mean) + 1

    if L_n % 2 == 0:
        L_n += 1
    if L_n > n_col:
        L_n = n_col - 1 if n_col % 2 == 0 else n_col

    return L_n


def _fill_aperture(out_idx: int,
                   aperture_bounds: list[tuple[int, int, int, int]],
                   buffers: dict[str, NDArray],
                   sources: dict[str, NDArray],
                   ) -> None:
    """Zero-fill aperture buffers and copy the source data for one frame.

    Args:
        out_idx: Current output frame index.
        aperture_bounds: Pre-computed list of (src_start, src_end,
            dest_start, dest_end) tuples.
        buffers: Dict mapping names to pre-allocated aperture arrays.
            Each is zeroed then filled.
        sources: Dict mapping the same names to the full-track arrays
            from which data is sliced.
    """
    src_start, src_end, dest_start, dest_end = aperture_bounds[out_idx]
    for name, buf in buffers.items():
        buf[:] = 0
        src = sources[name]
        if buf.ndim == 1:
            buf[dest_start:dest_end] = src[src_start:src_end]
        else:
            buf[:, dest_start:dest_end] = src[:, src_start:src_end]


def _remove_doppler_centroid(s_ap: NDArray[np.complexfloating],
                             f_dc_value: float,eta: NDArray[np.floating],
                             ) -> None:
    """Demodulate the Doppler centroid from the aperture data in-place.

    Args:
        s_ap: Aperture data, shape (n_range, L_n). Modified in-place.
        f_dc_value: Doppler centroid frequency (Hz) for this frame.
        eta: Azimuthal slow-time axis (s), shape (L_n,).
    """
    demod = np.exp(-2j * np.pi * f_dc_value * eta[None, :])
    s_ap *= demod


def _rcmc_interpolate(data: NDArray[np.complexfloating],
                      rcmc_bins: NDArray[np.floating],
                      interp_cval: float = 0.0,
                      ) -> NDArray[np.complexfloating]:
    """Apply range cell migration correction via interpolation.

    Shifts each azimuth column in the range direction by a fractional
    number of bins using cubic interpolation.

    Args:
        data: Complex 2D array, shape (n_range, L_n). Can be in the
            time domain or range-Doppler domain.
        rcmc_bins: Range shift per azimuth column in bins, shape (L_n,).
        interp_cval: Fill value for out-of-bounds samples.

    Returns:
        Interpolated complex array with RCMC applied, same shape as data.
    """
    n_range, L_n = data.shape
    range_bins = np.arange(n_range)

    new_pos = range_bins[:, None] + rcmc_bins[None, :]
    coords = np.zeros((2, n_range, L_n))
    coords[0] = new_pos
    coords[1] = np.arange(L_n)[None, :]

    return (
        map_coordinates(data.real, coords, order=3,
                        mode='constant', cval=interp_cval)
        + 1j * map_coordinates(data.imag, coords, order=3,
                               mode='constant', cval=interp_cval)
    )


# ---------------------------------------------------------------------------
# Public processors
# ---------------------------------------------------------------------------

def backscatter(s_t: NDArray[np.complexfloating],
                *,
                R_s: NDArray[np.floating],
                R_t: NDArray[np.floating],
                R_st: NDArray[np.floating],
                Vt: float | NDArray[np.floating],
                Vr: float | NDArray[np.floating],
                del_x: float | NDArray[np.floating],
                lamb: float,
                pri: float,
                dz: float,
                D: float | None = None,
                L_n: int | None = None,
                os_factor: int = 1,
                window_type: str = "hanning",
                window_alpha: float | None = None,
                n_mlk: int = 5,
                remove_doppler_centroid: bool = False,
                interp_cval: float = 0.0,
                sgn: Literal[-1,1] = 1,
                verbose: bool = False,
                ) -> tuple[NDArray[np.float32], NDArray[np.intp], int, NDArray[np.float64], NDArray[np.float64]]:
    """Range-Doppler backscatter processor with multilook detection.

    Produces a multilooked power image from range-compressed SAR data
    using range-Doppler processing with geometric range cell migration
    correction, phase compensation, and non-coherent multilook averaging.

    Args:
        s_t: Range-compressed complex data, shape (n_range, n_azimuth).
        R_s: Spacecraft position vectors, shape (3, n_azimuth).
        R_t: Target position vectors, shape (3, n_azimuth).
        R_st: Spacecraft-to-target vectors, shape (3, n_azimuth).
        Vt: Tangential velocity (m/s). Scalar or 1D array of length n_azimuth.
        Vr: Radial velocity (m/s). Scalar or 1D array of length n_azimuth.
        del_x: Along-track sample spacing (m). Scalar or 1D array of
            length n_azimuth.
        lamb: Radar wavelength (m).
        pri: Pulse repetition interval (s).
        dz: Range bin spacing (m).
        D: Real antenna length (m). Required when L_n is not provided.
        L_n: Synthetic aperture length in samples. If None, derived from D.
        os_factor: Oversampling factor for output grid (1 = Nyquist).
        window_type: Azimuth window type as accepted by form_window
            (default "hanning").
        window_alpha: Tukey window taper fraction. Only used when
            window_type is "tukey".
        n_mlk: Number of multilook Doppler bins. Forced odd internally.
        remove_doppler_centroid: If True, demodulate the Doppler centroid
            before azimuth compression (default False).
        interp_cval: Fill value for out-of-bounds samples during RCMC
            interpolation (default 0.0).
        sgn: Sign convention for phase exponential (-1 or +1).
        verbose: If True, print processing diagnostics (default False).

    Returns:
        s_foc: Detected (power) image, shape (n_range, n_output_frames).
        frames: Center frame indices into the original azimuth axis.
        aperture_step: Step size used between apertures.
        rho_a: Single-look azimuth resolution at each output frame (m).
        rho_df: Multilooked azimuth resolution at each output frame (m).

    Raises:
        ValueError: If neither L_n nor D is provided.
    """
    n_samp, n_col = s_t.shape

    # --- Array setup ---
    del_x, Vt, Vr = _coerce_arrays(n_col, del_x, Vt, Vr)
    del_x_mean = del_x.mean()

    # --- Aperture length ---
    L_n = _resolve_aperture_length(L_n, D, lamb, R_st, del_x_mean, n_col)
    L_sa = L_n * del_x
    L_t = L_n * pri
    c_frame = L_n // 2

    # --- Multilook setup ---
    if n_mlk <= 1:
        n_mlk = 1
        L_df = L_sa.copy() if isinstance(L_sa, np.ndarray) else L_sa
    else:
        if n_mlk % 2 == 0:
            n_mlk += 1
        n_mlk_2 = n_mlk // 2
        b = n_mlk_2 / L_t
        L_df = Vt / (2 * b)

    start_keep = (L_n - n_mlk) // 2
    end_keep = start_keep + n_mlk

    # --- Azimuth window ---
    azi_wnd = form_window(L_n, window_type, alpha=window_alpha)

    # --- Resolution ---
    R_st_norms = np.linalg.norm(R_st, axis=0)
    rho_a = determine_aperture_resolution(L_sa, lamb, R_st_norms, brd=1.0)
    rho_df = determine_aperture_resolution(L_df, lamb, R_st_norms, brd=1.0)

    # --- Step size and output grid ---
    aperture_step = determine_aperture_step(rho_df, del_x, os_factor)
    n_output_frames = (n_col + aperture_step - 1) // aperture_step

    if verbose:
        print(f"Synthetic Aperture Samples: {L_n}")
        print(f"\tSynthetic Aperture Length: {L_sa.mean() / 1e3:.2f} km")
        print(f"\tSynthetic Aperture Time: {L_t:.2f} s")
        print(f"\tNominal resolution: {rho_a.mean():.1f} m")
        print(f"Number of Multilooks: {n_mlk}")
        print(f"\tFinal resolution: {rho_df.mean():.1f} m")
        print(f"Oversampling Factor: {os_factor}")
        print(f"Aperture Step: {aperture_step}")
        print(f"Number of Output Frames: {n_output_frames}")

    # --- Precompute aperture bounds ---
    frames, aperture_bounds = determine_aperture_bounds(L_n, n_col, aperture_step)

    # --- Doppler centroid ---
    f_dc = sgn*-2 * Vr / lamb

    # --- Azimuthal time axis ---
    eta = (np.arange(L_n) - L_n // 2) * pri
    range_bins = np.arange(n_samp)

    # --- Output array ---
    s_foc = np.zeros((n_samp, n_output_frames), dtype=np.float32)

    # --- Aperture buffers ---
    s_ap = np.zeros((n_samp, L_n), dtype=np.complex64)
    R_t_ap = np.zeros((3, L_n), dtype=np.float64)
    R_s_ap = np.zeros((3, L_n), dtype=np.float64)

    buffers = {"s": s_ap, "R_t": R_t_ap, "R_s": R_s_ap}
    sources = {"s": s_t, "R_t": R_t, "R_s": R_s}

    # --- Processing loop ---
    for out_idx, frame in enumerate(frames):
        if verbose:
            print(f"Processing frame {frame + 1}/{n_col}, "
                  f"output column {out_idx + 1}/{n_output_frames}")

        # Fill aperture
        _fill_aperture(out_idx, aperture_bounds, buffers, sources)

        # Aperture center geometry
        G_0 = R_t_ap[:, c_frame]

        # --- Doppler centroid removal ---
        if remove_doppler_centroid:
            _remove_doppler_centroid(s_ap, f_dc[frame], eta)

        # --- Range Cell Migration Correction (geometric, time domain) ---
        H = aperture_range(R_s_ap, G_0)
        #H = np.linalg.norm(R_s_ap - G_0[:, None], axis=0)
        H_0 = H[c_frame]
        del_R = H - H_0
        R_bins = del_R / dz

        s_rcmc = _rcmc_interpolate(s_ap, R_bins, interp_cval)

        # --- Phase correction and windowing ---
        phase_adj = np.exp(sgn * -4j * np.pi * del_R[None, :] / lamb) # This works for LRS
        s_rcmc *= phase_adj * azi_wnd[None, :]

        # --- Azimuth compression ---
        S_t_k = fftshift(fft(s_rcmc, axis=1, workers=-1), axes=1)

        # --- Multilook detection ---
        s_foc[:, out_idx] = np.sum(
            np.abs(S_t_k[:, start_keep:end_keep]) ** 2, axis=1
        )

    return s_foc, frames, aperture_step, rho_a[frames], rho_df[frames]


def range_doppler(s_t: NDArray[np.complexfloating],
                  *,
                  R_s: NDArray[np.floating],
                  R_t: NDArray[np.floating],
                  R_st: NDArray[np.floating],
                  Vt: float | NDArray[np.floating],
                  Vr: float | NDArray[np.floating],
                  del_x: float | NDArray[np.floating],
                  lamb: float,
                  pri: float,
                  dz: float,
                  D: float | None = None,
                  L_n: int | None = None,
                  os_factor: int = 1,
                  window_type: str = "blackman",
                  window_alpha: float | None = None,
                  remove_doppler_centroid: bool = True,
                  interp_cval: float = 0.0,
                  sgn: Literal[-1,1] = -1,
                  verbose: bool = False,
                  ) -> tuple[NDArray[np.complexfloating] | NDArray[np.float32], NDArray[np.intp], int, NDArray[np.float64]]:
    """Range-Doppler SAR processor.

    Produces a focused SAR image from range-compressed data using
    range-Doppler processing with analytical range cell migration
    correction, azimuth matched filter compression, and optional
    Doppler centroid removal.

    Args:
        s_t: Range-compressed complex data, shape (n_range, n_azimuth).
        R_s: Spacecraft position vectors, shape (3, n_azimuth).
        R_t: Target position vectors, shape (3, n_azimuth).
        R_st: Spacecraft-to-target vectors, shape (3, n_azimuth).
        Vt: Tangential velocity (m/s). Scalar or 1D array of length n_azimuth.
        Vr: Radial velocity (m/s). Scalar or 1D array of length n_azimuth.
        del_x: Along-track sample spacing (m). Scalar or 1D array of
            length n_azimuth.
        lamb: Radar wavelength (m).
        pri: Pulse repetition interval (s).
        dz: Range bin spacing (m).
        D: Real antenna length (m). Required when L_n is not provided.
        L_n: Synthetic aperture length in samples. If None, derived from D.
        os_factor: Oversampling factor for output grid (1 = Nyquist).
        window_type: Azimuth window type as accepted by form_window
            (default "blackman").
        window_alpha: Tukey window taper fraction. Only used when
            window_type is "tukey".
        remove_doppler_centroid: If True, demodulate the Doppler centroid
            before azimuth compression (default True).
        interp_cval: Fill value for out-of-bounds samples during RCMC
            interpolation (default 0.0).
        sgn: Sign convention for phase exponential (-1 or +1).
        verbose: If True, print processing diagnostics (default False).

    Returns:
        s_foc: Complex Focused SAR image, shape (n_range, n_output_frames).
        frames: Center frame indices into the original azimuth axis.
        aperture_step: Step size used between apertures.
        rho_a: Azimuth resolution at each output frame (m).

    Raises:
        ValueError: If neither L_n nor D is provided.
    """
    n_samp, n_col = s_t.shape

    # --- Array setup ---
    del_x, Vt, Vr = _coerce_arrays(n_col, del_x, Vt, Vr)
    del_x_mean = del_x.mean()

    # --- Aperture length ---
    L_n = _resolve_aperture_length(L_n, D, lamb, R_st, del_x_mean, n_col)
    L_sa = L_n * del_x
    L_t = L_n * pri
    c_frame = L_n // 2

    # --- Azimuth window (applied in Doppler domain, so fftshift) ---
    azi_wnd = np.fft.fftshift(form_window(L_n, window_type, alpha=window_alpha))
    azi_brd = broadening_factor(form_window(L_n, window_type, alpha=window_alpha))

    # --- Resolution ---
    R_st_norms = np.linalg.norm(R_st, axis=0)
    rho_a = determine_aperture_resolution(L_sa, lamb, R_st_norms, brd=azi_brd)

    # --- Step size and output grid ---
    aperture_step = determine_aperture_step(rho_a, del_x, os_factor)
    n_output_frames = (n_col + aperture_step - 1) // aperture_step

    if verbose:
        print(f"Synthetic Aperture Samples: {L_n}")
        print(f"\tSynthetic Aperture Length: {L_sa.mean() / 1e3:.2f} km")
        print(f"\tSynthetic Aperture Time: {L_t:.2f} s")
        print(f"\tNominal resolution: {rho_a.mean():.1f} m")
        print(f"Oversampling Factor: {os_factor}")
        print(f"Aperture Step: {aperture_step}")
        print(f"Number of Output Frames: {n_output_frames}")

    # --- Precompute aperture bounds ---
    frames, aperture_bounds = determine_aperture_bounds(L_n, n_col, aperture_step)

    # --- Doppler centroid ---
    f_dc = sgn*-2 * Vr / lamb

    # --- Azimuthal time and frequency axes ---
    eta = (np.arange(L_n) - L_n // 2) * pri
    f_eta = np.fft.fftfreq(L_n, d=pri)

    # --- Output array ---

    s_foc = np.zeros((n_samp, n_output_frames), dtype=np.complex64)

    # --- Aperture buffers ---
    s_ap = np.zeros((n_samp, L_n), dtype=np.complex64)
    R_t_ap = np.zeros((3, L_n), dtype=np.float64)
    R_s_ap = np.zeros((3, L_n), dtype=np.float64)
    Vt_ap = np.zeros(L_n, dtype=np.float64)

    buffers = {"s": s_ap, "R_t": R_t_ap, "R_s": R_s_ap, "Vt": Vt_ap}
    sources = {"s": s_t, "R_t": R_t, "R_s": R_s, "Vt": Vt}

    # --- Processing loop ---
    for out_idx, frame in enumerate(frames):
        if verbose:
            print(f"Processing frame {frame + 1}/{n_col}, "
                  f"output column {out_idx + 1}/{n_output_frames}")

        # Fill aperture
        _fill_aperture(out_idx, aperture_bounds, buffers, sources)

        # Aperture center geometry
        G_0 = R_t_ap[:, c_frame]
        Vt_0 = Vt_ap[c_frame]

        # --- Doppler centroid removal ---
        if remove_doppler_centroid:
            _remove_doppler_centroid(s_ap, f_dc[frame], eta)

        # --- FFT to range-Doppler domain ---
        S_t_k = fft(s_ap, axis=1, workers=-1)

        # --- Range Cell Migration Correction (analytical, Doppler domain) ---
        H = aperture_range(R_s_ap, G_0)
        H_0 = H[c_frame]
        #del_R = (lamb ** 2 * H_0 * f_eta ** 2) / (8 * Vt_0 ** 2)
        x = np.clip(lamb * f_eta / (2.0 * Vt_0), -1.0 + 1e-12, 1.0 - 1e-12)
        del_R = H_0 * (1.0 / np.sqrt(1.0 - x * x) - 1.0)
        rcmc = del_R / dz

        S_t_k = _rcmc_interpolate(S_t_k, rcmc, interp_cval)

        # --- Azimuth compression ---
        k_a = (2 * Vt_0 ** 2) / (lamb * H_0)
        H_az = np.exp(sgn*1j * np.pi * f_eta[None, :] ** 2 / k_a)

        S_compressed = S_t_k * H_az * azi_wnd[None, :]
        s_final = ifft(S_compressed, axis=1, workers=-1)

        # --- Extract center column ---
        s_foc[:, out_idx] = s_final[:, c_frame]

    return s_foc, frames, aperture_step, rho_a[frames]