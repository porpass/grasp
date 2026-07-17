# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from scipy.fft import fft, fftshift
from numpy.typing import NDArray

from ..windows import form_window, broadening_factor
from .utils import determine_aperture_bounds, determine_aperture_step


def unfocused(
    s_t: NDArray[np.complexfloating],
    *,
    del_x: float | NDArray[np.floating],
    R0: float | NDArray[np.floating],
    lamb: float,
    L_n: int | None = None,
    os_factor: int = 1,
    presum: int = 1,
    window_type: str = "hanning",
    window_alpha: float | None = None,
    aperture_var_threshold: float = 0.05,
    coherent: bool = False,
    verbose: bool = False,
) -> tuple[NDArray[np.complexfloating] | NDArray[np.float32],
           NDArray[np.intp], int, NDArray[np.float64]]:
    """Unfocused SAR processor with non-coherent integration.

    Produces a focused image from range-compressed SAR data using a
    boxcar (or windowed) filter in the azimuth direction followed by
    Doppler-domain power summation. Aperture length is derived from
    the Fresnel zone constraint sqrt(lambda * R0) unless explicitly
    provided.

    Args:
        s_t: Range-compressed complex data, shape (n_range, n_azimuth).
        del_x: Along-track sample spacing (km). Scalar or 1D array of
            length n_azimuth.
        R0: Range to target (km). Scalar or 1D array of length n_azimuth.
        lamb: Radar wavelength (m).
        L_n: Synthetic aperture length in samples. If None, derived from
            the Fresnel zone constraint. When provided, overrides the
            automatic aperture sizing and forces a static aperture.
        os_factor: Oversampling factor for output grid (1 = Nyquist).
        presum: Presumming factor applied to data. Only used when L_n
            is derived automatically.
        window_type: Azimuth window type as accepted by form_window
            (default "hanning").
        window_alpha: Tukey window taper fraction. Only used when
            window_type is "tukey".
        aperture_var_threshold: Fractional variation in aperture length
            below which a static aperture is used (default 0.05 = 5%).
            Only used when L_n is derived automatically.
        coherent: If True, return the complex center column of each
            aperture. If False, return non-coherent (power) integration
            across all Doppler bins (default False).
        verbose: If True, print processing diagnostics (default False).

    Returns:
        s_out: Output image, shape (n_range, n_output_frames). Complex
            if coherent=True, float32 if coherent=False.
        frames: Center frame indices into the original azimuth axis.
        aperture_step: Step size used between apertures.
        rho_a: Azimuth resolution at each output frame (m).
    """
    n_samp, n_col = s_t.shape

    # --- Array setup ---
    R0 = np.atleast_1d(np.asarray(R0, dtype=np.float64))
    del_x = np.atleast_1d(np.asarray(del_x, dtype=np.float64))
    if R0.size == 1:
        R0 = np.full(n_col, R0[0])
    if del_x.size == 1:
        del_x = np.full(n_col, del_x[0])

    del_x_mean = del_x.mean()

    # --- Aperture length ---
    if L_n is not None:
        # User-specified: force static aperture
        if L_n % 2 == 0:
            L_n += 1
        if L_n > n_col:
            L_n = n_col - 1 if n_col % 2 == 0 else n_col

        L_n_all = np.full(n_col, L_n)
        L_n_mean = L_n
        static_aperture = True

        if verbose:
            print(f"User-specified aperture: L_n = {L_n}")
    else:
        # Derive from Fresnel zone constraint
        L_max_all = np.sqrt(lamb * R0)
        L_n_all = (L_max_all / del_x).astype(int) + 1
        L_n_all = L_n_all // presum
        L_n_all = np.where(L_n_all % 2 == 0, L_n_all + 1, L_n_all)
        max_L = n_col - 1 if n_col % 2 == 0 else n_col
        L_n_all = np.minimum(L_n_all, max_L)

        # Decide static vs. varying aperture
        L_n_mean = int(np.round(L_n_all.mean()))
        if L_n_mean % 2 == 0:
            L_n_mean += 1

        fractional_var = (L_n_all.max() - L_n_all.min()) / L_n_mean
        static_aperture = fractional_var <= aperture_var_threshold

        if verbose:
            if static_aperture:
                print(f"Static aperture: L_n = {L_n_mean} "
                      f"(variation {fractional_var:.2%} "
                      f"<= {aperture_var_threshold:.0%})")
            else:
                print(f"Varying aperture: L_n = {L_n_all.min()}"
                      f"--{L_n_all.max()} "
                      f"(variation {fractional_var:.2%} "
                      f"> {aperture_var_threshold:.0%})")

        if static_aperture:
            L_n_all[:] = L_n_mean

    # --- Resolution at each azimuth position ---
    azi_wnd = form_window(L_n_mean, window_type, alpha=window_alpha)
    azi_brd = broadening_factor(azi_wnd)

    if static_aperture:
        rho_a = np.full(n_col, azi_brd * L_n_mean * del_x_mean)
    else:
        rho_a = azi_brd * L_n_all * del_x

    # --- Step size and output grid ---
    aperture_step = determine_aperture_step(rho_a, del_x, os_factor)
    n_output_frames = (n_col + aperture_step - 1) // aperture_step

    if verbose:
        print(f"Nominal aperture: {L_n_mean} samples, "
              f"{L_n_mean * del_x_mean:.1f} m")
        print(f"Nominal resolution: {rho_a.mean():.1f} m")
        print(f"Aperture step: {aperture_step}, "
              f"output frames: {n_output_frames}")

    # --- Precompute static resources if aperture is constant ---
    if static_aperture:
        azi_wnd_static = form_window(L_n_mean, window_type, alpha=window_alpha)
        s_ap_static = np.zeros((n_samp, L_n_mean), dtype=np.complex64)
        frames, aperture_bounds = determine_aperture_bounds(
            L_n_mean, n_col, aperture_step)

    # --- Output grid ---
    frame_centers = np.arange(0, n_col, aperture_step)
    if not static_aperture:
        frames = np.clip(frame_centers, 0, n_col - 1)

    if coherent:
        s_out = np.zeros((n_samp, n_output_frames), dtype=np.complex64)
    else:
        s_out = np.zeros((n_samp, n_output_frames), dtype=np.float32)

    # --- Processing loop ---
    for out_idx, frame in enumerate(frame_centers):
        L_n_local = int(L_n_all[frame])
        c_frame_local = L_n_local // 2

        if verbose:
            print(f"Processing frame {frame + 1}/{n_col}, "
                  f"output column {out_idx + 1}/{n_output_frames}")

        if static_aperture:
            s_ap_static[:] = 0
            src_start, src_end, dest_start, dest_end = \
                aperture_bounds[out_idx]
            s_ap_static[:, dest_start:dest_end] = s_t[:, src_start:src_end]
            S_t_k = fftshift(
                fft(s_ap_static * azi_wnd_static[None, :],
                    axis=1, workers=-1),
                axes=1)
        else:
            half = L_n_local // 2
            src_start = max(0, frame - half)
            src_end = min(n_col, frame + half + 1)
            dest_start = src_start - (frame - half)
            dest_end = dest_start + (src_end - src_start)

            s_ap = np.zeros((n_samp, L_n_local), dtype=np.complex64)
            s_ap[:, dest_start:dest_end] = s_t[:, src_start:src_end]

            azi_wnd = form_window(L_n_local, window_type,
                                  alpha=window_alpha)
            S_t_k = fftshift(
                fft(s_ap * azi_wnd[None, :], axis=1, workers=-1),
                axes=1)

        if coherent:
            s_out[:, out_idx] = S_t_k[:, c_frame_local]
        else:
            s_out[:, out_idx] = np.sum(np.abs(S_t_k) ** 2, axis=1)

    return s_out, frames, aperture_step, rho_a[frames]