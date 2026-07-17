# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from ..processing.sar.utils import determine_aperture_step
from ..processing.windows import form_window
from scipy.ndimage import convolve1d

def multilook(
    s_foc: NDArray[np.complexfloating],
    rho_a: float | NDArray[np.floating],
    *,
    n_looks: int = 5,
    os_factor: int = 1,
    window_type: str = "hanning",
    window_alpha: float | None = None,
    coherent: bool = False,
    verbose: bool = False,
) -> tuple[NDArray[np.complexfloating] | NDArray[np.float32],
           NDArray[np.intp], NDArray[np.float64]]:
    """Multilook averaging of a complex SAR image.

    When coherent=False (default), detects the input image (power),
    then convolves along azimuth with a normalized window. When
    coherent=True, convolves the complex data directly, preserving
    phase information.

    Args:
        s_foc: Complex focused SAR image, shape (n_range, n_azimuth).
        n_looks: Number of looks (window length). Forced odd internally.
        rho_a: Single-look azimuth resolution per column (m),
            shape (n_azimuth,).
        os_factor: Oversampling factor for output grid (1 = Nyquist).
        window_type: Window type as accepted by form_window
            (default "hanning").
        window_alpha: Tukey window taper fraction. Only used when
            window_type is "tukey".
        coherent: If True, average the complex data (preserves phase).
            If False, average the detected power (default False).
        verbose: If True, print processing diagnostics (default False).

    Returns:
        s_mlk: Multilooked image, shape (n_range, n_output).
            Complex64 if coherent=True, float32 if coherent=False.
        frames: Output frame indices into the input azimuth axis.
        rho_mlk: Multilooked azimuth resolution per output column (m).
    """
    n_samp, n_col = s_foc.shape

    if n_looks % 2 == 0:
        n_looks += 1

    rho_a = np.atleast_1d(np.asarray(rho_a, dtype=np.float64))
    if rho_a.size == 1:
        rho_a = np.full(n_col, rho_a[0])

    # --- Multilooked resolution ---
    rho_mlk = n_looks * rho_a

    # --- Step size from resolution ---
    step = determine_aperture_step(rho_mlk, rho_a, os_factor)
    # --- Subsample ---
    #frames = determine_output_frames(n_col, step)
    frames = np.arange(0, n_col, step)

    if coherent:
        # Window and sum the complex data
        mlk_wnd = form_window(n_looks, window_type, alpha=window_alpha)
        mlk_wnd = mlk_wnd / mlk_wnd.sum()
        smoothed = convolve1d(s_foc, mlk_wnd, axis=1, mode='constant', cval=0.0)
        s_mlk = smoothed[:, frames].astype(np.complex64)
    else:
        # Detect then smooth
        intensity = np.abs(s_foc) ** 2
        mlk_wnd = form_window(n_looks, window_type, alpha=window_alpha)
        mlk_wnd = mlk_wnd / mlk_wnd.sum()
        smoothed = convolve1d(intensity, mlk_wnd, axis=1, mode='constant', cval=0.0)
        s_mlk = smoothed[:, frames].astype(np.float32)

    rho_mlk_out = rho_mlk[frames]

    if verbose:
        print(f"Multilook: {n_looks} looks")
        print(f"\tMultilooked resolution: {rho_mlk.mean():.1f} m")
        print(f"\tStep: {step}, output frames: {len(frames)}")
        print(f"\tInput shape: {s_foc.shape}")
        print(f"\tOutput shape: {s_mlk.shape}")

    return s_mlk, frames, rho_mlk_out