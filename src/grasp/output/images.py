# SPDX-License-Identifier: BSD-3-Clause
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from .utils import bytscl
import matplotlib


# Tone-mapping LUT used by simc for cluttergram display (see
# https://github.com/.../simc/curve.py). Maps a uint8 index [0, 255] to a
# value in [0, 1] with a soft-shoulder roll-off; saturates at index 189.
_TONE_CURVE = np.array([
    0.000000, 0.016373, 0.032740, 0.049095, 0.065431, 0.081741, 0.098021, 0.114263,
    0.130461, 0.146610, 0.162702, 0.178731, 0.194692, 0.210578, 0.226382, 0.242099,
    0.257722, 0.273246, 0.288663, 0.303967, 0.319153, 0.334213, 0.349143, 0.363935,
    0.378583, 0.393081, 0.407423, 0.421602, 0.435613, 0.449449, 0.463103, 0.476570,
    0.489843, 0.502916, 0.515783, 0.528437, 0.540873, 0.553083, 0.565062, 0.576804,
    0.588302, 0.599550, 0.610542, 0.621271, 0.631731, 0.641916, 0.651821, 0.661437,
    0.670760, 0.679783, 0.688499, 0.696903, 0.704988, 0.712749, 0.720178, 0.727269,
    0.734017, 0.740414, 0.746456, 0.752134, 0.757444, 0.762379, 0.766933, 0.771588,
    0.775488, 0.779346, 0.783161, 0.786935, 0.790666, 0.794357, 0.798005, 0.801613,
    0.805179, 0.808705, 0.812191, 0.815636, 0.819041, 0.822407, 0.825733, 0.829019,
    0.832266, 0.835475, 0.838644, 0.841775, 0.844868, 0.847923, 0.850940, 0.853920,
    0.856862, 0.859767, 0.862635, 0.865467, 0.868262, 0.871020, 0.873743, 0.876430,
    0.879081, 0.881697, 0.884278, 0.886824, 0.889335, 0.891812, 0.894254, 0.896663,
    0.899038, 0.901379, 0.903686, 0.905961, 0.908203, 0.910412, 0.912588, 0.914733,
    0.916845, 0.918925, 0.920974, 0.922992, 0.924978, 0.926934, 0.928859, 0.930753,
    0.932617, 0.934451, 0.936256, 0.938030, 0.939776, 0.941492, 0.943179, 0.944838,
    0.946468, 0.948070, 0.949644, 0.951191, 0.952709, 0.954200, 0.955665, 0.957102,
    0.958512, 0.959897, 0.961255, 0.962586, 0.963892, 0.965173, 0.966428, 0.967658,
    0.968863, 0.970044, 0.971200, 0.972332, 0.973440, 0.974524, 0.975584, 0.976621,
    0.977635, 0.978626, 0.979595, 0.980541, 0.981464, 0.982366, 0.983246, 0.984105,
    0.984942, 0.985758, 0.986553, 0.987327, 0.988081, 0.988815, 0.989528, 0.990222,
    0.990897, 0.991552, 0.992188, 0.992805, 0.993403, 0.993983, 0.994545, 0.995089,
    0.995614, 0.996123, 0.996614, 0.997088, 0.997545, 0.997985, 0.998409, 0.998817,
    0.999209, 0.999585, 0.999945, 1.000000,
] + [1.0] * 68, dtype=np.float64)


def _apply_tone_curve(scaled_01: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the simc tone-mapping LUT to [0, 1]-normalized data."""
    idx = np.clip((scaled_01 * 255.0).astype(np.int64), 0, 255)
    return _TONE_CURVE[idx]


def to_image(data: NDArray,
           output_path: str | Path,
           *,
           lower_percentile: float = 50,
           upper_percentile: float = 100,
           vmin: float | None = None,
           vmax: float | None = None,
           power: bool = False,
           invert: bool = False,
           is_db: bool = False,
           per_frame: bool = False,
           ) -> None:
    """Convert a radargram array to an 8-bit grayscale BMP image.

    Scales the input data to 0–255 using either explicit bounds or
    percentile-based clipping and saves as a grayscale BMP.

    If the input is not already in dB, a dB transformation is applied
    first: ``20 * log10(|data|)`` for amplitude (complex or real
    voltage) data, or ``10 * log10(|data|)`` for power data.

    Args:
        data: Input radargram array (complex, real, or pre-computed dB).
        output_path: File path for the output BMP image.
        lower_percentile: Lower percentile for automatic clipping.
            Ignored if ``vmin`` is provided.
        upper_percentile: Upper percentile for automatic clipping.
            Ignored if ``vmax`` is provided.
        vmin: Explicit minimum dB value for scaling. Overrides
            ``lower_percentile`` when set.
        vmax: Explicit maximum dB value for scaling. Overrides
            ``upper_percentile`` when set.
        power: If True, use ``10 * log10`` (power quantities).
            If False (default), use ``20 * log10`` (amplitude
            quantities). Ignored when ``is_db`` is True.
        invert: If True, invert the grayscale so that strong returns
            appear dark and weak returns appear bright.
        is_db: If True, treat ``data`` as already in dB and skip
            the dB conversion.
        per_frame: If True, compute vmin/vmax per column and apply the
            stretch column-by-column (frame-by-frame). Useful when
            trace-to-trace brightness varies along the orbit (e.g.,
            MARSIS altitude swings). Defaults to global scaling.
    """

    if is_db:
        db = data.astype(np.float64, copy=True)
    else:
        amplitude = np.abs(data).clip(min=np.finfo(np.float64).tiny)
        # Per-frame normalization: each column gets divided by its own
        # noise estimate so columns share a common ~0 dB noise floor before
        # the global vmin/vmax stretch.
        axis = 0 if per_frame else None
        noise = np.median(amplitude[:50, :], axis=axis)
        if power:
            db = 10 * np.log10(amplitude / noise)
        else:
            db = 20 * np.log10(amplitude / noise)

    if vmin is None:
        vmin = np.percentile(db, lower_percentile)
    if vmax is None:
        vmax = np.percentile(db, upper_percentile)

    scaled = bytscl(db, min_val=vmin, max_val=vmax)

    if invert:
        scaled = 255 - scaled

    img = Image.fromarray(scaled, "L")
    img.save(output_path)


def csim_to_image(data: NDArray,
                  output_path: str | Path,
                  *,
                  column_weights: NDArray | None = None,
                  apply_curve: bool = True,
                  invert: bool = False,
                  ) -> None:
    """Convert a clutter simulation array to an 8-bit grayscale BMP image.

    Mirrors simc's ``combined``/``combinedadj`` output: linear power,
    scaled to [0, 255] by ``data * (255.0 / data.max())``, optionally
    followed by the simc soft-shoulder tone curve.

    Args:
        data: Linear-power clutter simulation array.
        output_path: File path for the output BMP image.
        column_weights: Optional per-column multiplier of length
            ``data.shape[1]``. Each column is multiplied by its weight
            before the global max-normalize. Used to compensate for
            trace-to-trace gain variation (e.g., radar-equation ``R^4``
            path loss for spacecraft with varying altitude).
        apply_curve: If True (default), apply the simc tone-mapping
            LUT after the linear scale.
        invert: If True, invert the grayscale.
    """
    amplitude = np.abs(data).astype(np.float64, copy=False)
    if column_weights is not None:
        amplitude = amplitude * np.asarray(column_weights, dtype=np.float64)[None, :]
    peak = amplitude.max()
    if peak <= 0:
        scaled = np.zeros(amplitude.shape, dtype=np.uint8)
    else:
        scaled = (amplitude * (255.0 / peak)).astype(np.uint8)
        if apply_curve:
            scaled = (_TONE_CURVE[scaled] * 255).astype(np.uint8)
    if invert:
        scaled = 255 - scaled
    img = Image.fromarray(scaled, "L")
    img.save(output_path)


def radargram_with_dem(data: NDArray,
                       swath: dict,
                       output_path: str | Path,
                       *,
                       lower_percentile: float = 50,
                       upper_percentile: float = 100,
                       vmin: float | None = None,
                       vmax: float | None = None,
                       power: bool = False,
                       is_db: bool = False,
                       invert: bool = False,
                       dem_cmap: str = "terrain",
                       dem_position: str = "bottom",
                       draw_track: bool = True,
                       dem_height_ratio: float = 0.15,
                       ) -> None:
    """Save a combined radargram and DEM swath image.

    Renders the radargram as a grayscale image and the DEM swath
    with a colormap, stacks them vertically, and saves the result.
    The DEM strip is scaled to match the radargram width (one column
    per trace).

    The dB conversion uses ``20 * log10(|data|)`` for amplitude
    (complex or real voltage) data and ``10 * log10(|data|)`` for
    power data.

    Args:
        data: Input radargram array (complex, real, or pre-computed dB).
        swath: Dictionary returned by :func:`extract_dem_swath`.
        output_path: File path for the output image.
        lower_percentile: Lower percentile for automatic radargram
            clipping. Ignored if ``vmin`` is provided.
        upper_percentile: Upper percentile for automatic radargram
            clipping. Ignored if ``vmax`` is provided.
        vmin: Explicit minimum dB value for radargram scaling.
            Overrides ``lower_percentile`` when set.
        vmax: Explicit maximum dB value for radargram scaling.
            Overrides ``upper_percentile`` when set.
        power: If True, use ``10 * log10`` (power quantities).
            If False (default), use ``20 * log10`` (amplitude
            quantities). Ignored when ``is_db`` is True.
        invert: If True, invert the radargram grayscale so that
            strong returns appear dark and weak returns appear bright.
        is_db: If True, treat ``data`` as already in dB and skip
            the dB conversion.
        dem_cmap: Matplotlib colormap name for the DEM strip.
        dem_position: Position of the DEM strip relative to the
            radargram. Either ``"bottom"`` or ``"top"``.
        draw_track: If True, draw a horizontal line on the DEM
            strip indicating the ground track centerline.
        dem_height_ratio: Height of the DEM strip as a fraction of the
            radargram height. Defaults to 0.15.
    """
    if is_db:
        db = data.astype(np.float64, copy=True)
    else:
        amplitude = np.abs(data).clip(min=np.finfo(np.float64).tiny)
        noise = np.median(amplitude[:50, :])
        if power:
            db = 10 * np.log10(amplitude / noise)
        else:
            db = 20 * np.log10(amplitude / noise)

    # Compute percentiles from finite values only
    finite_db = db[np.isfinite(db)]
    if vmin is None:
        vmin = np.percentile(finite_db, lower_percentile) if finite_db.size > 0 else 0.0
    if vmax is None:
        vmax = np.percentile(finite_db, upper_percentile) if finite_db.size > 0 else 1.0

    # Replace non-finite values with vmin so they map to the dark end
    db = np.nan_to_num(db, nan=vmin, posinf=vmax, neginf=vmin)

    rgram_scaled = bytscl(db, min_val=vmin, max_val=vmax)

    if invert:
        rgram_scaled = 255 - rgram_scaled

    # Convert grayscale to RGB
    rgram_rgb = np.stack([rgram_scaled] * 3, axis=-1)

    # --- Render DEM swath to RGB ---
    n_traces = data.shape[1]
    elevation = swath["elevation"]

    # Normalize elevation to 0-1 for colormap using finite values only
    elev_finite = elevation[np.isfinite(elevation)]
    if elev_finite.size > 0:
        elev_min = np.nanmin(elev_finite)
        elev_max = np.nanmax(elev_finite)
    else:
        elev_min, elev_max = 0.0, 1.0

    if elev_max == elev_min:
        elev_norm = np.zeros_like(elevation)
    else:
        elev_norm = (elevation - elev_min) / (elev_max - elev_min)

    # NaN elevation pixels become transparent black (or cmap floor)
    elev_norm = np.clip(np.nan_to_num(elev_norm, nan=0.0), 0, 1)

    # Apply colormap
    colormap = matplotlib.colormaps[dem_cmap]
    dem_rgba = colormap(elev_norm)
    dem_rgb = (dem_rgba[:, :, :3] * 255).astype(np.uint8)

    # Mark NaN elevation pixels as black
    nan_mask = ~np.isfinite(elevation)
    dem_rgb[nan_mask] = [0, 0, 0]

    # Resize DEM strip to match radargram width
    dem_height = max(int(data.shape[0] * dem_height_ratio), 1)
    dem_img = Image.fromarray(dem_rgb, "RGB")
    dem_resized = dem_img.resize((n_traces, dem_height),
                                 Image.Resampling.NEAREST)
    dem_array = np.array(dem_resized)

    # Draw ground track line on the resized DEM
    if draw_track:
        center_idx = dem_height // 2
        dem_array[center_idx, :, :] = [255, 0, 0]
        if center_idx > 0:
            dem_array[center_idx - 1, :, :] = [255, 0, 0]

    # --- Stack and save ---
    if dem_position == "top":
        combined = np.vstack([dem_array, rgram_rgb])
    else:
        combined = np.vstack([rgram_rgb, dem_array])

    combined_img = Image.fromarray(combined, "RGB")
    combined_img.save(output_path)

def cluttergram_with_echomap(cluttergram: NDArray,
                             echomap: NDArray,
                             nadir_ct_idx: NDArray[np.intp],
                             fret_ct_idx: NDArray[np.intp],
                             output_path: str | Path,
                             *,
                             column_weights: NDArray | None = None,
                             apply_curve: bool = True,
                             invert: bool = False,
                             echomap_cmap: str = "gray",
                             echomap_position: str = "bottom",
                             echomap_height_ratio: float = 0.15,
                             nadir_color: tuple[int, int, int] = (50, 200, 200),
                             fret_color: tuple[int, int, int] = (255, 0, 255),
                             marker_radius: int = 3,
                             ) -> None:
    """Save a combined cluttergram and echo map image.

    Renders the cluttergram as a grayscale image and the echo map
    with nadir and first-return markers overlaid, stacks them
    vertically, and saves the result.

    Args:
        cluttergram: Simulated clutter power array, shape
            (n_samples, n_traces).
        echomap: Peak return power per cross-track bin, shape
            (n_ct_bins, n_traces).
        nadir_ct_idx: Cross-track bin index of the nadir return
            per trace, shape (n_traces,).
        fret_ct_idx: Cross-track bin index of the first return
            per trace, shape (n_traces,).
        output_path: File path for the output image.
        column_weights: Optional per-trace multiplier of length
            ``cluttergram.shape[1]``. Applied to the cluttergram before
            the global max-normalize to compensate for trace-to-trace
            gain variation (e.g., radar-equation ``R^4`` path loss).
        apply_curve: If True, apply the simc tone-mapping LUT to the
            cluttergram after the linear scale.
        invert: If True, invert the cluttergram grayscale.
        echomap_cmap: Matplotlib colormap name for the echo map
            background intensity.
        echomap_position: Position of the echo map strip relative
            to the cluttergram. Either ``"bottom"`` or ``"top"``.
        echomap_height_ratio: Height of the echo map strip as a
            fraction of the cluttergram height. Defaults to 0.15.
        nadir_color: RGB tuple for the nadir return marker.
        fret_color: RGB tuple for the first-return marker.
        marker_radius: Half-width of the marker in pixels.
    """
    n_samples, n_traces = cluttergram.shape

    # --- Render cluttergram to uint8 (simc combined/combinedadj style) ---
    amplitude = np.abs(cluttergram).astype(np.float64, copy=False)
    if column_weights is not None:
        amplitude = amplitude * np.asarray(column_weights, dtype=np.float64)[None, :]
    peak = amplitude.max()
    if peak <= 0:
        cgram_scaled = np.zeros(amplitude.shape, dtype=np.uint8)
    else:
        cgram_scaled = (amplitude * (255.0 / peak)).astype(np.uint8)
        if apply_curve:
            cgram_scaled = (_TONE_CURVE[cgram_scaled] * 255).astype(np.uint8)

    if invert:
        cgram_scaled = 255 - cgram_scaled

    cgram_rgb = np.stack([cgram_scaled] * 3, axis=-1)

    emap = np.abs(echomap.astype(np.float64, copy=False))
    if column_weights is not None:
        emap = emap * np.asarray(column_weights, dtype=np.float64)[None, :]
    emap_peak = emap.max()
    if emap_peak <= 0:
        emap_u8 = np.zeros(emap.shape, dtype=np.uint8)
    else:
        emap_u8 = (emap * (255.0 / emap_peak)).astype(np.uint8)
    emap_norm = emap_u8.astype(np.float64) / 255.0
    colormap = matplotlib.colormaps[echomap_cmap]
    emap_rgba = colormap(emap_norm)
    emap_rgb = (emap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Black out bins with zero power.
    zero_mask = echomap <= 0.0
    emap_rgb[zero_mask] = [0, 0, 0]

    # Resize echo map strip to match cluttergram width.
    emap_height = max(int(n_samples * echomap_height_ratio), 1)
    n_ct_bins = echomap.shape[0]
    emap_img = Image.fromarray(emap_rgb, "RGB")
    emap_resized = emap_img.resize((n_traces, emap_height),
                                    Image.Resampling.NEAREST)
    emap_array = np.array(emap_resized)

    # Scale factor from original echo map rows to resized rows.
    row_scale = emap_height / n_ct_bins

    # --- Draw nadir and FRET markers ---
    for col in range(n_traces):
        # Nadir marker.
        nr = int(nadir_ct_idx[col] * row_scale)
        r_lo = max(0, nr - marker_radius)
        r_hi = min(emap_height, nr + marker_radius + 1)
        emap_array[r_lo:r_hi, col, :] = nadir_color

        # FRET marker.
        fr = int(fret_ct_idx[col] * row_scale)
        r_lo = max(0, fr - marker_radius)
        r_hi = min(emap_height, fr + marker_radius + 1)
        emap_array[r_lo:r_hi, col, :] = fret_color

    # --- Stack and save ---
    if echomap_position == "top":
        combined = np.vstack([emap_array, cgram_rgb])
    else:
        combined = np.vstack([cgram_rgb, emap_array])

    combined_img = Image.fromarray(combined, "RGB")
    combined_img.save(output_path)