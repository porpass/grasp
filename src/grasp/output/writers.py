# SPDX-License-Identifier: BSD-3-Clause
"""GRaSP output file writers.

Provides functions to write GRaSP processing results in multiple
formats:

    - ``.img`` / ``.csv`` / ``.grsp`` — GRaSP native triplet
    - ``.sgy`` — SEG-Y Rev 1 for seismic interpretation software

Each writer can be called independently, or the native triplet can
be produced at once via :func:`write_grasp_output`.
"""
from datetime import date
import h5py
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from pyproj import CRS, Transformer
import segyio

from ..geospatial.extract_dem_swath import extract_dem_swath
from ..geospatial.crs import normalize_crs
from ..common.config import get_data_path
from ..grasp_types import CampbellResult, ContrastResult, SARResult, GeometryResult, ClutterResult
from .images import to_image, csim_to_image, radargram_with_dem, cluttergram_with_echomap

def export_data(data: NDArray,
                geometry: GeometryResult,
                *,
                out_dir: Path,
                base: str,
                instrument: str,
                dt: float,
                pri: float,
                presum: int = 1,
                sar: SARResult | None = None,
                ionosphere: CampbellResult | ContrastResult | None = None,
                output_format: str = "basic",
                byte_order: str = "little",
                verbose: bool = False,
                ):
    """Export processed data to output files.

    Args:
        data: Processed echo data, shape ``(n_samp, n_cols)``.
        geometry: Observation geometry.
        out_dir: Output directory.
        base: Base filename (without extension).
        instrument: Instrument identifier.
        dt: Sample spacing in seconds.
        pri: Pulse repetition interval in seconds.
        presum: Along-track presumming factor.
        sar: SAR processing result, if available.
        ionosphere: Ionospheric correction result, if available.
        output_format: Output format. Only ``"basic"`` is
            currently supported.
        byte_order: Byte order for binary output.
        verbose: If True, print progress messages.

    Raises:
        ValueError: If the output format is not supported.
    """
    output_format = output_format.lower()
    if output_format != "basic":
        raise ValueError(f"Unsupported output_format '{output_format}'")

    pri_eff = pri * presum
    if geometry.v_tangential is not None:
        dx = (geometry.v_tangential * pri_eff).mean()
    else:
        dx = -9999.0

    rho_a = sar.rho_a.mean() if sar is not None else -9999.0

    el_radius = (geometry.el_radius
                 if geometry.el_radius is not None
                 else np.full(len(geometry.latitude), -9999.0))

    iono_value = None
    if ionosphere is not None:
        if isinstance(ionosphere, CampbellResult):
            iono_value = ionosphere.e_values
        elif isinstance(ionosphere, ContrastResult):
            iono_value = ionosphere.a2

    write_grasp_output(
        data=data,
        output_dir=out_dir,
        base=base,
        instrument=instrument,
        dt=dt,
        dx=dx,
        rho_a=rho_a,
        ephemeris_time=geometry.et,
        geometry_epoch=geometry.epoch,
        latitude=geometry.latitude,
        longitude=geometry.longitude,
        altitude=geometry.altitude,
        sc_radius=geometry.sc_radius,
        el_radius=el_radius,
        solar_zenith_angle=None,
        iono_value=iono_value,
        byte_order=byte_order,
        verbose=verbose,
    )

def export_images(data: NDArray,
                  geometry: GeometryResult,
                  crs: CRS | None,
                  *,
                  out_dir: Path,
                  base: str,
                  dem: str | Path,
                  buffer_km: float,
                  lower_percentile: float | None = None,
                  upper_percentile: float | None = None,
                  vmin: float | None = None,
                  vmax: float | None = None,
                  power: bool | None = None,
                  invert: bool = False,
                  per_frame: bool = False,
                  verbose: bool = False,
                  ):
    """Export radargram and browse images.

    Args:
        data: Processed echo data, shape ``(n_samp, n_cols)``.
        geometry: Observation geometry.
        crs: Source coordinate reference system.
        out_dir: Output directory.
        base: Base filename (without extension).
        dem: DEM source for browse image. ``"HRSC-MOLA"``
            for the built-in DEM, a path to a custom DEM,
            or None to skip the browse image.
        buffer_km: Half-width of the DEM swath in kilometers.
        lower_percentile: Lower percentile for image scaling.
        upper_percentile: Upper percentile for image scaling.
        vmin: Explicit minimum dB value for scaling.
        vmax: Explicit maximum dB value for scaling.
        power: If True, use ``10 * log10`` scaling. If None,
            auto-detected from data dtype.
        invert: If True, invert the grayscale.
        per_frame: If True, compute vmin/vmax per column.
        verbose: If True, print progress messages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if power is None:
        power = not np.iscomplexobj(data)

    bmp_path = out_dir / f"{base}.bmp"
    if verbose:
        print(f"Writing radargram image: {bmp_path}")

    to_image(data, bmp_path, lower_percentile=lower_percentile, upper_percentile=upper_percentile, vmin=vmin,
             vmax=vmax, power=power, invert=invert, per_frame=per_frame,)

    dem = dem.upper()
    if isinstance(dem, str):
        if dem.upper() == "HRSC-MOLA":
            dem_path = get_data_path("mola_hrsc")
        elif dem.upper() == "LOLA":
            dem_path = get_data_path("lola_global")
        else:
            raise ValueError(f"Unknown DEM: {dem}")
    else:
        dem_path = Path(dem)

    if verbose:
        print(f"Extracting DEM swath from: {dem_path}")

    swath = extract_dem_swath(
        geometry.latitude,
        geometry.longitude,
        dem_path,
        crs,
        buffer_km=buffer_km,
    )

    browse_path = out_dir / f"{base}_browse.png"
    if verbose:
        print(f"Writing browse image: {browse_path}")

    radargram_with_dem(
        data,
        swath,
        browse_path,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        vmin=vmin,
        vmax=vmax,
        power=power,
        invert=invert,
    )


def export_segy(data: NDArray,
                latitude: NDArray,
                longitude: NDArray,
                obs_start: str | np.datetime64 | None,
                *,
                out_dir: Path,
                base: str,
                dt: float,
                instrument: str,
                product_id: str,
                product_type: str,
                src_crs: str | CRS | Path | None = None,
                tgt_crs: str | CRS | Path | None = None,
                verbose: bool = False,
                ):
    """Export radargram to SEG-Y format.

    Args:
        data: Processed echo data, shape ``(n_samp, n_cols)``.
        latitude: Latitude per trace in degrees, shape ``(n_col,)``.
        longitude: Longitude per trace in degrees, shape ``(n_col,)``.
        obs_start: Observation start time as an ISO-8601 string or
            ``numpy.datetime64`` scalar. If ``None``, the text-header
            start-time field is left blank.
        out_dir: Output directory.
        base: Base filename (without extension).
        dt: Sample spacing in seconds.
        instrument: Instrument identifier.
        product_id: Product identifier.
        product_type: Product type.
        src_crs: Source coordinate reference system.
        tgt_crs: Target CRS for reprojection.
        verbose: If True, print progress messages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sgy_path = out_dir / f"{base}.sgy"
    if verbose:
        print(f"Writing SEG-Y: {sgy_path}")

    write_segy(
        data,
        sgy_path,
        dt=dt,
        latitude=latitude,
        longitude=longitude,
        obs_start=obs_start,
        instrument=instrument,
        product_id=product_id,
        product_type=product_type,
        src_crs=src_crs,
        tgt_crs=tgt_crs,
        verbose=verbose,
    )



def write_grasp_img(data: NDArray,
                    filepath: str | Path,
                    byte_order: str = "big",
                    ) -> None:
    """Write a radargram to a flat binary ``.img`` file.

    Complex data is written as interleaved real/imaginary float32
    pairs. Real data is written as float32.

    Args:
        data: Radargram array, shape (n_samp, n_col). May be
            complex64/complex128 or float32/float64.
        filepath: Output file path.
        byte_order: Byte order for the output file. ``"big"`` for
            big-endian (default), ``"little"`` for little-endian.

    Raises:
        ValueError: If byte_order is not ``"big"`` or ``"little"``.
    """
    if byte_order not in ("big", "little"):
        raise ValueError(f"byte_order must be 'big' or 'little', "
                         f"got '{byte_order}'")

    bo_char = ">" if byte_order == "big" else "<"

    if np.iscomplexobj(data):
        # Interleave real and imaginary as float32 pairs
        out = np.empty((data.shape[0] * 2, data.shape[1]),
                       dtype=f"{bo_char}f4")
        out[0::2, :] = data.real
        out[1::2, :] = data.imag
    else:
        out = data.astype(f"{bo_char}f4")

    out.tofile(str(filepath))


def write_grasp_csv(filepath: str | Path,
                    ephemeris_time: NDArray,
                    geometry_epoch: NDArray,
                    latitude: NDArray,
                    longitude: NDArray,
                    altitude: NDArray,
                    sc_radius: NDArray,
                    el_radius: NDArray,
                    solar_zenith_angle: NDArray | None = None,
                    iono_value: NDArray | None = None,
                    ) -> None:
    """Write per-frame geometry data to a ``.csv`` file.

    Args:
        filepath: Output file path.
        ephemeris_time: SPICE ephemeris time per frame, shape (n_col,).
        geometry_epoch: UTC epoch string per frame, shape (n_col,).
        latitude: Latitude per frame in degrees, shape (n_col,).
        longitude: Longitude per frame in degrees, shape (n_col,).
        altitude: Spacecraft altitude per frame in meters,
            shape (n_col,).
        sc_radius: Spacecraft radius per frame in meters,
            shape (n_col,).
        el_radius: Ellipsoid radius per frame in meters,
            shape (n_col,).
        solar_zenith_angle: Solar zenith angle per frame in degrees,
            shape (n_col,). If None, column is filled with NaN.
        iono_value: Ionosphere correction value per frame,
            shape (n_col,). If None, column is filled with NaN.
    """
    n_col = len(ephemeris_time)

    if solar_zenith_angle is None:
        solar_zenith_angle = np.full(n_col, np.nan)

    if iono_value is None:
        iono_value = np.full(n_col, np.nan)

    header = ("FRAME_INDEX,EPHEMERIS_TIME,GEOMETRY_EPOCH,"
              "LATITUDE,LONGITUDE,ALTITUDE,"
              "SC_RADIUS,EL_RADIUS,"
              "SOLAR_ZENITH_ANGLE,IONO_VALUE")

    with open(filepath, "w") as f:
        f.write(header + "\n")
        for i in range(n_col):
            line = (
                f"{i},"
                f"{ephemeris_time[i]:.3f},"
                f"{geometry_epoch[i]},"
                f"{latitude[i]:.3f},"
                f"{longitude[i]:.3f},"
                f"{altitude[i]:.3f},"
                f"{sc_radius[i]:.3f},"
                f"{el_radius[i]:.3f},"
                f"{solar_zenith_angle[i]:.3f},"
                f"{iono_value[i]:.3f}"
            )
            f.write(line + "\n")


def write_grasp_grsp(filepath: str | Path,
                     img_path: str | Path,
                     csv_path: str | Path,
                     instrument: str,
                     product_id: str,
                     dtype: str,
                     n_samp: int,
                     n_col: int,
                     byte_order: str,
                     dt: float,
                     dx: float,
                     rho_a: float,
                     ) -> None:
    """Write a GRaSP descriptor file.

    The ``.grsp`` file serves as the entry point for reading GRaSP
    outputs. The first two lines contain absolute paths to the
    associated ``.img`` and ``.csv`` files, followed by key-value
    metadata describing the binary layout.

    Args:
        filepath: Output file path.
        img_path: Path to the associated ``.img`` file. Resolved
            to an absolute path before writing.
        csv_path: Path to the associated ``.csv`` file. Resolved
            to an absolute path before writing.
        instrument: Radar instrument identifier (e.g. ``"SHARAD"``).
        product_id: Observation product identifier.
        dtype: Data type string (``"COMPLEX64"`` or ``"FLOAT32"``).
        n_samp: Number of range samples per trace.
        n_col: Number of output columns (frames).
        byte_order: Byte order (``"BIG"`` or ``"LITTLE"``).
        dt: Range sample spacing in seconds.
        dx: Along-track sample spacing in meters.
        rho_a: Azimuth resolution in meters.
    """
    lines = [
        f"IMG_FILE = {Path(img_path).resolve()}",
        f"CSV_FILE = {Path(csv_path).resolve()}",
        f"INSTRUMENT = {instrument.upper()}",
        f"PRODUCT_ID = {product_id.upper()}",
        f"DTYPE = {dtype.upper()}",
        f"N_SAMP = {n_samp}",
        f"N_COL = {n_col}",
        f"BYTE_ORDER = {byte_order.upper()}",
        f"DT = {dt}",
        f"DX = {dx:.3f}",
        f"RHO_A = {rho_a:.3f}",
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_grasp_output(data: NDArray,
                       output_dir: str | Path,
                       base: str,
                       instrument: str,
                       dt: float,
                       dx: float,
                       rho_a: float,
                       ephemeris_time: NDArray,
                       geometry_epoch: NDArray,
                       latitude: NDArray,
                       longitude: NDArray,
                       altitude: NDArray,
                       sc_radius: NDArray,
                       el_radius: NDArray,
                       solar_zenith_angle: NDArray | None = None,
                       iono_value: NDArray | None = None,
                       byte_order: str = "big",
                       verbose: bool = False,
                       ) -> None:
    """Write a complete GRaSP output triplet (.img, .csv, .grsp).

    Builds filenames from the product_id, processing level, and
    method, then delegates to :func:`write_grasp_img`,
    :func:`write_grasp_csv`, and :func:`write_grasp_grsp`.

    Args:
        data: Radargram array, shape (n_samp, n_col).
        output_dir: Directory to write output files.
        base: Observation product identifier.
        instrument: Radar instrument identifier (e.g. ``"SHARAD"``).
        dt: Range sample spacing in seconds.
        dx: Along-track sample spacing in meters.
        rho_a: Azimuth resolution in meters.
        ephemeris_time: SPICE ephemeris time per frame, shape (n_col,).
        geometry_epoch: UTC epoch string per frame, shape (n_col,).
        latitude: Latitude per frame in degrees, shape (n_col,).
        longitude: Longitude per frame in degrees, shape (n_col,).
        altitude: Spacecraft altitude per frame in meters,
            shape (n_col,).
        sc_radius: Spacecraft radius per frame in meters,
            shape (n_col,).
        el_radius: Ellipsoid radius per frame in meters,
            shape (n_col,).
        solar_zenith_angle: Solar zenith angle per frame in degrees,
            shape (n_col,). If None, column is filled with NaN.
        iono_value: Ionosphere correction value per frame,
            shape (n_col,). If None, column is filled with NaN.
        byte_order: Byte order for the binary file. ``"big"``
            (default) or ``"little"``.
        verbose: If True, print progress messages.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_path = output_dir / f"{base}.img"
    csv_path = output_dir / f"{base}.csv"
    grsp_path = output_dir / f"{base}.grsp"

    # Determine dtype string
    if np.iscomplexobj(data):
        dtype_str = "COMPLEX64"
    else:
        dtype_str = "FLOAT32"

    n_samp, n_col = data.shape
    bo_str = byte_order.upper()

    if verbose:
        print(f"Writing GRaSP output: {base}")
        print(f"\tIMG:  {img_path}")
        print(f"\tCSV:  {csv_path}")
        print(f"\tGRSP: {grsp_path}")

    write_grasp_img(data, img_path, byte_order=byte_order)

    write_grasp_csv(csv_path,
                    ephemeris_time, geometry_epoch,
                    latitude, longitude, altitude,
                    sc_radius, el_radius,
                    solar_zenith_angle=solar_zenith_angle,
                    iono_value=iono_value)

    write_grasp_grsp(grsp_path,
                     img_path=img_path,
                     csv_path=csv_path,
                     instrument=instrument,
                     product_id=base,
                     dtype=dtype_str,
                     n_samp=n_samp,
                     n_col=n_col,
                     byte_order=bo_str,
                     dt=dt,
                     dx=dx,
                     rho_a=rho_a)

    if verbose:
        print("Output complete.")


def export_csim_data(clutter_result: ClutterResult,
                     geometry: GeometryResult,
                     *,
                     out_dir: Path,
                     base: str,
                     instrument: str,
                     bin_size: float,
                     byte_order: str = "big",
                     verbose: bool = False,
                     ) -> None:
    """Export clutter simulation results to GRaSP output files.

    Writes a ``.img`` file for the cluttergram, a ``.img`` file for
    the echomap, a ``.csv`` file with per-trace nadir and first-return
    diagnostics, and a ``.grsp`` descriptor for the cluttergram.

    Args:
        clutter_result: Output from :func:`simulate_clutter`.
        geometry: Observation geometry.
        out_dir: Output directory.
        base: Base filename (without extension).
        instrument: Instrument identifier.
        bin_size: Sampling period of the radar in seconds.
        byte_order: Byte order for binary output (``"big"`` or
            ``"little"``).
        verbose: If True, print progress messages.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emap_base = f"{base}_emap"

    img_path = out_dir / f"{base}.img"
    emap_path = out_dir / f"{emap_base}.img"
    csv_path = out_dir / f"{base}.csv"
    grsp_path = out_dir / f"{base}.grsp"

    if verbose:
        print(f"Writing CSIM output: {base}")
        print(f"\tCluttergram: {img_path}")
        print(f"\tEcho map:    {emap_path}")
        print(f"\tCSV:         {csv_path}")
        print(f"\tGRSP:        {grsp_path}")

    # Cluttergram binary.
    write_grasp_img(clutter_result.cluttergram, img_path,
                    byte_order=byte_order)

    # Echo map binary.
    write_grasp_img(clutter_result.echomap, emap_path,
                    byte_order=byte_order)

    # CSV with per-trace diagnostics.
    write_csim_csv(
        csv_path,
        latitude=geometry.latitude,
        longitude=geometry.longitude,
        nadir_twtt=clutter_result.nadir_twtt,
        fret_twtt=clutter_result.fret_twtt,
    )

    # Descriptor file for the cluttergram.
    n_samp, n_col = clutter_result.cluttergram.shape
    write_grasp_grsp(
        grsp_path,
        img_path=img_path,
        csv_path=csv_path,
        instrument=instrument,
        product_id=base,
        dtype="FLOAT32",
        n_samp=n_samp,
        n_col=n_col,
        byte_order=byte_order.upper(),
        dt=bin_size,
        dx=-9999.0,
        rho_a=-9999.0,
    )

    if verbose:
        print("CSIM data export complete.")


def write_csim_csv(filepath: str | Path,
                   latitude: NDArray,
                   longitude: NDArray,
                   nadir_twtt: NDArray,
                   fret_twtt: NDArray,
                   ) -> None:
    """Write per-trace clutter simulation diagnostics to a CSV file.

    Args:
        filepath: Output file path.
        latitude: Latitude per trace in degrees, shape (n_traces,).
        longitude: Longitude per trace in degrees, shape (n_traces,).
        nadir_twtt: Nadir two-way travel time per trace in seconds,
            shape (n_traces,).
        fret_twtt: First-return two-way travel time per trace in
            seconds, shape (n_traces,). NaN for traces with no
            valid return.
    """
    n_traces = len(latitude)
    header = "TRACE_INDEX,LATITUDE,LONGITUDE,NADIR_TWTT,FRET_TWTT"

    with open(filepath, "w") as f:
        f.write(header + "\n")
        for i in range(n_traces):
            line = (
                f"{i},"
                f"{latitude[i]:.6f},"
                f"{longitude[i]:.6f},"
                f"{nadir_twtt[i]:.12e},"
                f"{fret_twtt[i]:.12e}"
            )
            f.write(line + "\n")


def export_csim_images(clutter_result: ClutterResult,
                       *,
                       out_dir: Path,
                       base: str,
                       column_weights: NDArray | None = None,
                       apply_curve: bool = True,
                       invert: bool = False,
                       nadir_color: tuple[int, int, int] = (50, 200, 200),
                       fret_color: tuple[int, int, int] = (255, 0, 255),
                       marker_radius: int = 0,
                       verbose: bool = False,
                       ) -> None:
    """Export clutter simulation images.

    Produces a grayscale BMP of the cluttergram and a PNG browse
    image with the echomap strip and nadir/FRET markers. Both
    cluttergram and echomap use simc's ``combined``-style
    max-normalize on linear power. ``apply_curve`` adds the simc
    soft-shoulder tone-mapping LUT to the cluttergram (the echomap
    is rendered without the curve since its peak-power distribution
    is much tighter and the curve washes it out).

    Args:
        clutter_result: Output from :func:`simulate_clutter`.
        out_dir: Output directory.
        base: Base filename (without extension).
        column_weights: Optional per-trace multiplier of length
            ``n_traces``. Applied to both cluttergram and echomap
            before the max-normalize, e.g. radar-equation ``R^4``
            path-loss compensation for spacecraft with varying
            altitude.
        apply_curve: If True (default), apply the simc tone-mapping
            LUT to the cluttergram BMP and the cluttergram half of the
            browse image.
        invert: If True, invert the grayscale.
        nadir_color: RGB tuple for the nadir marker.
        fret_color: RGB tuple for the FRET marker.
        marker_radius: Half-width of the marker in pixels.
        verbose: If True, print progress messages.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiff_path = out_dir / f"{base}.bmp"
    browse_path = out_dir / f"{base}_browse.png"

    if verbose:
        print(f"Writing CSIM images:")
        print(f"\tCluttergram: {tiff_path}")
        print(f"\tBrowse:      {browse_path}")

    # Grayscale TIFF of cluttergram.
    csim_to_image(
        clutter_result.cluttergram,
        tiff_path,
        column_weights=column_weights,
        apply_curve=apply_curve,
        invert=invert,
    )

    # Browse PNG with echomap strip.
    cluttergram_with_echomap(
        clutter_result.cluttergram,
        clutter_result.echomap,
        clutter_result.nadir_ct_idx,
        clutter_result.fret_ct_idx,
        browse_path,
        column_weights=column_weights,
        apply_curve=apply_curve,
        invert=invert,
        nadir_color=nadir_color,
        fret_color=fret_color,
        marker_radius=marker_radius,
    )

    if verbose:
        print("CSIM image export complete.")

#####################################################################################################################
#
# SEG-Y
#
#####################################################################################################################
def write_segy(data: NDArray,
               filepath: str | Path,
               dt: float,
               latitude: NDArray,
               longitude: NDArray,
               *,
               obs_start: str | np.datetime64 | None = None,
               instrument: str = "",
               product_id: str = "",
               product_type: str = "",
               src_crs: str | CRS | Path | None = None,
               tgt_crs: str | CRS | Path | None = None,
               verbose: bool = False,
               ) -> Path:
    """Write a radargram to a SEG-Y Rev 1 file.

    Complex input data is automatically converted to power (float32)
    via ``10 * log10(|data|^2)``. Real input data is written as-is.

    Trace headers are populated with coordinates. If a target CRS is
    provided, coordinates are reprojected from the source CRS and
    stored in meters. Otherwise, latitude and longitude are stored
    in decimal degrees using the SEG-Y scalar convention.

    Args:
        data: Radargram array, shape (n_samp, n_col). May be complex
            or real valued.
        filepath: Output SEG-Y file path.
        dt: Range sample spacing in seconds.
        latitude: Latitude per trace in degrees, shape (n_col,).
        longitude: Longitude per trace in degrees, shape (n_col,).
        obs_start: Observation start time as an ISO-8601 string or
            ``numpy.datetime64`` scalar. If ``None``, the text-header
            start-time field is left blank.
        instrument: Instrument name for the text header.
        product_id: Product/observation identifier for the text header.
        product_type: Product type string for the text header.
        src_crs: Source coordinate reference system for the input
            lat/lon. Accepts a ``pyproj.CRS``, an authority string
            (e.g. ``"ESRI:104971"``), or a ``Path`` to a WKT/PRJ
            file. Required if ``tgt_crs`` is provided.
        tgt_crs: Target projected CRS for reprojection. Same accepted
            forms as ``src_crs``. If provided, coordinates are
            transformed to this CRS and stored in meters. If None,
            lat/lon are stored in decimal degrees.
        verbose: If True, print progress messages.

    Returns:
        Resolved path to the output SEG-Y file.

    Raises:
        ValueError: If ``tgt_crs`` is provided without ``src_crs``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Normalize CRS inputs (accept str / Path / CRS / None uniformly)
    src_crs = normalize_crs(src_crs)
    tgt_crs = normalize_crs(tgt_crs)

    if tgt_crs is not None and src_crs is None:
        raise ValueError("src_crs is required when tgt_crs is provided.")

    # Convert complex data to power
    if np.iscomplexobj(data):
        power = np.abs(data) ** 2
        power[power == 0] = np.finfo(np.float32).tiny
        data = (10.0 * np.log10(power)).astype(np.float32)
    else:
        data = np.asarray(data, dtype=np.float32)

    n_samp, n_col = data.shape

    # SEG-Y sample interval in microseconds (stored as integer)
    dt_us = int(round(dt * 1e6))

    # Coordinate setup
    if tgt_crs is not None:
        transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        x, y = transformer.transform(longitude, latitude)
        coord_units = 1  # meters
        scalar = 100  # store cm precision
    else:
        x = np.asarray(longitude, dtype=np.float64)
        y = np.asarray(latitude, dtype=np.float64)
        coord_units = 3  # decimal degrees
        scalar = 10000  # 4 decimal places

    if verbose:
        print(f"Writing SEG-Y: {filepath}")

    # Create initial SEG-Y from array
    segyio.tools.from_array2D(
        str(filepath),
        data.T,
        iline=189,
        xline=193,
        format=5,
        dt=dt_us,
    )

    # Build text header
    obs_start = np.datetime64(obs_start) if isinstance(obs_start, str) else obs_start
    if obs_start is None:
        obs_start = ""
    bbox_deg = [
        round(float(longitude[0]), 5),
        round(float(latitude[0]), 5),
        round(float(longitude[-1]), 5),
        round(float(latitude[-1]), 5),
    ]

    text_header = {
        1: f"INSTRUMENT: {instrument.upper()}",
        2: f"PRODUCT_TYPE: {product_type.upper()}",
        3: f"PRODUCT_ID: {product_id}",
        4: f"OBS_START: {obs_start}",
        5: f"N_SAMP: {n_samp}",
        6: f"N_COL: {n_col}",
        7: f"DT: {dt}",
        8: f"OUTPUT_FILE: {filepath.name}",
        9: f"DATE_CREATED: {date.today()}",
        10: f"BOUNDING_BOX_DEG: {bbox_deg}",
        39: "SEG-Y REV 1",
        40: "END TEXTUAL HEADER",
    }

    if tgt_crs is not None:
        bbox_m = [
            round(float(x[0]), 4),
            round(float(y[0]), 4),
            round(float(x[-1]), 4),
            round(float(y[-1]), 4),
        ]
        text_header[11] = f"BOUNDING_BOX_M: {bbox_m}"
        text_header[12] = f"SRC_CRS: {src_crs.to_wkt()[:60]}..."
        text_header[13] = f"TGT_CRS: {tgt_crs.to_wkt()[:60]}..."

    # Open and customize
    with segyio.open(str(filepath), "r+") as f:
        # Binary headers
        if product_id:
            try:
                obs_number = int(product_id)
            except ValueError:
                obs_number = 0
        else:
            obs_number = 0

        f.bin = {
            segyio.BinField.LineNumber: obs_number,
            segyio.BinField.ReelNumber: obs_number,
            segyio.BinField.MeasurementSystem: 1,
        }

        # Trace headers
        for tr in range(n_col):
            f.header[tr] = {
                segyio.TraceField.CoordinateUnits: coord_units,
                segyio.TraceField.SourceGroupScalar: -1 * scalar,
                segyio.TraceField.GroupX: int(x[tr] * scalar),
                segyio.TraceField.GroupY: int(y[tr] * scalar),
            }

        # Text header
        f.text[0] = segyio.tools.create_text_header(text_header)

    if verbose:
        print(f"SEG-Y complete: {filepath}")

    return filepath.resolve()

#####################################################################################################################
#
# HDF5
#
#####################################################################################################################
def save_hdf5(filepath: str | Path,
              state: dict[str, dict],
              verbose: bool = False,
              ) -> None:
    """Save observation state to an HDF5 file.

    The ``state`` dict should map group names to sub-dicts. Within
    each sub-dict, numpy arrays are written as HDF5 datasets and
    scalars, strings, booleans, and None values are written as
    group attributes.

    Args:
        filepath: Output HDF5 file path.
        state: Dict-of-dicts representing the observation state.
            Top-level keys become HDF5 groups. Values within each
            sub-dict become datasets (arrays) or attributes
            (scalars/strings).
        verbose: If True, print progress messages.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Saving state to: {filepath}")

    with h5py.File(str(filepath), "w") as f:
        for group_name, group_data in state.items():
            if group_data is None:
                continue
            grp = f.create_group(group_name)
            _write_group(grp, group_data, verbose=verbose)

    if verbose:
        print("Save complete.")


def _write_group(grp: h5py.Group,
                 data: dict,
                 verbose: bool = False,
                 ) -> None:
    """Write a dict into an HDF5 group.

    Arrays become datasets. Dicts become sub-groups (recursive).
    Everything else becomes an attribute.

    Args:
        grp: HDF5 group to write into.
        data: Dict of key-value pairs to write.
        verbose: If True, print dataset/attribute names.
    """
    for key, value in data.items():
        if value is None:
            grp.attrs[key] = "__NONE__"
            continue

        if isinstance(value, dict):
            sub = grp.create_group(key)
            _write_group(sub, value, verbose=verbose)
            continue

        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.datetime64):
                # HDF5 doesn't support datetime64; store as ISO strings
                str_arr = np.datetime_as_string(value)
                dt = h5py.string_dtype()
                grp.create_dataset(key, data=str_arr.astype(object), dtype=dt)
            elif value.dtype.kind == 'U' or value.dtype.kind == 'O':
                # String arrays
                dt = h5py.string_dtype()
                grp.create_dataset(key, data=value.astype(object), dtype=dt)
            else:
                grp.create_dataset(key, data=value, compression="gzip",
                                   compression_opts=4)
            if verbose:
                print(f"\tDataset: {grp.name}/{key} {value.shape} {value.dtype}")
            continue

        if isinstance(value, (bool, np.bool_)):
            grp.attrs[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            grp.attrs[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            grp.attrs[key] = float(value)
        elif isinstance(value, str):
            grp.attrs[key] = value
        elif isinstance(value, Path):
            grp.attrs[key] = str(value)
        elif isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                grp.create_dataset(key, data=arr, compression="gzip",
                                   compression_opts=4)
            except (ValueError, TypeError):
                # Fall back to string representation
                grp.attrs[key] = str(value)
        else:
            grp.attrs[key] = str(value)

        if verbose and key in grp.attrs:
            print(f"\tAttr:    {grp.name}/{key} = {grp.attrs[key]}")


