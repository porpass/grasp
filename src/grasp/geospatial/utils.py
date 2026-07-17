# SPDX-License-Identifier: BSD-3-Clause
import os

import rasterio
from pyproj import Transformer
from rasterio.crs import CRS


from .mola_info import (MOLA_METADATA, MOLA_VALID, MOLA_PPD_CODES, MOLA_DATA_IDS, MOLA_FILENAME_TEMPLATES,
                       MOLA_REGIONS, MOLA_TYPES, MOLA_RESOLUTIONS, MOLA_RES_MAP)
import re

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from .mola_info import get_mola_root



def _parse_mola_filename(mola_filename):
    """Parse a standard MOLA PDS filename.

    MEGDR data files are named according to the scheme ``MEGkxxdyyyrv.IMG``,
    where:

    - ``k``: ``A`` for areoid, ``C`` for counts, ``R`` for radius,
      ``T`` for topography.
    - ``xx``: latitude of pixel in upper-left corner of the image.
    - ``d``: ``N`` for north latitude, ``S`` for south.
    - ``yyy``: longitude of the pixel in the upper-left corner of the image
      (0-360 east).
    - ``r``: map resolution in pixels per degree (``C`` = 4 pix/deg,
      ``E`` = 16 pix/deg, ``F`` = 32 pix/deg, ``G`` = 64 pix/deg,
      ``H`` = 128 pix/deg).
    - ``v``: version letter.

    North and south polar maps have names in the form ``MEGk_p_rrr_v.IMG``,
    where:

    - ``k``: ``T`` for topography, ``R`` for radius, or ``C`` for counts.
    - ``p``: ``N`` for north pole or ``S`` for south pole.
    - ``rrr``: resolution in pixels per degree (128, 256, or 512).
    - ``v``: product version.

    Args:
        mola_filename: MOLA filename in the PDS naming standard.

    Returns:
        Dictionary with parsed components (``region``, ``type``, ``hemisphere``,
        ``res``, and, for global files, ``lat`` and ``lon``), or ``None`` if
        the filename does not match a known MOLA pattern.

    Raises:
        ValueError: If the resolution character cannot be mapped to a known
            pixels-per-degree value.
    """
    mola_filename = os.path.basename(mola_filename).split('.')[0].lower()
    pattern1 = re.compile(r'^meg(?P<type>[acrt])(?P<lat>\d{2})(?P<hemisphere>[ns])(?P<lon>\d{3})(?P<res>[efgh])b$')
    pattern2 = re.compile(r'^meg(?P<type>[crt])_(?P<hemisphere>[ns])_(?P<res>128|256|512)$')
    match1 = pattern1.match(mola_filename)
    if match1:
        result = match1.groupdict()
        res_char = result['res'].lower()
        if res_char not in MOLA_RES_MAP:
            raise ValueError(f"Invalid res {result['res']}")
        result['res'] = MOLA_RES_MAP[res_char]
        return {'region': 'GLOBAL', **result}

    match2 = pattern2.match(mola_filename)
    if match2:
        result = match2.groupdict()
        result['res'] = int(result['res'])
        return {'region': 'POLAR', **result}
    return None


def set_mola_metadata(mola_filename: str) -> None:
    """Populate the ``MOLA_METADATA`` dictionary from a MOLA filename.

    Reads region (``GLOBAL`` or ``POLAR``), data type, and resolution from the
    filename and writes the corresponding sample format, record geometry,
    projection, and reference radii into the module-level ``MOLA_METADATA``.

    Args:
        mola_filename: MOLA filename in the PDS naming standard.

    Raises:
        ValueError: If the parsed MOLA data type or resolution is not
            recognized.
        NotImplementedError: For tiled global resolutions (>=64 pix/deg) and
            other regions that have not yet been wired up.
    """
    fileInfo = _parse_mola_filename(mola_filename)
    metadata = MOLA_METADATA
    if fileInfo['region'] == 'POLAR':
        metadata['COORDINATE_SYSTEM_TYPE'] = "BODY-FIXED ROTATING"
        metadata['COORDINATE_SYSTEM_NAME'] = "PLANETOCENTRIC"
        if fileInfo['type'] == "c":
            metadata['SAMPLE_TYPE'] = "MSB_UNSIGNED_INTEGER"
            metadata['SAMPLE_BITS'] = 8
            metadata['OFFSET'] = 0
            metadata['SCALING_FACTOR'] = 1
        elif fileInfo['type'] == "t":
            metadata['SAMPLE_TYPE'] = "MSB_UNSIGNED_INTEGER"
            metadata['SAMPLE_BITS'] = 16
            metadata['SCALING_FACTOR'] = 0.25
            metadata['OFFSET'] = -8000
        elif fileInfo['type'] == "r":
            metadata['SAMPLE_TYPE'] = "IEEE_REAL"
            metadata['SAMPLE_BITS'] = 32
            metadata['OFFSET'] = 3396000
            metadata['SCALING_FACTOR'] = 1
        else:
            raise ValueError(f"Invalid type {fileInfo['type']}")
        if fileInfo['res'] == 128:
            metadata['FILE_RECORDS'] = 10240
            metadata['RECORD_BYTES'] = 10240
            metadata['LINES'] = 10240
            metadata['LINE_SAMPLES'] = 10240
            metadata['MAP_RESOLUTION'] = 128
            metadata['MAP_SCALE'] = 0.460
        #elif fileInfo['res'] == 256:
        #elif fileInfo['res'] == 512:
        else:
            raise ValueError(f"Invalid res: {fileInfo['res']}")
        metadata['MAP_PROJECTION_TYPE'] = "POLAR STEREOGRAPHIC"
        if fileInfo['type'] in ["c", "t"]:
            metadata['A_AXIS_RADIUS'] = 3376.2
            metadata['B_AXIS_RADIUS'] = 3376.2
            metadata['C_AXIS_RADIUS'] = 3376.2
        elif fileInfo['type'] == "r":
            metadata['A_AXIS_RADIUS'] = 3396.0
            metadata['B_AXIS_RADIUS'] = 3396.0
            metadata['C_AXIS_RADIUS'] = 3396.0
    elif fileInfo['region'] == 'GLOBAL':
        # 4, 16, 32 are not tiled
        if fileInfo['res'] == 4:
            metadata['FILE_RECORDS'] = 720
            metadata['RECORD_BYTES'] = 2880
        elif fileInfo['res'] == 16:
            metadata['FILE_RECORDS'] = 2280
            metadata['RECORD_BYTES'] = 11520
        elif fileInfo['res'] == 32:
            metadata['FILE_RECORDS'] = 5760
            metadata['RECORD_BYTES'] = 23040
        elif fileInfo['res'] == 64:
            # Tiling begins
            raise NotImplementedError

        metadata['MAP_PROJECTION_TYPE'] = "SIMPLE CYLINDRICAL"
        metadata['A_AXIS_RADIUS'] = 3396.0
        metadata['B_AXIS_RADIUS'] = 3396.0
        metadata['C_AXIS_RADIUS'] = 3396.0
    else:
        raise NotImplementedError
    #elif fileInfo['region'] == 'POLAR':


def select_mola_file(lon: float,
                     lat: float,
                     dataType: str = "radius",
                     map_resolution: int = 128,
                     region: str = "global") -> Path:
    """Select the MOLA file containing the requested longitude/latitude.

    Args:
        lon: Longitude in degrees, in the range ``[0, 360]``.
        lat: Latitude in degrees, in the range ``[-90, 90]``.
        dataType: MOLA data type. Accepted values: ``counts``, ``areoid``,
            ``radius``, ``topography``.
        map_resolution: Map resolution in pixels per degree. Accepted values:
            4, 16, 32, 64, 128, 256, 512.
        region: Map region. Accepted values: ``GLOBAL``, ``POLAR``.

    Returns:
        Absolute path to the matching MOLA file.

    Raises:
        ValueError: If any input is outside its supported set, if the
            requested combination is not valid, or if the resolved file does
            not exist on disk.
    """
    dataType = dataType.upper()
    region = region.upper()
    lon = lon + 360 if lon < 0 else lon  # Force 0-360° longitudes
    #
    # Check inputs
    #
    if dataType not in MOLA_TYPES:
        raise ValueError(f"{dataType.upper()} is not a supported MOLA data type")
    if map_resolution not in MOLA_RESOLUTIONS:
        raise ValueError(f"{map_resolution} is not a valid resolution")
    if region not in MOLA_REGIONS:
        raise ValueError(f"{region} is not a valid region")
    if lon < -180 or lon > 360:
        raise ValueError(f"{lon} is not a valid longitude")
    #
    # Check if VALID
    #
    try:
        supported_res = MOLA_VALID.get(dataType).get(region).get('MAP_RESOLUTIONS')
    except:
        raise ValueError(f"Invalid request: {dataType} {region} {map_resolution}")
    if map_resolution not in supported_res:
        raise ValueError(f"{map_resolution} is not a valid resolution for {dataType} {region}")
    base = MOLA_FILENAME_TEMPLATES[region]  # Get basename template
    # Set data type ID
    k = MOLA_DATA_IDS.get(dataType)
    if region == "GLOBAL":
        resStr = str(map_resolution)
        r = MOLA_PPD_CODES.get(region).get(resStr)
        if map_resolution in [4,16,32]:
            xx = "90"
            yyy = "000"
            d= "n"
        elif map_resolution == 64:
            d = "n"
            # Determine upper right corner latitude
            if 0 <= lat < 90:
                xx = "90"
            elif -90 < lat < 0:
                xx = "00"
            else:
                raise ValueError(f"Invalid latitude: {lat}")
            # Determine upper right corner longitude
            if 0 <= lon <= 180:
                yyy = "000"
            elif 180 < lon <= 360:
                yyy = "180"
            else:
                raise ValueError(f"Invalid longitude: {lon}")
        elif map_resolution == 128:
            # Determine upper right corner latitude
            if 44 <= lat <= 88:
                xx = "88"
                d = "n"
            elif 0 <= lat < 44:
                xx = "44"
                d = "n"
            elif -44 <= lat < 0:
                xx = "00"
                d = "n"
            elif -88 <= lat < -44:
                xx = "44"
                d = "s"
            else:
                raise ValueError(f"Invalid latitude: {lat}")
            # Determine upper right corner longitude
            if 0 <= lon <= 90:
                yyy = "000"
            elif 90 < lon <= 180:
                yyy = "090"
            elif 180 < lon <= 270:
                yyy = "180"
            elif 270 < lon <= 360:
                yyy = "270"
            else:
                raise ValueError(f"Invalid longitude: {lon}")
        else:
            raise ValueError(f"Invalid map_resolution: {map_resolution}")
        base = base.format(k, xx,d, yyy, r)
    elif region == "POLAR":
        d = "n" if lat > 0 else "s"
        base = base.format(k, d, map_resolution)
    else:
        raise ValueError(f"Invalid request: {dataType} {region} {map_resolution}")
    mola_file = os.path.join(get_mola_root(), base)
    if os.path.isfile(mola_file):
        return mola_file
    else:
        raise ValueError(f"File {mola_file} does not exist")

def coords_to_pixels(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    transform: rasterio.transform.Affine,
    shape: tuple[int, int],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    """Convert map coordinates to pixel indices.

    Args:
        x: X coordinates in the raster's CRS.
        y: Y coordinates in the raster's CRS.
        transform: Affine geotransform of the raster.
        shape: (n_rows, n_cols) of the raster.

    Returns:
        A tuple (row, col, in_bounds) where row and col are
        integer pixel indices and in_bounds is True where the
        coordinates fall within the raster extent.
    """
    inv = ~transform
    col_f, row_f = inv * (x, y)
    col = np.round(col_f).astype(int)
    row = np.round(row_f).astype(int)
    in_bounds = (
        (row >= 0) & (row < shape[0])
        & (col >= 0) & (col < shape[1])
    )
    return row, col, in_bounds

def read_dem_window(dem_path: str | Path,
                    lat_min: float,
                    lat_max: float,
                    lon_min: float,
                    lon_max: float,
                    *,
                    source_crs: CRS | None = None,
                    pad_pixels: int = 2,
                    ) -> tuple[NDArray[np.floating],
                               rasterio.transform.Affine,
                               CRS,
                               float | None]:
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        scale = src.scales[0] if src.scales[0] is not None else 1.0
        offset = src.offsets[0] if src.offsets[0] is not None else 0.0
        dem_nodata = src.nodata

        if source_crs is not None:
            from_crs = source_crs
        elif dem_crs.is_projected:
            from_crs = dem_crs.geodetic_crs
        else:
            from_crs = dem_crs

        to_dem = Transformer.from_crs(from_crs, dem_crs, always_xy=True)
        x0, y0 = to_dem.transform(lon_min, lat_min)
        x1, y1 = to_dem.transform(lon_max, lat_max)

        x_lo, x_hi = min(x0, x1), max(x0, x1)
        y_lo, y_hi = min(y0, y1), max(y0, y1)

        inv = ~src.transform
        col_a, row_a = inv * (x_lo, y_lo)
        col_b, row_b = inv * (x_hi, y_hi)

        row_start = max(int(np.floor(min(row_a, row_b))) - pad_pixels, 0)
        row_stop = min(int(np.ceil(max(row_a, row_b))) + pad_pixels, src.height)
        col_start = max(int(np.floor(min(col_a, col_b))) - pad_pixels, 0)
        col_stop = min(int(np.ceil(max(col_a, col_b))) + pad_pixels, src.width)

        if row_start >= row_stop or col_start >= col_stop:
            raise ValueError(
                "Bounding box does not intersect the DEM. "
                f"Requested rows [{row_start}, {row_stop}), "
                f"cols [{col_start}, {col_stop}) in a "
                f"{src.height}×{src.width} raster."
            )

        window = rasterio.windows.Window.from_slices(
            (row_start, row_stop), (col_start, col_stop),
        )

        dem_data = src.read(1, window=window).astype(np.float64)
        dem_transform = src.window_transform(window)

    if scale != 1.0 or offset != 0.0:
        dem_data = dem_data * scale + offset
        if dem_nodata is not None:
            dem_nodata = dem_nodata * scale + offset

    return dem_data, dem_transform, dem_crs, dem_nodata
