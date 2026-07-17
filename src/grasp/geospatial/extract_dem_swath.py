# SPDX-License-Identifier: BSD-3-Clause
"""DEM swath extraction and local cartesian projection utilities.

This module provides functions for extracting elevation swaths from
georeferenced DEMs along radar sounder ground tracks, transforming
them into local cartesian coordinates, and preparing them for
visualization alongside radargram imagery.

Example usage::

    from grasp.geospatial.dem import extract_dem_swath

    swath = extract_dem_swath(lats, lons, "/path/to/mola_hrsc.tif",
                              buffer_km=10.0)
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any

import rasterio
from rasterio.crs import CRS
from pyproj import CRS as ProjCRS
from pyproj import Transformer


def extract_dem_swath(lats: NDArray,
                      lons: NDArray,
                      dem_path: str | Path,
                      track_crs: CRS,
                      *,
                      buffer_km: float = 10.0,
                      ) -> dict[str, NDArray]:
    """Extract a DEM swath along a ground track.

    Extracts elevation data from a georeferenced DEM in a corridor
    centered on the input ground track. The result is returned in
    a local cartesian frame with along-track and cross-track distances
    in kilometers.

    The planetary radius is derived from the DEM's CRS, making this
    function planet-agnostic (works with Mars, Moon, etc.).

    If ``track_crs`` is provided and differs from the DEM's CRS, the
    ground track coordinates are reprojected before sampling. If
    ``None``, the track and DEM are assumed to share the same CRS.

    Antimeridian crossings are handled automatically by wrapping
    sample longitudes and extracting from both edges of the raster
    when necessary.

    Args:
        lats: 1D array of ground track latitudes (degrees).
        lons: 1D array of ground track longitudes (degrees).
        dem_path: Path to a georeferenced DEM file readable by rasterio.
        buffer_km: Half-width of the cross-track swath in kilometers.
        track_crs: CRS of the input ground track coordinates. If
            ``None``, assumed to match the DEM's CRS.

    Returns:
        A dictionary containing:
            - ``elevation``: 2D array (cross-track × along-track) of
              elevation values in the DEM's native units.
            - ``along_track_km``: 1D array of cumulative along-track
              distances in kilometers.
            - ``cross_track_km``: 1D array of cross-track offsets in
              kilometers (negative = left, positive = right).
            - ``lats``: 2D latitude grid of the extracted swath.
            - ``lons``: 2D longitude grid of the extracted swath.
            - ``track_lats``: 1D input ground track latitudes
              (in DEM CRS, after reprojection if applicable).
            - ``track_lons``: 1D input ground track longitudes
              (in DEM CRS, after reprojection if applicable).

    Raises:
        ValueError: If ``lats`` and ``lons`` have different lengths,
            or if fewer than two ground track points are provided.
        FileNotFoundError: If ``dem_path`` does not exist.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    if lats.shape != lons.shape:
        raise ValueError(
            f"lats and lons must have the same shape: "
            f"got {lats.shape} and {lons.shape}"
        )
    if lats.size < 2:
        raise ValueError("At least two ground track points are required.")

    with rasterio.open(dem_path) as ds:
        dem_crs = ds.crs

        # Reproject track coordinates if a different CRS is provided
        track_lats, track_lons = _reproject_track(
            lats, lons, track_crs, dem_crs
        )

        # Get planetary radius from the DEM CRS
        #radius_km = ds.crs.ellipsoid.semi_major_metre / 1000.0
        proj_crs = ProjCRS(ds.crs.to_wkt())
        radius_km = proj_crs.ellipsoid.semi_major_metre / 1000.0

        # Compute along-track distances
        along_track_km = _compute_along_track_distances(
            track_lats, track_lons, radius_km
        )

        # Compute cross-track perpendicular bearings at each point
        bearings = _compute_track_bearings(track_lats, track_lons)

        # Build the sample grid (lat/lon for each cross-track/along-track point)
        n_cross = _compute_n_cross_samples(buffer_km, ds.transform, radius_km, ds.crs.is_projected)

        #cross_track_km = np.linspace(-buffer_km, buffer_km, n_cross)
        cross_track_km = np.linspace(buffer_km, -buffer_km, n_cross)

        sample_lats, sample_lons = _build_sample_grid(
            track_lats, track_lons, bearings, cross_track_km, radius_km
        )

        # Extract elevation values from the DEM
        elevation = _sample_dem(ds, sample_lats, sample_lons)

    return {
        "elevation": elevation,
        "along_track_km": along_track_km,
        "cross_track_km": cross_track_km,
        "lats": sample_lats,
        "lons": sample_lons,
        "track_lats": track_lats,
        "track_lons": track_lons,
    }


def _reproject_track(lats: NDArray,
                     lons: NDArray,
                     track_crs: CRS,
                     dem_crs: CRS,
                     ) -> tuple[NDArray, NDArray]:
    """Reproject ground track coordinates to the DEM's CRS if needed.

    Args:
        lats: Ground track latitudes in the track CRS.
        lons: Ground track longitudes in the track CRS.
        track_crs: CRS of the input coordinates
        dem_crs: CRS of the target DEM.

    Returns:
        Tuple of (latitudes, longitudes) in the DEM's CRS.
    """
    if dem_crs.is_projected:
        target_crs = dem_crs.geodetic_crs
    else:
        target_crs = dem_crs

    if track_crs == target_crs:
        return lats.copy(), lons.copy()

    transformer = Transformer.from_crs(track_crs, target_crs, always_xy=True)
    new_lons, new_lats = transformer.transform(lons, lats)
    return np.asarray(new_lats, dtype=np.float64), np.asarray(new_lons, dtype=np.float64)


def _compute_along_track_distances(lats: NDArray,
                                   lons: NDArray,
                                   radius_km: float,
                                   ) -> NDArray:
    """Compute cumulative along-track distances using the Haversine formula.

    Args:
        lats: Ground track latitudes (degrees).
        lons: Ground track longitudes (degrees).
        radius_km: Planetary radius in kilometers.

    Returns:
        1D array of cumulative distances in kilometers, starting at 0.
    """
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)

    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) *
         np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    segment_distances = radius_km * c
    return np.concatenate([[0.0], np.cumsum(segment_distances)])


def _compute_track_bearings(lats: NDArray,
                            lons: NDArray,
                            ) -> NDArray:
    """Compute forward azimuth (bearing) at each ground track point.

    Uses the forward azimuth formula for a sphere. At interior points,
    the bearing is averaged from the incoming and outgoing segments.
    Endpoints use the adjacent segment's bearing.

    Args:
        lats: Ground track latitudes (degrees).
        lons: Ground track longitudes (degrees).

    Returns:
        1D array of bearings in radians (clockwise from north).
    """
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    n = len(lats)

    # Forward azimuths for each segment
    dlon = np.diff(lon_rad)
    y = np.sin(dlon) * np.cos(lat_rad[1:])
    x = (np.cos(lat_rad[:-1]) * np.sin(lat_rad[1:]) -
         np.sin(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.cos(dlon))
    segment_bearings = np.arctan2(y, x)

    # Assign bearings to each point
    bearings = np.empty(n, dtype=np.float64)
    bearings[0] = segment_bearings[0]
    bearings[-1] = segment_bearings[-1]

    if n > 2:
        # Average adjacent segment bearings for interior points
        # Use circular mean to handle wrapping
        b1 = segment_bearings[:-1]
        b2 = segment_bearings[1:]
        bearings[1:-1] = np.arctan2(
            np.sin(b1) + np.sin(b2),
            np.cos(b1) + np.cos(b2)
        )

    return bearings


def _compute_n_cross_samples(buffer_km: float,
                             transform,
                             radius_km: float,
                             is_projected: bool,
                             ) -> int:
    """Determine the number of cross-track samples to match DEM resolution.

        Uses the DEM pixel size to compute an appropriate number of
        cross-track samples spanning ``2 * buffer_km``. Handles both
        geographic DEMs (pixel size in degrees) and projected DEMs
        (pixel size in metres).

        Args:
            buffer_km: Half-width of the cross-track swath in kilometers.
            transform: Rasterio affine transform of the DEM.
            radius_km: Planetary radius in kilometers.
            is_projected: If True, the DEM is in a projected CRS and
                ``transform.a`` is in metres. If False, ``transform.a``
                is in degrees.

        Returns:
            Number of cross-track sample points (always odd, so the
            track centerline is a sample point).
        """
    if is_projected:
        pixel_km = abs(transform.a) / 1000.0
    else:
        pixel_deg = abs(transform.a)
        pixel_km = np.deg2rad(pixel_deg) * radius_km

    n_cross = max(int(np.round(2 * buffer_km / pixel_km)), 3)

    if n_cross % 2 == 0:
        n_cross += 1

    return n_cross


def _build_sample_grid(track_lats: NDArray,
                       track_lons: NDArray,
                       bearings: NDArray,
                       cross_track_km: NDArray,
                       radius_km: float,
                       ) -> tuple[NDArray, NDArray]:
    """Build a 2D grid of sample lat/lon points for the swath.

    At each ground track point, sample points are placed along the
    perpendicular (cross-track) direction at the specified offsets.

    Args:
        track_lats: Ground track latitudes (degrees).
        track_lons: Ground track longitudes (degrees).
        bearings: Along-track bearings in radians at each point.
        cross_track_km: 1D array of cross-track offsets in km.
        radius_km: Planetary radius in kilometers.

    Returns:
        Tuple of 2D arrays (sample_lats, sample_lons) with shape
        ``(n_cross, n_along)``.
    """
    n_along = len(track_lats)
    n_cross = len(cross_track_km)

    sample_lats = np.empty((n_cross, n_along), dtype=np.float64)
    sample_lons = np.empty((n_cross, n_along), dtype=np.float64)

    lat_rad = np.deg2rad(track_lats)
    lon_rad = np.deg2rad(track_lons)

    for i, offset_km in enumerate(cross_track_km):
        # Perpendicular bearing: track bearing + 90° (right) or -90° (left)
        # Positive offset = right of track, negative = left
        if offset_km >= 0:
            perp_bearing = bearings - np.pi / 2  # left of track
        else:
            perp_bearing = bearings + np.pi / 2  # right of track

        angular_dist = abs(offset_km) / radius_km

        # Destination point given start, bearing, and angular distance
        dest_lat = np.arcsin(
            np.sin(lat_rad) * np.cos(angular_dist) +
            np.cos(lat_rad) * np.sin(angular_dist) * np.cos(perp_bearing)
        )
        dest_lon = lon_rad + np.arctan2(
            np.sin(perp_bearing) * np.sin(angular_dist) * np.cos(lat_rad),
            np.cos(angular_dist) - np.sin(lat_rad) * np.sin(dest_lat)
        )

        sample_lats[i, :] = np.rad2deg(dest_lat)
        sample_lons[i, :] = np.rad2deg(dest_lon)

    return sample_lats, sample_lons


def _sample_dem(ds,
                sample_lats: NDArray,
                sample_lons: NDArray,
                ) -> NDArray:
    """Sample elevation values from a DEM at the given coordinates.

    Handles both geographic and projected CRS by transforming sample
    coordinates to the DEM's native coordinate system before sampling.
    Only the bounding window of the sample points is read from disk,
    avoiding loading the full raster into memory.

    Antimeridian crossings are handled for geographic DEMs by wrapping
    longitudes to the DEM's native range. When sample points span both
    edges of the raster, two separate windows are read and combined.

    Args:
        ds: Open rasterio dataset.
        sample_lats: 2D array of sample latitudes (degrees).
        sample_lons: 2D array of sample longitudes (degrees).

    Returns:
        2D array of elevation values with the same shape as
        the input coordinate arrays.
    """
    shape = sample_lats.shape
    nodata = ds.nodata
    height, width = ds.height, ds.width

    flat_lats = sample_lats.ravel()
    flat_lons = sample_lons.ravel()

    if ds.crs.is_projected:
        geographic_crs = ds.crs.geodetic_crs
        transformer = Transformer.from_crs(
            geographic_crs, ds.crs, always_xy=True
        )
        xs, ys = transformer.transform(flat_lons, flat_lats)
    else:
        bounds = ds.bounds
        lon_min = bounds.left
        lon_range = bounds.right - lon_min
        xs = ((flat_lons - lon_min) % lon_range) + lon_min
        ys = flat_lats

    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    rows = np.full(xs.size, -1, dtype=np.int32)
    cols = np.full(xs.size, -1, dtype=np.int32)

    if finite_mask.any():
        valid_rows, valid_cols = rasterio.transform.rowcol(
            ds.transform, xs[finite_mask], ys[finite_mask]
        )
        rows[finite_mask] = np.asarray(valid_rows)
        cols[finite_mask] = np.asarray(valid_cols)

    elevation = np.full(rows.size, np.nan, dtype=np.float64)

    # Check for antimeridian crossing: columns span both edges of the raster
    col_span = int(cols.max()) - int(cols.min())
    antimeridian_crossing = (not ds.crs.is_projected) and (col_span > width // 2)

    if antimeridian_crossing:
        # Split into left-edge and right-edge groups
        mid_col = width // 2
        left_mask = cols < mid_col
        right_mask = ~left_mask

        for mask in (left_mask, right_mask):
            if not mask.any():
                continue

            _read_window(ds, dem_data=None, rows=rows, cols=cols,
                         mask=mask, elevation=elevation,
                         height=height, width=width)
    else:
        mask = np.ones(rows.size, dtype=bool)
        _read_window(ds, dem_data=None, rows=rows, cols=cols,
                     mask=mask, elevation=elevation,
                     height=height, width=width)

    if nodata is not None:
        elevation[elevation == nodata] = np.nan

    return elevation.reshape(shape)


def _read_window(ds,
                 dem_data,
                 rows: NDArray,
                 cols: NDArray,
                 mask: NDArray,
                 elevation: NDArray,
                 height: int,
                 width: int,
                 pad: int = 2,
                 ) -> None:
    """Read a windowed subset of a DEM and sample elevation values.

    Computes the bounding window for the masked subset of sample
    points, reads only that region from disk, and writes the sampled
    values into the corresponding positions in the output array.

    Args:
        ds: Open rasterio dataset.
        dem_data: Unused, reserved for future caching.
        rows: 1D array of row indices for all sample points.
        cols: 1D array of column indices for all sample points.
        mask: Boolean mask selecting which sample points belong
            to this window.
        elevation: 1D output array to write elevation values into.
        height: Total raster height in pixels.
        width: Total raster width in pixels.
        pad: Pixel buffer to add around the bounding window.
    """
    sub_rows = rows[mask]
    sub_cols = cols[mask]

    # Clamp to valid pixel range before computing window bounds
    valid_pixels = ((sub_rows >= 0) & (sub_rows < height) &
                    (sub_cols >= 0) & (sub_cols < width))

    if not valid_pixels.any():
        return

    valid_rows = sub_rows[valid_pixels]
    valid_cols = sub_cols[valid_pixels]

    row_min = max(int(valid_rows.min()) - pad, 0)
    row_max = min(int(valid_rows.max()) + pad + 1, height)
    col_min = max(int(valid_cols.min()) - pad, 0)
    col_max = min(int(valid_cols.max()) + pad + 1, width)

    window = rasterio.windows.Window.from_slices(
        (row_min, row_max), (col_min, col_max)
    )
    window_data = ds.read(1, window=window)

    local_rows = sub_rows - row_min
    local_cols = sub_cols - col_min
    win_h, win_w = window_data.shape

    local_valid = ((local_rows >= 0) & (local_rows < win_h) &
                   (local_cols >= 0) & (local_cols < win_w))

    # Map back to the full elevation array
    full_indices = np.where(mask)[0]
    elevation[full_indices[local_valid]] = window_data[
        local_rows[local_valid], local_cols[local_valid]
    ]