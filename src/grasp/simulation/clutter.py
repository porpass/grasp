# SPDX-License-Identifier: BSD-3-Clause
"""Surface clutter simulation for orbital radar sounder data.

This module generates synthetic cluttergrams — images of off-nadir surface
return power — from a digital elevation model and spacecraft navigation
data. The simulation discretizes the surface into triangular facets on
both sides of the ground track, computes two-way travel time and
scattered power for each facet, and bins the results into a radargram-
shaped array.

The core algorithm follows the approach of the MultiSim clutter simulator
(O'Connell, University of Arizona), re-implemented as vectorized numpy
operations for integration into GRaSP.  Coordinate transforms follow the
same strategy as the original: grid points are transformed **directly**
between geocentric XYZ and the DEM's projected CRS, avoiding an
intermediate geographic (lon/lat) step that introduces longitude-wrapping
artifacts near the 0°/360° boundary.

References:
    O'Connell, J. (2021). Multi Clutter Simulator.
    https://github.com/oconnellj2/clutter-simulator

    Choudhary, P., Holt, J. W., & Kempf, S. D. (2016).
    Surface clutter and echo location analysis for the
    interpretation of SHARAD data from Mars.
    IEEE Geoscience and Remote Sensing Letters, 13(9), 1285-1289.


TODO Clutter simulator misbehaves on polar observations; bug I keep encountering.
  Doesn't occur in DEM Extraction though.

"""
from ..grasp_types import ClutterResult

import numpy as np
import rasterio
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from ..constants import C
from ..geospatial.utils import coords_to_pixels, read_dem_window
from ..geospatial.crs import gcs_2000_crs, geocent_crs
from ..spice.utils import get_radii


# ---------------------------------------------------------------------------
# Geometry helpers (all operate on (N, 3) arrays)
# ---------------------------------------------------------------------------

def _direction_vectors(xyz: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute unit direction-of-travel vectors for a sequence of XYZ positions.

    For interior points the direction is estimated from the symmetric
    difference of neighbours; endpoints use the one-sided difference.

    Args:
        xyz: Spacecraft positions in geocentric Cartesian coordinates,
            shape (n_traces, 3).

    Returns:
        Unit direction vectors, shape (n_traces, 3).
    """
    d = np.empty_like(xyz)
    d[0] = xyz[1] - xyz[0]
    d[-1] = xyz[-1] - xyz[-2]
    d[1:-1] = xyz[2:] - xyz[:-2]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return d / norms


def _cross_track_vectors(direction: NDArray[np.floating],
                         nadir_vec: NDArray[np.floating],
                         ) -> NDArray[np.floating]:
    """Compute unit cross-track vectors perpendicular to flight and nadir.

    The cross-track vector points to the right side of the spacecraft
    (looking in the direction of travel).

    Args:
        direction: Unit along-track direction vectors, shape (n_traces, 3).
        nadir_vec: Unit vectors from ground nadir to spacecraft,
            shape (n_traces, 3).

    Returns:
        Unit cross-track vectors, shape (n_traces, 3).
    """
    cross = np.cross(direction, nadir_vec)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return cross / norms


def _build_grid(xcvr: NDArray[np.floating],
                along: NDArray[np.floating],
                cross: NDArray[np.floating],
                at_step: float,
                at_dist: float,
                ct_step: float,
                ct_dist: float,
                ) -> NDArray[np.floating]:
    """Build a rectangular grid of 3D points for one side of one trace.

    The grid is centred on the spacecraft position along-track and extends
    from nadir outward cross-track.

    Args:
        xcvr: Spacecraft XYZ position, shape (3,).
        along: Unit along-track vector, shape (3,).
        cross: Unit cross-track vector (pointing toward the side of
            interest), shape (3,).
        at_step: Along-track facet dimension in metres.
        at_dist: Along-track half-extent in metres.
        ct_step: Cross-track facet dimension in metres.
        ct_dist: Cross-track extent in metres (nadir to far edge).

    Returns:
        Grid of XYZ positions, shape (n_at, n_ct, 3), where
        n_at = 2 * int(at_dist / at_step) + 1 and
        n_ct = int(ct_dist / ct_step) + 1.
    """
    n_at = int(2 * (at_dist / at_step)) + 1
    n_ct = int(ct_dist / ct_step) + 1

    at_offsets = np.linspace(-at_dist, at_dist, n_at, dtype=np.float64)
    ct_offsets = np.arange(n_ct, dtype=np.float64) * ct_step

    grid = (
        xcvr[None, None, :]
        + at_offsets[:, None, None] * along[None, None, :]
        + ct_offsets[None, :, None] * cross[None, None, :]
    )
    return grid


def _project_grid_to_dem(grid: NDArray[np.floating],
                         dem_data: NDArray[np.floating],
                         dem_transform: rasterio.transform.Affine,
                         dem_crs: CRS,
                         dem_nodata: float | None,
                         xyz_crs: CRS,
                         ) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Replace grid Z values with DEM elevations.

    Grid points are transformed directly from geocentric XYZ to the
    DEM's projected coordinate system, sampled from the DEM, then
    transformed back to XYZ with the new elevation.  No intermediate
    geographic (lon/lat) step is used, which avoids longitude-wrapping
    artifacts near the 0°/360° boundary.

    Args:
        grid: XYZ grid positions, shape (n_at, n_ct, 3).
        dem_data: DEM elevation array, shape (rows, cols).
        dem_transform: Affine geotransform of the DEM.
        dem_crs: Coordinate reference system of the DEM.
        dem_nodata: DEM no-data value (or None).
        xyz_crs: Geocentric Cartesian CRS.

    Returns:
        A tuple of:
          - ground: Grid with Z replaced by DEM elevation, shape
            (n_at, n_ct, 3), in geocentric XYZ.
          - nodata_mask: Boolean mask, True where DEM had no data,
            shape (n_at, n_ct).
    """
    orig_shape = grid.shape[:2]
    pts = grid.reshape(-1, 3)

    # XYZ → DEM projected CRS (direct, no lon/lat intermediate).
    to_dem = Transformer.from_crs(xyz_crs, dem_crs, always_xy=True)
    dem_x, dem_y, _ = to_dem.transform(pts[:, 0], pts[:, 1], pts[:, 2])

    row, col, in_bounds = coords_to_pixels(
        dem_x, dem_y, dem_transform, dem_data.shape,
    )

    elev = np.full(pts.shape[0], np.nan, dtype=np.float64)
    elev[in_bounds] = dem_data[row[in_bounds], col[in_bounds]]

    nodata_mask = ~in_bounds
    if dem_nodata is not None:
        nodata_mask |= (elev == dem_nodata)

    # DEM projected CRS (with new elevation) → XYZ (direct).
    to_xyz = Transformer.from_crs(dem_crs, xyz_crs, always_xy=True)
    gx, gy, gz = to_xyz.transform(dem_x, dem_y, elev)

    ground = np.column_stack([gx, gy, gz]).reshape(*orig_shape, 3)
    nodata_mask = nodata_mask.reshape(orig_shape)

    return ground, nodata_mask


def _facets_from_grid(grid: NDArray[np.floating],
                      nodata_mask: NDArray[np.bool_],
                      ct_step: float,
                      ) -> tuple[NDArray[np.floating],
                                 NDArray[np.floating],
                                 NDArray[np.floating],
                                 NDArray[np.bool_],
                                 NDArray[np.floating],
                                ]:
    """Generate triangular facets from a projected grid.

    Each grid cell (i, j) is split into two triangles. For an (n_at, n_ct)
    grid this produces 2 * (n_at - 1) * (n_ct - 1) facets.

    Args:
        grid: Surface XYZ positions, shape (n_at, n_ct, 3).
        nodata_mask: Boolean mask of no-data grid nodes,
            shape (n_at, n_ct).
        ct_step: Cross-track step size in metres, used to compute
            the cross-track distance of each facet centroid.

    Returns:
        A tuple of:
          - centers: Facet centroids, shape (n_facets, 3).
          - normals: Facet normal vectors (not normalised; magnitude
            equals twice the facet area), shape (n_facets, 3).
          - areas: Facet areas, shape (n_facets,).
          - nd_mask: True where any vertex had no data, shape (n_facets,).
          - ct_dists: Cross-track distance of each facet centroid in
            metres (measured in grid-index space), shape (n_facets,).
    """
    n_at_m1 = grid.shape[0] - 1
    n_ct_m1 = grid.shape[1] - 1

    # Upper triangles: (i,j+1), (i,j), (i+1,j)
    v0_u = grid[:-1, 1:, :]
    v1_u = grid[:-1, :-1, :]
    v2_u = grid[1:, :-1, :]

    # Lower triangles: (i+1,j), (i+1,j+1), (i,j+1)
    v0_l = grid[1:, :-1, :]
    v1_l = grid[1:, 1:, :]
    v2_l = grid[:-1, 1:, :]

    v0 = np.concatenate([v0_u.reshape(-1, 3), v0_l.reshape(-1, 3)], axis=0)
    v1 = np.concatenate([v1_u.reshape(-1, 3), v1_l.reshape(-1, 3)], axis=0)
    v2 = np.concatenate([v2_u.reshape(-1, 3), v2_l.reshape(-1, 3)], axis=0)

    centers = (v0 + v1 + v2) / 3.0

    e1 = v0 - v1
    e2 = v2 - v1
    normals = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(normals, axis=1)

    # Nodata propagation.
    nd_u = (
        nodata_mask[:-1, 1:]
        | nodata_mask[:-1, :-1]
        | nodata_mask[1:, :-1]
    )
    nd_l = (
        nodata_mask[1:, :-1]
        | nodata_mask[1:, 1:]
        | nodata_mask[:-1, 1:]
    )
    nd_mask = np.concatenate([nd_u.ravel(), nd_l.ravel()], axis=0)

    # Cross-track distance of each facet centroid (in grid-index units).
    # Upper triangle centroid CT index: mean of j+1, j, j = (j + 1/3)
    # Lower triangle centroid CT index: mean of j, j+1, j+1 = (j + 2/3)
    j_u = np.arange(n_ct_m1, dtype=np.float64) + 1.0 / 3.0
    j_l = np.arange(n_ct_m1, dtype=np.float64) + 2.0 / 3.0

    # Tile across along-track dimension.
    ct_idx_u = np.tile(j_u, n_at_m1)
    ct_idx_l = np.tile(j_l, n_at_m1)
    ct_dists = np.concatenate([ct_idx_u, ct_idx_l]) * ct_step

    return centers, normals, areas, nd_mask, ct_dists


def _compute_twtt_power(xcvr: NDArray[np.floating],
                        centers: NDArray[np.floating],
                        normals: NDArray[np.floating],
                        areas: NDArray[np.floating],
                        nd_mask: NDArray[np.bool_],
                        speed_of_light: float,
                        ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute two-way travel time and received power for all facets.

    The power model is a simplified radar equation:
        P = (A * cos(theta))^2 / R^4
    where A is the facet area, theta is the incidence angle, and R is the
    slant range from the transceiver to the facet centroid.

    Args:
        xcvr: Spacecraft position, shape (3,).
        centers: Facet centroids, shape (n_facets, 3).
        normals: Facet normal vectors, shape (n_facets, 3).
        areas: Facet areas, shape (n_facets,).
        nd_mask: True for invalid facets, shape (n_facets,).
        speed_of_light: Speed of light in m/s.

    Returns:
        A tuple of:
          - twtt: Two-way travel time in seconds, shape (n_facets,).
          - power: Received power (arbitrary units), shape (n_facets,).
            Zero for invalid facets or facets facing away from the
            spacecraft.
    """
    vsc = xcvr[None, :] - centers
    slant_range = np.linalg.norm(vsc, axis=1)

    twtt = 2.0 * slant_range / speed_of_light

    norm_mag = np.linalg.norm(normals, axis=1)
    dot = np.sum(vsc * normals, axis=1)
    denom = slant_range * norm_mag
    denom = np.where(denom == 0.0, 1.0, denom)
    cos_theta = np.abs(dot / denom)

    power = np.abs((areas * cos_theta) ** 2 / slant_range ** 4)

    power = np.where(nd_mask | (cos_theta == 0.0), 0.0, power)

    return twtt, power


def _bin_power(twtt: NDArray[np.floating],
               power: NDArray[np.floating],
               bin_size: float,
               datum: float,
               n_samples: int,
               ) -> NDArray[np.floating]:
    """Bin scattered power into range samples by two-way travel time.

    Maps each facet's absolute two-way travel time to a sample index
    using the datum (the absolute TWTT of sample 0).

    Args:
        twtt: Two-way travel time per facet, shape (n_facets,).
        power: Received power per facet, shape (n_facets,).
        bin_size: Sampling period of the radar in seconds.
        datum: Absolute two-way travel time of sample 0 for this
            trace in seconds, corrected for the difference between
            the ellipsoidal and spherical reference surfaces.
        n_samples: Number of range samples (trace length).

    Returns:
        Binned power trace, shape (n_samples,).
    """
    raw_bins = (twtt - datum) / bin_size
    finite = np.isfinite(raw_bins)
    # Clip before casting to int. Out-of-range bins get -1 (still filtered
    # out by the `valid` mask below), avoiding the int-overflow cast warning
    # on very large finite values.
    bounded = np.clip(np.where(finite, raw_bins, -1.0), -1.0, float(n_samples))
    bins = bounded.astype(int)
    valid = finite & (bins >= 0) & (bins < n_samples) & (power > 0.0)
    trace = np.zeros(n_samples, dtype=np.float64)
    np.add.at(trace, bins[valid], power[valid])
    return trace


def _bin_echomap(ct_dists: NDArray[np.floating],
                 power: NDArray[np.floating],
                 ct_step: float,
                 n_ct_bins: int,
                 side_sign: float,
                 ) -> NDArray[np.floating]:
    """Bin peak power into cross-track bins for the echo map.

    Args:
        ct_dists: Cross-track distance of each facet in metres
            (positive, measured from nadir), shape (n_facets,).
        power: Received power per facet, shape (n_facets,).
        ct_step: Cross-track bin spacing in metres.
        n_ct_bins: Total number of cross-track bins (left + right).
        side_sign: +1.0 for right side, -1.0 for left side.

    Returns:
        Echo map column, shape (n_ct_bins,). Values are peak power
        per cross-track bin.
    """
    center_bin = n_ct_bins // 2
    # Signed cross-track distance: positive = right, negative = left.
    signed_ct = side_sign * ct_dists
    ct_bins = (signed_ct / ct_step + center_bin).astype(int)

    valid = (ct_bins >= 0) & (ct_bins < n_ct_bins) & (power > 0.0)
    col = np.zeros(n_ct_bins, dtype=np.float64)
    # Peak power per bin.
    np.maximum.at(col, ct_bins[valid], power[valid])
    return col


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_clutter(lat: NDArray[np.floating],
                     lon: NDArray[np.floating],
                     alt: NDArray[np.floating],
                     datum: NDArray[np.floating],
                     dem_path: str,
                     target: str,
                     bin_size: float,
                     n_samples: int,
                     at_step: float,
                     at_dist: float,
                     ct_step: float,
                     ct_dist: float,
                     *,
                     n_center: int | None = None,
                     speed_of_light: float = C,
                     ) -> ClutterResult:
    """Simulate surface clutter for a radar sounder observation.

    Constructs triangular surface facets from a DEM on both sides of the
    ground track, computes two-way travel time and scattered power for each
    facet, and bins the results into a cluttergram array. Also produces an
    echo map showing peak power as a function of cross-track distance and
    records nadir and first-return two-way travel times.

    The output is datumed so that the ellipsoid nadir return is placed
    at sample ``n_center`` for each trace.

    Args:
        lat: Spacecraft geodetic latitude in degrees, shape (n_traces,).
        lon: Spacecraft longitude in degrees, shape (n_traces,).
        alt: Spacecraft altitude above the reference ellipsoid in metres,
            shape (n_traces,).
        datum: Absolute two-way travel time corresponding to sample 0
            for each trace in seconds, shape (n_traces,). Produced by
            ``remove_rwot_offset`` during preprocessing.
        dem_path: Path to a rasterio-readable DEM file.
        target: "``MARS``", "``MOON``", or "``PHOBOS``"
        bin_size: Sampling period of the radar in seconds.
        n_samples: Number of range samples per trace (trace length).
        at_step: Along-track facet dimension in metres.
        at_dist: Along-track half-extent of the grid in metres. Must be
            >= at_step.
        ct_step: Cross-track facet dimension in metres.
        ct_dist: Cross-track extent from nadir in metres. Must be
            >= ct_step.
        n_center: Sample index at which the ellipsoid surface return
            is centred. Defaults to ``n_samples // 2``.
        speed_of_light: Speed of light in m/s. Defaults to the value
            defined in ``grasp.constants``.

    Returns:
        A :class:`ClutterResult` containing the cluttergram, echo map,
        nadir and first-return information.

    Raises:
        ValueError: If input array shapes are inconsistent or facet
            parameters are invalid.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)

    if lat.ndim != 1 or lon.ndim != 1 or alt.ndim != 1:
        raise ValueError("lat, lon, alt must be 1D arrays")
    n_traces = lat.size
    if lon.size != n_traces or alt.size != n_traces:
        raise ValueError("lat, lon, alt must have the same length")
    if at_dist < at_step:
        raise ValueError("at_dist must be >= at_step")
    if ct_dist < ct_step:
        raise ValueError("ct_dist must be >= ct_step")

    if n_center is None:
        n_center = n_samples // 2

    # -----------------------------------------------------------------
    # Coordinate reference systems
    # -----------------------------------------------------------------
    # Geographic (lon/lat) and geocentric (XYZ) on the sphere.
    lle_crs = gcs_2000_crs(target.upper())
    xyz_crs = geocent_crs(target.upper())
    #
    # SPICE ellipsoid (triaxial), used for the datum correction below.
    # This is intentionally NOT the sphere from the DEM CRS — the datum
    # was authored on the SPICE ellipsoid.
    #
    _, radii = get_radii(target)  # returns (a_eq, b_eq, c_polar) in km
    a = radii[0] * 1000.0  # m
    b = radii[2] * 1000.0  # polar in m
    # -----------------------------------------------------------------
    # Navigation → geocentric XYZ
    # -----------------------------------------------------------------
    to_xyz = Transformer.from_crs(lle_crs, xyz_crs, always_xy=True)
    nav_x, nav_y, nav_z = to_xyz.transform(lon, lat, alt)
    nav_xyz = np.column_stack([nav_x, nav_y, nav_z])

    # -----------------------------------------------------------------
    # Read DEM corridor
    # -----------------------------------------------------------------
    buffer_deg = np.degrees(ct_dist / a) * 1.5
    lon_span = float(lon.max()) - float(lon.min())
    if lon_span > 180.0:
        dem_data, dem_transform, dem_crs, dem_nodata = read_dem_window(
            dem_path,
            lat_min=float(lat.min()) - buffer_deg,
            lat_max=float(lat.max()) + buffer_deg,
            lon_min=0.0,
            lon_max=359.99,
            source_crs=lle_crs,
        )
    else:
        dem_data, dem_transform, dem_crs, dem_nodata = read_dem_window(
            dem_path,
            lat_min=float(lat.min()) - buffer_deg,
            lat_max=float(lat.max()) + buffer_deg,
            lon_min=float(lon.min()) - buffer_deg,
            lon_max=float(lon.max()) + buffer_deg,
            source_crs=lle_crs,
        )
    # -----------------------------------------------------------------
    # Datum correction: ellipsoid → sphere
    # -----------------------------------------------------------------
    # The datum from remove_rwot_offset is referenced to the SPICE
    # ellipsoid (semi-axes body_a × body_b), but the DEM elevations
    # and therefore the facet geometry are referenced to a sphere of
    # radius body_a.  At polar latitudes the ellipsoid radius is up to
    # (body_a - body_b) ≈ 20 km smaller than the sphere, which would
    # shift all clutter bins by thousands of samples without this
    # correction.  Converting the datum to the spherical frame keeps
    # the simulated cluttergram aligned with the real radargram.
    lat_rad = np.radians(lat)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    el_radius = np.sqrt(
        ((a ** 2 * cos_lat) ** 2 + (b ** 2 * sin_lat) ** 2) /
        ((a * cos_lat) ** 2 + (b * sin_lat) ** 2)
    )
    dem_sphere_radius = CRS(dem_crs).geodetic_crs.ellipsoid.semi_major_metre
    datum_correction = 2.0 * (dem_sphere_radius - el_radius) / speed_of_light
    # -----------------------------------------------------------------
    # Nadir ground points (directly via DEM CRS, no lon/lat hop)
    # -----------------------------------------------------------------
    to_dem_crs = Transformer.from_crs(xyz_crs, dem_crs, always_xy=True)
    nad_dem_x, nad_dem_y, _ = to_dem_crs.transform(nav_x, nav_y, nav_z)

    nad_row, nad_col, nad_in_bounds = coords_to_pixels(
        nad_dem_x, nad_dem_y, dem_transform, dem_data.shape,
    )
    nad_elev = np.zeros(n_traces, dtype=np.float64)
    nad_elev[nad_in_bounds] = dem_data[
        nad_row[nad_in_bounds], nad_col[nad_in_bounds]
    ]

    to_xyz_from_dem = Transformer.from_crs(dem_crs, xyz_crs, always_xy=True)
    gnd_x, gnd_y, gnd_z = to_xyz_from_dem.transform(
        nad_dem_x, nad_dem_y, nad_elev,
    )
    gnd_xyz = np.column_stack([gnd_x, gnd_y, gnd_z])

    # -----------------------------------------------------------------
    # Along-track and cross-track vectors
    # -----------------------------------------------------------------
    direction = _direction_vectors(nav_xyz)
    nadir_vec = nav_xyz - gnd_xyz
    nadir_norms = np.linalg.norm(nadir_vec, axis=1, keepdims=True)
    nadir_norms = np.where(nadir_norms == 0.0, 1.0, nadir_norms)
    nadir_vec = nadir_vec / nadir_norms
    cross_vec = _cross_track_vectors(direction, nadir_vec)

    # -----------------------------------------------------------------
    # Output arrays
    # -----------------------------------------------------------------
    n_ct_bins = 2 * int(ct_dist / ct_step) + 1

    cluttergram = np.zeros((n_samples, n_traces), dtype=np.float64)
    echomap = np.zeros((n_ct_bins, n_traces), dtype=np.float64)
    nadir_twtt = np.full(n_traces, np.nan, dtype=np.float64)
    fret_twtt = np.full(n_traces, np.inf, dtype=np.float64)
    nadir_ct_idx = np.full(n_traces, n_ct_bins // 2, dtype=np.intp)
    fret_ct_idx = np.zeros(n_traces, dtype=np.intp)

    # Nadir TWTT from spacecraft to nadir DEM surface.
    nadir_range = np.linalg.norm(nav_xyz - gnd_xyz, axis=1)
    nadir_twtt[:] = 2.0 * nadir_range / speed_of_light

    # -----------------------------------------------------------------
    # Main simulation loop
    # -----------------------------------------------------------------
    for i in range(n_traces):
        xcvr = nav_xyz[i]
        for side_sign in [1.0, -1.0]:
            side_cross = side_sign * cross_vec[i]

            grid = _build_grid(
                xcvr, direction[i], side_cross,
                at_step, at_dist, ct_step, ct_dist,
            )

            ground, nd_mask = _project_grid_to_dem(
                grid, dem_data, dem_transform, dem_crs,
                dem_nodata, xyz_crs,
            )

            centers, normals, areas, facet_nd, ct_dists = _facets_from_grid(
                ground, nd_mask, ct_step,
            )

            twtt, power = _compute_twtt_power(
                xcvr, centers, normals, areas, facet_nd,
                speed_of_light,
            )

            # Cluttergram — datum_correction bridges the ellipsoidal
            # datum (from remove_rwot_offset) to the spherical DEM
            # reference used by the facet geometry.
            cluttergram[:, i] += _bin_power(
                twtt, power, bin_size,
                datum[i] + datum_correction[i],
                n_samples,
            )

            # Echo map.
            echomap[:, i] = np.maximum(
                echomap[:, i],
                _bin_echomap(ct_dists, power, ct_step, n_ct_bins, side_sign),
            )

            # First return: minimum TWTT with nonzero power.
            valid_power = power > 0.0
            if np.any(valid_power):
                min_idx = np.argmin(np.where(valid_power, twtt, np.inf))
                if twtt[min_idx] < fret_twtt[i]:
                    fret_twtt[i] = twtt[min_idx]
                    center_bin = n_ct_bins // 2
                    fret_ct_idx[i] = int(
                        side_sign * ct_dists[min_idx] / ct_step + center_bin
                    )

    # Replace inf with nan for traces with no valid FRET.
    fret_twtt[np.isinf(fret_twtt)] = np.nan

    return ClutterResult(
        cluttergram=cluttergram,
        echomap=echomap,
        nadir_twtt=nadir_twtt,
        fret_twtt=fret_twtt,
        nadir_ct_idx=nadir_ct_idx,
        fret_ct_idx=fret_ct_idx,
    )