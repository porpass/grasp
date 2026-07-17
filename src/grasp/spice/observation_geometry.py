# SPDX-License-Identifier: BSD-3-Clause

from numpy.typing import NDArray
import numpy as np
from typing import Any
import spiceypy as sp
from .utils import et2utc, utc2et

from ..grasp_types import GeodeticResult, GeometryResult, IllumResult, VelocityComponents, StateVectors


def compute_geometry(
        ets: NDArray[np.floating[Any]],
        obsrvr_str: str,
        target_str: str,
        fix_ref_str: str,
        *,
        ab_corr: str = None,
        intercept_method_str: str = "NEAR POINT/ELLIPSOID",
        compute_illum: bool = False,
        utc: bool = False,
    ) -> GeometryResult:
    """Compute full observation geometry for a set of ephemeris times.

    Wraps ``compute_state_vectors``, ``compute_geodetic_position``,
    and ``decompose_velocity`` into a single call that returns a
    fully populated ``GeometryResult``. Optionally computes
    illumination angles via ``compute_sza``.

    Args:
        ets: Observation times, shape (N,). If ``utc=True``, these
            are UTC strings that will be converted to ephemeris
            time internally.
        obsrvr_str: SPICE observer body name (e.g. ``'MRO'``).
        target_str: SPICE target body name (e.g. ``'MARS'``).
        fix_ref_str: SPICE body-fixed reference frame
            (e.g. ``'IAU_MARS'``).
        ab_corr: Aberration correction flag (e.g. ``'LT+S'``).
        intercept_method_str: Surface intercept method passed to
            ``spice.subpnt``.
        compute_illum: If True, compute illumination angles
            (phase, solar zenith, emission).
        utc: If True, treat ``ets`` as UTC strings and convert
            to ephemeris time before processing.

    Returns:
        GeometryResult with all available fields populated.
    """
    ets = np.asarray(ets)

    if utc:
        epoch = np.char.replace(ets.astype(str), ' ', 'T').astype('datetime64[ms]')
        ets = utc2et(ets)
    else:
        strs = et2utc(ets)
        if isinstance(strs, np.ndarray):
            strs = np.char.replace(strs, ' ', 'T')
        epoch = strs.astype('datetime64[ms]')

    state = compute_state_vectors(
        ets, obsrvr_str, target_str, fix_ref_str,
        ab_corr=ab_corr,
        intercept_method_str=intercept_method_str,
        utc=False,
    )

    geodetic = compute_geodetic_position(
        state.r_t, state.r_s, state.r_st,
    )

    velocity = decompose_velocity(state.r_s, state.v_s)

    phase_angle = None
    solar_zenith_angle = None
    emission_angle = None

    if compute_illum:
        illum = compute_sza(
            state.et, obsrvr_str, target_str, state.r_t,
            ab_corr=ab_corr or 'LT',
        )
        phase_angle = illum.phase
        solar_zenith_angle = illum.solar
        emission_angle = illum.emission

    return GeometryResult(
        et=ets,
        epoch=epoch,
        longitude=geodetic.longitude,
        latitude=geodetic.latitude,
        altitude=geodetic.altitude,
        sc_radius=geodetic.sc_radius,
        el_radius=geodetic.el_radius,
        r_s=state.r_s,
        v_s=state.v_s,
        r_t=state.r_t,
        v_t=state.v_t,
        r_st=state.r_st,
        v_radial=velocity.v_radial,
        v_tangential=velocity.v_tangential,
        phase_angle=phase_angle,
        solar_zenith_angle=solar_zenith_angle,
        emission_angle=emission_angle,
    )

def compute_sza(ets: NDArray[np.floating[Any]],
                obsrvr_str: str,
                target_str: str,
                spoint: NDArray[np.floating[Any]],
                *,
                ab_corr: str = 'LT',
                utc: bool = False,
                ) -> IllumResult:
    """Compute illumination angles at a series of surface points via SPICE.

    Wraps the SPICE ``illum_c`` function to compute the phase, solar incidence
    (solar zenith angle), and emission angles at each surface intercept point
    along an observation ground track. One set of angles is computed per epoch.

    Args:
        ets: Array of observation times. If ``utc`` is True, these are UTC
            strings converted internally to ephemeris time (ET). If ``utc`` is
            False, these are ET values in seconds past J2000. Shape (n_col,).
        obsrvr_str: SPICE observer string for the spacecraft
            (e.g. ``'SELENE'``).
        target_str: SPICE target body string (e.g. ``'MOON'``).
        spoint: Array of surface intercept points in body-fixed Cartesian
            coordinates (km), one per epoch. Shape (3, n_col).
        ab_corr: Aberration correction flag passed to ``illum_c``.
            Defaults to ``'LT'``.
        utc: If True, ``ets`` contains UTC timestamps which are converted to
            ET via ``spiceypy.utc2et`` before calling ``illum_c``. If False,
            ``ets`` are already in ET seconds past J2000. Defaults to False.

    Returns:
        IllumResult: Dataclass containing three arrays of shape (n_col,):

            - ``phase``: Phase angle in radians at each epoch.
            - ``solar``: Solar incidence angle (SZA) in radians at each epoch.
            - ``emission``: Emission angle in radians at each epoch.
    """
    ets = np.asarray(ets)
    n_col = len(ets)
    p = np.zeros(n_col, dtype=np.float32)
    s = np.zeros(n_col, dtype=np.float32)
    e = np.zeros(n_col, dtype=np.float32)
    for _e in range(n_col):
        if utc:
            et = sp.utc2et(str(ets[_e]))
        else:
            et = ets[_e]
        phase, solar, emissn = sp.illum(
            target_str,
            et,
            ab_corr,
            obsrvr_str,
            spoint[:, _e],
        )
        p[_e] = phase
        s[_e] = solar
        e[_e] = emissn
    return IllumResult(
        phase=p,
        solar=s,
        emission=e,
    )


def compute_geodetic_position(r_t: NDArray[np.floating[Any]],
                              r_s: NDArray[np.floating[Any]],
                              r_st: NDArray[np.floating[Any]],
                              ) -> GeodeticResult:
    """Compute observation geodectic position from SPICE-derived position vectors.

    Derives altitude, spacecraft and ellipsoid radii, and planetocentric
    latitude/longitude of the sub-spacecraft surface intercept point.

    Args:
        r_t (NDArray[np.floating]): Sub-spacecraft point on the reference
            ellipsoid relative to COM. Shape (3,), (3,1), (1,3), (3,N),
            or (N,3).
        r_s (NDArray[np.floating]): Spacecraft position vector relative to
            COM. Shape (3,), (3,1), (1,3), (3,N), or (N,3).
        r_st (NDArray[np.floating]): Observer-to-target surface vector
            (r_s to r_t). Shape (3,), (3,1), (1,3), (3,N), or (N,3).

    Returns:
        GeometryResult: Frozen dataclass with fields:
            - **longitude** (*NDArray[np.floating]*): Planetocentric east
              longitude in degrees [0, 360). Shape (N,) or scalar.
            - **latitude** (*NDArray[np.floating]*): Planetocentric latitude
              in degrees. Shape (N,) or scalar.
            - **altitude** (*NDArray[np.floating]*): Altitude above the
              reference ellipsoid in km. Shape (N,) or scalar.
            - **sc_radius** (*NDArray[np.floating]*): Spacecraft radius from
              target COM in km. Shape (N,) or scalar.
            - **el_radius** (*NDArray[np.floating]*): Ellipsoid radius at
              sub-spacecraft point in km. Shape (N,) or scalar.
    """
    r_s  = np.atleast_2d(r_s) / 1000 # m -> km
    r_t  = np.atleast_2d(r_t) / 1000 # m -> km
    r_st = np.atleast_2d(r_st) / 1000 # m -> km

    if r_s.shape[0] != 3:
        r_s = r_s.T
    if r_t.shape[0] != 3:
        r_t = r_t.T
    if r_st.shape[0] != 3:
        r_st = r_st.T

    alt       = np.linalg.norm(r_st, axis=0)
    sc_radius = np.linalg.norm(r_s, axis=0)
    el_radius = np.linalg.norm(r_t, axis=0)

    _, lon, lat = zip(*[sp.reclat(r_t[:, _e]) for _e in range(r_t.shape[1])])
    lon_deg = np.array(lon) * sp.dpr()
    lat_deg = np.array(lat) * sp.dpr()
    lon_deg[lon_deg < 0] += 360.0

    return GeodeticResult(longitude=lon_deg.squeeze(),
                          latitude=lat_deg.squeeze(),
                          altitude=1000*alt.squeeze(),
                          sc_radius=1000*sc_radius.squeeze(),
                          el_radius=1000*el_radius.squeeze(),
                          )


def compute_state_vectors(ets: NDArray[np.floating[Any]],
                          obsrvr_str: str,
                          target_str: str,
                          fix_ref_str: str,
                          *,
                          ab_corr: str = None,
                          intercept_method_str: str = "NEAR POINT/ELLIPSOID",
                          utc: bool = False,
                         ) -> StateVectors:
    """Compute spacecraft and target position/velocity state vectors via SPICE.

    For each epoch, queries SPICE for the spacecraft state and sub-spacecraft
    surface intercept point, returning all vectors in a frozen dataclass.

    Args:
        ets (NDArray[np.floating]): Ephemeris times. Shape (N,). If ``utc=True``,
            these should be UTC strings convertible via ``spice.utc2et``.
        obsrvr_str (str): SPICE observer body name (e.g. ``'MRO'``).
        target_str (str): SPICE target body name (e.g. ``'MARS'``).
        fix_ref_str (str): SPICE body-fixed reference frame (e.g. ``'IAU_MARS'``).
        ab_corr (str, optional): Aberration correction flag (e.g. ``'LT+S'``).
            Defaults to None.
        intercept_method_str (str, optional): Surface intercept method passed to
            ``spice.subpnt``. Defaults to ``'NEAR POINT/ELLIPSOID'``.
        utc (bool, optional): If True, treat ``ets`` as UTC strings and convert
            via ``spice.utc2et``. Defaults to False.

    Returns:
        StateVectors: Frozen dataclass with fields:
            - **et** (*NDArray[np.floating]*): Ephemeris times. Shape (N,).
            - **r_s** (*NDArray[np.floating]*): Spacecraft position relative to
              target COM. Shape (3, N).
            - **v_s** (*NDArray[np.floating]*): Spacecraft velocity relative to
              target COM. Shape (3, N).
            - **r_t** (*NDArray[np.floating]*): Sub-spacecraft point on target
              ellipsoid relative to COM. Shape (3, N).
            - **v_t** (*NDArray[np.floating]*): Target velocity relative to COM.
              Shape (3, N).
            - **r_st** (*NDArray[np.floating]*): Observer-to-target surface
              vector. Shape (3, N).
    """
    ets = np.asarray(ets)
    n_col = len(ets)
    r_s  = np.zeros((3, n_col), dtype=np.float32)
    v_s  = np.zeros((3, n_col), dtype=np.float32)
    r_t  = np.zeros((3, n_col), dtype=np.float32)
    v_t  = np.zeros((3, n_col), dtype=np.float32)  # Zero by definition in body-fixed frame
    r_st = np.zeros((3, n_col), dtype=np.float32)

    for _e in range(n_col):
        if utc:
            et = sp.utc2et(str(ets[_e]))
        else:
            et = ets[_e]
        starg, lt = sp.spkezr(obsrvr_str, et, fix_ref_str, ab_corr, target_str)
        spoint, trgepx, srfvec = sp.subpnt(intercept_method_str, target_str, et, fix_ref_str, ab_corr, obsrvr_str)
        r_s[:, _e]  = 1000*starg[:3]
        v_s[:, _e]  = 1000*starg[3:]
        r_t[:, _e]  = 1000*spoint[:]
        r_st[:, _e] = 1000*srfvec[:]

    return StateVectors(et=ets.astype(float),
                        r_s=r_s,
                        v_s=v_s,
                        r_t=r_t,
                        v_t=v_t,
                        r_st=r_st)


def decompose_velocity(r: NDArray[np.floating[Any]],
                       v: NDArray[np.floating[Any]]) -> VelocityComponents:
    """Decompose a velocity vector into radial and tangential components
    relative to a position vector.

    Args:
        r (NDArray[np.floating]): Position vector(s). Shape (3,), (3,1), (1,3),
            (3,N), or (N,3).
        v (NDArray[np.floating]): Velocity vector(s). Shape (3,), (3,1), (1,3),
            (3,N), or (N,3).

    Returns:
        VelocityComponents: Frozen dataclass with fields:
            - **v_radial** (*float or ndarray*): Radial speed, component of v
              along r. Shape (N,) or scalar.
            - **v_tangential** (*float or ndarray*): Tangential speed, component
              of v perpendicular to r. Shape (N,) or scalar.
    """
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)

    if r.shape[0] != 3:
        r = r.T
        v = v.T

    R = np.linalg.norm(r, axis=0)
    V = np.linalg.norm(v, axis=0)
    v_r = (np.einsum('ie,ie->e', r, v) / R)
    v_t = np.sqrt(V**2 - v_r**2)

    return VelocityComponents(v_radial=v_r.squeeze(),
                              v_tangential=v_t.squeeze())