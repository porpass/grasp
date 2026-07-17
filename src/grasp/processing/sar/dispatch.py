# SPDX-License-Identifier: BSD-3-Clause
import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Literal

from ...common.utils import assert_gt_0
from ...grasp_types import SARResult
from .unfocused import unfocused
from .range_doppler import range_doppler, backscatter
from .utils import check_aperture

def sar_process(data: NDArray,
                method: str,
                presum: int,
                *,
                # Instrument Parameters
                pri: float,
                dz: float,
                lamb: float,
                # SAR Parameters
                l_n: int | None = None,
                d: float,
                os_factor: int = 1,
                number_of_looks: int = 5,
                remove_doppler_centroid: bool = False,
                aperture_var_threshold: float = 0.05,
                coherent: bool = False,
                # SAR Geometry
                r_st: NDArray[np.floating],
                r_s: NDArray[np.floating],
                r_t: NDArray[np.floating],
                vt: float | NDArray[np.floating],
                vr: float | NDArray[np.floating],
                # Azimuth Window
                window_type: str = "hanning",
                window_alpha: float | None = None,
                # Other
                interp_cval: float = 0.0,
                sgn: Literal[-1,1] = +1,
                ) -> tuple[NDArray, SARResult]:
    """Apply SAR azimuth processing to echo data.

    Computes along-track spacing from the effective PRI and
    tangential velocity, then applies the selected SAR method.
    Unfocused SAR uses a boxcar aperture sized by the Fresnel
    zone. Range-Doppler and Backscatter use the full focused
    SAR geometry with RCMC and phase compensation.

    The range-Doppler and backscatter methods use the start
    frequency rather than center frequency for wavelength
    calculation, which produces better imaging results for
    SHARAD.

    Args:
        data: Range-compressed echo data, shape
            ``(n_samp, n_cols)``.
        method: SAR method — ``"UNFOCUSED"``,
            ``"RANGE-DOPPLER"``, or ``"BACKSCATTER"``.
        presum: Along-track presumming factor.
        pri: Pulse repetition interval in seconds.
        dz: Range bin spacing in meters.
        lamb: Wavelength
        l_n: Aperture length in number of pulses. If None,
            computed from the Fresnel zone.
        d: Antenna length in meters.
        os_factor: Oversampling factor.
        number_of_looks: Number of looks for the backscatter
            method.
        remove_doppler_centroid: If True, remove the Doppler
            centroid before processing.
        aperture_var_threshold: Variance threshold for aperture
            selection.
        coherent: If True, apply coherent integration in the
            unfocused method.
        r_st: Spacecraft-to-target position vectors, shape
            ``(3, n_cols)``.
        r_s: Spacecraft position vectors, shape
            ``(3, n_cols)``.
        r_t: Target position vectors, shape ``(3, n_cols)``.
        vt: Tangential velocity in m/s. Scalar or array of
            length ``n_cols``.
        vr: Radial velocity in m/s. Scalar or array of
            length ``n_cols``.
        window_type: Azimuth window type.
        window_alpha: Tukey window taper fraction.
        sgn: Sign convention for phase exponential (-1 or +1).
        interp_cval: Fill value for RCMC interpolation
            outside the data bounds.

    Returns:
        A tuple ``(data, result)`` where:

        - ``data``: SAR-processed echo data.
        - ``result``: A ``SARResult`` containing the method,
            output frame indices, aperture step size,
            along-track resolution, and Doppler-filtered
            resolution (backscatter only).

    Raises:
        ValueError: If the SAR method is not supported or
            required parameters are invalid.
    """
    assert_gt_0(pri)
    assert_gt_0(dz)
    assert_gt_0(presum)

    pri_eff = pri * presum
    del_x = vt * pri_eff

    # --- Aperture guards ---
    # Clamp on input-column count (hard limit) and warn on Doppler-aliasing
    # (soft limit; long-mode SHARAD intentionally operates here and relies on
    # Doppler-window edge suppression to clean up the aliased signal).
    n_col = data.shape[1]
    R0_mean   = float(np.mean(np.linalg.norm(r_st, axis=0)))
    vt_mean   = float(np.mean(np.atleast_1d(vt)))
    delx_mean = float(np.mean(np.atleast_1d(del_x)))
    l_n_safe, ap_msgs = check_aperture(
        L_n_requested=l_n,
        n_col=n_col,
        lamb=lamb,
        vt=vt_mean,
        pri_eff=pri_eff,
        R0=R0_mean,
        del_x=delx_mean,
    )
    for m in ap_msgs:
        warnings.warn(m, RuntimeWarning, stacklevel=2)

    rho_df = None
    method = method.upper()

    if method == "UNFOCUSED":
        R0 = np.linalg.norm(r_st, axis=0)
        s_out, frames, aperture_step, rho_a = unfocused(data, del_x=del_x, R0=R0, lamb=lamb, L_n=l_n_safe,
                                                        os_factor=os_factor, presum=presum, window_type=window_type,
                                                        window_alpha=window_alpha, coherent=coherent,
                                                        aperture_var_threshold=aperture_var_threshold)
    elif method == "RANGE-DOPPLER":
        s_out, frames, aperture_step, rho_a = range_doppler(data, R_s=r_s, R_t=r_t, R_st=r_st, Vt=vt, Vr=vr,
                                                            del_x=del_x, lamb=lamb, pri=pri_eff, dz=dz, D=d,
                                                            L_n=l_n_safe, os_factor=os_factor, window_type=window_type,
                                                            window_alpha=window_alpha,
                                                            remove_doppler_centroid=remove_doppler_centroid,
                                                            interp_cval=interp_cval, sgn=sgn, verbose=True)

    elif method == "BACKSCATTER":
        s_out, frames, aperture_step, rho_a, rho_df = backscatter(data, R_s=r_s, R_t=r_t, R_st=r_st, Vt=vt, Vr=vr,
                                                                  del_x=del_x, lamb=lamb, pri=pri_eff, dz=dz,
                                                                  D=d, L_n=l_n_safe, os_factor=os_factor,
                                                                  window_type=window_type, window_alpha=window_alpha,
                                                                  n_mlk=number_of_looks,
                                                                  remove_doppler_centroid=remove_doppler_centroid,
                                                                  interp_cval=interp_cval, sgn=sgn, verbose=True)
    else:
        raise ValueError(f"{method} is not a supported SAR method")

    result = SARResult(
        method=method,
        frames=frames,
        aperture_step=aperture_step,
        rho_a=rho_a,
        rho_df=rho_df,
    )

    return s_out, result