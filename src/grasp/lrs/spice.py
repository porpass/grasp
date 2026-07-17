# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import SpiceParams

SPICE_PARAMS = SpiceParams(
        target_int = 301,
        target_str = "MOON",
        observer_int = -131,
        observer_str =  "SELENE",
        method = "INTERCEPT/ELLIPSOID",
        method2 = "ELLIPSOID",
        ab_corr = "None",
        fix_ref = "IAU_MOON",
)