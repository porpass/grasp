# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import SpiceParams

SPICE_PARAMS = SpiceParams(
        target_int = 499,
        target_str = "MARS",
        observer_int = -74,
        observer_str = "MRO",
        method = "INTERCEPT/ELLIPSOID",
        method2 = "ELLIPSOID",
        ab_corr = "LT+S",
        fix_ref = "IAU_MARS",
)