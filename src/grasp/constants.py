# SPDX-License-Identifier: BSD-3-Clause
"""Physical constants used across GRaSP.

Most are re-exported from scipy.constants under domain-relevant names.
PLASMA_FREQ_COEFF is the only derived value, included because it
appears throughout the ionosphere literature (Cartacci, Picardi, etc.)
and is implicit in the contrast method's a3/a4 derivations.
"""
import math

from scipy.constants import (
    c as C,
    epsilon_0 as EPSILON_0,
    mu_0 as MU_0,
    k as K_B,
    e as ELEMENTARY_CHARGE,
    m_e as ELECTRON_MASS,
)

# f_p[Hz] = PLASMA_FREQ_COEFF * sqrt(N_e[m^-3])
PLASMA_FREQ_COEFF = 8.9787
FREE_SPACE_IMPEDANCE = math.sqrt(MU_0 / EPSILON_0)

# SHARAD-empirical ionospheric calibration (Campbell et al. 2014 / 2016).
# delay_seconds = (E / 1e16) * CAMPBELL_DELAY_COEFF
# TEC          = CAMPBELL_TEC_COEFF * E
CAMPBELL_DELAY_COEFF = 1.4547e-6
CAMPBELL_TEC_COEFF = 0.29