# SPDX-License-Identifier: BSD-3-Clause

from .campbell import campbell_method as ionosphere_campbell
from .contrast import contrast_method as ionosphere_contrast
from .ionospheric_compensation import ionospheric_compensation

#TODO(Campbell & Watters 2016 §3): Along-track 7th-order polynomial
#   fit of the quadratic phase for MARSIS.
__all__ = [
    "ionosphere_campbell",
    "ionospheric_compensation",
    "ionosphere_contrast",
]