# SPDX-License-Identifier: BSD-3-Clause
"""
Signal processing window utilities.

This module re-exports commonly used time-domain window functions and
frequency-domain bandlimiting helpers from `grasp.signal_processing.common.windows`.

Only the names listed in __all__ are considered part of the public API.
"""

from .windows import (broadening_factor, form_window, form_window_bandlimited,
                      )
from .utils import to_complex_baseband

from .range_compression import (create_complex_baseband_chirp,
                                create_sharad_calibrated_chirp,
                                create_filter, range_compress)
from .emi import (adaptive_spectral_notch, suppress_emi, threshold_emi)
from .ionosphere import ionosphere_campbell, ionospheric_compensation, ionosphere_contrast

__all__ = [
    "adaptive_spectral_notch",
    "broadening_factor",
    "create_complex_baseband_chirp",
    "create_filter",
    "create_sharad_calibrated_chirp",
    "form_window",
    "form_window_bandlimited",
    "ionosphere_campbell",
    "ionospheric_compensation",
    "ionosphere_contrast",
    "suppress_emi",
    "threshold_emi",
    "to_complex_baseband",
]