# SPDX-License-Identifier: BSD-3-Clause
from importlib.metadata import metadata, version
from email.utils import parseaddr

# TODO Audit __init__
__version__ = version("grasp")
_meta = metadata("grasp")
__author__ = parseaddr(_meta["Author-email"])[0]
__description__ = _meta["Summary"]
__license__ = _meta["License-Expression"]

# Custom fields with no PEP 621 home — keep as literals
__funding__ = "NASA Planetary Data Archival, Restoration, and Tools (PDART)"
__grants__ = "80NSSC20K1057"

###########################################################################
#
# Common module functions to include
#
###########################################################################
from .common import determine_observation_years

###########################################################################
#
# Input module functions to include
#
###########################################################################
from .input import (check_supported, grab_radar_params, identify_file,
                    load, load_job,read)
###########################################################################
#
# Processing module functions to include
#
###########################################################################
from .processing import (adaptive_spectral_notch, broadening_factor, create_complex_baseband_chirp,
                         create_filter, create_sharad_calibrated_chirp, form_window,
                         form_window_bandlimited, ionosphere_campbell, ionospheric_compensation,
                         ionosphere_contrast, range_compress, suppress_emi, threshold_emi, to_complex_baseband,
                         )

###########################################################################
#
# The radar_sounder class
#
###########################################################################
from .instantiator import radar_sounder

###########################################################################
#
# The Processor
#
###########################################################################
from .grasp import (
    print_job_summary,
    process_job as process
)
###########################################################################
#
# SPICE Routines TODO Audit SPICE routines
#
###########################################################################
from .spice import (compute_geodetic_position, compute_geometry,
                    compute_sza, compute_state_vectors,
                    decompose_velocity, et2utc,
                    find_mk_files, furnish, get_radii,
                    grab_spice_params, print_mk_paths,
                    unload, utc2et)
###########################################################################
#
# Geospatial Routines TODO Audit Geospatial routines
#
###########################################################################
from .geospatial.extract_dem_swath import extract_dem_swath
from .geospatial.crs import (
    MARS_LLE as mars_lle_crs,
    MOON_LLE as moon_lle_crs,
    PHOBOS_LLE as phobos_lle_crs,
    gcs_2000_crs, geocent_crs
)

#
# Azimuth Processing TODO Audit Azimuth Processing
#
from .processing.sar.unfocused import unfocused
from .processing.sar.range_doppler import backscatter, range_doppler
from .processing.sar.utils import (determine_aperture_bounds, determine_output_frames,
                                   determine_aperture_resolution, determine_aperture_step,
                                   max_unaliased_aperture, check_aperture)
from .postprocessing.multilook import multilook
#
# Output module functions to include TODO Audit Output module
#
from .output.images import to_image, radargram_with_dem
from .output.plotting import plot_dem_swath
from .output.writers import write_grasp_output, export_segy

# TODO Audit Simulation module
from .simulation.clutter import simulate_clutter

# TODO audit plotting module

__all__ = [
    ###################################################################################################################
    #
    # METADATA
    #
    ###################################################################################################################
    "__author__",
    "__license__",
    "__version__",
    "__description__",
    "__funding__",
    "__grants__",
    ###################################################################################################################
    #
    # Common Functions
    #
    ###################################################################################################################
    "determine_observation_years",
    ###################################################################################################################
    #
    # Input Functions
    #
    ###################################################################################################################
    "check_supported",
    "grab_radar_params",
    "identify_file",
    "load",
    "load_job",
    "read",
    ###################################################################################################################
    #
    # Classes
    #
    ###################################################################################################################
    "radar_sounder",
    ###################################################################################################################
    #
    # Processor
    #
    ###################################################################################################################
    "process",
    "print_job_summary",
    ###################################################################################################################
    #
    # SPICE Functions
    #
    ###################################################################################################################
    "compute_geodetic_position",
    "compute_geometry",
    "compute_state_vectors",
    "compute_sza",
    "decompose_velocity",
    "et2utc",
    "find_mk_files",
    "furnish",
    "get_radii",
    "grab_spice_params",
    "print_mk_paths",
    "unload",
    "utc2et",
    ###################################################################################################################
    #
    # Geospatial Functions
    #
    ###################################################################################################################
    "extract_dem_swath",
    "mars_lle_crs",
    "moon_lle_crs",
    "phobos_lle_crs",
    ###################################################################################################################
    #
    # Window Functions
    #
    ###################################################################################################################
    "form_window",
    "form_window_bandlimited",
    "create_complex_baseband_chirp",
    "to_complex_baseband",
    ###################################################################################################################
    #
    # Range Processing Functions
    #
    ###################################################################################################################
    "range_compress",
    ###################################################################################################################
    #
    # EMI Functions
    #
    ###################################################################################################################
    "adaptive_spectral_notch",
    "suppress_emi",
    "threshold_emi",
    ###################################################################################################################
    #
    # Ionospheric Correction Functions
    #
    ###################################################################################################################
    "ionosphere_campbell",
    "ionospheric_compensation",
    "ionosphere_contrast",
    ###################################################################################################################
    #
    # Common Azimuth Processing Functions
    #
    ###################################################################################################################
    "determine_aperture_resolution",
    "determine_aperture_step",
    "determine_aperture_bounds",
    "max_unaliased_aperture",
    "check_aperture",
    ###################################################################################################################
    #
    # Azimuth Processing
    #
    ###################################################################################################################
    "unfocused",
    "backscatter",
    "multilook",
    "simulate_clutter",
    ###################################################################################################################
    #
    # Plotting Functions
    #
    ###################################################################################################################
    "plot_dem_swath",
    ###################################################################################################################
    #
    # Output Functions
    #
    ###################################################################################################################
    # Output Functions
    "radargram_with_dem",
    "to_image",
    "write_grasp_output",
    "export_segy",
]