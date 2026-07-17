# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from typing import Any

##############################################################################################################
#
# PDS EDRs
#
##############################################################################################################

def make_sharad_edr_aux_dict(n_cols: int,
                             ) -> dict[str, NDArray]:
    """
    Initialize a pre-allocated SHARAD EDR Aux data dictionary.

    This function creates a standard dictionary for SHARAD EDR Aux
    data. Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    Args:
        n_cols: Number of data columns.

    Returns:
        Preallocated dictionary for SHARAD EDR Aux data.
    """
    aux_dict: dict[str, NDArray[Any]] = {
                'SCET_BLOCK_WHOLE': np.zeros(n_cols, dtype=np.uint32),
                'SCET_BLOCK_FRAC': np.zeros(n_cols, dtype=np.uint16),
                'EPHEMERIS_TIME': np.zeros(n_cols, dtype=np.float64),
                'GEOMETRY_EPOCH': np.full(n_cols, np.datetime64('NaT', 'ms')),
                'SOLAR_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'ORBIT_NUMBER': np.zeros(n_cols, dtype=np.int32),
                'X_MARS_SC_POSITION_VECTOR': np.zeros(n_cols, dtype=np.float64),
                'Y_MARS_SC_POSITION_VECTOR': np.zeros(n_cols, dtype=np.float64),
                'Z_MARS_SC_POSITION_VECTOR': np.zeros(n_cols, dtype=np.float64),
                'SPACECRAFT_ALTITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_EAST_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_PLANETOCENTRIC_LATITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_PLANETOGRAPHIC_LATITUDE': np.zeros(n_cols, dtype=np.float64),
                'X_MARS_SC_VELOCITY_VECTOR': np.zeros(n_cols, dtype=np.float64),
                'Y_MARS_SC_VELOCITY_VECTOR': np.zeros(n_cols, dtype=np.float64),
                'Z_MARS_SC_VELOCITY_VECTOR': np.zeros(n_cols, dtype=np.float64),
                'MARS_SC_RADIAL_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'MARS_SC_TANGENTIAL_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'LOCAL_TRUE_SOLAR_TIME': np.zeros(n_cols, dtype=np.float64),
                'SOLAR_ZENITH_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'SC_PITCH_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'SC_YAW_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'SC_ROLL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'MRO_SAMX_INNER_GIMBAL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'MRO_SAMX_OUTER_GIMBAL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'MRO_SAPX_INNER_GIMBAL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'MRO_SAPX_OUTER_GIMBAL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'MRO_HGA_INNER_GIMBAL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'MRO_HGA_OUTER_GIMBAL_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'DES_TEMP': np.zeros(n_cols, dtype=np.float32),
                'DES_5V': np.zeros(n_cols, dtype=np.float32),
                'DES_12V': np.zeros(n_cols, dtype=np.float32),
                'DES_2V5': np.zeros(n_cols, dtype=np.float32),
                'RX_TEMP': np.zeros(n_cols, dtype=np.float32),
                'TX_TEMP': np.zeros(n_cols, dtype=np.float32),
                'TX_LEV': np.zeros(n_cols, dtype=np.float32),
                'TX_CURR': np.zeros(n_cols, dtype=np.float32),
                'CORRUPTED_DATA_FLAG': np.zeros(n_cols, dtype=np.int16),
            }
    return aux_dict


def make_sharad_edr_sci_dict(n_cols: int,
                             n_samp: int = 3600,
                             ) -> dict[str, Any]:
    """
    Initialize a preallocated SHARAD EDR science data dictionary.

    This function creates a standard dictionary for SHARAD EDR science
    data. Most keys are initialized with 1D NumPy arrays of length
    ``n_cols`` and a specific data type. OST line and pack segmentation
    fields are stored with flattened key names (e.g.,
    ``OST_LINE_OPERATIVE_MODE``, ``PSFS_DMA_ERROR``). Some keys contain
    2D arrays:

        - ``S_COEFFS``: shape (8, n_cols)
        - ``C_COEFFS``: shape (7, n_cols)
        - ``ECHO_SAMPLES``: shape (n_samp, n_cols)

    Args:
        n_cols: Number of data columns.
        n_samp: Number of samples in the science data after decompression.

    Returns:
        Preallocated dictionary for SHARAD EDR science data.
    """
    sci_dict: dict[str, Any] = {
        "SCET_BLOCK_WHOLE": np.zeros(n_cols, dtype=np.uint32),
        "SCET_BLOCK_FRAC": np.zeros(n_cols, dtype=np.uint16),
        "TLM_COUNTER": np.zeros(n_cols, dtype=np.uint32),
        "FMT_LENGTH": np.zeros(n_cols, dtype=np.uint16),
        "BIT_RESOLUTION": np.zeros(n_cols, dtype=np.uint16),
        "SCET_OST_WHOLE": np.zeros(n_cols, dtype=np.uint32),
        "SCET_OST_FRAC": np.zeros(n_cols, dtype=np.uint16),
        "OST_LINE_NUMBER": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_PULSE_REPETITION_INTERVAL": np.zeros(n_cols, dtype=np.uint16),
        "OST_LINE_PHASE_COMPENSATION_TYPE": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_DATA_TAKE_LENGTH": np.zeros(n_cols, dtype=np.uint32),
        "OST_LINE_OPERATIVE_MODE": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_MANUAL_GAIN_CONTROL": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_COMPRESSION_SELECTION": np.zeros(n_cols, dtype=bool),
        "OST_LINE_CLOSED_LOOP_TRACKING": np.zeros(n_cols, dtype=bool),
        "OST_LINE_TRACKING_DATA_STORAGE": np.zeros(n_cols, dtype=bool),
        "OST_LINE_TRACKING_PRE_SUMMING": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_TRACKING_LOGIC_SELECTION": np.zeros(n_cols, dtype=bool),
        "OST_LINE_THRESHOLD_LOGIC_SELECTION": np.zeros(n_cols, dtype=bool),
        "OST_LINE_SAMPLE_NUMBER": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_ALPHA_BETA": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_REFERENCE_BIT": np.zeros(n_cols, dtype=bool),
        "OST_LINE_THRESHOLD": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_THRESHOLD_INCREMENT": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_INITIAL_ECHO_VALUE": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_EXPECTED_ECHO_SHIFT": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_WINDOW_LEFT_SHIFT": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_WINDOW_RIGHT_SHIFT": np.zeros(n_cols, dtype=np.uint8),
        "DATA_BLOCK_ID": np.zeros(n_cols, dtype=np.uint32),
        "SCIENCE_DATA_SOURCE_COUNTER": np.zeros(n_cols, dtype=np.uint16),
        "PSFS_SCIENTIFIC_DATA_TYPE": np.zeros(n_cols, dtype=bool),
        "PSFS_SEGMENTATION_FLAG": np.zeros(n_cols, dtype=np.uint8),
        "PSFS_DMA_ERROR": np.zeros(n_cols, dtype=bool),
        "PSFS_TC_OVERRUN": np.zeros(n_cols, dtype=bool),
        "PSFS_FIFO_FULL": np.zeros(n_cols, dtype=bool),
        "PSFS_TEST": np.zeros(n_cols, dtype=bool),
        "DATA_BLOCK_FIRST_PRI": np.zeros(n_cols, dtype=np.uint32),
        "TIME_DATA_BLOCK_WHOLE": np.zeros(n_cols, dtype=np.uint32),
        "TIME_DATA_BLOCK_FRAC": np.zeros(n_cols, dtype=np.uint16),
        "SDI_BIT_FIELD": np.zeros(n_cols, dtype=np.uint16),
        "TIME_N": np.zeros(n_cols, dtype=np.float32),
        "RADIUS_N": np.zeros(n_cols, dtype=np.float32),
        "TANGENTIAL_VELOCITY_N": np.zeros(n_cols, dtype=np.float32),
        "RADIAL_VELOCITY_N": np.zeros(n_cols, dtype=np.float32),
        "TLP": np.zeros(n_cols, dtype=np.float32),
        "TIME_WPF": np.zeros(n_cols, dtype=np.float32),
        "DELTA_TIME": np.zeros(n_cols, dtype=np.float32),
        "TLP_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "RADIUS_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "TANGENTIAL_VELOCITY_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "RADIAL_VELOCITY_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "END_TLP": np.zeros(n_cols, dtype=np.float32),
        "S_COEFFS": np.zeros((8, n_cols), dtype=np.float32),
        "C_COEFFS": np.zeros((7, n_cols), dtype=np.float32),
        "SLOPE": np.zeros(n_cols, dtype=np.float32),
        "TOPOGRAPHY": np.zeros(n_cols, dtype=np.float32),
        "PHASE_COMPENSATION_STEP": np.zeros(n_cols, dtype=np.float32),
        "RECEIVE_WINDOW_OPENING_TIME": np.zeros(n_cols, dtype=np.float32),
        "RECEIVE_WINDOW_POSITION": np.zeros(n_cols, dtype=np.uint32),
        "ECHO_SAMPLES": np.zeros((n_samp, n_cols), dtype=np.int8),
    }
    return sci_dict

##############################################################################################################
#
# PDS RDRs
#
##############################################################################################################
def make_sharad_rdr_sci_dict(n_cols: int,
                             n_samp: int = 667,
                             ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD RDR science data dictionary.

    This function creates a standard dictionary for SHARAD RDR science
    data. Most keys are initialized with 1D NumPy arrays of length
    ``n_cols`` with appropriate numeric or boolean data types. OST line
    and pack segmentation fields are stored with flattened key names
    (e.g., ``OST_LINE_OPERATIVE_MODE``, ``PSFS_DMA_ERROR``). Several
    fields are initialized as 2D arrays, including:

        - ``S_COEFFS``: shape (n_cols, 8)
        - ``C_COEFFS``: shape (n_cols, 7)
        - ``ECHO_SAMPLES_REAL``: shape (667, n_cols)
        - ``ECHO_SAMPLES_IMAGINARY``: shape (667, n_cols)
        - ``MARS_SC_POSITION_VECTOR``: shape (n_cols, 3)
        - ``MARS_SC_VELOCITY_VECTOR``: shape (n_cols, 3)
        - ``ECHO_SAMPLES``: shape (667, n_cols), complex values

    The ``GEOMETRY_EPOCH`` field is initialized as a ``datetime64[ms]``
    array filled with ``NaT``.

    Args:
        n_cols: Number of data columns.
        n_samp: Number of samples in the science data

    Returns:
        Preallocated dictionary for SHARAD RDR science data.
    """
    sci_dict: dict[str, NDArray] = {
        "SCET_BLOCK_WHOLE": np.zeros(n_cols, dtype=np.uint32),
        "SCET_BLOCK_FRAC": np.zeros(n_cols, dtype=np.uint16),
        "TLM_COUNTER": np.zeros(n_cols, dtype=np.uint32),
        "FMT_LENGTH": np.zeros(n_cols, dtype=np.uint16),
        "SCET_OST_WHOLE": np.zeros(n_cols, dtype=np.uint32),
        "SCET_OST_FRAC": np.zeros(n_cols, dtype=np.uint16),
        "OST_LINE_NUMBER": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_PULSE_REPETITION_INTERVAL": np.zeros(n_cols, dtype=np.uint16),
        "OST_LINE_PHASE_COMPENSATION_TYPE": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_DATA_TAKE_LENGTH": np.zeros(n_cols, dtype=np.uint32),
        "OST_LINE_OPERATIVE_MODE": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_MANUAL_GAIN_CONTROL": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_COMPRESSION_SELECTION": np.zeros(n_cols, dtype=bool),
        "OST_LINE_CLOSED_LOOP_TRACKING": np.zeros(n_cols, dtype=bool),
        "OST_LINE_TRACKING_DATA_STORAGE": np.zeros(n_cols, dtype=bool),
        "OST_LINE_TRACKING_PRE_SUMMING": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_TRACKING_LOGIC_SELECTION": np.zeros(n_cols, dtype=bool),
        "OST_LINE_THRESHOLD_LOGIC_SELECTION": np.zeros(n_cols, dtype=bool),
        "OST_LINE_SAMPLE_NUMBER": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_ALPHA_BETA": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_REFERENCE_BIT": np.zeros(n_cols, dtype=bool),
        "OST_LINE_THRESHOLD": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_THRESHOLD_INCREMENT": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_INITIAL_ECHO_VALUE": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_EXPECTED_ECHO_SHIFT": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_WINDOW_LEFT_SHIFT": np.zeros(n_cols, dtype=np.uint8),
        "OST_LINE_WINDOW_RIGHT_SHIFT": np.zeros(n_cols, dtype=np.uint8),
        "DATA_BLOCK_ID": np.zeros(n_cols, dtype=np.uint32),
        "SCIENCE_DATA_SOURCE_COUNTER": np.zeros(n_cols, dtype=np.uint16),
        "PSFS_SCIENTIFIC_DATA_TYPE": np.zeros(n_cols, dtype=bool),
        "PSFS_SEGMENTATION_FLAG": np.zeros(n_cols, dtype=np.uint8),
        "PSFS_DMA_ERROR": np.zeros(n_cols, dtype=bool),
        "PSFS_TC_OVERRUN": np.zeros(n_cols, dtype=bool),
        "PSFS_FIFO_FULL": np.zeros(n_cols, dtype=bool),
        "PSFS_TEST": np.zeros(n_cols, dtype=bool),
        "DATA_BLOCK_FIRST_PRI": np.zeros(n_cols, dtype=np.uint32),
        "TIME_DATA_BLOCK_WHOLE": np.zeros(n_cols, dtype=np.uint32),
        "TIME_DATA_BLOCK_FRAC": np.zeros(n_cols, dtype=np.uint16),
        "SDI_BIT_FIELD": np.zeros(n_cols, dtype=np.uint16),
        "TIME_N": np.zeros(n_cols, dtype=np.float32),
        "RADIUS_N": np.zeros(n_cols, dtype=np.float32),
        "TANGENTIAL_VELOCITY_N": np.zeros(n_cols, dtype=np.float32),
        "RADIAL_VELOCITY_N": np.zeros(n_cols, dtype=np.float32),
        "TLP": np.zeros(n_cols, dtype=np.float32),
        "TIME_WPF": np.zeros(n_cols, dtype=np.float32),
        "DELTA_TIME": np.zeros(n_cols, dtype=np.float32),
        "TLP_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "RADIUS_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "TANGENTIAL_VELOCITY_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "RADIAL_VELOCITY_INTERPOLATE": np.zeros(n_cols, dtype=np.float32),
        "END_TLP": np.zeros(n_cols, dtype=np.float32),
        "S_COEFFS": np.zeros((8, n_cols), dtype=np.float32),
        "C_COEFFS": np.zeros((7, n_cols), dtype=np.float32),
        "SLOPE": np.zeros(n_cols, dtype=np.float32),
        "TOPOGRAPHY": np.zeros(n_cols, dtype=np.float32),
        "PHASE_COMPENSATION_STEP": np.zeros(n_cols, dtype=np.float32),
        "RECEIVE_WINDOW_OPENING_TIME": np.zeros(n_cols, dtype=np.float32),
        "ANTENNA_RELATIVE_GAIN": np.zeros(n_cols, dtype=np.float32),
        "ECHO_SAMPLES_REAL": np.zeros((n_samp, n_cols), dtype=np.float32),
        "ECHO_SAMPLES_IMAGINARY": np.zeros((n_samp, n_cols), dtype=np.float32),
        "N_PRE": np.zeros(n_cols, dtype=np.uint16),
        "BLOCK_NR": np.zeros(n_cols, dtype=np.uint16),
        "BLOCK_ROWS": np.zeros(n_cols, dtype=np.uint16),
        "DOPPLER_BW": np.zeros(n_cols, dtype=np.float32),
        "DOPPLER_CENTROID": np.zeros(n_cols, dtype=np.float32),
        "AZ_TIME_SPACING": np.zeros(n_cols, dtype=np.float32),
        "AZ_RES": np.zeros(n_cols, dtype=np.float32),
        "T_INT": np.zeros(n_cols, dtype=np.float32),
        "AVG_TAN_VELOCITY": np.zeros(n_cols, dtype=np.float32),
        "RANGE_SHIFT": np.zeros(n_cols, dtype=np.int16),
        "EPHEMERIS_TIME": np.zeros(n_cols, dtype=np.float64),
        "GEOMETRY_EPOCH": np.full(n_cols, np.datetime64("NaT", "ms")),
        "SOLAR_LONGITUDE": np.zeros(n_cols, dtype=np.float64),
        "ORBIT_NUMBER": np.zeros(n_cols, dtype=np.int32),
        "MARS_SC_POSITION_VECTOR": np.zeros((3, n_cols), dtype=np.float64),
        "SPACECRAFT_ALTITUDE": np.zeros(n_cols, dtype=np.float64),
        "SUB_SC_EAST_LONGITUDE": np.zeros(n_cols, dtype=np.float64),
        "SUB_SC_PLANETOCENTRIC_LATITUDE": np.zeros(n_cols, dtype=np.float64),
        "SUB_SC_PLANETOGRAPHIC_LATITUDE": np.zeros(n_cols, dtype=np.float64),
        "MARS_SC_VELOCITY_VECTOR": np.zeros((3, n_cols), dtype=np.float64),
        "MARS_SC_RADIAL_VELOCITY": np.zeros(n_cols, dtype=np.float64),
        "MARS_SC_TANGENTIAL_VELOCITY": np.zeros(n_cols, dtype=np.float64),
        "LOCAL_TRUE_SOLAR_TIME": np.zeros(n_cols, dtype=np.float64),
        "SOLAR_ZENITH_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "SC_PITCH_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "SC_YAW_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "SC_ROLL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "MRO_SAMX_INNER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "MRO_SAMX_OUTER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "MRO_SAPX_INNER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "MRO_SAPX_OUTER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "MRO_HGA_INNER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "MRO_HGA_OUTER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float64),
        "DES_TEMP": np.zeros(n_cols, dtype=np.float32),
        "DES_5V": np.zeros(n_cols, dtype=np.float32),
        "DES_12V": np.zeros(n_cols, dtype=np.float32),
        "DES_2V5": np.zeros(n_cols, dtype=np.float32),
        "RX_TEMP": np.zeros(n_cols, dtype=np.float32),
        "TX_TEMP": np.zeros(n_cols, dtype=np.float32),
        "TX_LEV": np.zeros(n_cols, dtype=np.float32),
        "TX_CURR": np.zeros(n_cols, dtype=np.float32),
        "QUALITY_CODE": np.zeros(n_cols, dtype=np.uint8),
        "IMAGE": np.zeros((n_samp, n_cols), dtype=complex),
    }

    return sci_dict

##############################################################################################################
#
# PDS US_RDRs
#
##############################################################################################################
def make_sharad_usrdr_aux_dict(n_cols: int,
                               ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD US_RDR auxiliary data dictionary.

    This function creates a standard auxiliary data dictionary for SHARAD
    US_RDR products. All values are 1D NumPy arrays of length ``n_cols``.
    Most fields use floating-point data types, while ``RADARGRAM_COLUMN``
    uses integers and ``TIME`` is a ``datetime64[ms]`` array initialized
    to ``NaT``.

    Args:
        n_cols: Number of radargram columns / data records.

    Returns:
        Preallocated dictionary for SHARAD US_RDR auxiliary data.
    """
    aux_dict: dict[str, NDArray] = {
        "RADARGRAM_COLUMN": np.zeros(n_cols, dtype=np.uint32),
        "TIME": np.full(n_cols, np.datetime64("NaT", "ms")),
        "LATITUDE": np.zeros(n_cols, dtype=np.float32),
        "LONGITUDE": np.zeros(n_cols, dtype=np.float32),
        "MARS_RADIUS": np.zeros(n_cols, dtype=np.float32),
        "SPACECRAFT_RADIUS": np.zeros(n_cols, dtype=np.float32),
        "RADIAL_VELOCITY": np.zeros(n_cols, dtype=np.float32),
        "TANGENTIAL_VELOCITY": np.zeros(n_cols, dtype=np.float32),
        "SZA": np.zeros(n_cols, dtype=np.float32),
        "PHASE/1.0E16": np.zeros(n_cols, dtype=np.float32),
    }

    return aux_dict


def make_sharad_usrdr_sci_dict(n_cols: int,
                               n_samp: int = 3600,
                               ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD US_RDR science data dictionary.

    This function creates a minimal science data dictionary for SHARAD
    US_RDR products. The dictionary contains a single key, ``"IMAGE"``,
    whose value is a 2D ``float32`` NumPy array of shape
    ``(n_samp, n_cols)``.

    Args:
        n_cols: Number of records (columns) in the science file.
        n_samp: Number of samples per record.

    Returns:
        Preallocated dictionary for SHARAD US_RDR science data.
    """
    sci_dict: dict[str, NDArray] = {
        "IMAGE": np.zeros((n_samp, n_cols), dtype=np.float32),
    }

    return sci_dict

##############################################################################################################
#
# PDS SCSs
#
##############################################################################################################
def make_sharad_usscs_rtrn_dict(n_cols: int,
                                    ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD US_RDR CSC return data dictionary.

    This function creates a dictionary for storing CSC return geometry
    and footprint-related parameters for SHARAD US_RDR products. All
    values are 1D NumPy arrays of length ``n_cols``. Most fields use
    floating-point data types, while fields representing indices or
    line numbers use integers.

    Args:
        n_cols: Number of records / columns in the return data.

    Returns:
        Preallocated dictionary for SHARAD US_RDR CSC return data.
    """
    aux_dict: dict[str, NDArray] = {
        "COLUMN": np.zeros(n_cols, dtype=np.uint32),
        "SPACECRAFTLON": np.zeros(n_cols, dtype=np.float32),
        "SPACECRAFTLAT": np.zeros(n_cols, dtype=np.float32),
        "SPACECRAFTHGT": np.zeros(n_cols, dtype=np.float32),
        "NADIRHGT": np.zeros(n_cols, dtype=np.float32),
        "NADIRLINE": np.zeros(n_cols, dtype=np.uint32),
        "NADIRAREOIDRAD": np.zeros(n_cols, dtype=np.float32),
        "NADIRELLIPSOIDRAD": np.zeros(n_cols, dtype=np.float32),
        "FIRSTLON": np.zeros(n_cols, dtype=np.float32),
        "FIRSTLAT": np.zeros(n_cols, dtype=np.float32),
        "FIRSTHGT": np.zeros(n_cols, dtype=np.float32),
        "FIRSTLINE": np.zeros(n_cols, dtype=np.uint32),
    }

    return aux_dict


def make_sharad_usscs_sim_dict(n_cols: int,
                               n_samp: int = 3600,
                               ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD US SCS SIM data dictionary.

    This function creates a dictionary for storing CSIM (simulated)
    SHARAD US_RDR data. The dictionary contains three 2D ``float32``
    NumPy arrays of shape ``(n_samp, n_cols)``:

        - ``LEFT``
        - ``RIGHT``
        - ``COMBINED``

    Args:
        n_cols: Number of records (columns) in the science file.
        n_samp: Number of samples per record.

    Returns:
        Preallocated dictionary for SHARAD US_RDR CSIM data.
    """
    sci_dict: dict[str, NDArray] = {
        "LEFT": np.zeros((n_samp, n_cols), dtype=np.float32),
        "RIGHT": np.zeros((n_samp, n_cols), dtype=np.float32),
        "COMBINED": np.zeros((n_samp, n_cols), dtype=np.float32),
    }
    return sci_dict

##############################################################################################################
#
# CO-SHARPS Decoder Products (DEC_DATA; US EDRs)
#
##############################################################################################################
def make_sharad_usedr_sci_dict(n_cols: int,
                               n_samp: int = 3600,
                               ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD US EDR science data dictionary.

    This function creates a minimal science data dictionary for SHARAD
    US EDR products. The dictionary contains a single key,
    ``ECHO_SAMPLES``, whose value is a 2D NumPy array of shape
    ``(n_samp, n_cols)`` with ``float32`` data type.

    Args:
        n_cols: Number of data columns.
        n_samp: Number of samples in the science data after decompression.

    Returns:
        Preallocated dictionary for SHARAD US EDR science data.
    """
    sci_dict: dict[str, NDArray] = {
        "ECHO_SAMPLES": np.zeros((n_samp, n_cols), dtype=np.float32),
    }
    return sci_dict


def make_sharad_usedr_aux_dict(n_cols: int,
                               ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD US EDR auxiliary data dictionary.

    This function creates a minimal auxiliary data dictionary for SHARAD
    US EDR products. All values are 1D ``float32`` NumPy arrays of length
    ``n_cols``.

    Note:
        The purpose of the third column (``"JUNK"``) is currently unknown
        and requires clarification from the SHARAD US EDR AUX specification
        or upstream data source.

    Args:
        n_cols: Number of data columns.

    Returns:
        Preallocated dictionary for SHARAD US EDR auxiliary data.
    """
    aux_dict: dict[str, NDArray] = {
        "TIME": np.zeros(n_cols, dtype=np.float32),
        "RECEIVE_WINDOW_OPENING_TIME": np.zeros(n_cols, dtype=np.float32),
        "JUNK": np.zeros(n_cols, dtype=np.float32),
    }

    return aux_dict


def make_sharad_usedr_orb_dict(n_cols: int,
                               ) -> dict[str, NDArray[Any]]:
    """
    Initialize a preallocated SHARAD US EDR orbit/geometry auxiliary
    data dictionary.

    This function creates a standard orbit/geometry auxiliary data
    dictionary for SHARAD US EDR products. All values are 1D NumPy arrays
    of length ``n_cols``. Most fields use the ``float32`` data type, while
    ``GEOMETRY_EPOCH`` is a ``datetime64[ms]`` array initialized to ``NaT``.

    Args:
        n_cols: Number of data columns.

    Returns:
        Preallocated dictionary for SHARAD US EDR orbit/geometry auxiliary data.
    """
    aux_dict: dict[str, NDArray[Any]] = {
        "GEOMETRY_EPOCH": np.full(n_cols, np.datetime64("NaT", "ms"), dtype="datetime64[ms]"),
        "SUB_SC_PLANETOCENTRIC_LATITUDE": np.zeros(n_cols, dtype=np.float32),
        "SUB_SC_EAST_LONGITUDE": np.zeros(n_cols, dtype=np.float32),
        "SPACECRAFT_RADIUS": np.zeros(n_cols, dtype=np.float32),
        "MARS_SC_TANGENTIAL_VELOCITY": np.zeros(n_cols, dtype=np.float32),
        "MARS_SC_RADIAL_VELOCITY": np.zeros(n_cols, dtype=np.float32),
        "X_MARS_SC_POSITION_VECTOR": np.zeros(n_cols, dtype=np.float32),
        "Y_MARS_SC_POSITION_VECTOR": np.zeros(n_cols, dtype=np.float32),
        "Z_MARS_SC_POSITION_VECTOR": np.zeros(n_cols, dtype=np.float32),
        "X_MARS_SC_VELOCITY_VECTOR": np.zeros(n_cols, dtype=np.float32),
        "Y_MARS_SC_VELOCITY_VECTOR": np.zeros(n_cols, dtype=np.float32),
        "Z_MARS_SC_VELOCITY_VECTOR": np.zeros(n_cols, dtype=np.float32),
        "SC_ROLL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "SC_PITCH_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "SC_YAW_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MRO_HGA_INNER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MRO_HGA_OUTER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MRO_SAPX_INNER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MRO_SAPX_OUTER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MRO_SAMX_INNER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MRO_SAMX_OUTER_GIMBAL_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "SOLAR_ZENITH_ANGLE": np.zeros(n_cols, dtype=np.float32),
        "MAGNETIC_FIELD": np.zeros(n_cols, dtype=np.float32),
        "MARS_SUN_DISTANCE": np.zeros(n_cols, dtype=np.float32),
    }

    return aux_dict


##############################################################################################################
#
# CO-SHARPS FPB Files
#
##############################################################################################################
def make_sharad_fpb_sim_dict(n_cols: int,
                             n_samp: int = 3600,
                             ) -> dict[str, NDArray]:
    """
    Initialize a preallocated SHARAD FPB SIM data dictionary.

    This function creates a dictionary for storing CSIM (simulated)
    SHARAD FPB data. The dictionary contains one 2D ``float32``
    NumPy array of shape ``(n_samp, n_cols)``:

        - ``COMBINED``

    Args:
        n_cols: Number of records (columns) in the science file.
        n_samp: Number of samples per record.

    Returns:
        Preallocated dictionary for SHARAD US_RDR CSIM data.
    """
    sci_dict: dict[str, NDArray] = {
        "COMBINED": np.zeros((n_samp, n_cols), dtype=np.float32),
    }
    return sci_dict


def make_sharad_fpb_rtrn_dict(n_cols: int,) -> dict[str, NDArray]:
    """Initialize a preallocated SHARAD FPB SIM nadir or first return
    data dictionary.

    This function creates a dictionary for storing NADIR or FIRST RETURN
    geometry . All values are 1D NumPy arrays of length ``n_cols``. Most fields use
    floating-point data types, while fields representing indices or
    line numbers use integers.

    Args:
        n_cols: Number of records / columns in the return data.

    Returns:
        Preallocated dictionary for SHARAD US_RDR CSC return data.
    """
    aux_dict: dict[str, NDArray] = {
        "LATITUDE": np.zeros(n_cols, dtype=np.float32),
        "LONGITUDE": np.zeros(n_cols, dtype=np.float32),
        "ELEV_IAU2000": np.zeros(n_cols, dtype=np.float32),
        "SAMPLE": np.zeros(n_cols, dtype=np.uint32),
    }

    return aux_dict