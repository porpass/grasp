# SPDX-License-Identifier: BSD-3-Clause
import bitstring
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import struct
from typing import Any

from ..common.utils import calc_nrecs, decode_datetime, parse_pds_lbl



##############################################################################################################
#
# Geometry
#
##############################################################################################################
def make_marsis_edr_geo_dict(n_cols: int,
                             ) -> dict[str, NDArray]:
    """
    Initialize a pre-allocated MARSIS EDR Aux data dictionary.

    This function creates a standard dictionary for MARSIS EDR Aux
    data. Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type. Four keys have 2D arrays as their
    values, with ``n_cols`` being the length of axis 0 and 3 being the
    length of axis 1:

        - TARGET_SC_POSITION_VECTOR
        - TARGET_SC_VELOCITY_VECTOR
        - DIPOLE_UNIT_VECTOR
        - MONOPOLE_UNIT_VECTOR

    Args:
        n_cols: Number of data columns.

    Returns:
        Preallocated dictionary for MARSIS EDR Aux data.
    """
    aux_dict: dict[str, NDArray] = {
                'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=np.uint32),
                'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=np.uint16),
                'GEOMETRY_EPHEMERIS_TIME': np.zeros(n_cols, dtype=np.float64),
                'GEOMETRY_EPOCH': np.full(n_cols, np.datetime64('NaT', 'ms')),
                'MARS_SOLAR_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'MARS_SUN_DISTANCE': np.zeros(n_cols, dtype=np.float64),
                'ORBIT_NUMBER': np.zeros(n_cols, dtype=np.uint32),
                'TARGET_NAME': np.zeros(n_cols, dtype=str),
                'TARGET_SC_POSITION_VECTOR': np.zeros((3, n_cols), dtype=np.float64),
                'SPACECRAFT_ALTITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_LATITUDE': np.zeros(n_cols, dtype=np.float64),
                'TARGET_SC_VELOCITY_VECTOR': np.zeros((3, n_cols), dtype=np.float64),
                'TARGET_SC_RADIAL_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'TARGET_SC_TANG_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'LOCAL_TRUE_SOLAR_TIME': np.zeros(n_cols, dtype=np.float64),
                'SOLAR_ZENITH_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'DIPOLE_UNIT_VECTOR': np.zeros((3, n_cols), dtype=np.float64),
                'MONOPOLE_UNIT_VECTOR': np.zeros((3, n_cols), dtype=np.float64),
                }
    return aux_dict


##############################################################################################################
#
# EDR Dictionaries
#
##############################################################################################################

def make_marsis_edr_f_dict(n_cols: int,
                           mode: str,
                           state: str,
                           form: str,
                           ) -> dict[str, NDArray[Any]]:
    """Initialize a preallocated MARSIS EDR science data dictionary.

    This function dispatches to mode- and form-specific factory functions that
    construct standardized MARSIS EDR science dictionaries.

    Args:
        n_cols: Number of data columns (records) to preallocate.
        mode: Observing mode (e.g., ``"SS1"``, ``"SS3"``, ``"SS4"``).
        state: Instrument state ('ACQ' or 'TRK')
        form: Data form (e.g., ``"CMP"``, ``"RAW"``).

    Returns:
        A preallocated MARSIS EDR science dictionary mapping field names to
        NumPy arrays.

    Raises:
        ValueError: If ``mode`` is invalid or not yet supported, or if the
            requested ``form`` is not supported for the given ``mode``.
    """
    mode = mode.upper()
    form = form.upper()
    state = state.upper()

    dispatch = {
        ("SS1", "ACQ", "CMP"): make_marsis_edr_acq_cmp_f_dict,
        ("SS1", "TRK", "CMP"): make_marsis_edr_ss1_trk_cmp_f_dict,
        ("SS2", "ACQ", "CMP"): make_marsis_edr_acq_cmp_f_dict,
        #("SS2", "TRK", "CMP"): make_marsis_edr_ss2_trk_cmp_f_dict,
        ("SS3", "ACQ", "CMP"): make_marsis_edr_acq_cmp_f_dict,
        ("SS3", "TRK", "CMP"): make_marsis_edr_ss3_trk_cmp_f_dict,
        ("SS3", "TRK", "RAW"): make_marsis_edr_ss3_trk_raw_f_dict,
        ("SS4", "ACQ", "CMP"): make_marsis_edr_ss4_acq_cmp_f_dict,
        ("SS4", "TRK", "CMP"): make_marsis_edr_ss4_trk_cmp_f_dict,
        #("SS5", "TRK", "CMP"): make_marsis_edr_ss5_trk_cmp_f_dict,
    }

    try:
        factory = dispatch[(mode, state, form)]
    except KeyError:
        raise ValueError(f"State {state}, Form {form} for mode {mode} is not yet supported") from None
    return factory(n_cols)

##############################################################################################################
#
# General ACQ Dictionary (SS1-SS3 are the same)
#
##############################################################################################################

def make_marsis_edr_acq_cmp_f_dict(n_cols: int,
                                   n_samp: int = 1024,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR ACQ CMP science data dictionary.

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples in the science data (default = 512).

    Returns:
        Preallocated dictionary for MARSIS EDR SS1 ACQ CMP science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        'AH0': np.zeros(n_cols, dtype=float),
        'AH2': np.zeros(n_cols, dtype=float),
        'AH4': np.zeros(n_cols, dtype=float),
        'AH6': np.zeros(n_cols, dtype=float),
        'AR1': np.zeros(n_cols, dtype=float),
        'AR3': np.zeros(n_cols, dtype=float),
        'AR5': np.zeros(n_cols, dtype=float),
        'AR7': np.zeros(n_cols, dtype=float),
        'AT0': np.zeros(n_cols, dtype=float),
        'AT2': np.zeros(n_cols, dtype=float),
        'AT4': np.zeros(n_cols, dtype=float),
        'AT6': np.zeros(n_cols, dtype=float),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
        'AGC_PIS_PT_VALUE': np.zeros((2, n_cols), dtype=float),
        'AGC_PIS_LEVELS': np.zeros((2, n_cols), dtype=int),
        'K_PIM': np.zeros(n_cols, dtype=int),
        'PIS_MAX_DATA_EXP': np.zeros((2, n_cols), dtype=int),
        'AGC_NPM_PT_VALUE': np.zeros(n_cols, dtype=float),
        'AGC_NPM_LEVELS': np.zeros(n_cols, dtype=int),
        'NPM_INT': np.zeros((2, n_cols), dtype=float),
        'X': np.zeros(n_cols, dtype=int),
        'AGC_COLL_X': np.zeros((2, n_cols), dtype=float),
        'AGC_COLL_X_LEVELS': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_ACQ_COMP': np.zeros(n_cols, dtype=int),
        'RX_TRIG_ACQ_PROGR': np.zeros(n_cols, dtype=int),
        'AGC_SA_FOR_TRK_FRAME': np.zeros((2, n_cols), dtype=float),
        'RX_TRIG_SA_FOR_TRK_FRAME': np.zeros((2, n_cols), dtype=int),
        'DET_THRESH': np.zeros((2, n_cols), dtype=float),
        'K_DET_THRES': np.zeros((2, n_cols), dtype=float),
        'K_DET_THRES_MIN': np.zeros((2, n_cols), dtype=float),
        'PHI_ACQ_F1_RE': np.zeros(n_cols, dtype=float),
        'PHI_ACQ_F1_IM': np.zeros(n_cols, dtype=float),
        'PHI_ACQ_F2_RE': np.zeros(n_cols, dtype=float),
        'PHI_ACQ_F2_IM': np.zeros(n_cols, dtype=float),
        'N_D': np.zeros(n_cols, dtype=int),
        'K_AGC': np.zeros(n_cols, dtype=float),
        'AREF': np.zeros(n_cols, dtype=float),
        'REF_FUN_FLAG': np.zeros((2, n_cols), dtype=int),
        'I_LE': np.zeros((2, n_cols), dtype=int),
        'T_LE': np.zeros((2, n_cols), dtype=float),
        'MAX_RE_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
        'NS_LED': np.zeros(n_cols, dtype=int),
        'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
        'REAL_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'PIS_F1': np.zeros((128, n_cols), dtype=int),
        'PIS_F2': np.zeros((128, n_cols), dtype=int),
    }
    return sci_dict

##############################################################################################################
#
# SS1
#
##############################################################################################################

def make_marsis_edr_ss1_trk_cmp_f_dict(n_cols:int,
                                       n_samp:int=512,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS1 TRK Sci data dictionary.

    This function creates a dictionary for MARSIS EDR SS1 TRK Sci data.
    Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    Args:
        n_cols: Number of science data records
        n_samp: Number of samples in the science data (default = 512)

    Returns:
        Preallocated dictionary for MARSIS EDR SS1 TRK Sci data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray] = {
                'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
                'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
                'MODE_DURATION': np.zeros(n_cols, dtype=int),
                'MODE_SELECTION': np.zeros(n_cols, dtype=int),
                'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
                'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
                'PIM_RX': np.zeros(n_cols, dtype=int),
                'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
                'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
                'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
                'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
                'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
                'TX_POWER': np.zeros(n_cols, dtype=int),
                'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
                'IE_FM': np.zeros(n_cols, dtype=int),
                'FM_FRAMES': np.zeros(n_cols, dtype=int),
                'FRAME_ID': np.zeros(n_cols, dtype=int),
                'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
                'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
                'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
                'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
                'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
                'H_SCET_PAR': np.zeros(n_cols, dtype=float),
                'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
                'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
                'N_0': np.zeros(n_cols, dtype=int),
                'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
                'NB_MIN': np.zeros(n_cols, dtype=int),
                'M_OCOG': np.zeros((2, n_cols), dtype=float),
                'INDEX_OCOG': np.zeros((2, n_cols), dtype=int),
                'TRK_THRESHOLD': np.zeros((2, n_cols), dtype=float),
                'INI_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
                'LAST_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
                'INI_IND_FSRM': np.zeros((2, n_cols), dtype=int),
                'LAST_IND_FSRM': np.zeros((2, n_cols), dtype=int),
                'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
                'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
                'NA_SCET_PAR': np.zeros((2, n_cols), dtype=int),
                'A2_INI_CM': np.zeros((2, n_cols), dtype=float),
                'A2_OPT': np.zeros((2, n_cols), dtype=float),
                'REF_CA_OPT': np.zeros((2, n_cols), dtype=float),
                'DELTA_T': np.zeros((2, n_cols), dtype=int),
                'SF': np.zeros((2, n_cols), dtype=float),
                'I_C': np.zeros((2, n_cols), dtype=int),
                'AGC_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=float),
                'AGC_SA_LEVELS_CURRENT_FRAME': np.zeros((2, n_cols), dtype=int),
                'RX_TRIG_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=int),
                'RX_TRIG_SA_PROGR': np.zeros((2, n_cols), dtype=int),
                'INI_IND_OCOG': np.zeros(n_cols, dtype=int),
                'LAST_IND_OCOG': np.zeros(n_cols, dtype=int),
                'OCOG': np.zeros((2, n_cols), dtype=float),
                'A': np.zeros((2, n_cols), dtype=float),
                'C_LOL': np.zeros((2, n_cols), dtype=int),
                'MAX_RE_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_ZERO_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_ZERO_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_ZERO_F2_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_ZERO_F2_MON': np.zeros(n_cols, dtype=int),
                'AGC_PIS_PT_VALUE': np.zeros((2, n_cols), dtype=float),
                'AGC_PIS_LEVELS': np.zeros((2, n_cols), dtype=int),
                'K_PIM': np.zeros(n_cols, dtype=int),
                'PIS_MAX_DATA_EXP': np.zeros((2, n_cols), dtype=int),
                'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
                'REAL_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
                'IMAG_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
                'REAL_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
                'IMAG_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
                'REAL_ECHO_ZERO_F1_MON': np.zeros((n_samp, n_cols), dtype=int),
                'IMAG_ECHO_ZERO_F1_MON': np.zeros((n_samp, n_cols), dtype=int),
                'REAL_ECHO_ZERO_F2_MON': np.zeros((n_samp, n_cols), dtype=int),
                'IMAG_ECHO_ZERO_F2_MON': np.zeros((n_samp, n_cols), dtype=int),
                'PIS_F1': np.zeros((128, n_cols), dtype=int),
                'PIS_F2': np.zeros((128, n_cols), dtype=int),
    }
    return sci_dict

##############################################################################################################
#
# SS2
#
##############################################################################################################
def make_marsis_edr_ss2_trk_cmp_f_dict(n_cols: int,
                                       n_samp: int = 256,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS2 TRK CMP science data dictionary.

    SS2 TRK echo data is float32, 256 samples per channel, 2 channels
    (F1 and F2 dipole only, no monopole, no real/imag split).

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples in the echo data (default = 256).

    Returns:
        Preallocated dictionary for MARSIS EDR SS2 TRK CMP science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        'M_OCOG': np.zeros((2, n_cols), dtype=float),
        'INDEX_OCOG': np.zeros((2, n_cols), dtype=int),
        'TRK_THRESHOLD': np.zeros((2, n_cols), dtype=float),
        'INI_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
        'LAST_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
        'INI_IND_FSRM': np.zeros((2, n_cols), dtype=int),
        'LAST_IND_FSRM': np.zeros((2, n_cols), dtype=int),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
        'NA_SCET_PAR': np.zeros((2, n_cols), dtype=int),
        'A2_INI_CM': np.zeros((2, n_cols), dtype=float),
        'A2_OPT': np.zeros((2, n_cols), dtype=float),
        'REF_CA_OPT': np.zeros((2, n_cols), dtype=float),
        'DELTA_T': np.zeros((2, n_cols), dtype=int),
        'SF': np.zeros((2, n_cols), dtype=float),
        'I_C': np.zeros((2, n_cols), dtype=int),
        'AGC_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=float),
        'AGC_SA_LEVELS_CURRENT_FRAME': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_SA_PROGR': np.zeros((2, n_cols), dtype=int),
        'INI_IND_OCOG': np.zeros(n_cols, dtype=int),
        'LAST_IND_OCOG': np.zeros(n_cols, dtype=int),
        'OCOG': np.zeros((2, n_cols), dtype=float),
        'A': np.zeros((2, n_cols), dtype=float),
        'C_LOL': np.zeros((2, n_cols), dtype=int),
        'SS2_DCEX': np.zeros((3, n_cols), dtype=int),
        'AGC_PIS_PT_VALUE': np.zeros((2, n_cols), dtype=float),
        'AGC_PIS_LEVELS': np.zeros((2, n_cols), dtype=int),
        'K_PIM': np.zeros(n_cols, dtype=int),
        'PIS_MAX_DATA_EXP': np.zeros((2, n_cols), dtype=int),
        'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
        'ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=float),
        'ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=float),
        'PIS_F1': np.zeros((128, n_cols), dtype=int),
        'PIS_F2': np.zeros((128, n_cols), dtype=int),
    }
    return sci_dict

##############################################################################################################
#
# SS3
#
##############################################################################################################

def make_marsis_edr_ss3_trk_cmp_f_dict(n_cols: int,
                                       n_samp: int = 512,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS3 TRK CMP science data dictionary.

    This function creates a dictionary for MARSIS EDR SS3 TRK CMP science data.
    Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    SS3 has 3 Doppler filters (MINUS1, ZERO, PLUS1) for both dipole
    channels (F1, F2), resulting in 12 echo arrays.

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples in the science data (default = 512).

    Returns:
        Preallocated dictionary for MARSIS EDR SS3 TRK CMP science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        'M_OCOG': np.zeros((2, n_cols), dtype=float),
        'INDEX_OCOG': np.zeros((2, n_cols), dtype=int),
        'TRK_THRESHOLD': np.zeros((2, n_cols), dtype=float),
        'INI_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
        'LAST_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
        'INI_IND_FSRM': np.zeros((2, n_cols), dtype=int),
        'LAST_IND_FSRM': np.zeros((2, n_cols), dtype=int),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
        'NA_SCET_PAR': np.zeros((2, n_cols), dtype=int),
        'A2_INI_CM': np.zeros((2, n_cols), dtype=float),
        'A2_OPT': np.zeros((2, n_cols), dtype=float),
        'REF_CA_OPT': np.zeros((2, n_cols), dtype=float),
        'DELTA_T': np.zeros((2, n_cols), dtype=int),
        'SF': np.zeros((2, n_cols), dtype=float),
        'I_C': np.zeros((2, n_cols), dtype=int),
        'AGC_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=float),
        'AGC_SA_LEVELS_CURRENT_FRAME': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_SA_PROGR': np.zeros((2, n_cols), dtype=int),
        'INI_IND_OCOG': np.zeros(n_cols, dtype=int),
        'LAST_IND_OCOG': np.zeros(n_cols, dtype=int),
        'OCOG': np.zeros((2, n_cols), dtype=float),
        'A': np.zeros((2, n_cols), dtype=float),
        'C_LOL': np.zeros((2, n_cols), dtype=int),
        'MAX_RE_EXP_MINUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_MINUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_PLUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_PLUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_MINUS1_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_MINUS1_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_PLUS1_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_PLUS1_F2_DIP': np.zeros(n_cols, dtype=int),
        'AGC_PIS_PT_VALUE': np.zeros((2, n_cols), dtype=float),
        'AGC_PIS_LEVELS': np.zeros((2, n_cols), dtype=int),
        'K_PIM': np.zeros(n_cols, dtype=int),
        'PIS_MAX_DATA_EXP': np.zeros((2, n_cols), dtype=int),
        'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
        'REAL_ECHO_MINUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_MINUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_PLUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_PLUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_MINUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_MINUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_PLUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_PLUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'PIS_F1': np.zeros((128, n_cols), dtype=int),
        'PIS_F2': np.zeros((128, n_cols), dtype=int),
    }
    return sci_dict


def make_marsis_edr_ss3_trk_raw_f_dict(n_cols: int,
                                       n_samp: int = 980,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS3 RAW TRK Sci data dictionary.
    This is for MARSIS Flash Memory and Super Frame observations!

    This function creates a dictionary for MARSIS EDR SS3 TRK RAW Sci data.
    Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    Args:
        n_cols: Number of science data records
        n_samp: Number of samples in the science data (default = 980)

    Returns:
        Preallocated dictionary for MARSIS EDR SS3 TRK RAW Sci data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
                'SCET_STAR_WHOLE_B1': np.zeros(n_cols, dtype=int),
                'SCET_STAR_FRAC_B1': np.zeros(n_cols, dtype=int),
                'OST_LINE_NUMBER_B1': np.zeros(n_cols, dtype=int),
                'MODE_DURATION_B1': np.zeros(n_cols, dtype=int),
                'MODE_SELECTION_B1': np.zeros(n_cols, dtype=int),
                'DCG_CONFIGURATION_B1': np.zeros([2, n_cols], dtype=int),
                'PI_BAND_SEL_B1': np.zeros([2, n_cols], dtype=int),
                'PIM_RX_B1': np.zeros(n_cols, dtype=int),
                'REF_ALG_SEL_B1': np.zeros(n_cols, dtype=int),
                'LOL_LOGIC_MF_B1': np.zeros([2, n_cols], dtype=int),
                'PRESENT_TRACKING_B1': np.zeros(n_cols, dtype=int),
                'F_NPM_ADDRESS_B1': np.zeros(n_cols, dtype=int),
                'SLOPE_ADDRESS_B1': np.zeros(n_cols, dtype=int),
                'TX_POWER_B1': np.zeros(n_cols, dtype=int),
                'A2_0_OST_ABSCISSA_B1': np.zeros(n_cols, dtype=int),
                'IE_FM_B1': np.zeros(n_cols, dtype=int),
                'FM_FRAMES_B1': np.zeros(n_cols, dtype=int),
                'FRAME_ID_B1': np.zeros(n_cols, dtype=int),
                'FIRST_PRI_OF_FRAME_B1': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_WHOLE_B1': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_FRAC_B1': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_WHOLE_B1': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_FRAC_B1': np.zeros(n_cols, dtype=int),
                'NA_SCET_PAR_B1': np.zeros(n_cols, dtype=int),
                'BAND_B1': np.zeros(n_cols, dtype=int),
                'CHANNEL_B1': np.zeros(n_cols, dtype=int),
                'SCIENCE_DATA_TYPE_B1': np.zeros(n_cols, dtype=int),
                'SCIENCE_DATA_AMOUNT_B1': np.zeros(n_cols, dtype=int),
                'REAL_ECHO_ZERO_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'SCET_STAR_WHOLE_B2': np.zeros(n_cols, dtype=int),
                'SCET_STAR_FRAC_B2': np.zeros(n_cols, dtype=int),
                'OST_LINE_NUMBER_B2': np.zeros(n_cols, dtype=int),
                'MODE_DURATION_B2': np.zeros(n_cols, dtype=int),
                'MODE_SELECTION_B2': np.zeros(n_cols, dtype=int),
                'DCG_CONFIGURATION_B2': np.zeros([2, n_cols], dtype=int),
                'PI_BAND_SEL_B2': np.zeros([2, n_cols], dtype=int),
                'PIM_RX_B2': np.zeros(n_cols, dtype=int),
                'REF_ALG_SEL_B2': np.zeros(n_cols, dtype=int),
                'LOL_LOGIC_MF_B2': np.zeros([2, n_cols], dtype=int),
                'PRESENT_TRACKING_B2': np.zeros(n_cols, dtype=int),
                'F_NPM_ADDRESS_B2': np.zeros(n_cols, dtype=int),
                'SLOPE_ADDRESS_B2': np.zeros(n_cols, dtype=int),
                'TX_POWER_B2': np.zeros(n_cols, dtype=int),
                'A2_0_OST_ABSCISSA_B2': np.zeros(n_cols, dtype=int),
                'IE_FM_B2': np.zeros(n_cols, dtype=int),
                'FM_FRAMES_B2': np.zeros(n_cols, dtype=int),
                'FRAME_ID_B2': np.zeros(n_cols, dtype=int),
                'FIRST_PRI_OF_FRAME_B2': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_WHOLE_B2': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_FRAC_B2': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_WHOLE_B2': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_FRAC_B2': np.zeros(n_cols, dtype=int),
                'NA_SCET_PAR_B2': np.zeros(n_cols, dtype=int),
                'BAND_B2': np.zeros(n_cols, dtype=int),
                'CHANNEL_B2': np.zeros(n_cols, dtype=int),
                'SCIENCE_DATA_TYPE_B2': np.zeros(n_cols, dtype=int),
                'SCIENCE_DATA_AMOUNT_B2': np.zeros(n_cols, dtype=int),
                'REAL_ECHO_ZERO_F2_DIP': np.zeros([n_samp, n_cols], dtype=int),
    }
    return sci_dict


def make_marsis_rdr_ss3_trk_cmp_dict(n_cols: int,
                                   n_samp: int = 512,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS RDR SS3 TRK CMP Sci data dictionary.

    This function creates a dictionary for MARSIS RDR SS3 TRK Sci data.
    Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    Args:
        n_cols: Number of science data records
        n_samp: Number of samples in the science data (default = 512)

    Returns:
        Preallocated dictionary for MARSIS RDR SS3 TRK Sci data.

    """
    sci_dict: dict[str, NDArray[Any]] = {
                'CENTRAL_FREQUENCY': np.zeros([2, n_cols], dtype=np.float32),
                'SLOPE': np.zeros(n_cols, dtype=np.float32),
                'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=np.uint32),
                'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=np.uint16),
                'H_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'VT_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'VR_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'NA_SCET_PAR': np.zeros([2, n_cols], dtype=np.uint16),
                'ECHO_MODULUS_MINUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_MINUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_MODULUS_ZERO_F1_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_ZERO_F1_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_MODULUS_PLUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_PLUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_MODULUS_MINUS1_F2_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_MINUS1_F2_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_MODULUS_ZERO_F2_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_ZERO_F2_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_MODULUS_PLUS1_F2_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_PLUS1_F2_DIP': np.zeros([n_samp, n_cols], dtype=np.float32),
                'GEOMETRY_EPHEMERIS_TIME': np.zeros(n_cols, dtype=np.float64),
                'GEOMETRY_EPOCH': np.full(n_cols, np.datetime64('NaT', 'ms')),
                'MARS_SOLAR_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'MARS_SUN_DISTANCE': np.zeros(n_cols, dtype=np.float64),
                'ORBIT_NUMBER': np.zeros(n_cols, dtype=np.uint32),
                'TARGET_NAME': np.zeros(n_cols, dtype=str),
                'TARGET_SC_POSITION_VECTOR': np.zeros([3, n_cols], dtype=np.float64),
                'SPACECRAFT_ALTITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_LATITUDE': np.zeros(n_cols, dtype=np.float64),
                'TARGET_SC_VELOCITY_VECTOR': np.zeros([3, n_cols], dtype=np.float64),
                'TARGET_SC_RADIAL_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'TARGET_SC_TANG_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'LOCAL_TRUE_SOLAR_TIME': np.zeros(n_cols, dtype=np.float64),
                'SOLAR_ZENITH_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'DIPOLE_UNIT_VECTOR': np.zeros([3, n_cols], dtype=np.float64),
                'MONOPOLE_UNIT_VECTOR': np.zeros([3, n_cols], dtype=np.float64)
    }
    return sci_dict


def make_marsis_rdr_ss3_trk_raw_dict(n_cols: int,
                                     n_samp: int = 980,
                                     ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS RDR SS3 TRK RAW Sci data dictionary.

    This is for MARSIS Flash Memory and Super Frame Observations

    This function creates a dictionary for MARSIS RDR SS3 TRK RAW Sci data.
    Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    Args:
        n_cols: Number of science data records
        n_samp: Number of samples in the science data (default = 980)

    Returns:
        Preallocated dictionary for MARSIS RDR SS3 TRK RAW Sci data.

    """
    sci_dict: dict[str, NDArray[Any]] = {
                'CENTRAL_FREQUENCY': np.zeros([2, n_cols], dtype=np.float32),
                'SLOPE': np.zeros(n_cols, dtype=np.float32),
                'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=np.uint32),
                'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=np.int16),
                'H_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'VT_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'VR_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=np.float32),
                'NA_SCET_PAR': np.zeros([2, n_cols], dtype=np.float32),
                'ECHO_MODULUS_B1': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_B1': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_MODULUS_B2': np.zeros([n_samp, n_cols], dtype=np.float32),
                'ECHO_PHASE_B2': np.zeros([n_samp, n_cols], dtype=np.float32),
                'GEOMETRY_EPHEMERIS_TIME': np.zeros(n_cols, dtype=np.float64),
                'GEOMETRY_EPOCH': np.full(n_cols, np.datetime64('NaT', 'ms')),
                'MARS_SOLAR_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'MARS_SUN_DISTANCE': np.zeros(n_cols, dtype=np.float64),
                'ORBIT_NUMBER': np.zeros(n_cols, dtype=np.uint32),
                'TARGET_NAME': np.zeros(n_cols, dtype=str),
                'TARGET_SC_POSITION_VECTOR': np.zeros([3, n_cols], dtype=np.float64),
                'SPACECRAFT_ALTITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_LONGITUDE': np.zeros(n_cols, dtype=np.float64),
                'SUB_SC_LATITUDE': np.zeros(n_cols, dtype=np.float64),
                'TARGET_SC_VELOCITY_VECTOR': np.zeros([3, n_cols], dtype=np.float64),
                'TARGET_SC_RADIAL_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'TARGET_SC_TANG_VELOCITY': np.zeros(n_cols, dtype=np.float64),
                'LOCAL_TRUE_SOLAR_TIME': np.zeros(n_cols, dtype=np.float64),
                'SOLAR_ZENITH_ANGLE': np.zeros(n_cols, dtype=np.float64),
                'DIPOLE_UNIT_VECTOR': np.zeros([3, n_cols], dtype=np.float64),
                'MONOPOLE_UNIT_VECTOR': np.zeros([3, n_cols], dtype=np.float64)
                }
    return sci_dict
##############################################################################################################
#
# SS4
#
##############################################################################################################
def make_marsis_edr_ss4_acq_cmp_f_dict(n_cols: int,
                                       n_samp: int = 1024,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS4 ACQ CMP science data dictionary.

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples in the science data (default = 1024).

    Returns:
        Preallocated dictionary for MARSIS EDR SS4 ACQ CMP science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        'AH0': np.zeros(n_cols, dtype=float),
        'AH2': np.zeros(n_cols, dtype=float),
        'AH4': np.zeros(n_cols, dtype=float),
        'AH6': np.zeros(n_cols, dtype=float),
        'AR1': np.zeros(n_cols, dtype=float),
        'AR3': np.zeros(n_cols, dtype=float),
        'AR5': np.zeros(n_cols, dtype=float),
        'AR7': np.zeros(n_cols, dtype=float),
        'AT0': np.zeros(n_cols, dtype=float),
        'AT2': np.zeros(n_cols, dtype=float),
        'AT4': np.zeros(n_cols, dtype=float),
        'AT6': np.zeros(n_cols, dtype=float),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
        'AGC_PIS_PT_VALUE': np.zeros((2, n_cols), dtype=float),
        'AGC_PIS_LEVELS': np.zeros((2, n_cols), dtype=int),
        'K_PIM': np.zeros(n_cols, dtype=int),
        'PIS_MAX_DATA_EXP': np.zeros((2, n_cols), dtype=int),
        'AGC_NPM_PT_VALUE': np.zeros(n_cols, dtype=float),
        'AGC_NPM_LEVELS': np.zeros(n_cols, dtype=int),
        'NPM_INT': np.zeros((2, n_cols), dtype=float),
        'X': np.zeros(n_cols, dtype=int),
        'AGC_COLL_X': np.zeros((2, n_cols), dtype=float),
        'AGC_COLL_X_LEVELS': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_ACQ_COMP': np.zeros(n_cols, dtype=int),
        'RX_TRIG_ACQ_PROGR': np.zeros(n_cols, dtype=int),
        'AGC_SA_FOR_TRK_FRAME': np.zeros((2, n_cols), dtype=float),
        'RX_TRIG_SA_FOR_TRK_FRAME': np.zeros((2, n_cols), dtype=int),
        'DET_THRESH': np.zeros((2, n_cols), dtype=float),
        'K_DET_THRES': np.zeros((2, n_cols), dtype=float),
        'K_DET_THRES_MIN': np.zeros((2, n_cols), dtype=float),
        'PHI_ACQ_F1_RE': np.zeros(n_cols, dtype=float),
        'PHI_ACQ_F1_IM': np.zeros(n_cols, dtype=float),
        'PHI_ACQ_F2_RE': np.zeros(n_cols, dtype=float),
        'PHI_ACQ_F2_IM': np.zeros(n_cols, dtype=float),
        'N_D': np.zeros(n_cols, dtype=int),
        'K_AGC': np.zeros(n_cols, dtype=float),
        'AREF': np.zeros(n_cols, dtype=float),
        'REF_FUN_FLAG': np.zeros((2, n_cols), dtype=int),
        'I_LE': np.zeros((2, n_cols), dtype=int),
        'T_LE': np.zeros((2, n_cols), dtype=float),
        'MAX_RE_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F2_DIP': np.zeros(n_cols, dtype=int),
        'NS_LED': np.zeros(n_cols, dtype=int),
        'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
        'REAL_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'PIS_F1': np.zeros((128, n_cols), dtype=int),
        'PIS_F2': np.zeros((128, n_cols), dtype=int),
    }
    return sci_dict



def make_marsis_edr_ss4_trk_cmp_f_dict(n_cols: int,
                                       n_samp: int = 512,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS4 TRK CMP Sci data dictionary.

    This function creates a dictionary for MARSIS EDR SS4 TRK Sci data.
    Each key is initialized with an empty NumPy array of length
    ``n_cols`` and a specific data type.

    Args:
        n_cols: Number of science data records
        n_samp: Number of samples in the science data (default = 980)

    Returns:
        Preallocated dictionary for MARSIS EDR SS4 TRK Sci data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
                'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
                'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
                'MODE_DURATION': np.zeros(n_cols, dtype=int),
                'MODE_SELECTION': np.zeros(n_cols, dtype=int),
                'DCG_CONFIGURATION': np.zeros([2, n_cols], dtype=int),
                'PI_BAND_SEL': np.zeros([2, n_cols], dtype=int),
                'PIM_RX': np.zeros(n_cols, dtype=int),
                'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
                'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
                'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
                'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
                'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
                'TX_POWER': np.zeros(n_cols, dtype=int),
                'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
                'IE_FM': np.zeros(n_cols, dtype=int),
                'FM_FRAMES': np.zeros(n_cols, dtype=int),
                'FRAME_ID': np.zeros(n_cols, dtype=int),
                'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
                'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
                'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
                'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
                'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
                'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
                'H_SCET_PAR': np.zeros(n_cols, dtype=float),
                'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
                'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
                'N_0': np.zeros(n_cols, dtype=int),
                'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
                'NB_MIN': np.zeros(n_cols, dtype=int),
                'M_OCOG': np.zeros([2, n_cols], dtype=float),
                'INDEX_OCOG': np.zeros([2, n_cols], dtype=int),
                'TRK_THRESHOLD': np.zeros([2, n_cols], dtype=float),
                'INI_IND_TRK_THRESHOLD': np.zeros([2, n_cols], dtype=int),
                'LAST_IND_TRK_THRESHOLD': np.zeros([2, n_cols], dtype=int),
                'INI_IND_FSRM': np.zeros([2, n_cols], dtype=int),
                'LAST_IND_FSRM': np.zeros([2, n_cols], dtype=int),
                'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
                'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
                'NA_SCET_PAR': np.zeros([2, n_cols], dtype=int),
                'A2_INI_CM': np.zeros([2, n_cols], dtype=float),
                'A2_OPT': np.zeros([2, n_cols], dtype=float),
                'REF_CA_OPT': np.zeros([2, n_cols], dtype=float),
                'DELTA_T': np.zeros([2, n_cols], dtype=int),
                'SF': np.zeros([2, n_cols], dtype=float),
                'I_C': np.zeros([2, n_cols], dtype=int),
                'AGC_SA_FOR_NEXT_FRAME': np.zeros([2, n_cols], dtype=float),
                'AGC_SA_LEVELS_CURRENT_FRAME': np.zeros([2, n_cols], dtype=int),
                'RX_TRIG_SA_FOR_NEXT_FRAME': np.zeros([2, n_cols], dtype=int),
                'RX_TRIG_SA_PROGR': np.zeros([2, n_cols], dtype=int),
                'INI_IND_OCOG': np.zeros(n_cols, dtype=int),
                'LAST_IND_OCOG': np.zeros(n_cols, dtype=int),
                'OCOG': np.zeros([2, n_cols], dtype=float),
                'A': np.zeros([2, n_cols], dtype=float),
                'C_LOL': np.zeros([2, n_cols], dtype=int),
                'MAX_RE_EXP_MINUS2_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_MINUS2_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_MINUS1_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_MINUS1_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_PLUS1_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_PLUS1_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_PLUS2_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_PLUS2_F1_DIP': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_MINUS2_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_MINUS2_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_MINUS1_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_MINUS1_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_ZERO_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_ZERO_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_PLUS1_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_PLUS1_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_RE_EXP_PLUS2_F1_MON': np.zeros(n_cols, dtype=int),
                'MAX_IM_EXP_PLUS2_F1_MON': np.zeros(n_cols, dtype=int),
                'AGC_PIS_PT_VALUE': np.zeros([2, n_cols], dtype=float),
                'AGC_PIS_LEVELS': np.zeros([2, n_cols], dtype=int),
                'K_PIM': np.zeros(n_cols, dtype=int),
                'PIS_MAX_DATA_EXP': np.zeros([2, n_cols], dtype=int),
                'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
                'REAL_ECHO_MINUS2_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_MINUS2_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_MINUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_MINUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_ZERO_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_ZERO_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_PLUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_PLUS1_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_PLUS2_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_PLUS2_F1_DIP': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_MINUS2_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_MINUS2_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_MINUS1_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_MINUS1_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_ZERO_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_ZERO_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_PLUS1_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_PLUS1_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'REAL_ECHO_PLUS2_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'IMAG_ECHO_PLUS2_F1_MON': np.zeros([n_samp, n_cols], dtype=int),
                'PIS_F1': np.zeros([128, n_cols], dtype=int),
                'PIS_F2': np.zeros([128, n_cols], dtype=int),
    }
    return sci_dict

##############################################################################################################
#
# SS5
#
##############################################################################################################
def make_marsis_edr_ss5_trk_cmp_f_dict(n_cols: int,
                                       n_samp: int = 512,
                                       ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR SS5 TRK CMP science data dictionary.

    SS5 has 3 Doppler filters (MINUS1, ZERO, PLUS1) for both dipole
    channels (F1, F2), resulting in 12 echo arrays. The exponent fields
    cover F1 DIP and F1 MON (unlike SS3 which has F1 DIP and F2 DIP).

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples in the science data (default = 512).

    Returns:
        Preallocated dictionary for MARSIS EDR SS5 TRK CMP science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        'M_OCOG': np.zeros((2, n_cols), dtype=float),
        'INDEX_OCOG': np.zeros((2, n_cols), dtype=int),
        'TRK_THRESHOLD': np.zeros((2, n_cols), dtype=float),
        'INI_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
        'LAST_IND_TRK_THRESHOLD': np.zeros((2, n_cols), dtype=int),
        'INI_IND_FSRM': np.zeros((2, n_cols), dtype=int),
        'LAST_IND_FSRM': np.zeros((2, n_cols), dtype=int),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        'NB_SCET_PAR': np.zeros(n_cols, dtype=int),
        'NA_SCET_PAR': np.zeros((2, n_cols), dtype=int),
        'A2_INI_CM': np.zeros((2, n_cols), dtype=float),
        'A2_OPT': np.zeros((2, n_cols), dtype=float),
        'REF_CA_OPT': np.zeros((2, n_cols), dtype=float),
        'DELTA_T': np.zeros((2, n_cols), dtype=int),
        'SF': np.zeros((2, n_cols), dtype=float),
        'I_C': np.zeros((2, n_cols), dtype=int),
        'AGC_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=float),
        'AGC_SA_LEVELS_CURRENT_FRAME': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_SA_FOR_NEXT_FRAME': np.zeros((2, n_cols), dtype=int),
        'RX_TRIG_SA_PROGR': np.zeros((2, n_cols), dtype=int),
        'INI_IND_OCOG': np.zeros(n_cols, dtype=int),
        'LAST_IND_OCOG': np.zeros(n_cols, dtype=int),
        'OCOG': np.zeros((2, n_cols), dtype=float),
        'A': np.zeros((2, n_cols), dtype=float),
        'C_LOL': np.zeros((2, n_cols), dtype=int),
        # Exponents — F1 DIP
        'MAX_RE_EXP_MINUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_MINUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_PLUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_PLUS1_F1_DIP': np.zeros(n_cols, dtype=int),
        # Exponents — F1 MON (SS5-specific, SS3 has F2 DIP here)
        'MAX_RE_EXP_MINUS1_F1_MON': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_MINUS1_F1_MON': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_ZERO_F1_MON': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_ZERO_F1_MON': np.zeros(n_cols, dtype=int),
        'MAX_RE_EXP_PLUS1_F1_MON': np.zeros(n_cols, dtype=int),
        'MAX_IM_EXP_PLUS1_F1_MON': np.zeros(n_cols, dtype=int),
        'AGC_PIS_PT_VALUE': np.zeros((2, n_cols), dtype=float),
        'AGC_PIS_LEVELS': np.zeros((2, n_cols), dtype=int),
        'K_PIM': np.zeros(n_cols, dtype=int),
        'PIS_MAX_DATA_EXP': np.zeros((2, n_cols), dtype=int),
        'PROCESSING_PRF': np.zeros(n_cols, dtype=float),
        # Echo data — dipole only, 3 Doppler filters x 2 frequencies
        'REAL_ECHO_MINUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_MINUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_PLUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_PLUS1_F1_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_MINUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_MINUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_ZERO_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'REAL_ECHO_PLUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'IMAG_ECHO_PLUS1_F2_DIP': np.zeros((n_samp, n_cols), dtype=int),
        'PIS_F1': np.zeros((128, n_cols), dtype=int),
        'PIS_F2': np.zeros((128, n_cols), dtype=int),
    }
    return sci_dict
##############################################################################################################
#
# AIS
#
##############################################################################################################
def make_marsis_edr_ais_f_dict(n_cols: int,
                               n_samp: int = 12800,
                               ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR AIS science data dictionary.

    AIS echo data is a single dipole channel with 12800 uint16 samples
    per record. The echo data may be reshaped downstream depending on
    the frequency stepping scheme.

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples in the echo data (default = 12800).

    Returns:
        Preallocated dictionary for MARSIS EDR AIS science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        # First set of polynomials (even indices)
        'AH0': np.zeros(n_cols, dtype=float),
        'AH2': np.zeros(n_cols, dtype=float),
        'AH4': np.zeros(n_cols, dtype=float),
        'AH6': np.zeros(n_cols, dtype=float),
        'AR1': np.zeros(n_cols, dtype=float),
        'AR3': np.zeros(n_cols, dtype=float),
        'AR5': np.zeros(n_cols, dtype=float),
        'AR7': np.zeros(n_cols, dtype=float),
        'AT0': np.zeros(n_cols, dtype=float),
        'AT2': np.zeros(n_cols, dtype=float),
        'AT4': np.zeros(n_cols, dtype=float),
        'AT6': np.zeros(n_cols, dtype=float),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        # AIS-specific fields
        'NB_160_DEC': np.zeros(n_cols, dtype=int),
        'AGC_AIS_LAST_PRI_OF_CURRENT_FRAME': np.zeros(n_cols, dtype=float),
        'AGC_AIS_LEVEL_LAST_PRI_OF_CURRENT_FRAME': np.zeros(n_cols, dtype=int),
        'RX_TRIG_AIS': np.zeros(n_cols, dtype=int),
        'RX_TRIG_AIS_PROGR': np.zeros(n_cols, dtype=int),
        'AIS_MAXIMUM_OUTPUT_DATA_EXP': np.zeros(n_cols, dtype=int),
        # Second set of polynomials (odd indices)
        'AH1': np.zeros(n_cols, dtype=float),
        'AH3': np.zeros(n_cols, dtype=float),
        'AH5': np.zeros(n_cols, dtype=float),
        'AH7': np.zeros(n_cols, dtype=float),
        'AR0': np.zeros(n_cols, dtype=float),
        'AR2': np.zeros(n_cols, dtype=float),
        'AR4': np.zeros(n_cols, dtype=float),
        'AR6': np.zeros(n_cols, dtype=float),
        'AT1': np.zeros(n_cols, dtype=float),
        'AT3': np.zeros(n_cols, dtype=float),
        'AT5': np.zeros(n_cols, dtype=float),
        'AT7': np.zeros(n_cols, dtype=float),
        # Echo data
        'ECHO_DIP': np.zeros((n_samp, n_cols), dtype=int),
    }
    return sci_dict

##############################################################################################################
#
# CAL & RXO
#
##############################################################################################################

def make_marsis_edr_cal_f_dict(n_cols: int,
                               n_samp: int = 156800,
                               ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR CAL science data dictionary.

    CAL echo data consists of two dipole channels (F1, F2) with 156800
    signed int8 samples each per record.

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples per echo channel (default = 156800).

    Returns:
        Preallocated dictionary for MARSIS EDR CAL science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        # First set of polynomials (even indices)
        'AH0': np.zeros(n_cols, dtype=float),
        'AH2': np.zeros(n_cols, dtype=float),
        'AH4': np.zeros(n_cols, dtype=float),
        'AH6': np.zeros(n_cols, dtype=float),
        'AR1': np.zeros(n_cols, dtype=float),
        'AR3': np.zeros(n_cols, dtype=float),
        'AR5': np.zeros(n_cols, dtype=float),
        'AR7': np.zeros(n_cols, dtype=float),
        'AT0': np.zeros(n_cols, dtype=float),
        'AT2': np.zeros(n_cols, dtype=float),
        'AT4': np.zeros(n_cols, dtype=float),
        'AT6': np.zeros(n_cols, dtype=float),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        # CAL-specific fields
        'NB_160_DEC': np.zeros(n_cols, dtype=int),
        'AGC_CAL_PT_VALUE': np.zeros(n_cols, dtype=float),
        'AGC_CAL_LEVEL': np.zeros(n_cols, dtype=int),
        'RX_TRIG_CAL_COMP': np.zeros(n_cols, dtype=int),
        'RX_TRIG_CAL_PROGR': np.zeros(n_cols, dtype=int),
        # Second set of polynomials (odd indices)
        'AH1': np.zeros(n_cols, dtype=float),
        'AH3': np.zeros(n_cols, dtype=float),
        'AH5': np.zeros(n_cols, dtype=float),
        'AH7': np.zeros(n_cols, dtype=float),
        'AR0': np.zeros(n_cols, dtype=float),
        'AR2': np.zeros(n_cols, dtype=float),
        'AR4': np.zeros(n_cols, dtype=float),
        'AR6': np.zeros(n_cols, dtype=float),
        'AT1': np.zeros(n_cols, dtype=float),
        'AT3': np.zeros(n_cols, dtype=float),
        'AT5': np.zeros(n_cols, dtype=float),
        'AT7': np.zeros(n_cols, dtype=float),
        # Echo data
        'ECHO_F1_DIP': np.zeros((n_samp, n_cols), dtype=np.int8),
        'ECHO_F2_DIP': np.zeros((n_samp, n_cols), dtype=np.int8),
    }
    return sci_dict


def make_marsis_edr_rxo_f_dict(n_cols: int,
                               n_samp: int = 156800,
                               ) -> dict[str, NDArray[Any]]:
    """
    Initialize a pre-allocated MARSIS EDR RXO science data dictionary.

    RXO echo data consists of two dipole channels (F1, F2) with 156800
    signed int8 samples each per record.

    Args:
        n_cols: Number of science data records.
        n_samp: Number of samples per echo channel (default = 156800).

    Returns:
        Preallocated dictionary for MARSIS EDR RXO science data.

    TODO (low-priority; post-beta): Install exact dtypes
    """
    sci_dict: dict[str, NDArray[Any]] = {
        'SCET_STAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_STAR_FRAC': np.zeros(n_cols, dtype=int),
        'OST_LINE_NUMBER': np.zeros(n_cols, dtype=int),
        'MODE_DURATION': np.zeros(n_cols, dtype=int),
        'MODE_SELECTION': np.zeros(n_cols, dtype=int),
        'DCG_CONFIGURATION': np.zeros((2, n_cols), dtype=int),
        'PI_BAND_SEL': np.zeros((2, n_cols), dtype=int),
        'PIM_RX': np.zeros(n_cols, dtype=int),
        'REF_ALG_SEL': np.zeros(n_cols, dtype=int),
        'LOL_LOGIC_MF': np.zeros(n_cols, dtype=int),
        'PRESET_TRACKING': np.zeros(n_cols, dtype=int),
        'F_NPM_ADDRESS': np.zeros(n_cols, dtype=int),
        'SLOPE_ADDRESS': np.zeros(n_cols, dtype=int),
        'TX_POWER': np.zeros(n_cols, dtype=int),
        'A2_0_OST_ABSCISSA': np.zeros(n_cols, dtype=int),
        'IE_FM': np.zeros(n_cols, dtype=int),
        'FM_FRAMES': np.zeros(n_cols, dtype=int),
        'FRAME_ID': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_TYPE': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER': np.zeros(n_cols, dtype=int),
        'SCIENTIFIC_DATA_SEGM_FLAG': np.zeros(n_cols, dtype=int),
        'FIRST_PRI_OF_FRAME': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_FRAME_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PERICENTER_FRAC': np.zeros(n_cols, dtype=int),
        'SCET_PAR_WHOLE': np.zeros(n_cols, dtype=int),
        'SCET_PAR_FRAC': np.zeros(n_cols, dtype=int),
        'H_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VT_SCET_PAR': np.zeros(n_cols, dtype=float),
        'VR_SCET_PAR': np.zeros(n_cols, dtype=float),
        'N_0': np.zeros(n_cols, dtype=int),
        'DELTA_S_MIN': np.zeros(n_cols, dtype=float),
        'NB_MIN': np.zeros(n_cols, dtype=int),
        # First set of polynomials (even indices)
        'AH0': np.zeros(n_cols, dtype=float),
        'AH2': np.zeros(n_cols, dtype=float),
        'AH4': np.zeros(n_cols, dtype=float),
        'AH6': np.zeros(n_cols, dtype=float),
        'AR1': np.zeros(n_cols, dtype=float),
        'AR3': np.zeros(n_cols, dtype=float),
        'AR5': np.zeros(n_cols, dtype=float),
        'AR7': np.zeros(n_cols, dtype=float),
        'AT0': np.zeros(n_cols, dtype=float),
        'AT2': np.zeros(n_cols, dtype=float),
        'AT4': np.zeros(n_cols, dtype=float),
        'AT6': np.zeros(n_cols, dtype=float),
        'DELTA_S_SCET_PAR': np.zeros(n_cols, dtype=float),
        # RXO-specific fields
        'NB_160_DEC': np.zeros(n_cols, dtype=int),
        'AGC_RO_PT_VALUE': np.zeros(n_cols, dtype=float),
        'AGC_RO_LEVEL': np.zeros(n_cols, dtype=int),
        'RX_TRIG_RO_COMP': np.zeros(n_cols, dtype=int),
        'RX_TRIG_RO_PROGR': np.zeros(n_cols, dtype=int),
        # Second set of polynomials (odd indices)
        'AH1': np.zeros(n_cols, dtype=float),
        'AH3': np.zeros(n_cols, dtype=float),
        'AH5': np.zeros(n_cols, dtype=float),
        'AH7': np.zeros(n_cols, dtype=float),
        'AR0': np.zeros(n_cols, dtype=float),
        'AR2': np.zeros(n_cols, dtype=float),
        'AR4': np.zeros(n_cols, dtype=float),
        'AR6': np.zeros(n_cols, dtype=float),
        'AT1': np.zeros(n_cols, dtype=float),
        'AT3': np.zeros(n_cols, dtype=float),
        'AT5': np.zeros(n_cols, dtype=float),
        'AT7': np.zeros(n_cols, dtype=float),
        # Echo data
        'ECHO_F1_DIP': np.zeros((n_samp, n_cols), dtype=np.int8),
        'ECHO_F2_DIP': np.zeros((n_samp, n_cols), dtype=np.int8),
    }
    return sci_dict
