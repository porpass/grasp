# SPDX-License-Identifier: BSD-3-Clause
import bitstring
import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import struct
from typing import Any

from ..common.utils import calc_nrecs, decode_datetime, parse_pds_lbl
from .dictionaries import (
    make_marsis_edr_geo_dict,
    make_marsis_edr_acq_cmp_f_dict,
    make_marsis_edr_ss1_trk_cmp_f_dict,
    make_marsis_edr_ss2_trk_cmp_f_dict,
    make_marsis_edr_ss3_trk_cmp_f_dict,
    make_marsis_edr_ss3_trk_raw_f_dict,
    make_marsis_edr_ss4_acq_cmp_f_dict,
    make_marsis_edr_ss4_trk_cmp_f_dict,
    make_marsis_edr_ss5_trk_cmp_f_dict,
    make_marsis_rdr_ss3_trk_cmp_dict,
    make_marsis_rdr_ss3_trk_raw_dict,
    make_marsis_edr_ais_f_dict,
    make_marsis_edr_cal_f_dict,
    make_marsis_edr_rxo_f_dict
)

##############################################################################################################
#
# Byte Structure and Date Formats
#
##############################################################################################################
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # DATE FORMAT USED IN MARSIS DATA
AUX_BYTE_FORMAT = ">IHd23c2dI6c19d"   # BYTE FORMAT FOR MARSIS AUXILIARY DATA FILES
# BYTE FORMATS FOR MARSIS SCIENCE DATA FILES
EDR_SCI_ACQ_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH12ffH2f2BB2BfB2fB2f2B2H2f2H6f4fH2f2B2h2f4BHf3B4096B256h"
EDR_SCI_SS1_TRK_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH2f2H2f8H3If3H6f2H2f2H2f2B6H4f2h3H20B2f5Bf4097B256h"
EDR_SCI_SS2_TRK_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH2f2H2f2H2H2H2H3IfH2H2f2f2f2H2f2H2f2B2H2H2H2f2f2h3H20B2f2BB2BfB512f256h"
EDR_SCI_SS3_TRK_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH2f2H2f8H3If3H6f2H2f2H2f2B6H4f2h3H20B2f5BfB6144B256h"
RDR_SCI_SS3_TRK_CMP_FORMAT = "<2ffIH4f2H6144fd23c2dI6c3d3d3d4d3d3d"
RDR_SCI_SS3_TRK_RAW_FORMAT = "<3fIH4f2H3920fd23c2dI6c19d"
EDR_SCI_SS4_ACQ_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH12ffH2f2BB2BfB2fB2f2B2H2f2H6f4fH2f2B2h2f4BHf3B2048B256h"
EDR_SCI_SS4_TRK_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH2f2H2f8H3If3H6f2H2f2H2f2B6H4f2h3H20B2f5BfB10240B256h"
EDR_SCI_SS5_TRK_CMP_FORMAT = ">I2H12sH6s2IHIHIH3fIfH2f2H2f8H3If3H6f2H2f2H2f2B6H4f2h3H20B2f5BfB6144B256h"
EDR_SCI_AIS_FORMAT = ">I2H12sH6s2IHIHIH3fIfH12ffHfB2HB12f72B12800H"
EDR_SCI_CAL_AUX_FORMAT = ">I2H12sH6s2IHIHIH3fIfH12ffHfB2HB12f72B"
EDR_SCI_RXO_AUX_FORMAT = ">I2H12sH6s2IHIHIH3fIfH12ffHfB2HB12f72B"


##############################################################################################################
#
# WISHLIST
#
# TODO (low-priority; post-beta): Install functionality for SS1, SS2, SS4, and SS5 RDRs
#  Don't know if these actually exist
# TODO (low-priority; post-beta): Install make_marsis_r_ss1_trk_cmp_f_dict function
#  Don't know if these actually exist
# TODO (low-priority; post-beta): Install make_marsis_r_ss2_trk_cmp_f_dict function
#  Don't know if these actually exist
# TODO (low-priority; post-beta): Reinstall make_marsis_r_ss4_trk_cmp_f_dict function
#  Don't know if these actually exist
# TODO (low-priority; post-beta): Reinstall make_marsis_r_ss5_trk_cmp_f_dict function
#  Don't know if these actually exist
# TODO (low-priority; post-beta): Install Dict and Reader for MARSIS Optim products (upgrade)

##############################################################################################################

##############################################################################################################
#
# Main Wrapper Functions
#
##############################################################################################################
def parse(path: str | Path,
          prod_type: str,
          role: str,
          mode: str | None = None,
          state: str | None = None,
          form: str | None = None,
          rec_len: int | None = None,
          ) ->  dict[str, Any]:
    """Dispatch a MARSIS file to the appropriate product-specific parser.

    Args:
        path: Path to the MARSIS file.
        prod_type: Product type (``"EDR"`` or ``"RDR"``).
        role: File role (``"LABEL"``, ``"SCIENCE"``, or ``"AUXILIARY"``).
        mode: Operative mode string (e.g. ``"SS3"``).
        state: Instrument state string (e.g. ``"TRK"``, ``"ACQ"``).
        form: Data form code (e.g. ``"CMP"``, ``"UNC"``).
        rec_len: Record length in bytes.

    Returns:
        Dictionary of parsed fields produced by the dispatched parser.

    Raises:
        ValueError: If ``prod_type`` or ``role`` is not supported.
    """
    if role == "LABEL":
        return parse_marsis_label(path)
    elif prod_type == "EDR":
        if role == "AUXILIARY":
            return parse_marsis_edr_geo(path, rec_len)
        elif role == "SCIENCE":
            return parse_marsis_edr_f(path, mode, state, form, rec_len)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "RDR":
        if role == "SCIENCE":
            return parse_marsis_rdr_f(path, mode, form, rec_len)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    else:
        raise ValueError(f"Unsupported product type: {prod_type}")


def parse_marsis_label(lblFile: str | Path) -> dict[str, Any]:
    """
    Parse a PDS3 label file into a Python dictionary

    This is a wrapper for product-specific label parsing. This function is defunct,
    but can still be used as it simply calls a generic PDS3 LBL reader (PVL)

    Args:
        lblFile:  Path to the label file

    Returns:
        Dictionary containing the parsed PDS3 label metadata as native
        Python types.
    """
    return parse_pds_lbl(lblFile)

##############################################################################################################
#
# Parsers
#
##############################################################################################################
def parse_marsis_edr_geo(geo_path: str | Path,
                         rec_len: int,
                         ) -> dict[str, NDArray[Any]]:
    """Parse a MARSIS EDR auxiliary/geometry file into a preallocated dictionary.

    SS1–SS5 share the same auxiliary (G) record format. This function reads the
    fixed-length binary records, unpacks them using ``AUX_BYTE_FORMAT``, and
    fills the arrays returned by ``make_marsis_edr_geo_dict``.

    Args:
        geo_path: Path to the MARSIS EDR auxiliary/geometry file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary containing NumPy arrays filled with parsed MARSIS EDR
        auxiliary/geometry fields.

    Raises:
        ValueError: If ``rec_len`` is not positive, the file size is not an
            integer multiple of ``rec_len``, or a short read occurs.
        OSError: If the file cannot be accessed.
    """
    if rec_len <= 0:
        raise ValueError(f"rec_len must be > 0, got {rec_len}")

    path_str = os.fspath(geo_path)
    file_size = os.path.getsize(path_str)

    if file_size % rec_len != 0:
        raise ValueError(
            f"File size ({file_size} bytes) is not a multiple of rec_len ({rec_len} bytes): {path_str}"
        )

    n_recs = file_size // rec_len
    aux_dict = make_marsis_edr_geo_dict(n_recs)

    unpack = struct.Struct(AUX_BYTE_FORMAT).unpack

    with open(path_str, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise ValueError(
                    f"Short read: expected {rec_len} bytes, got {len(column)} at frame {frame} in {path_str}"
                )

            try:
                item = unpack(column)

                aux_dict["SCET_FRAME_WHOLE"][frame] = item[0]
                aux_dict["SCET_FRAME_FRAC"][frame] = item[1]
                aux_dict["GEOMETRY_EPHEMERIS_TIME"][frame] = item[2]
                aux_dict["GEOMETRY_EPOCH"][frame] = decode_datetime(item[3:26], DATE_FORMAT)
                aux_dict["MARS_SOLAR_LONGITUDE"][frame] = item[26]
                aux_dict["MARS_SUN_DISTANCE"][frame] = item[27]
                aux_dict["ORBIT_NUMBER"][frame] = item[28]
                aux_dict["TARGET_NAME"][frame] = (
                    b"".join(item[29:35]).decode("utf-8", errors="replace").strip()
                )
                aux_dict["TARGET_SC_POSITION_VECTOR"][:, frame] = item[35:38]
                aux_dict["SPACECRAFT_ALTITUDE"][frame] = item[38]
                aux_dict["SUB_SC_LONGITUDE"][frame] = item[39]
                aux_dict["SUB_SC_LATITUDE"][frame] = item[40]
                aux_dict["TARGET_SC_VELOCITY_VECTOR"][:, frame] = item[41:44]
                aux_dict["TARGET_SC_RADIAL_VELOCITY"][frame] = item[44]
                aux_dict["TARGET_SC_TANG_VELOCITY"][frame] = item[45]
                aux_dict["LOCAL_TRUE_SOLAR_TIME"][frame] = item[46]
                aux_dict["SOLAR_ZENITH_ANGLE"][frame] = item[47]
                aux_dict["DIPOLE_UNIT_VECTOR"][:, frame] = item[48:51]
                aux_dict["MONOPOLE_UNIT_VECTOR"][:, frame] = item[51:54]

            except struct.error as exc:
                raise ValueError(f"Error unpacking frame {frame} of {path_str}: {exc}") from exc
    return aux_dict


def parse_marsis_edr_f(sci_path: str | Path,
                       mode: str,
                       state: str,
                       form: str,
                       rec_len: int,
                       ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR science data file.

    This function dispatches to mode- and form-specific MARSIS EDR
    science parsers. Only a subset of MARSIS EDR modes and forms
    are currently supported.

    Args:
        sci_path: Path to the MARSIS EDR science file.
        mode: Observation mode (e.g., ``"SS1"``, ``"SS3"``, ``"SS4"``).
        state: Instrument state (ACQ, TRK)
        form: Data form (e.g., ``"CMP"``, ``"RAW"``).
        rec_len: Length of a single science data record in bytes.

    Returns:
        A dictionary containing the parsed MARSIS EDR science data,
        with values stored as NumPy arrays.

    Raises:
        ValueError: If the specified mode is unsupported.
        ValueError: If the specified form is unsupported for the given mode.

    Notes:
        - SS2 and SS5 EDR products are not yet supported.
        - This function is a thin dispatcher; all parsing logic is
          implemented in the mode-specific helper functions.
    """
    if mode == "SS1":
        if state == "TRK":
            if form == "CMP":
                return parse_marsis_edr_ss1_trk_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        elif state == "ACQ":
            if form == "CMP":
                return parse_marsis_edr_acq_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported state {state!r} for mode {mode}")
        else:
            raise ValueError(f"Unsupported state {state!r} for mode {mode}")

    elif mode == "SS2":
        if state == "TRK":
            return parse_marsis_edr_ss2_trk_cmp_f(sci_path, rec_len)
        elif state == "ACQ":
            if form == "CMP":
                return parse_marsis_edr_acq_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        else:
            raise ValueError(f"Unsupported state {state!r} for mode {mode}")

    elif mode == "SS3":
        if state == "TRK":
            if form == "CMP":
                return parse_marsis_edr_ss3_trk_cmp_f(sci_path, rec_len)
            elif form == "RAW":
                return parse_marsis_edr_ss3_trk_raw_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        elif state == "ACQ":
            if form == "CMP":
                return parse_marsis_edr_acq_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        else:
            raise ValueError(f"Unsupported state {state!r} for mode {mode}")

    elif mode == "SS4":
        if state == "TRK":
            if form == "CMP":
                return parse_marsis_edr_ss4_trk_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        elif state == "ACQ":
            if form == "CMP":
                return parse_marsis_edr_ss4_acq_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        else:
            raise ValueError(f"Unsupported state {state!r} for mode {mode}")

    elif mode == "SS5":
        if state == "TRK":
            if form == "CMP":
                return parse_marsis_edr_ss5_trk_cmp_f(sci_path, rec_len)
            else:
                raise ValueError(f"Unsupported form {form!r} for mode {mode}")
        else:
            raise ValueError(f"Unsupported state {state!r} for mode {mode}")

    elif mode == "AIS":
        return parse_marsis_edr_ais_f(sci_path, rec_len)

    elif mode == "CAL":
        return parse_marsis_edr_cal_f(sci_path, rec_len)

    elif mode == "RXO":
        return parse_marsis_edr_rxo_f(sci_path, rec_len)

    else:
        raise ValueError(f"Unsupported MARSIS EDR mode {mode!r}")


def parse_marsis_rdr_f(sci_path: str | Path,
                       mode: str,
                       form: str,
                       rec_len: int,
                       ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS RDR science data file.

    This function dispatches to mode- and form-specific MARSIS RDR
    science parsers. Currently, only SS3 RDR products are supported.

    Args:
        sci_path: Path to the MARSIS RDR science file.
        mode: Observation mode (e.g., ``"SS3"``).
        form: Data form (e.g., ``"CMP"``, ``"RAW"``).
        rec_len: Length of a single science data record in bytes.

    Returns:
        A dictionary containing the parsed MARSIS RDR science data,
        with values stored as NumPy arrays.

    Raises:
        ValueError: If the specified mode is unsupported.
        ValueError: If the specified form is unsupported for the given mode.

    Notes:
        - SS1, SS2, SS4, and SS5 RDR products are not yet supported.
    """
    mode = mode.upper()
    if mode != "SS3":
        raise ValueError(f"Unsupported MARSIS RDR mode {mode!r}")

    if form == "CMP":
        return parse_marsis_rdr_ss3_trk_cmp(sci_path, rec_len)
    if form == "RAW":
        return parse_marsis_rdr_ss3_trk_raw(sci_path, rec_len)

    raise ValueError(f"Unsupported form {form!r} for mode {mode}")

#######################################################################################################################
#
# General ACQ Parser (SS1-SS3 are identical)
#
#######################################################################################################################
def parse_marsis_edr_acq_cmp_f(sci_path: str | Path,
                               rec_len: int,
                               ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS1-SS3 ACQ CMP science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_SS1_ACQ_CMP_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_edr_ss1_acq_cmp_f_dict`.

    Args:
        sci_path: Path to the MARSIS SS1-SS3 ACQ CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_edr_acq_cmp_f_dict(n_recs)

    unpack_record = struct.Struct(EDR_SCI_ACQ_CMP_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            # idx 0: SCET_STAR_WHOLE
            sci_dict["SCET_STAR_WHOLE"][frame] = item[0]
            # idx 1: SCET_STAR_FRAC
            sci_dict["SCET_STAR_FRAC"][frame] = item[1]
            # idx 2: OST_LINE_NUMBER
            sci_dict["OST_LINE_NUMBER"][frame] = item[2]

            # idx 3: OST_LINE (bitfield)
            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict["MODE_DURATION"][frame] = ost_bits[8:32].uint
            sci_dict["MODE_SELECTION"][frame] = ost_bits[34:38].uint
            sci_dict["DCG_CONFIGURATION"][frame, 0] = ost_bits[38:40].uint
            sci_dict["DCG_CONFIGURATION"][frame, 1] = ost_bits[40:42].uint
            sci_dict["PI_BAND_SEL"][frame, 0] = ost_bits[42:45].uint
            sci_dict["PI_BAND_SEL"][frame, 1] = ost_bits[45:48].uint
            sci_dict["PIM_RX"][frame] = ost_bits[48]
            sci_dict["REF_ALG_SEL"][frame] = ost_bits[49:51].uint
            sci_dict["LOL_LOGIC_MF"][frame] = ost_bits[51:53].uint
            sci_dict["PRESET_TRACKING"][frame] = ost_bits[53]
            sci_dict["F_NPM_ADDRESS"][frame] = ost_bits[54:56].uint
            sci_dict["SLOPE_ADDRESS"][frame] = ost_bits[56:60].uint
            sci_dict["TX_POWER"][frame] = ost_bits[60:64].uint
            sci_dict["A2_0_OST_ABSCISSA"][frame] = ost_bits[64:76].uint
            sci_dict["IE_FM"][frame] = ost_bits[76:80].uint
            sci_dict["FM_FRAMES"][frame] = ost_bits[80:96].uint

            # idx 4: FRAME_ID
            sci_dict["FRAME_ID"][frame] = item[4]

            # idx 5: ANCILLARY_DATA_HEADER (bitfield)
            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict["SCIENTIFIC_DATA_TYPE"][frame] = sci_bits[0:2].uint
            sci_dict["SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER"][frame] = sci_bits[2:16].uint
            sci_dict["SCIENTIFIC_DATA_SEGM_FLAG"][frame] = sci_bits[16:18].uint

            # idx 6-7: FIRST_PRI_OF_FRAME, SCET_FRAME_WHOLE
            sci_dict["FIRST_PRI_OF_FRAME"][frame] = item[6]
            sci_dict["SCET_FRAME_WHOLE"][frame] = item[7]
            # idx 8: SCET_FRAME_FRAC
            sci_dict["SCET_FRAME_FRAC"][frame] = item[8]
            # idx 9: SCET_PERICENTER_WHOLE
            sci_dict["SCET_PERICENTER_WHOLE"][frame] = item[9]
            # idx 10: SCET_PERICENTER_FRAC
            sci_dict["SCET_PERICENTER_FRAC"][frame] = item[10]
            # idx 11: SCET_PAR_WHOLE
            sci_dict["SCET_PAR_WHOLE"][frame] = item[11]
            # idx 12: SCET_PAR_FRAC
            sci_dict["SCET_PAR_FRAC"][frame] = item[12]
            # idx 13-15: H_SCET_PAR, VT_SCET_PAR, VR_SCET_PAR
            sci_dict["H_SCET_PAR"][frame] = item[13]
            sci_dict["VT_SCET_PAR"][frame] = item[14]
            sci_dict["VR_SCET_PAR"][frame] = item[15]
            # idx 16: N_0
            sci_dict["N_0"][frame] = item[16]
            # idx 17: DELTA_S_MIN
            sci_dict["DELTA_S_MIN"][frame] = item[17]
            # idx 18: NB_MIN
            sci_dict["NB_MIN"][frame] = item[18]

            # idx 19-30: Polynomial coefficients
            sci_dict["AH0"][frame] = item[19]
            sci_dict["AH2"][frame] = item[20]
            sci_dict["AH4"][frame] = item[21]
            sci_dict["AH6"][frame] = item[22]
            sci_dict["AR1"][frame] = item[23]
            sci_dict["AR3"][frame] = item[24]
            sci_dict["AR5"][frame] = item[25]
            sci_dict["AR7"][frame] = item[26]
            sci_dict["AT0"][frame] = item[27]
            sci_dict["AT2"][frame] = item[28]
            sci_dict["AT4"][frame] = item[29]
            sci_dict["AT6"][frame] = item[30]

            # idx 31: DELTA_S_SCET_PAR
            sci_dict["DELTA_S_SCET_PAR"][frame] = item[31]
            # idx 32: NB_SCET_PAR
            sci_dict["NB_SCET_PAR"][frame] = item[32]
            # idx 33-34: AGC_PIS_PT_VALUE (2x)
            sci_dict["AGC_PIS_PT_VALUE"][:, frame] = item[33:35]
            # idx 35-36: AGC_PIS_LEVELS (2x)
            sci_dict["AGC_PIS_LEVELS"][:, frame] = item[35:37]
            # idx 37: K_PIM
            sci_dict["K_PIM"][frame] = item[37]
            # idx 38-39: PIS_MAX_DATA_EXP (2x)
            sci_dict["PIS_MAX_DATA_EXP"][:, frame] = item[38:40]
            # idx 40: AGC_NPM_PT_VALUE
            sci_dict["AGC_NPM_PT_VALUE"][frame] = item[40]
            # idx 41: AGC_NPM_LEVELS
            sci_dict["AGC_NPM_LEVELS"][frame] = item[41]
            # idx 42-43: NPM_INT (2x)
            sci_dict["NPM_INT"][:, frame] = item[42:44]
            # idx 44: X
            sci_dict["X"][frame] = item[44]
            # idx 45-46: AGC_COLL_X (2x)
            sci_dict["AGC_COLL_X"][:, frame] = item[45:47]
            # idx 47-48: AGC_COLL_X_LEVELS (2x)
            sci_dict["AGC_COLL_X_LEVELS"][:, frame] = item[47:49]
            # idx 49: RX_TRIG_ACQ_COMP
            sci_dict["RX_TRIG_ACQ_COMP"][frame] = item[49]
            # idx 50: RX_TRIG_ACQ_PROGR
            sci_dict["RX_TRIG_ACQ_PROGR"][frame] = item[50]
            # idx 51-52: AGC_SA_FOR_TRK_FRAME (2x)
            sci_dict["AGC_SA_FOR_TRK_FRAME"][:, frame] = item[51:53]
            # idx 53-54: RX_TRIG_SA_FOR_TRK_FRAME (2x)
            sci_dict["RX_TRIG_SA_FOR_TRK_FRAME"][:, frame] = item[53:55]
            # idx 55-56: DET_THRESH (2x)
            sci_dict["DET_THRESH"][:, frame] = item[55:57]
            # idx 57-58: K_DET_THRES (2x)
            sci_dict["K_DET_THRES"][:, frame] = item[57:59]
            # idx 59-60: K_DET_THRES_MIN (2x)
            sci_dict["K_DET_THRES_MIN"][:, frame] = item[59:61]
            # idx 61-64: PHI_ACQ
            sci_dict["PHI_ACQ_F1_RE"][frame] = item[61]
            sci_dict["PHI_ACQ_F1_IM"][frame] = item[62]
            sci_dict["PHI_ACQ_F2_RE"][frame] = item[63]
            sci_dict["PHI_ACQ_F2_IM"][frame] = item[64]
            # idx 65: N_D
            sci_dict["N_D"][frame] = item[65]
            # idx 66: K_AGC
            sci_dict["K_AGC"][frame] = item[66]
            # idx 67: AREF
            sci_dict["AREF"][frame] = item[67]
            # idx 68-69: REF_FUN_FLAG (2x)
            sci_dict["REF_FUN_FLAG"][:, frame] = item[68:70]
            # idx 70-71: I_LE (2x)
            sci_dict["I_LE"][:, frame] = item[70:72]
            # idx 72-73: T_LE (2x)
            sci_dict["T_LE"][:, frame] = item[72:74]
            # idx 74-77: MAX exponents
            sci_dict["MAX_RE_EXP_ZERO_F1_DIP"][frame] = item[74]
            sci_dict["MAX_IM_EXP_ZERO_F1_DIP"][frame] = item[75]
            sci_dict["MAX_RE_EXP_ZERO_F2_DIP"][frame] = item[76]
            sci_dict["MAX_IM_EXP_ZERO_F2_DIP"][frame] = item[77]
            # idx 78: NS_LED
            sci_dict["NS_LED"][frame] = item[78]
            # idx 79: PROCESSING_PRF
            sci_dict["PROCESSING_PRF"][frame] = item[79]

            # idx 80-82: SPARE_4 (skip)

            # idx 83-4178: Echo data (4 x 1024 uint8 samples)
            sci_dict["REAL_ECHO_ZERO_F1_DIP"][:, frame] = item[83:1107]
            sci_dict["IMAG_ECHO_ZERO_F1_DIP"][:, frame] = item[1107:2131]
            sci_dict["REAL_ECHO_ZERO_F2_DIP"][:, frame] = item[2131:3155]
            sci_dict["IMAG_ECHO_ZERO_F2_DIP"][:, frame] = item[3155:4179]

            # idx 4179-4434: PIS data (2 x 128 int16)
            sci_dict["PIS_F1"][:, frame] = item[4179:4307]
            sci_dict["PIS_F2"][:, frame] = item[4307:]

    return sci_dict

#######################################################################################################################
#
# SS1
#
#######################################################################################################################
def parse_marsis_edr_ss1_acq_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """Parse a MARSIS EDR SS1 ACQ CMP science file.

    Thin wrapper around :func:`parse_marsis_edr_acq_cmp_f`; SS1, SS2, and SS3
    share the same ACQ CMP record format.

    Args:
        sci_path: Path to the science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of NumPy arrays with parsed science fields.
    """
    return parse_marsis_edr_acq_cmp_f(sci_path, rec_len)


def parse_marsis_edr_ss1_trk_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS1 TRK CMP science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_SS1_TRK_CMP_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_edr_ss1_trk_cmp_f_dict`.

    Args:
        sci_path: Path to the MARSIS SS1 TRK CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``recLen``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_edr_ss1_trk_cmp_f_dict(n_recs)

    unpack_record = struct.Struct(EDR_SCI_SS1_TRK_CMP_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict["SCET_STAR_WHOLE"][frame] = item[0]
            sci_dict["SCET_STAR_FRAC"][frame] = item[1]
            sci_dict["OST_LINE_NUMBER"][frame] = item[2]

            # ---- OST line (bitfield) ----
            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict["MODE_DURATION"][frame] = ost_bits[8:32].uint
            sci_dict["MODE_SELECTION"][frame] = ost_bits[34:38].uint
            sci_dict["DCG_CONFIGURATION"][frame, 0] = ost_bits[38:40].uint
            sci_dict["DCG_CONFIGURATION"][frame, 1] = ost_bits[40:42].uint
            sci_dict["PI_BAND_SEL"][frame, 0] = ost_bits[42:45].uint
            sci_dict["PI_BAND_SEL"][frame, 1] = ost_bits[45:48].uint
            sci_dict["PIM_RX"][frame] = ost_bits[48]
            sci_dict["REF_ALG_SEL"][frame] = ost_bits[49:51].uint
            sci_dict["LOL_LOGIC_MF"][frame] = ost_bits[51:53].uint
            sci_dict["PRESET_TRACKING"][frame] = ost_bits[53]
            sci_dict["F_NPM_ADDRESS"][frame] = ost_bits[54:56].uint
            sci_dict["SLOPE_ADDRESS"][frame] = ost_bits[56:60].uint
            sci_dict["TX_POWER"][frame] = ost_bits[60:64].uint
            sci_dict["A2_0_OST_ABSCISSA"][frame] = ost_bits[64:76].uint
            sci_dict["IE_FM"][frame] = ost_bits[76:80].uint
            sci_dict["FM_FRAMES"][frame] = ost_bits[80:96].uint

            sci_dict["FRAME_ID"][frame] = item[4]

            # ---- ancillary bitfield ----
            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict["SCIENTIFIC_DATA_TYPE"][frame] = sci_bits[0:2].uint
            sci_dict["SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER"][frame] = sci_bits[2:16].uint
            sci_dict["SCIENTIFIC_DATA_SEGM_FLAG"][frame] = sci_bits[16:18].uint

            sci_dict["FIRST_PRI_OF_FRAME"][frame] = item[6]
            sci_dict["SCET_FRAME_WHOLE"][frame] = item[7]
            sci_dict["SCET_FRAME_FRAC"][frame] = item[8]
            sci_dict["SCET_PERICENTER_WHOLE"][frame] = item[9]
            sci_dict["SCET_PERICENTER_FRAC"][frame] = item[10]
            sci_dict["SCET_PAR_WHOLE"][frame] = item[11]
            sci_dict["SCET_PAR_FRAC"][frame] = item[12]
            sci_dict["H_SCET_PAR"][frame] = item[13]
            sci_dict["VT_SCET_PAR"][frame] = item[14]
            sci_dict["VR_SCET_PAR"][frame] = item[15]
            sci_dict["N_0"][frame] = item[16]
            sci_dict["DELTA_S_MIN"][frame] = item[17]
            sci_dict["NB_MIN"][frame] = item[18]
            sci_dict["M_OCOG"][:, frame] = item[19:21]
            sci_dict["INDEX_OCOG"][:, frame] = item[21:23]
            sci_dict["TRK_THRESHOLD"][:, frame] = item[23:25]
            sci_dict["INI_IND_TRK_THRESHOLD"][:, frame] = item[25:27]
            sci_dict["LAST_IND_TRK_THRESHOLD"][:, frame] = item[27:29]
            sci_dict["INI_IND_FSRM"][:, frame] = item[29:31]
            sci_dict["LAST_IND_FSRM"][:, frame] = item[31:33]

            # item[33:36] are spares
            sci_dict["DELTA_S_SCET_PAR"][frame] = item[36]
            sci_dict["NB_SCET_PAR"][frame] = item[37]
            sci_dict["NA_SCET_PAR"][:, frame] = item[38:40]
            sci_dict["A2_INI_CM"][:, frame] = item[40:42]
            sci_dict["A2_OPT"][:, frame] = item[42:44]
            sci_dict["REF_CA_OPT"][:, frame] = item[44:46]
            sci_dict["DELTA_T"][:, frame] = item[46:48]
            sci_dict["SF"][:, frame] = item[48:50]
            sci_dict["I_C"][:, frame] = item[50:52]
            sci_dict["AGC_SA_FOR_NEXT_FRAME"][:, frame] = item[52:54]
            sci_dict["AGC_SA_LEVELS_CURRENT_FRAME"][:, frame] = item[54:56]
            sci_dict["RX_TRIG_SA_FOR_NEXT_FRAME"][:, frame] = item[56:58]
            sci_dict["RX_TRIG_SA_PROGR"][:, frame] = item[58:60]
            sci_dict["INI_IND_OCOG"][frame] = item[60]
            sci_dict["LAST_IND_OCOG"][frame] = item[61]
            sci_dict["OCOG"][:, frame] = item[62:64]
            sci_dict["A"][:, frame] = item[64:66]
            sci_dict["C_LOL"][:, frame] = item[66:68]

            # item[68:70] are spares
            sci_dict["MAX_RE_EXP_ZERO_F1_DIP"][frame] = item[71]
            sci_dict["MAX_IM_EXP_ZERO_F1_DIP"][frame] = item[72]
            sci_dict["MAX_RE_EXP_ZERO_F2_DIP"][frame] = item[73]
            sci_dict["MAX_IM_EXP_ZERO_F2_DIP"][frame] = item[74]
            sci_dict["MAX_RE_EXP_ZERO_F1_MON"][frame] = item[75]
            sci_dict["MAX_IM_EXP_ZERO_F1_MON"][frame] = item[76]
            sci_dict["MAX_RE_EXP_ZERO_F2_MON"][frame] = item[77]
            sci_dict["MAX_IM_EXP_ZERO_F2_MON"][frame] = item[78]

            # item[79:90] are spares
            sci_dict["AGC_PIS_PT_VALUE"][:, frame] = item[91:93]
            sci_dict["AGC_PIS_LEVELS"][:, frame] = item[93:95]
            sci_dict["K_PIM"][frame] = item[95]
            sci_dict["PIS_MAX_DATA_EXP"][:, frame] = item[96:98]
            sci_dict["PROCESSING_PRF"][frame] = item[98]

            # item[99] is a spare.
            sci_dict["REAL_ECHO_ZERO_F1_DIP"][:, frame] = item[100:612]
            sci_dict["IMAG_ECHO_ZERO_F1_DIP"][:, frame] = item[612:1124]
            sci_dict["REAL_ECHO_ZERO_F2_DIP"][:, frame] = item[1124:1636]
            sci_dict["IMAG_ECHO_ZERO_F2_DIP"][:, frame] = item[1636:2148]
            sci_dict["REAL_ECHO_ZERO_F1_MON"][:, frame] = item[2148:2660]
            sci_dict["IMAG_ECHO_ZERO_F1_MON"][:, frame] = item[2660:3172]
            sci_dict["REAL_ECHO_ZERO_F2_MON"][:, frame] = item[3172:3684]
            sci_dict["IMAG_ECHO_ZERO_F2_MON"][:, frame] = item[3684:4196]

            sci_dict["PIS_F1"][:, frame] = item[4196:4324]
            sci_dict["PIS_F2"][:, frame] = item[4324:]

    return sci_dict

#######################################################################################################################
#
# SS2
#
#######################################################################################################################
def parse_marsis_edr_ss2_acq_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """Parse a MARSIS EDR SS2 ACQ CMP science file.

    Thin wrapper around :func:`parse_marsis_edr_acq_cmp_f`; SS1, SS2, and SS3
    share the same ACQ CMP record format.

    Args:
        sci_path: Path to the science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of NumPy arrays with parsed science fields.
    """
    return parse_marsis_edr_acq_cmp_f(sci_path, rec_len)


def parse_marsis_edr_ss2_trk_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS2 TRK CMP science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_SS2_TRK_CMP_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_edr_ss2_trk_cmp_f_dict`.

    SS2 TRK echo data is float32, 256 samples per channel, with 2 dipole
    channels (F1, F2). No real/imaginary split, no monopole channels.

    Args:
        sci_path: Path to the MARSIS SS2 TRK CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields.

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)
    sci_dict = make_marsis_edr_ss2_trk_cmp_f_dict(n_recs)
    unpack_record = struct.Struct(EDR_SCI_SS2_TRK_CMP_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][frame, 0] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][frame, 1] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][frame, 0] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][frame, 1] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]

            # TRK-specific fields
            sci_dict['M_OCOG'][:, frame] = item[19:21]
            sci_dict['INDEX_OCOG'][:, frame] = item[21:23]
            sci_dict['TRK_THRESHOLD'][:, frame] = item[23:25]
            sci_dict['INI_IND_TRK_THRESHOLD'][:, frame] = item[25:27]
            sci_dict['LAST_IND_TRK_THRESHOLD'][:, frame] = item[27:29]
            sci_dict['INI_IND_FSRM'][:, frame] = item[29:31]
            sci_dict['LAST_IND_FSRM'][:, frame] = item[31:33]

            # item[33:36] are SPARE_4
            sci_dict['DELTA_S_SCET_PAR'][frame] = item[36]
            sci_dict['NB_SCET_PAR'][frame] = item[37]
            sci_dict['NA_SCET_PAR'][:, frame] = item[38:40]
            sci_dict['A2_INI_CM'][:, frame] = item[40:42]
            sci_dict['A2_OPT'][:, frame] = item[42:44]
            sci_dict['REF_CA_OPT'][:, frame] = item[44:46]
            sci_dict['DELTA_T'][:, frame] = item[46:48]
            sci_dict['SF'][:, frame] = item[48:50]
            sci_dict['I_C'][:, frame] = item[50:52]
            sci_dict['AGC_SA_FOR_NEXT_FRAME'][:, frame] = item[52:54]
            sci_dict['AGC_SA_LEVELS_CURRENT_FRAME'][:, frame] = item[54:56]
            sci_dict['RX_TRIG_SA_FOR_NEXT_FRAME'][:, frame] = item[56:58]
            sci_dict['RX_TRIG_SA_PROGR'][:, frame] = item[58:60]
            sci_dict['INI_IND_OCOG'][frame] = item[60]
            sci_dict['LAST_IND_OCOG'][frame] = item[61]
            sci_dict['OCOG'][:, frame] = item[62:64]
            sci_dict['A'][:, frame] = item[64:66]
            sci_dict['C_LOL'][:, frame] = item[66:68]

            # SS2-specific
            sci_dict['SS2_DCEX'][:, frame] = item[68:71]

            # item[71:91] are SPARE_5 (20x uint8)
            sci_dict['AGC_PIS_PT_VALUE'][:, frame] = item[91:93]
            sci_dict['AGC_PIS_LEVELS'][:, frame] = item[93:95]
            sci_dict['K_PIM'][frame] = item[95]
            sci_dict['PIS_MAX_DATA_EXP'][:, frame] = item[96:98]
            sci_dict['PROCESSING_PRF'][frame] = item[98]

            # item[99] is SPARE_6

            # Echo data: 2 x 256 float32
            sci_dict['ECHO_ZERO_F1_DIP'][:, frame] = item[100:356]
            sci_dict['ECHO_ZERO_F2_DIP'][:, frame] = item[356:612]

            # PIS data: 2 x 128 int16
            sci_dict['PIS_F1'][:, frame] = item[612:740]
            sci_dict['PIS_F2'][:, frame] = item[740:]

    return sci_dict
#######################################################################################################################
#
# SS3
#
#######################################################################################################################
def parse_marsis_edr_ss3_acq_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """Parse a MARSIS EDR SS3 ACQ CMP science file.

    Thin wrapper around :func:`parse_marsis_edr_acq_cmp_f`; SS1, SS2, and SS3
    share the same ACQ CMP record format.

    Args:
        sci_path: Path to the science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of NumPy arrays with parsed science fields.
    """
    return parse_marsis_edr_acq_cmp_f(sci_path, rec_len)


def parse_marsis_edr_ss3_trk_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS3 TRK CMP science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_SS3_TRK_CMP_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_edr_ss3_trk_cmp_f_dict`.

    SS3 has 3 Doppler filters (MINUS1, ZERO, PLUS1) for both dipole
    channels (F1, F2), resulting in 12 echo arrays (6 real + 6 imaginary).

    Args:
        sci_path: Path to the MARSIS SS3 TRK CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_edr_ss3_trk_cmp_f_dict(n_recs)

    unpack_record = struct.Struct(EDR_SCI_SS3_TRK_CMP_FORMAT).unpack

    with open(sci_path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][0, frame] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][1, frame] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][0, frame] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][1, frame] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]
            sci_dict['M_OCOG'][:, frame] = item[19:21]
            sci_dict['INDEX_OCOG'][:, frame] = item[21:23]
            sci_dict['TRK_THRESHOLD'][:, frame] = item[23:25]
            sci_dict['INI_IND_TRK_THRESHOLD'][:, frame] = item[25:27]
            sci_dict['LAST_IND_TRK_THRESHOLD'][:, frame] = item[27:29]
            sci_dict['INI_IND_FSRM'][:, frame] = item[29:31]
            sci_dict['LAST_IND_FSRM'][:, frame] = item[31:33]

            # item[33:36] are spares
            sci_dict['DELTA_S_SCET_PAR'][frame] = item[36]
            sci_dict['NB_SCET_PAR'][frame] = item[37]
            sci_dict['NA_SCET_PAR'][:, frame] = item[38:40]
            sci_dict['A2_INI_CM'][:, frame] = item[40:42]
            sci_dict['A2_OPT'][:, frame] = item[42:44]
            sci_dict['REF_CA_OPT'][:, frame] = item[44:46]
            sci_dict['DELTA_T'][:, frame] = item[46:48]
            sci_dict['SF'][:, frame] = item[48:50]
            sci_dict['I_C'][:, frame] = item[50:52]
            sci_dict['AGC_SA_FOR_NEXT_FRAME'][:, frame] = item[52:54]
            sci_dict['AGC_SA_LEVELS_CURRENT_FRAME'][:, frame] = item[54:56]
            sci_dict['RX_TRIG_SA_FOR_NEXT_FRAME'][:, frame] = item[56:58]
            sci_dict['RX_TRIG_SA_PROGR'][:, frame] = item[58:60]
            sci_dict['INI_IND_OCOG'][frame] = item[60]
            sci_dict['LAST_IND_OCOG'][frame] = item[61]
            sci_dict['OCOG'][:, frame] = item[62:64]
            sci_dict['A'][:, frame] = item[64:66]
            sci_dict['C_LOL'][:, frame] = item[66:68]

            # item[68:70] are spares
            sci_dict['MAX_RE_EXP_MINUS1_F1_DIP'][frame] = item[71]
            sci_dict['MAX_IM_EXP_MINUS1_F1_DIP'][frame] = item[72]
            sci_dict['MAX_RE_EXP_ZERO_F1_DIP'][frame] = item[73]
            sci_dict['MAX_IM_EXP_ZERO_F1_DIP'][frame] = item[74]
            sci_dict['MAX_RE_EXP_PLUS1_F1_DIP'][frame] = item[75]
            sci_dict['MAX_IM_EXP_PLUS1_F1_DIP'][frame] = item[76]
            sci_dict['MAX_RE_EXP_MINUS1_F2_DIP'][frame] = item[77]
            sci_dict['MAX_IM_EXP_MINUS1_F2_DIP'][frame] = item[78]
            sci_dict['MAX_RE_EXP_ZERO_F2_DIP'][frame] = item[79]
            sci_dict['MAX_IM_EXP_ZERO_F2_DIP'][frame] = item[80]
            sci_dict['MAX_RE_EXP_PLUS1_F2_DIP'][frame] = item[81]
            sci_dict['MAX_IM_EXP_PLUS1_F2_DIP'][frame] = item[82]

            # item[83:91] are spares
            sci_dict['AGC_PIS_PT_VALUE'][:, frame] = item[91:93]
            sci_dict['AGC_PIS_LEVELS'][:, frame] = item[93:95]
            sci_dict['K_PIM'][frame] = item[95]
            sci_dict['PIS_MAX_DATA_EXP'][:, frame] = item[96:98]
            sci_dict['PROCESSING_PRF'][frame] = item[98]

            # item[99] is a spare
            sci_dict['REAL_ECHO_MINUS1_F1_DIP'][:, frame] = item[100:612]
            sci_dict['IMAG_ECHO_MINUS1_F1_DIP'][:, frame] = item[612:1124]
            sci_dict['REAL_ECHO_ZERO_F1_DIP'][:, frame] = item[1124:1636]
            sci_dict['IMAG_ECHO_ZERO_F1_DIP'][:, frame] = item[1636:2148]
            sci_dict['REAL_ECHO_PLUS1_F1_DIP'][:, frame] = item[2148:2660]
            sci_dict['IMAG_ECHO_PLUS1_F1_DIP'][:, frame] = item[2660:3172]
            sci_dict['REAL_ECHO_MINUS1_F2_DIP'][:, frame] = item[3172:3684]
            sci_dict['IMAG_ECHO_MINUS1_F2_DIP'][:, frame] = item[3684:4196]
            sci_dict['REAL_ECHO_ZERO_F2_DIP'][:, frame] = item[4196:4708]
            sci_dict['IMAG_ECHO_ZERO_F2_DIP'][:, frame] = item[4708:5220]
            sci_dict['REAL_ECHO_PLUS1_F2_DIP'][:, frame] = item[5220:5732]
            sci_dict['IMAG_ECHO_PLUS1_F2_DIP'][:, frame] = item[5732:6244]

            sci_dict['PIS_F1'][:, frame] = item[6244:6372]
            sci_dict['PIS_F2'][:, frame] = item[6372:]

    return sci_dict


def parse_marsis_edr_ss3_trk_raw_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS3 TRK RAW science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_SS3_TRK_RAW_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_r_ss3_trk_raw_dict`.

    Args:
        sci_path: Path to the MARSIS SS3 TRK RAW EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``recLen``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_edr_ss3_trk_raw_f_dict(n_recs)

    with open(sci_path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            bits = bitstring.BitArray(column)
            try:
                for chan in [1,2]:
                    offset = 0 if chan == 1 else 8192
                    prefix = f"B{chan}"
                    prefix2 = f"F{chan}"
                    sci_dict[f'SCET_STAR_WHOLE_{prefix}'][frame] = bits[offset+0:offset+32].uintbe  # 4 bytes
                    sci_dict[f'SCET_STAR_FRAC_{prefix}'][frame] = bits[offset+32:offset+48].uintbe  # 2 bytes
                    sci_dict[f'OST_LINE_NUMBER_{prefix}'][frame] = bits[offset+48:offset+64].uintbe  # 2 bytes
                    sci_dict[f'MODE_DURATION_{prefix}'][frame] = bits[offset+72:offset+96].uint  # 24-bit
                    sci_dict[f'MODE_SELECTION_{prefix}'][frame] = bits[offset+98:offset+102].uint  # 4-bit
                    sci_dict[f'DCG_CONFIGURATION_{prefix}'][frame, 0] = bits[offset+102:offset+104].uint
                    sci_dict[f'DCG_CONFIGURATION_{prefix}'][frame, 1] = bits[offset+104:offset+106].uint
                    sci_dict[f'PI_BAND_SEL_{prefix}'][frame, 0] = bits[offset+106:offset+109].uint
                    sci_dict[f'PI_BAND_SEL_{prefix}'][frame, 1] = bits[offset+109:offset+112].uint
                    sci_dict[f'PIM_RX_{prefix}'][frame] = bits[offset+112:offset+113].uint
                    sci_dict[f'REF_ALG_SEL_{prefix}'][frame] = bits[offset+113:offset+115].uint
                    sci_dict[f'LOL_LOGIC_MF_{prefix}'][frame, 0] = bits[offset+115:offset+116].uint
                    sci_dict[f'LOL_LOGIC_MF_{prefix}'][frame, 1] = bits[offset+116:offset+117].uint
                    sci_dict[f'PRESENT_TRACKING_{prefix}'][frame] = bits[offset+117:offset+118].uint
                    sci_dict[f'F_NPM_ADDRESS_{prefix}'][frame] = bits[offset+118:offset+120].uint
                    sci_dict[f'SLOPE_ADDRESS_{prefix}'][frame] = bits[offset+120:offset+124].uint
                    sci_dict[f'TX_POWER_{prefix}'][frame] = bits[offset+124:offset+128].uint
                    sci_dict[f'A2_0_OST_ABSCISSA_{prefix}'][frame] = bits[offset+128:offset+140].uint
                    sci_dict[f'IE_FM_{prefix}'][frame] = bits[offset+140:offset+144].uint
                    sci_dict[f'FM_FRAMES_{prefix}'][frame] = bits[offset+144:offset+160].uint
                    sci_dict[f'FRAME_ID_{prefix}'][frame] = bits[offset+160:offset+176].uint
                    sci_dict[f'FIRST_PRI_OF_FRAME_{prefix}'][frame] = bits[offset+176:offset+208].uint
                    sci_dict[f'SCET_FRAME_WHOLE_{prefix}'][frame] = bits[offset+208:offset+240].uint
                    sci_dict[f'SCET_FRAME_FRAC_{prefix}'][frame] = bits[offset+240:offset+256].uint
                    sci_dict[f'SCET_PERICENTER_WHOLE_{prefix}'][frame] = bits[offset+256:offset+288].uint
                    sci_dict[f'SCET_PERICENTER_FRAC_{prefix}'][frame] = bits[offset+288:offset+304].uint
                    sci_dict[f'NA_SCET_PAR_{prefix}'][frame] = bits[offset+304:offset+320].uint
                    sci_dict[f'BAND_{prefix}'][frame] = bits[offset+320:offset+322].uint
                    sci_dict[f'CHANNEL_{prefix}'][frame] = bits[offset+322:offset+324].uint
                    sci_dict[f'SCIENCE_DATA_TYPE_{prefix}'][frame] = bits[offset+324:offset+326].uint
                    sci_dict[f'SCIENCE_DATA_AMOUNT_{prefix}'][frame] = bits[offset+326:offset+352].uint
                    if chan == 1:
                        sci_dict[f'REAL_ECHO_ZERO_{prefix2}_DIP'][:, frame] = struct.unpack('>980b', column[44:1024])
                    else:
                        sci_dict[f'REAL_ECHO_ZERO_{prefix2}_DIP'][:, frame] = struct.unpack('>980b', column[1068:])
            except struct.error:
                raise ValueError(f"Error parsing Science data for frame {frame}")
    return sci_dict

def parse_marsis_rdr_ss3_trk_cmp(sci_path: str | Path,
                                 rec_len: int,
                                 ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS RDR SS3 TRK CMP science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``RDR_SCI_SS3_TRK_CMP_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_rdr_ss3_trk_cmp_f_dict`.

    Args:
        sci_path: Path to the MARSIS SS3 TRK CMP RDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``recLen``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_rdr_ss3_trk_cmp_dict(n_recs)

    record_unpacker = struct.Struct(RDR_SCI_SS3_TRK_CMP_FORMAT).unpack

    with open(sci_path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            item = record_unpacker(column)
            try:
                sci_dict['CENTRAL_FREQUENCY'][:, frame] = item[0:2]
                sci_dict['SLOPE'][frame] = item[2]
                sci_dict['SCET_FRAME_WHOLE'][frame] = item[3]
                sci_dict['SCET_FRAME_FRAC'][frame] = item[4]
                sci_dict['H_SCET_PAR'][frame] = item[5]
                sci_dict['VT_SCET_PAR'][frame] = item[6]
                sci_dict['VR_SCET_PAR'][frame] = item[7]
                sci_dict['DELTA_S_SCET_PAR'][frame] = item[8]
                sci_dict['NA_SCET_PAR'][:, frame] = item[9:11]
                sci_dict['ECHO_MODULUS_MINUS1_F1_DIP'][:, frame] = item[11:523]
                sci_dict['ECHO_PHASE_MINUS1_F1_DIP'][:, frame] = item[523:1035]
                sci_dict['ECHO_MODULUS_ZERO_F1_DIP'][:, frame] = item[1035:1547]
                sci_dict['ECHO_PHASE_ZERO_F1_DIP'][:, frame] = item[1547:2059]
                sci_dict['ECHO_MODULUS_PLUS1_F1_DIP'][:, frame]= item[2059:2571]
                sci_dict['ECHO_PHASE_PLUS1_F1_DIP'][:, frame] = item[2571:3083]
                sci_dict['ECHO_MODULUS_MINUS1_F2_DIP'][:, frame] = item[3083:3595]
                sci_dict['ECHO_PHASE_MINUS1_F2_DIP'][:, frame] = item[3595:4107]
                sci_dict['ECHO_MODULUS_ZERO_F2_DIP'][:, frame] = item[4107:4619]
                sci_dict['ECHO_PHASE_ZERO_F2_DIP'][:, frame] = item[4619:5131]
                sci_dict['ECHO_MODULUS_PLUS1_F2_DIP'][:, frame] = item[5131:5643]
                sci_dict['ECHO_PHASE_PLUS1_F2_DIP'][:, frame] = item[5643:6155]
                sci_dict['GEOMETRY_EPHEMERIS_TIME'][frame] = item[6155]
                sci_dict['GEOMETRY_EPOCH'][frame] = decode_datetime(item[6156:6179], DATE_FORMAT)
                sci_dict['MARS_SOLAR_LONGITUDE'][frame] = item[6179]
                sci_dict['MARS_SUN_DISTANCE'][frame] = item[6180]
                sci_dict['ORBIT_NUMBER'][frame] = item[6181]
                sci_dict['TARGET_NAME'][frame] = "".join([x.decode('UTF-8') for x in item[6182:6188]])
                sci_dict['TARGET_SC_POSITION_VECTOR'][:, frame] = item[6188:6191]
                sci_dict['SPACECRAFT_ALTITUDE'][frame] = item[6191]
                sci_dict['SUB_SC_LONGITUDE'][frame] = item[6192]
                sci_dict['SUB_SC_LATITUDE'][frame] = item[6193]
                sci_dict['TARGET_SC_VELOCITY_VECTOR'][:, frame] = item[6194:6197]
                sci_dict['TARGET_SC_RADIAL_VELOCITY'][frame] = item[6197]
                sci_dict['TARGET_SC_TANG_VELOCITY'][frame] = item[6198]
                sci_dict['LOCAL_TRUE_SOLAR_TIME'][frame] = item[6199]
                sci_dict['SOLAR_ZENITH_ANGLE'][frame] = item[6200]
                sci_dict['DIPOLE_UNIT_VECTOR'][:, frame] = item[6201:6204]
                sci_dict['MONOPOLE_UNIT_VECTOR'][:, frame] = item[6204:6207]
            except struct.error:
                print(f"Error parsing frame {frame} of {sci_path}")
    return sci_dict


def parse_marsis_rdr_ss3_trk_raw(sci_path: str | Path,
                                 rec_len: int,
                                 ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS RDR SS3 TRK RAW science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``RDR_SCI_SS3_TRK_RAW_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_rdr_ss3_trk_cmp_f_dict`.

    Args:
        sci_path: Path to the MARSIS SS3 TRK RAW RDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``recLen``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.

    TODO (low-priority; post-beta): Phase values are all 0 for SS3 RAW RDRs. Why?
    """

    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_rdr_ss3_trk_raw_dict(n_recs)

    edr_unpacker = struct.Struct(RDR_SCI_SS3_TRK_RAW_FORMAT).unpack

    with open(sci_path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            item = edr_unpacker(column)
            try:
                sci_dict['CENTRAL_FREQUENCY'][:, frame] = item[0:2]
                sci_dict['SLOPE'][frame] = item[2]
                sci_dict['SCET_FRAME_WHOLE'][frame] = item[3]
                sci_dict['SCET_FRAME_FRAC'][frame] = item[4]
                sci_dict['H_SCET_PAR'][frame] = item[5]
                sci_dict['VT_SCET_PAR'][frame] = item[6]
                sci_dict['VR_SCET_PAR'][frame] = item[7]
                sci_dict['DELTA_S_SCET_PAR'][frame] = item[8]
                sci_dict['NA_SCET_PAR'][:, frame] = item[9:11]
                sci_dict['ECHO_MODULUS_B1'][:, frame] = item[11:991]
                sci_dict['ECHO_PHASE_B1'][:, frame] = item[991:1971]
                sci_dict['ECHO_MODULUS_B2'][:, frame] = item[1971:2951]
                sci_dict['ECHO_PHASE_B2'][:, frame] = item[2951:3931]
                sci_dict['GEOMETRY_EPHEMERIS_TIME'][frame] = item[3931]
                # Convert ASCII characters to a date string and parse
                sci_dict['GEOMETRY_EPOCH'][frame] = decode_datetime(item[3932:3955], DATE_FORMAT)
                sci_dict['MARS_SOLAR_LONGITUDE'][frame] = item[3955]
                sci_dict['MARS_SUN_DISTANCE'][frame] = item[3956]
                sci_dict['ORBIT_NUMBER'][frame] = item[3957]
                sci_dict['TARGET_NAME'][frame] = "".join([x.decode('UTF-8') for x in item[3958:3964]])
                sci_dict['TARGET_SC_POSITION_VECTOR'][:, frame] = item[3964:3967]
                sci_dict['SPACECRAFT_ALTITUDE'][frame] = item[3967]
                sci_dict['SUB_SC_LONGITUDE'][frame] = item[3968]
                sci_dict['SUB_SC_LATITUDE'][frame] = item[3969]
                sci_dict['TARGET_SC_VELOCITY_VECTOR'][:, frame] = item[3970:3973]
                sci_dict['TARGET_SC_RADIAL_VELOCITY'][frame] = item[3973]
                sci_dict['TARGET_SC_TANG_VELOCITY'][frame] = item[3974]
                sci_dict['LOCAL_TRUE_SOLAR_TIME'][frame] = item[3975]
                sci_dict['SOLAR_ZENITH_ANGLE'][frame] = item[3976]
                sci_dict['DIPOLE_UNIT_VECTOR'][:, frame] = item[3977:3980]
                sci_dict['MONOPOLE_UNIT_VECTOR'][:, frame] = item[3980:3983]
            except struct.error:
                print(f"Error parsing frame {frame} of {sci_path}")
    return sci_dict
#######################################################################################################################
#
# SS4
#
#######################################################################################################################
def parse_marsis_edr_ss4_acq_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS4 ACQ CMP science file into a preallocated dictionary.

    SS4 ACQ has only F1 dipole echo channels (real and imaginary),
    no F2 dipole channels.

    Args:
        sci_path: Path to the MARSIS SS4 ACQ CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields.

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)
    sci_dict = make_marsis_edr_ss4_acq_cmp_f_dict(n_recs)
    unpack_record = struct.Struct(EDR_SCI_SS4_ACQ_CMP_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict["SCET_STAR_WHOLE"][frame] = item[0]
            sci_dict["SCET_STAR_FRAC"][frame] = item[1]
            sci_dict["OST_LINE_NUMBER"][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict["MODE_DURATION"][frame] = ost_bits[8:32].uint
            sci_dict["MODE_SELECTION"][frame] = ost_bits[34:38].uint
            sci_dict["DCG_CONFIGURATION"][frame, 0] = ost_bits[38:40].uint
            sci_dict["DCG_CONFIGURATION"][frame, 1] = ost_bits[40:42].uint
            sci_dict["PI_BAND_SEL"][frame, 0] = ost_bits[42:45].uint
            sci_dict["PI_BAND_SEL"][frame, 1] = ost_bits[45:48].uint
            sci_dict["PIM_RX"][frame] = ost_bits[48]
            sci_dict["REF_ALG_SEL"][frame] = ost_bits[49:51].uint
            sci_dict["LOL_LOGIC_MF"][frame] = ost_bits[51:53].uint
            sci_dict["PRESET_TRACKING"][frame] = ost_bits[53]
            sci_dict["F_NPM_ADDRESS"][frame] = ost_bits[54:56].uint
            sci_dict["SLOPE_ADDRESS"][frame] = ost_bits[56:60].uint
            sci_dict["TX_POWER"][frame] = ost_bits[60:64].uint
            sci_dict["A2_0_OST_ABSCISSA"][frame] = ost_bits[64:76].uint
            sci_dict["IE_FM"][frame] = ost_bits[76:80].uint
            sci_dict["FM_FRAMES"][frame] = ost_bits[80:96].uint

            sci_dict["FRAME_ID"][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict["SCIENTIFIC_DATA_TYPE"][frame] = sci_bits[0:2].uint
            sci_dict["SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER"][frame] = sci_bits[2:16].uint
            sci_dict["SCIENTIFIC_DATA_SEGM_FLAG"][frame] = sci_bits[16:18].uint

            sci_dict["FIRST_PRI_OF_FRAME"][frame] = item[6]
            sci_dict["SCET_FRAME_WHOLE"][frame] = item[7]
            sci_dict["SCET_FRAME_FRAC"][frame] = item[8]
            sci_dict["SCET_PERICENTER_WHOLE"][frame] = item[9]
            sci_dict["SCET_PERICENTER_FRAC"][frame] = item[10]
            sci_dict["SCET_PAR_WHOLE"][frame] = item[11]
            sci_dict["SCET_PAR_FRAC"][frame] = item[12]
            sci_dict["H_SCET_PAR"][frame] = item[13]
            sci_dict["VT_SCET_PAR"][frame] = item[14]
            sci_dict["VR_SCET_PAR"][frame] = item[15]
            sci_dict["N_0"][frame] = item[16]
            sci_dict["DELTA_S_MIN"][frame] = item[17]
            sci_dict["NB_MIN"][frame] = item[18]

            sci_dict["AH0"][frame] = item[19]
            sci_dict["AH2"][frame] = item[20]
            sci_dict["AH4"][frame] = item[21]
            sci_dict["AH6"][frame] = item[22]
            sci_dict["AR1"][frame] = item[23]
            sci_dict["AR3"][frame] = item[24]
            sci_dict["AR5"][frame] = item[25]
            sci_dict["AR7"][frame] = item[26]
            sci_dict["AT0"][frame] = item[27]
            sci_dict["AT2"][frame] = item[28]
            sci_dict["AT4"][frame] = item[29]
            sci_dict["AT6"][frame] = item[30]

            sci_dict["DELTA_S_SCET_PAR"][frame] = item[31]
            sci_dict["NB_SCET_PAR"][frame] = item[32]
            sci_dict["AGC_PIS_PT_VALUE"][:, frame] = item[33:35]
            sci_dict["AGC_PIS_LEVELS"][:, frame] = item[35:37]
            sci_dict["K_PIM"][frame] = item[37]
            sci_dict["PIS_MAX_DATA_EXP"][:, frame] = item[38:40]
            sci_dict["AGC_NPM_PT_VALUE"][frame] = item[40]
            sci_dict["AGC_NPM_LEVELS"][frame] = item[41]
            sci_dict["NPM_INT"][:, frame] = item[42:44]
            sci_dict["X"][frame] = item[44]
            sci_dict["AGC_COLL_X"][:, frame] = item[45:47]
            sci_dict["AGC_COLL_X_LEVELS"][:, frame] = item[47:49]
            sci_dict["RX_TRIG_ACQ_COMP"][frame] = item[49]
            sci_dict["RX_TRIG_ACQ_PROGR"][frame] = item[50]
            sci_dict["AGC_SA_FOR_TRK_FRAME"][:, frame] = item[51:53]
            sci_dict["RX_TRIG_SA_FOR_TRK_FRAME"][:, frame] = item[53:55]
            sci_dict["DET_THRESH"][:, frame] = item[55:57]
            sci_dict["K_DET_THRES"][:, frame] = item[57:59]
            sci_dict["K_DET_THRES_MIN"][:, frame] = item[59:61]
            sci_dict["PHI_ACQ_F1_RE"][frame] = item[61]
            sci_dict["PHI_ACQ_F1_IM"][frame] = item[62]
            sci_dict["PHI_ACQ_F2_RE"][frame] = item[63]
            sci_dict["PHI_ACQ_F2_IM"][frame] = item[64]
            sci_dict["N_D"][frame] = item[65]
            sci_dict["K_AGC"][frame] = item[66]
            sci_dict["AREF"][frame] = item[67]
            sci_dict["REF_FUN_FLAG"][:, frame] = item[68:70]
            sci_dict["I_LE"][:, frame] = item[70:72]
            sci_dict["T_LE"][:, frame] = item[72:74]
            sci_dict["MAX_RE_EXP_ZERO_F1_DIP"][frame] = item[74]
            sci_dict["MAX_IM_EXP_ZERO_F1_DIP"][frame] = item[75]
            sci_dict["MAX_RE_EXP_ZERO_F2_DIP"][frame] = item[76]
            sci_dict["MAX_IM_EXP_ZERO_F2_DIP"][frame] = item[77]
            sci_dict["NS_LED"][frame] = item[78]
            sci_dict["PROCESSING_PRF"][frame] = item[79]

            # idx 80-82: SPARE_4 (skip)

            # Only F1 dipole echo channels (2 x 1024 uint8)
            sci_dict["REAL_ECHO_ZERO_F1_DIP"][:, frame] = item[83:1107]
            sci_dict["IMAG_ECHO_ZERO_F1_DIP"][:, frame] = item[1107:2131]

            # PIS data
            sci_dict["PIS_F1"][:, frame] = item[2131:2259]
            sci_dict["PIS_F2"][:, frame] = item[2259:]

    return sci_dict


def parse_marsis_edr_ss4_trk_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS4 TRK CMP science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_SS4_TRK_CMP_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_edr_ss4_trk_cmp_f_dict`.

    Args:
        sci_path: Path to the MARSIS SS4 TRK CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields. Values are NumPy arrays (mostly 1D with
        length ``n_recs``; some are 2D with shape ``(n_recs, 2)`` or
        ``(n_samp, n_recs)`` depending on the field).

    Raises:
        ValueError: If the file size is not an integer multiple of ``recLen``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict = make_marsis_edr_ss4_trk_cmp_f_dict(n_recs)

    unpack_record = struct.Struct(EDR_SCI_SS4_TRK_CMP_FORMAT).unpack

    with open(sci_path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][frame, 0] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][frame, 1] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][frame, 0] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][frame, 1] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]
            sci_dict['M_OCOG'][:, frame] = item[19:21]
            sci_dict['INDEX_OCOG'][:, frame] = item[21:23]
            sci_dict['TRK_THRESHOLD'][:, frame] = item[23:25]
            sci_dict['INI_IND_TRK_THRESHOLD'][:, frame] = item[25:27]
            sci_dict['LAST_IND_TRK_THRESHOLD'][:, frame] = item[27:29]
            sci_dict['INI_IND_FSRM'][:, frame] = item[29:31]
            sci_dict['LAST_IND_FSRM'][:, frame] = item[31:33]

            # item[33:36] are spares
            sci_dict['DELTA_S_SCET_PAR'][frame] = item[36]
            sci_dict['NB_SCET_PAR'][frame] = item[37]
            sci_dict['NA_SCET_PAR'][:, frame] = item[38:40]
            sci_dict['A2_INI_CM'][:, frame] = item[40:42]
            sci_dict['A2_OPT'][:, frame] = item[42:44]
            sci_dict['REF_CA_OPT'][:, frame] = item[44:46]
            sci_dict['DELTA_T'][:, frame] = item[46:48]
            sci_dict['SF'][:, frame] = item[48:50]
            sci_dict['I_C'][:, frame] = item[50:52]
            sci_dict['AGC_SA_FOR_NEXT_FRAME'][:, frame] = item[52:54]
            sci_dict['AGC_SA_LEVELS_CURRENT_FRAME'][:, frame] = item[54:56]
            sci_dict['RX_TRIG_SA_FOR_NEXT_FRAME'][:, frame] = item[56:58]
            sci_dict['RX_TRIG_SA_PROGR'][:, frame] = item[58:60]
            sci_dict['INI_IND_OCOG'][frame] = item[60]
            sci_dict['LAST_IND_OCOG'][frame] = item[61]
            sci_dict['OCOG'][:, frame] = item[62:64]
            sci_dict['A'][:, frame] = item[64:66]
            sci_dict['C_LOL'][:, frame] = item[66:68]

            # item[68:70] are spares
            sci_dict['MAX_RE_EXP_MINUS2_F1_DIP'][frame] = item[71]
            sci_dict['MAX_IM_EXP_MINUS2_F1_DIP'][frame] = item[72]
            sci_dict['MAX_RE_EXP_MINUS1_F1_DIP'][frame] = item[73]
            sci_dict['MAX_IM_EXP_MINUS1_F1_DIP'][frame] = item[74]
            sci_dict['MAX_RE_EXP_ZERO_F1_DIP'][frame] = item[75]
            sci_dict['MAX_IM_EXP_ZERO_F1_DIP'][frame] = item[76]
            sci_dict['MAX_RE_EXP_PLUS1_F1_DIP'][frame] = item[77]
            sci_dict['MAX_IM_EXP_PLUS1_F1_DIP'][frame] = item[78]
            sci_dict['MAX_RE_EXP_PLUS2_F1_DIP'][frame] = item[79]
            sci_dict['MAX_IM_EXP_PLUS2_F1_DIP'][frame] = item[80]
            sci_dict['MAX_RE_EXP_MINUS2_F1_MON'][frame] = item[81]
            sci_dict['MAX_IM_EXP_MINUS2_F1_MON'][frame] = item[82]
            sci_dict['MAX_RE_EXP_MINUS1_F1_MON'][frame] = item[83]
            sci_dict['MAX_IM_EXP_MINUS1_F1_MON'][frame] = item[84]
            sci_dict['MAX_RE_EXP_ZERO_F1_MON'][frame] = item[85]
            sci_dict['MAX_IM_EXP_ZERO_F1_MON'][frame] = item[86]
            sci_dict['MAX_RE_EXP_PLUS1_F1_MON'][frame] = item[87]
            sci_dict['MAX_IM_EXP_PLUS1_F1_MON'][frame] = item[88]
            sci_dict['MAX_RE_EXP_PLUS2_F1_MON'][frame] = item[89]
            sci_dict['MAX_IM_EXP_PLUS2_F1_MON'][frame] = item[90]

            # item[91] is a spare
            sci_dict['AGC_PIS_PT_VALUE'][:, frame] = item[91:93]
            sci_dict['AGC_PIS_LEVELS'][:, frame] = item[93:95]
            sci_dict['K_PIM'][frame] = item[95]
            sci_dict['PIS_MAX_DATA_EXP'][:, frame] = item[96:98]
            sci_dict['PROCESSING_PRF'][frame] = item[98]

            # item[99] is a spare
            sci_dict['REAL_ECHO_MINUS2_F1_DIP'][:, frame] = item[100:612]
            sci_dict['IMAG_ECHO_MINUS2_F1_DIP'][:, frame] = item[612:1124]
            sci_dict['REAL_ECHO_MINUS1_F1_DIP'][:, frame] = item[1124:1636]
            sci_dict['IMAG_ECHO_MINUS1_F1_DIP'][:, frame] = item[1636:2148]
            sci_dict['REAL_ECHO_ZERO_F1_DIP'][:, frame] = item[2148:2660]
            sci_dict['IMAG_ECHO_ZERO_F1_DIP'][:, frame] = item[2660:3172]
            sci_dict['REAL_ECHO_PLUS1_F1_DIP'][:, frame] = item[3172:3684]
            sci_dict['IMAG_ECHO_PLUS1_F1_DIP'][:, frame] = item[3684:4196]
            sci_dict['REAL_ECHO_PLUS2_F1_DIP'][:, frame] = item[4196:4708]
            sci_dict['IMAG_ECHO_PLUS2_F1_DIP'][:, frame] = item[4708:5220]
            sci_dict['REAL_ECHO_MINUS2_F1_MON'][:, frame] = item[5220:5732]
            sci_dict['IMAG_ECHO_MINUS2_F1_MON'][:, frame] = item[5732:6244]
            sci_dict['REAL_ECHO_MINUS1_F1_MON'][:, frame] = item[6244:6756]
            sci_dict['IMAG_ECHO_MINUS1_F1_MON'][:, frame] = item[6756:7268]
            sci_dict['REAL_ECHO_ZERO_F1_MON'][:, frame] = item[7268:7780]
            sci_dict['IMAG_ECHO_ZERO_F1_MON'][:, frame] = item[7780:8292]
            sci_dict['REAL_ECHO_PLUS1_F1_MON'][:, frame] = item[8292:8804]
            sci_dict['IMAG_ECHO_PLUS1_F1_MON'][:, frame] = item[8804:9316]
            sci_dict['REAL_ECHO_PLUS2_F1_MON'][:, frame] = item[9316:9828]
            sci_dict['IMAG_ECHO_PLUS2_F1_MON'][:, frame] = item[9828:10340]
            sci_dict['PIS_F1'][:, frame] = item[10340:10468]
            sci_dict['PIS_F2'][:, frame] = item[10468:]

    return sci_dict

#######################################################################################################################
#
# SS5
#
#######################################################################################################################
def parse_marsis_edr_ss5_trk_cmp_f(sci_path: str | Path,
                                   rec_len: int,
                                   ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR SS5 TRK CMP science file into a preallocated dictionary.

    The binary layout is identical to SS3 TRK CMP. The only difference is
    in the exponent field naming: SS5 uses F1_MON where SS3 uses F2_DIP.

    Args:
        sci_path: Path to the MARSIS SS5 TRK CMP EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields.

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)
    sci_dict = make_marsis_edr_ss5_trk_cmp_f_dict(n_recs)
    unpack_record = struct.Struct(EDR_SCI_SS5_TRK_CMP_FORMAT).unpack

    with open(sci_path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][frame, 0] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][frame, 1] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][frame, 0] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][frame, 1] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]

            # TRK-specific fields
            sci_dict['M_OCOG'][:, frame] = item[19:21]
            sci_dict['INDEX_OCOG'][:, frame] = item[21:23]
            sci_dict['TRK_THRESHOLD'][:, frame] = item[23:25]
            sci_dict['INI_IND_TRK_THRESHOLD'][:, frame] = item[25:27]
            sci_dict['LAST_IND_TRK_THRESHOLD'][:, frame] = item[27:29]
            sci_dict['INI_IND_FSRM'][:, frame] = item[29:31]
            sci_dict['LAST_IND_FSRM'][:, frame] = item[31:33]

            # item[33:36] are spares
            sci_dict['DELTA_S_SCET_PAR'][frame] = item[36]
            sci_dict['NB_SCET_PAR'][frame] = item[37]
            sci_dict['NA_SCET_PAR'][:, frame] = item[38:40]
            sci_dict['A2_INI_CM'][:, frame] = item[40:42]
            sci_dict['A2_OPT'][:, frame] = item[42:44]
            sci_dict['REF_CA_OPT'][:, frame] = item[44:46]
            sci_dict['DELTA_T'][:, frame] = item[46:48]
            sci_dict['SF'][:, frame] = item[48:50]
            sci_dict['I_C'][:, frame] = item[50:52]
            sci_dict['AGC_SA_FOR_NEXT_FRAME'][:, frame] = item[52:54]
            sci_dict['AGC_SA_LEVELS_CURRENT_FRAME'][:, frame] = item[54:56]
            sci_dict['RX_TRIG_SA_FOR_NEXT_FRAME'][:, frame] = item[56:58]
            sci_dict['RX_TRIG_SA_PROGR'][:, frame] = item[58:60]
            sci_dict['INI_IND_OCOG'][frame] = item[60]
            sci_dict['LAST_IND_OCOG'][frame] = item[61]
            sci_dict['OCOG'][:, frame] = item[62:64]
            sci_dict['A'][:, frame] = item[64:66]
            sci_dict['C_LOL'][:, frame] = item[66:68]

            # item[68:71] are SPARE_5 (3x uint16)

            # Exponents — F1 DIP
            sci_dict['MAX_RE_EXP_MINUS1_F1_DIP'][frame] = item[71]
            sci_dict['MAX_IM_EXP_MINUS1_F1_DIP'][frame] = item[72]
            sci_dict['MAX_RE_EXP_ZERO_F1_DIP'][frame] = item[73]
            sci_dict['MAX_IM_EXP_ZERO_F1_DIP'][frame] = item[74]
            sci_dict['MAX_RE_EXP_PLUS1_F1_DIP'][frame] = item[75]
            sci_dict['MAX_IM_EXP_PLUS1_F1_DIP'][frame] = item[76]

            # Exponents — F1 MON (SS5-specific)
            sci_dict['MAX_RE_EXP_MINUS1_F1_MON'][frame] = item[77]
            sci_dict['MAX_IM_EXP_MINUS1_F1_MON'][frame] = item[78]
            sci_dict['MAX_RE_EXP_ZERO_F1_MON'][frame] = item[79]
            sci_dict['MAX_IM_EXP_ZERO_F1_MON'][frame] = item[80]
            sci_dict['MAX_RE_EXP_PLUS1_F1_MON'][frame] = item[81]
            sci_dict['MAX_IM_EXP_PLUS1_F1_MON'][frame] = item[82]

            # item[83:91] are SPARE_6 (8x uint8)
            sci_dict['AGC_PIS_PT_VALUE'][:, frame] = item[91:93]
            sci_dict['AGC_PIS_LEVELS'][:, frame] = item[93:95]
            sci_dict['K_PIM'][frame] = item[95]
            sci_dict['PIS_MAX_DATA_EXP'][:, frame] = item[96:98]
            sci_dict['PROCESSING_PRF'][frame] = item[98]

            # item[99] is SPARE_7

            # Echo data: 12 x 512 uint8
            sci_dict['REAL_ECHO_MINUS1_F1_DIP'][:, frame] = item[100:612]
            sci_dict['IMAG_ECHO_MINUS1_F1_DIP'][:, frame] = item[612:1124]
            sci_dict['REAL_ECHO_ZERO_F1_DIP'][:, frame] = item[1124:1636]
            sci_dict['IMAG_ECHO_ZERO_F1_DIP'][:, frame] = item[1636:2148]
            sci_dict['REAL_ECHO_PLUS1_F1_DIP'][:, frame] = item[2148:2660]
            sci_dict['IMAG_ECHO_PLUS1_F1_DIP'][:, frame] = item[2660:3172]
            sci_dict['REAL_ECHO_MINUS1_F2_DIP'][:, frame] = item[3172:3684]
            sci_dict['IMAG_ECHO_MINUS1_F2_DIP'][:, frame] = item[3684:4196]
            sci_dict['REAL_ECHO_ZERO_F2_DIP'][:, frame] = item[4196:4708]
            sci_dict['IMAG_ECHO_ZERO_F2_DIP'][:, frame] = item[4708:5220]
            sci_dict['REAL_ECHO_PLUS1_F2_DIP'][:, frame] = item[5220:5732]
            sci_dict['IMAG_ECHO_PLUS1_F2_DIP'][:, frame] = item[5732:6244]

            sci_dict['PIS_F1'][:, frame] = item[6244:6372]
            sci_dict['PIS_F2'][:, frame] = item[6372:]

    return sci_dict
#######################################################################################################################
#
# AIS
#
#######################################################################################################################
def parse_marsis_edr_ais_f(sci_path: str | Path,
                           rec_len: int,
                           ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR AIS science file into a preallocated dictionary.

    The file is treated as a sequence of fixed-length binary records. Each record
    is unpacked with ``EDR_SCI_AIS_FORMAT`` and values are written into
    arrays preallocated by :func:`make_marsis_edr_ais_f_dict`.

    AIS records contain a single dipole echo array of 12800 uint16 samples.

    Args:
        sci_path: Path to the MARSIS AIS EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields.

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If record unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)
    sci_dict = make_marsis_edr_ais_f_dict(n_recs)
    unpack_record = struct.Struct(EDR_SCI_AIS_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            item = unpack_record(column)

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][frame, 0] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][frame, 1] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][frame, 0] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][frame, 1] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]

            # First set of polynomials (even indices)
            sci_dict['AH0'][frame] = item[19]
            sci_dict['AH2'][frame] = item[20]
            sci_dict['AH4'][frame] = item[21]
            sci_dict['AH6'][frame] = item[22]
            sci_dict['AR1'][frame] = item[23]
            sci_dict['AR3'][frame] = item[24]
            sci_dict['AR5'][frame] = item[25]
            sci_dict['AR7'][frame] = item[26]
            sci_dict['AT0'][frame] = item[27]
            sci_dict['AT2'][frame] = item[28]
            sci_dict['AT4'][frame] = item[29]
            sci_dict['AT6'][frame] = item[30]

            sci_dict['DELTA_S_SCET_PAR'][frame] = item[31]

            # AIS-specific fields
            sci_dict['NB_160_DEC'][frame] = item[32]
            sci_dict['AGC_AIS_LAST_PRI_OF_CURRENT_FRAME'][frame] = item[33]
            sci_dict['AGC_AIS_LEVEL_LAST_PRI_OF_CURRENT_FRAME'][frame] = item[34]
            sci_dict['RX_TRIG_AIS'][frame] = item[35]
            sci_dict['RX_TRIG_AIS_PROGR'][frame] = item[36]
            sci_dict['AIS_MAXIMUM_OUTPUT_DATA_EXP'][frame] = item[37]

            # Second set of polynomials (odd indices)
            sci_dict['AH1'][frame] = item[38]
            sci_dict['AH3'][frame] = item[39]
            sci_dict['AH5'][frame] = item[40]
            sci_dict['AH7'][frame] = item[41]
            sci_dict['AR0'][frame] = item[42]
            sci_dict['AR2'][frame] = item[43]
            sci_dict['AR4'][frame] = item[44]
            sci_dict['AR6'][frame] = item[45]
            sci_dict['AT1'][frame] = item[46]
            sci_dict['AT3'][frame] = item[47]
            sci_dict['AT5'][frame] = item[48]
            sci_dict['AT7'][frame] = item[49]

            # item[50:122] are SPARE_4 (72 bytes)

            # Echo data: 12800 uint16
            sci_dict['ECHO_DIP'][:, frame] = item[122:]

    return sci_dict

##############################################################################################################
#
# CAL & RXO
#
##############################################################################################################
# Echo data layout (after 256-byte aux header):
#   Bytes 257-157056:  ECHO_F1_DIP  156800 x signed int8
#   Bytes 157057-313856: ECHO_F2_DIP  156800 x signed int8

ECHO_F1_OFFSET = 256  # byte offset within record
ECHO_F2_OFFSET = 157056  # byte offset within record
ECHO_NSAMP = 156800  # samples per channel


def parse_marsis_edr_cal_f(sci_path: str | Path,
                           rec_len: int,
                           ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR CAL science file into a preallocated dictionary.

    The auxiliary header (256 bytes) is unpacked with struct. The echo data
    (2 x 156800 signed int8) is read with np.frombuffer for performance,
    since struct.unpack on 313600 elements per frame would be very slow.

    Args:
        sci_path: Path to the MARSIS CAL EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields.

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If auxiliary header unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)
    sci_dict = make_marsis_edr_cal_f_dict(n_recs)
    unpack_aux = struct.Struct(EDR_SCI_CAL_AUX_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            # Unpack auxiliary header (first 256 bytes)
            item = unpack_aux(column[:256])

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][frame, 0] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][frame, 1] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][frame, 0] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][frame, 1] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]

            # First set of polynomials (even indices)
            sci_dict['AH0'][frame] = item[19]
            sci_dict['AH2'][frame] = item[20]
            sci_dict['AH4'][frame] = item[21]
            sci_dict['AH6'][frame] = item[22]
            sci_dict['AR1'][frame] = item[23]
            sci_dict['AR3'][frame] = item[24]
            sci_dict['AR5'][frame] = item[25]
            sci_dict['AR7'][frame] = item[26]
            sci_dict['AT0'][frame] = item[27]
            sci_dict['AT2'][frame] = item[28]
            sci_dict['AT4'][frame] = item[29]
            sci_dict['AT6'][frame] = item[30]

            sci_dict['DELTA_S_SCET_PAR'][frame] = item[31]

            # CAL-specific fields
            sci_dict['NB_160_DEC'][frame] = item[32]
            sci_dict['AGC_CAL_PT_VALUE'][frame] = item[33]
            sci_dict['AGC_CAL_LEVEL'][frame] = item[34]
            sci_dict['RX_TRIG_CAL_COMP'][frame] = item[35]
            sci_dict['RX_TRIG_CAL_PROGR'][frame] = item[36]

            # item[37] is SPARE_4

            # Second set of polynomials (odd indices)
            sci_dict['AH1'][frame] = item[38]
            sci_dict['AH3'][frame] = item[39]
            sci_dict['AH5'][frame] = item[40]
            sci_dict['AH7'][frame] = item[41]
            sci_dict['AR0'][frame] = item[42]
            sci_dict['AR2'][frame] = item[43]
            sci_dict['AR4'][frame] = item[44]
            sci_dict['AR6'][frame] = item[45]
            sci_dict['AT1'][frame] = item[46]
            sci_dict['AT3'][frame] = item[47]
            sci_dict['AT5'][frame] = item[48]
            sci_dict['AT7'][frame] = item[49]

            # item[50:122] are SPARE_5 (72 bytes)

            # Echo data via np.frombuffer (much faster than struct for 313600 int8)
            sci_dict['ECHO_F1_DIP'][:, frame] = np.frombuffer(
                column, dtype=np.int8, count=ECHO_NSAMP, offset=ECHO_F1_OFFSET
            )
            sci_dict['ECHO_F2_DIP'][:, frame] = np.frombuffer(
                column, dtype=np.int8, count=ECHO_NSAMP, offset=ECHO_F2_OFFSET
            )

    return sci_dict


def parse_marsis_edr_rxo_f(sci_path: str | Path,
                           rec_len: int,
                           ) -> dict[str, NDArray[Any]]:
    """
    Parse a MARSIS EDR RXO science file into a preallocated dictionary.

    The auxiliary header (256 bytes) is unpacked with struct. The echo data
    (2 x 156800 signed int8) is read with np.frombuffer for performance.

    Args:
        sci_path: Path to the MARSIS RXO EDR science file.
        rec_len: Record length in bytes.

    Returns:
        Dictionary of science fields.

    Raises:
        ValueError: If the file size is not an integer multiple of ``rec_len``.
        EOFError: If an incomplete record is encountered while reading.
        struct.error: If auxiliary header unpacking fails.
    """
    n_recs = calc_nrecs(sci_path, rec_len)
    sci_dict = make_marsis_edr_rxo_f_dict(n_recs)
    unpack_aux = struct.Struct(EDR_SCI_RXO_AUX_FORMAT).unpack

    with open(sci_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            if len(column) != rec_len:
                raise EOFError(
                    f"Incomplete record at frame {frame}: "
                    f"expected {rec_len} bytes, got {len(column)} bytes"
                )

            # Unpack auxiliary header (first 256 bytes)
            item = unpack_aux(column[:256])

            sci_dict['SCET_STAR_WHOLE'][frame] = item[0]
            sci_dict['SCET_STAR_FRAC'][frame] = item[1]
            sci_dict['OST_LINE_NUMBER'][frame] = item[2]

            ost_bits = bitstring.BitArray(bytes=item[3])
            sci_dict['MODE_DURATION'][frame] = ost_bits[8:32].uint
            sci_dict['MODE_SELECTION'][frame] = ost_bits[34:38].uint
            sci_dict['DCG_CONFIGURATION'][frame, 0] = ost_bits[38:40].uint
            sci_dict['DCG_CONFIGURATION'][frame, 1] = ost_bits[40:42].uint
            sci_dict['PI_BAND_SEL'][frame, 0] = ost_bits[42:45].uint
            sci_dict['PI_BAND_SEL'][frame, 1] = ost_bits[45:48].uint
            sci_dict['PIM_RX'][frame] = ost_bits[48]
            sci_dict['REF_ALG_SEL'][frame] = ost_bits[49:51].uint
            sci_dict['LOL_LOGIC_MF'][frame] = ost_bits[51:53].uint
            sci_dict['PRESET_TRACKING'][frame] = ost_bits[53]
            sci_dict['F_NPM_ADDRESS'][frame] = ost_bits[54:56].uint
            sci_dict['SLOPE_ADDRESS'][frame] = ost_bits[56:60].uint
            sci_dict['TX_POWER'][frame] = ost_bits[60:64].uint
            sci_dict['A2_0_OST_ABSCISSA'][frame] = ost_bits[64:76].uint
            sci_dict['IE_FM'][frame] = ost_bits[76:80].uint
            sci_dict['FM_FRAMES'][frame] = ost_bits[80:96].uint

            sci_dict['FRAME_ID'][frame] = item[4]

            sci_bits = bitstring.BitArray(bytes=item[5])
            sci_dict['SCIENTIFIC_DATA_TYPE'][frame] = sci_bits[0:2].uint
            sci_dict['SCIENTIFIC_DATA_SOURCE_SEQ_COUNTER'][frame] = sci_bits[2:16].uint
            sci_dict['SCIENTIFIC_DATA_SEGM_FLAG'][frame] = sci_bits[16:18].uint

            sci_dict['FIRST_PRI_OF_FRAME'][frame] = item[6]
            sci_dict['SCET_FRAME_WHOLE'][frame] = item[7]
            sci_dict['SCET_FRAME_FRAC'][frame] = item[8]
            sci_dict['SCET_PERICENTER_WHOLE'][frame] = item[9]
            sci_dict['SCET_PERICENTER_FRAC'][frame] = item[10]
            sci_dict['SCET_PAR_WHOLE'][frame] = item[11]
            sci_dict['SCET_PAR_FRAC'][frame] = item[12]
            sci_dict['H_SCET_PAR'][frame] = item[13]
            sci_dict['VT_SCET_PAR'][frame] = item[14]
            sci_dict['VR_SCET_PAR'][frame] = item[15]
            sci_dict['N_0'][frame] = item[16]
            sci_dict['DELTA_S_MIN'][frame] = item[17]
            sci_dict['NB_MIN'][frame] = item[18]

            # First set of polynomials (even indices)
            sci_dict['AH0'][frame] = item[19]
            sci_dict['AH2'][frame] = item[20]
            sci_dict['AH4'][frame] = item[21]
            sci_dict['AH6'][frame] = item[22]
            sci_dict['AR1'][frame] = item[23]
            sci_dict['AR3'][frame] = item[24]
            sci_dict['AR5'][frame] = item[25]
            sci_dict['AR7'][frame] = item[26]
            sci_dict['AT0'][frame] = item[27]
            sci_dict['AT2'][frame] = item[28]
            sci_dict['AT4'][frame] = item[29]
            sci_dict['AT6'][frame] = item[30]

            sci_dict['DELTA_S_SCET_PAR'][frame] = item[31]

            # RXO-specific fields
            sci_dict['NB_160_DEC'][frame] = item[32]
            sci_dict['AGC_RO_PT_VALUE'][frame] = item[33]
            sci_dict['AGC_RO_LEVEL'][frame] = item[34]
            sci_dict['RX_TRIG_RO_COMP'][frame] = item[35]
            sci_dict['RX_TRIG_RO_PROGR'][frame] = item[36]

            # item[37] is SPARE_4

            # Second set of polynomials (odd indices)
            sci_dict['AH1'][frame] = item[38]
            sci_dict['AH3'][frame] = item[39]
            sci_dict['AH5'][frame] = item[40]
            sci_dict['AH7'][frame] = item[41]
            sci_dict['AR0'][frame] = item[42]
            sci_dict['AR2'][frame] = item[43]
            sci_dict['AR4'][frame] = item[44]
            sci_dict['AR6'][frame] = item[45]
            sci_dict['AT1'][frame] = item[46]
            sci_dict['AT3'][frame] = item[47]
            sci_dict['AT5'][frame] = item[48]
            sci_dict['AT7'][frame] = item[49]

            # item[50:122] are SPARE_5 (72 bytes)

            # Echo data via np.frombuffer
            sci_dict['ECHO_F1_DIP'][:, frame] = np.frombuffer(
                column, dtype=np.int8, count=ECHO_NSAMP, offset=ECHO_F1_OFFSET
            )
            sci_dict['ECHO_F2_DIP'][:, frame] = np.frombuffer(
                column, dtype=np.int8, count=ECHO_NSAMP, offset=ECHO_F2_OFFSET
            )

    return sci_dict