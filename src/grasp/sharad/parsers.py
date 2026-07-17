# SPDX-License-Identifier: BSD-3-Clause
import bitstring
import os
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

import struct

from typing import Any

from ..common.utils import (calc_nrecs,
                            check_frame_length,
                            convert_value,
                            decode_datetime,
                            parse_pds_lbl,
                            parse_pds_xml)

from .dictionaries import (make_sharad_edr_aux_dict,
                            make_sharad_edr_sci_dict,
                            make_sharad_fpb_rtrn_dict,
                            make_sharad_fpb_sim_dict,
                            make_sharad_rdr_sci_dict,
                            make_sharad_usedr_aux_dict,
                            make_sharad_usedr_orb_dict,
                            make_sharad_usedr_sci_dict,
                            make_sharad_usrdr_aux_dict,
                            make_sharad_usrdr_sci_dict,
                            make_sharad_usscs_rtrn_dict,
                            make_sharad_usscs_sim_dict)

##############################################################################################################
#
# Notes on Products
#
##############################################################################################################
#
# SHARAD EDR -- Engineering Data Records available through the PDS Archive
#     - Auxiliary and Science Files
# SHARAD US EDR -- Engineering Data Records available through the CO-SHARPS
#     - Auxiliary, Orbit, Header, HK (not supported), various LOG files (not supported), Science Files.
# SHARAD RDR -- Reduced Data Records available through the PDS Archive
#     - Science Files only!
# SHARAD US_RDR -- Reduced Data Records available through the PDS Archive
#     - Geometry and Science Files
# SHARAD US_RDR CSC -- SHARAD Surface Clutter Simulations available through the PDS Archive
#
##############################################################################################################
#
# Wishlist
#
##############################################################################################################
#     TODO (low-priority; post-beta): Install routines for CO-SHARPS DEC HK Files
#     TODO (low-priority; post-beta): Install routines for CO-SHARPS QDA Files
#     TODO (low-priority; post-beta): Install routines for CO-SHARPS UPB Files
##############################################################################################################
#
# Byte Structure and Date Formats
#
##############################################################################################################
AUX_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
AUX_BYTE_FORMAT = ">IHd23cdi23d8fh"
ANC_BYTE_FORMAT = ">IHI2HIH2B16sB3sH2sB3sI2H31fI"
RDR_BYTE_FORMAT = "<IHIHIH3BI3B2?12BIH6B2I2H12f8f7f5f667f667f3H6fhd23cdi3d4d3d13d8fB"
ANC_BYTE_LEN = int(186)
BIT_RES_MAPPING = {1972: 4, 2872: 6, 3772: 8}
PRI_MAPPING = {1: 1428, 2: 1492, 3: 1290, 4: 2856, 5: 2984, 6: 2580}
###############################################################################################################
#
# Wrappers
#
##############################################################################################################
def parse_sharad_lbl(lblFile: str,
                     ) -> dict[str, Any]:
    """Parse a PDS3 label file into a Python dictionary

    This is a wrapper for product-specific label parsing. This function is defunct,
    but can still be used as it simply calls a generic PDS3 LBL reader (PVL)

    Args:
        lblFile:  Path to the label file

    Returns:
        Dictionary containing the parsed PDS3 label metadata as native
        Python types.
    """
    return parse_pds_lbl(lblFile)


def parse_sharad_xml(xmlFile: str,
                     ) -> dict[str, Any]:
    """Parse a PDS4 label file into a Python dictionary

    This is a wrapper for product-specific label parsing. This function is defunct,
    but can still be used as it simply calls a generic PDS4 XML reader (pds4tools)

    Args:
        xmlFile:  Path to the label file

    Returns:
        Dictionary containing the parsed PDS4 label metadata as native
        Python types.
    """
    return parse_pds_xml(xmlFile)


def parse(iFile: str,
          prod_type: str,
          role: str,
          rec_len: int | None = None,
          ) -> dict[str, Any]:
    """Read a SHARAD PDS or CO-SHARPS File into a Python dictionary

    Wrapper that will parse a SHARAD data file retrieved either from the PDS or CO-SHARPS.
    The following prod_types and roles are currently supported

    PDS
        - EDR
            - AUXILIARY, SCIENCE, LABEL
        - RDR
            - SCIENCE, LABEL
        - US_RDR
            - GEOMETRY, SCIENCE, CSIM, RTRN, LABEL
        - SCS
            - SIM, RTRN, EMAP

    CO-SHARPS
        - US EDR
            - AUXILIARY, ORBIT, HEADER, SCIENCE
        - UPB
            - UNFOCPOW, ORBIT
        - FPB
            - RGRAMS, GEOMS
        - QDA
            - MLK

    Args:
        iFile:  Path to the input PDS file
        prod_type: The product type (EDR, DEC, RDR, RGRAM, UPB, FPB, QDA, SCS)
        role: The file role (SCIENCE, LABEL, AUXILIARY, ORBIT, HEADER, GEOMETRY, RTRN, CSIM)
        rec_len: The length of each data record (can be None for some products).

    Returns:
        Dictionary containing the parse data

    Raises:
        ValueError: If the prodType or role is not supported.
    """
    if prod_type == "EDR":
        if role == "AUXILIARY":
            return parse_sharad_edr_aux(iFile, rec_len)
        elif role == "SCIENCE":
            return parse_sharad_edr_sci(iFile, rec_len)
        elif role == "LABEL":
            return parse_sharad_lbl(iFile)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "DEC":
        if role == "AUXILIARY":
            return parse_sharad_usedr_aux(iFile)
        elif role == "ORBIT":
            return parse_sharad_usedr_orb(iFile)
        elif role == "HEADER":
            return parse_sharad_usedr_hdr(iFile)
        elif role == "SCIENCE":
            return parse_sharad_usedr_sci(iFile, rec_len)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "RDR":
        if role == "SCIENCE":
            return parse_sharad_rdr_sci(iFile, rec_len)
        elif role == "LABEL":
            return parse_sharad_lbl(iFile)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "RGRAM":
        if role == "GEOMETRY":
            return parse_sharad_usrdr_aux(iFile)
        elif role == "SCIENCE":
            return parse_sharad_usrdr_sci(iFile, rec_len)
        elif role == "LABEL":
            return parse_sharad_lbl(iFile)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "UPB":
        if role == "GEOM":
            raise NotImplementedError("GEOM reading for SHARAD UPBs not supported yet")
        elif role == "SCIENCE":
            return parse_sharad_upb_sci(iFile, rec_len)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "FPB":
        if role == "GEOMETRY":
            return parse_sharad_usrdr_aux(iFile)
        elif role == "SCIENCE":
            return parse_sharad_fpb_sci(iFile, rec_len)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "FPB_SIM":
        if role == "SIM":
            return parse_sharad_fpb_sim(iFile, rec_len)
        elif role == "RTRN":
            return parse_sharad_fpb_rtrn(iFile)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "QDA":
        if role == "GEOM":
            raise NotImplementedError("GEOM reading for SHARAD QDAs not supported yet")
        elif role == "SCIENCE":
            return parse_sharad_qda_sci(iFile, rec_len)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    elif prod_type == "SCS":
        if role == "RTRN":
            return parse_sharad_usscs_rtrn(iFile)
        elif role == "SIM":
            return parse_sharad_usscs_sim(iFile, rec_len)
        elif role == "EMAP":
            return parse_sharad_usscs_emap(iFile)
        elif role == "LABEL":
            return parse_sharad_lbl(iFile)
        else:
            raise ValueError(f"Unsupported role: {role} for {prod_type}")
    else:
        raise ValueError(f"Unsupported product type: {prod_type}")

##############################################################################################################
#
# PDS EDRs
#
##############################################################################################################

def parse_sharad_edr_aux(aux_path: str | Path,
                         rec_len: int,
                         ) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD EDR auxiliary (AUX) binary file into a preallocated
    auxiliary data dictionary.

    This function reads a SHARAD EDR AUX file consisting of fixed-length
    records, unpacks each record using ``AUX_BYTE_FORMAT``, and fills the
    arrays returned by :func:`make_sharad_edr_aux_dict`. The number of
    records is inferred from the file size and the provided record length.

    For each record, the fields are mapped into the corresponding entries
    of ``aux_dict`` (e.g., ``SCET_BLOCK_WHOLE``, ``EPHEMERIS_TIME``,
    ``GEOMETRY_EPOCH``, position/velocity vectors, attitudes, and
    housekeeping parameters). The ``GEOMETRY_EPOCH`` field is decoded
    from ASCII bytes using :func:`decode_datetime` and ``AUX_DATE_FORMAT``.

    If a record cannot be unpacked due to a struct error, the function
    prints an error message and continues with the next record. Any
    corresponding array entries for that frame remain in their
    preallocated state.

    Args:
        aux_path: Path to the SHARAD EDR AUX binary file.
        rec_len: Length in bytes of a single AUX record.

    Returns:
        A dictionary of NumPy arrays containing the parsed SHARAD EDR
        auxiliary data. Array lengths are equal to the inferred number
        of records in the file.

    Raises:
        ValueError: If error occurs during parsing.
    """
    n_recs = calc_nrecs(aux_path, rec_len)

    aux_dict: dict[str, NDArray[Any]] = make_sharad_edr_aux_dict(n_recs)
    record_unpacker = struct.Struct(AUX_BYTE_FORMAT).unpack

    with open(aux_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            check_frame_length(frame, column, rec_len)
            try:
                item = record_unpacker(column)
            except struct.error:
                raise ValueError(f"Error parsing frame {frame} of {aux_path}")
            aux_dict["SCET_BLOCK_WHOLE"][frame] = item[0]
            aux_dict["SCET_BLOCK_FRAC"][frame] = item[1]
            aux_dict["EPHEMERIS_TIME"][frame] = item[2]
            # Convert ASCII characters to a date string and parse
            aux_dict["GEOMETRY_EPOCH"][frame] = decode_datetime(item[3:26], AUX_DATE_FORMAT)
            aux_dict["SOLAR_LONGITUDE"][frame] = item[26]
            aux_dict["ORBIT_NUMBER"][frame] = item[27]
            aux_dict["X_MARS_SC_POSITION_VECTOR"][frame] = item[28]
            aux_dict["Y_MARS_SC_POSITION_VECTOR"][frame] = item[29]
            aux_dict["Z_MARS_SC_POSITION_VECTOR"][frame] = item[30]
            aux_dict["SPACECRAFT_ALTITUDE"][frame] = item[31]
            aux_dict["SUB_SC_EAST_LONGITUDE"][frame] = item[32]
            aux_dict["SUB_SC_PLANETOCENTRIC_LATITUDE"][frame] = item[33]
            aux_dict["SUB_SC_PLANETOGRAPHIC_LATITUDE"][frame] = item[34]
            aux_dict["X_MARS_SC_VELOCITY_VECTOR"][frame] = item[35]
            aux_dict["Y_MARS_SC_VELOCITY_VECTOR"][frame] = item[36]
            aux_dict["Z_MARS_SC_VELOCITY_VECTOR"][frame] = item[37]
            aux_dict["MARS_SC_RADIAL_VELOCITY"][frame] = item[38]
            aux_dict["MARS_SC_TANGENTIAL_VELOCITY"][frame] = item[39]
            aux_dict["LOCAL_TRUE_SOLAR_TIME"][frame] = item[40]
            aux_dict["SOLAR_ZENITH_ANGLE"][frame] = item[41]
            aux_dict["SC_PITCH_ANGLE"][frame] = item[42]
            aux_dict["SC_YAW_ANGLE"][frame] = item[43]
            aux_dict["SC_ROLL_ANGLE"][frame] = item[44]
            aux_dict["MRO_SAMX_INNER_GIMBAL_ANGLE"][frame] = item[45]
            aux_dict["MRO_SAMX_OUTER_GIMBAL_ANGLE"][frame] = item[46]
            aux_dict["MRO_SAPX_INNER_GIMBAL_ANGLE"][frame] = item[47]
            aux_dict["MRO_SAPX_OUTER_GIMBAL_ANGLE"][frame] = item[48]
            aux_dict["MRO_HGA_INNER_GIMBAL_ANGLE"][frame] = item[49]
            aux_dict["MRO_HGA_OUTER_GIMBAL_ANGLE"][frame] = item[50]
            aux_dict["DES_TEMP"][frame] = item[51]
            aux_dict["DES_5V"][frame] = item[52]
            aux_dict["DES_12V"][frame] = item[53]
            aux_dict["DES_2V5"][frame] = item[54]
            aux_dict["RX_TEMP"][frame] = item[55]
            aux_dict["TX_TEMP"][frame] = item[56]
            aux_dict["TX_LEV"][frame] = item[57]
            aux_dict["TX_CURR"][frame] = item[58]
            aux_dict["CORRUPTED_DATA_FLAG"][frame] = item[59]


    return aux_dict

def parse_sharad_edr_sci(edr_path: str | Path,
                         rec_len: int,
                         n_samp: int = 3600,
                         ) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD EDR science (SCI) binary file into a preallocated
    science data dictionary.

    This function reads a SHARAD EDR SCI file consisting of fixed-length
    records. Each record is split into an ancillary header and a science
    data segment. The ancillary header is unpacked using
    ``ANC_BYTE_FORMAT``, decoded into bit fields where required (e.g.,
    OST line, pack segmentation / FPGA status), and used to populate the
    arrays returned by :func:`make_sharad_edr_sci_dict`.

    The science data portion of each record is decoded into
    ``ECHO_SAMPLES`` according to the per-record bit resolution:

    * 8-bit: interpreted as signed 8-bit integers via ``np.frombuffer``.
    * 6-bit: decoded using :func:`_load_sharad_6bit`.
    * 4-bit: decoded using :func:`_load_sharad_4bit`.

    The number of records is inferred from the file size and the
    provided record length.

    Args:
        edr_path: Path to the SHARAD EDR SCI binary file.
        rec_len: Length in bytes of a single record.
        n_samp: Number of science samples per record after decompression.

    Returns:
        A dictionary of NumPy arrays containing the parsed SHARAD EDR
        science data. Array lengths are equal to the inferred number of
        records in the file.

    Raises:
        ValueError: If an unsupported bit resolution is encountered.
    """
    n_recs = calc_nrecs(edr_path, rec_len)

    sci_dict: dict[str, NDArray[Any]] = make_sharad_edr_sci_dict(n_recs)

    record_unpacker = struct.Struct(ANC_BYTE_FORMAT).unpack

    with open(edr_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            check_frame_length(frame, column, rec_len)

            # Separate ancillary and science data
            anc_data = column[:ANC_BYTE_LEN]
            sci_data = column[ANC_BYTE_LEN:]
            try:
                item = record_unpacker(anc_data)
            except struct.error:
                raise ValueError(f"Error parsing frame {frame} of {edr_path}")

            sci_dict["SCET_BLOCK_WHOLE"][frame] = item[0]
            sci_dict["SCET_BLOCK_FRAC"][frame] = item[1]
            sci_dict["TLM_COUNTER"][frame] = item[2]
            sci_dict["FMT_LENGTH"][frame] = item[3]
            sci_dict["BIT_RESOLUTION"][frame] = BIT_RES_MAPPING.get(item[3], 8)
            sci_dict["SCET_OST_WHOLE"][frame] = item[5]
            sci_dict["SCET_OST_FRAC"][frame] = item[6]
            sci_dict["OST_LINE_NUMBER"][frame] = item[8]

            # OST bitstring
            ost_bits = bitstring.BitArray(item[9])
            sci_dict["OST_LINE_PULSE_REPETITION_INTERVAL"][frame] = (PRI_MAPPING.get(ost_bits[0:4].uint, 0))
            sci_dict["OST_LINE_PHASE_COMPENSATION_TYPE"][frame] = ost_bits[4:8].uint
            sci_dict["OST_LINE_DATA_TAKE_LENGTH"][frame] = ost_bits[10:32].uint
            sci_dict["OST_LINE_OPERATIVE_MODE"][frame] = ost_bits[32:40].uint
            sci_dict["OST_LINE_MANUAL_GAIN_CONTROL"][frame] = ost_bits[40:48].uint
            sci_dict["OST_LINE_COMPRESSION_SELECTION"][frame] = ost_bits[48]
            sci_dict["OST_LINE_CLOSED_LOOP_TRACKING"][frame] = ost_bits[49]
            sci_dict["OST_LINE_TRACKING_DATA_STORAGE"][frame] = ost_bits[50]
            sci_dict["OST_LINE_TRACKING_PRE_SUMMING"][frame] = ost_bits[51:54].uint
            sci_dict["OST_LINE_TRACKING_LOGIC_SELECTION"][frame] = ost_bits[54]
            sci_dict["OST_LINE_THRESHOLD_LOGIC_SELECTION"][frame] = ost_bits[55]
            sci_dict["OST_LINE_SAMPLE_NUMBER"][frame] = ost_bits[56:60].uint
            sci_dict["OST_LINE_ALPHA_BETA"][frame] = ost_bits[61:63].uint
            sci_dict["OST_LINE_REFERENCE_BIT"][frame] = ost_bits[63]
            sci_dict["OST_LINE_THRESHOLD"][frame] = ost_bits[64:72].uint
            sci_dict["OST_LINE_THRESHOLD_INCREMENT"][frame] = ost_bits[72:80].uint
            sci_dict["OST_LINE_INITIAL_ECHO_VALUE"][frame] = ost_bits[84:87].uint
            sci_dict["OST_LINE_EXPECTED_ECHO_SHIFT"][frame] = ost_bits[87:90].uint
            sci_dict["OST_LINE_WINDOW_LEFT_SHIFT"][frame] = ost_bits[90:93].uint
            sci_dict["OST_LINE_WINDOW_RIGHT_SHIFT"][frame] = ost_bits[93:96].uint

            # Data block ID + counters
            sci_dict["DATA_BLOCK_ID"][frame] = bitstring.BitArray(item[11]).uint
            sci_dict["SCIENCE_DATA_SOURCE_COUNTER"][frame] = item[12]

            # PACK_SEGMENTATION_AND_FPGA_STATUS bitstring
            psfs = bitstring.BitArray(item[13])
            sci_dict["PSFS_SCIENTIFIC_DATA_TYPE"][frame] = psfs[0]
            sci_dict["PSFS_SEGMENTATION_FLAG"][frame] = psfs[1:3].uint
            sci_dict["PSFS_DMA_ERROR"][frame] = psfs[12]
            sci_dict["PSFS_TC_OVERRUN"][frame] = psfs[13]
            sci_dict["PSFS_FIFO_FULL"][frame] = psfs[14]
            sci_dict["PSFS_TEST"][frame] = psfs[15]

            # Remaining ancillary fields
            sci_dict["DATA_BLOCK_FIRST_PRI"][frame] = bitstring.BitArray(item[15]).uint
            sci_dict["TIME_DATA_BLOCK_WHOLE"][frame] = item[16]
            sci_dict["TIME_DATA_BLOCK_FRAC"][frame] = item[17]
            sci_dict["SDI_BIT_FIELD"][frame] = item[18]
            sci_dict["TIME_N"][frame] = item[19]
            sci_dict["RADIUS_N"][frame] = item[20]
            sci_dict["TANGENTIAL_VELOCITY_N"][frame] = item[21]
            sci_dict["RADIAL_VELOCITY_N"][frame] = item[22]
            sci_dict["TLP"][frame] = item[23]
            sci_dict["TIME_WPF"][frame] = item[24]
            sci_dict["DELTA_TIME"][frame] = item[25]
            sci_dict["TLP_INTERPOLATE"][frame] = item[26]
            sci_dict["RADIUS_INTERPOLATE"][frame] = item[27]
            sci_dict["TANGENTIAL_VELOCITY_INTERPOLATE"][frame] = item[28]
            sci_dict["RADIAL_VELOCITY_INTERPOLATE"][frame] = item[29]
            sci_dict["END_TLP"][frame] = item[30]
            sci_dict["S_COEFFS"][:, frame] = item[31:39]
            sci_dict["C_COEFFS"][:, frame] = item[39:46]
            sci_dict["SLOPE"][frame] = item[46]
            sci_dict["TOPOGRAPHY"][frame] = item[47]
            sci_dict["PHASE_COMPENSATION_STEP"][frame] = item[48]
            sci_dict["RECEIVE_WINDOW_OPENING_TIME"][frame] = item[49]
            sci_dict["RECEIVE_WINDOW_POSITION"][frame] = item[50]

            # Science data decoding
            bit_res = int(sci_dict["BIT_RESOLUTION"][frame])
            if bit_res == 8:
                sci_dict["ECHO_SAMPLES"][:, frame] = np.frombuffer(
                    sci_data, dtype=np.int8, count=n_samp
                )
            elif bit_res == 6:
                sci_dict["ECHO_SAMPLES"][:, frame] = _load_sharad_6bit(sci_data)
            elif bit_res == 4:
                sci_dict["ECHO_SAMPLES"][:, frame] = _load_sharad_4bit(sci_data)
            else:
                raise ValueError(f"Unsupported bit resolution {bit_res} for frame {frame} in {edr_path}")
    return sci_dict


##############################################################################################################
#
# PDS RDRs
#
##############################################################################################################
def parse_sharad_rdr_sci(rdr_path: str | Path,
                         rec_len: int,
                         n_samp: int = 667,
                         ) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD RDR science (SCI) binary file into a preallocated
    science data dictionary.

    This function reads a SHARAD RDR SCI file consisting of fixed-length
    records. Each record is unpacked using ``RDR_BYTE_FORMAT`` and used
    to populate the arrays returned by :func:`make_sharad_rdr_sci_dict`.
    The number of records is inferred from the file size and the provided
    record length.

    For each record, the function populates:

    * Ancillary scalar fields (SCET, OST, tracking, thresholds, time,
      geometry, attitude, housekeeping, etc.).
    * Coefficient arrays:
        - ``S_COEFFS``: row ``frame`` is filled from ``item[51:59]``.
        - ``C_COEFFS``: row ``frame`` is filled from ``item[59:66]``.
    * Echo samples:
        - ``ECHO_SAMPLES_REAL[:, frame]`` from ``item[71:738]``.
        - ``ECHO_SAMPLES_IMAGINARY[:, frame]`` from ``item[738:1405]``.

    At the end, the complex echo array ``ECHO_SAMPLES`` is formed as::

        ECHO_SAMPLES = ECHO_SAMPLES_REAL + 1j * ECHO_SAMPLES_IMAGINARY

    Notes:
        The ``n_samp`` parameter defaults to 667, matching the expected
        number of echo samples per record (i.e., the length of the
        ``ECHO_SAMPLES_REAL`` / ``ECHO_SAMPLES_IMAGINARY`` slices). It is
        currently not used directly in the unpacking logic but is kept
        for interface compatibility and documentation.

    Args:
        rdr_path: Path to the SHARAD RDR SCI binary file.
        rec_len: Length in bytes of a single RDR record.
        n_samp: Number of echo samples per record (nominally 667).

    Returns:
        A dictionary of NumPy arrays containing the parsed SHARAD RDR
        science data. Array lengths equal the inferred number of records
        in the file.

    Raises:
        ValueError: If the file size is not an integer multiple of
            ``rec_len``.
        struct.error: If a record cannot be unpacked with
            ``RDR_BYTE_FORMAT``.
    """
    n_recs = calc_nrecs(rdr_path, rec_len)

    sci_dict: dict[str, NDArray[Any]] = make_sharad_rdr_sci_dict(n_recs, n_samp=n_samp)
    record_unpacker = struct.Struct(RDR_BYTE_FORMAT).unpack

    with open(rdr_path, "rb") as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            check_frame_length(frame, column, rec_len)
            try:
                item = record_unpacker(column)
            except struct.error:
                raise ValueError(f"Error parsing frame {frame} of {rdr_path}")

            sci_dict["SCET_BLOCK_WHOLE"][frame] = item[0]
            sci_dict["SCET_BLOCK_FRAC"][frame] = item[1]
            sci_dict["TLM_COUNTER"][frame] = item[2]
            sci_dict["FMT_LENGTH"][frame] = item[3]
            sci_dict["SCET_OST_WHOLE"][frame] = item[4]
            sci_dict["SCET_OST_FRAC"][frame] = item[5]
            sci_dict["OST_LINE_NUMBER"][frame] = item[6]
            sci_dict["OST_LINE_PULSE_REPETITION_INTERVAL"][frame] = item[7]
            sci_dict["OST_LINE_PHASE_COMPENSATION_TYPE"][frame] = item[8]
            sci_dict["OST_LINE_DATA_TAKE_LENGTH"][frame] = item[9]
            sci_dict["OST_LINE_OPERATIVE_MODE"][frame] = item[10]
            sci_dict["OST_LINE_MANUAL_GAIN_CONTROL"][frame] = item[11]
            sci_dict["OST_LINE_COMPRESSION_SELECTION"][frame] = item[12]
            sci_dict["OST_LINE_CLOSED_LOOP_TRACKING"][frame] = item[13]
            sci_dict["OST_LINE_TRACKING_DATA_STORAGE"][frame] = item[14]
            sci_dict["OST_LINE_TRACKING_PRE_SUMMING"][frame] = item[15]
            sci_dict["OST_LINE_TRACKING_LOGIC_SELECTION"][frame] = item[16]
            sci_dict["OST_LINE_THRESHOLD_LOGIC_SELECTION"][frame] = item[17]
            sci_dict["OST_LINE_SAMPLE_NUMBER"][frame] = item[18]
            sci_dict["OST_LINE_ALPHA_BETA"][frame] = item[19]
            sci_dict["OST_LINE_REFERENCE_BIT"][frame] = item[20]
            sci_dict["OST_LINE_THRESHOLD"][frame] = item[21]
            sci_dict["OST_LINE_THRESHOLD_INCREMENT"][frame] = item[22]
            sci_dict["OST_LINE_INITIAL_ECHO_VALUE"][frame] = item[23]
            sci_dict["OST_LINE_EXPECTED_ECHO_SHIFT"][frame] = item[24]
            sci_dict["OST_LINE_WINDOW_LEFT_SHIFT"][frame] = item[25]
            sci_dict["OST_LINE_WINDOW_RIGHT_SHIFT"][frame] = item[26]
            sci_dict["DATA_BLOCK_ID"][frame] = item[27]
            sci_dict["SCIENCE_DATA_SOURCE_COUNTER"][frame] = item[28]
            sci_dict["PSFS_SCIENTIFIC_DATA_TYPE"][frame] = item[29]
            sci_dict["PSFS_SEGMENTATION_FLAG"][frame] = item[30]
            sci_dict["PSFS_DMA_ERROR"][frame] = item[31]
            sci_dict["PSFS_TC_OVERRUN"][frame] = item[32]
            sci_dict["PSFS_FIFO_FULL"][frame] = item[33]
            sci_dict["PSFS_TEST"][frame] = item[34]
            sci_dict["DATA_BLOCK_FIRST_PRI"][frame] = item[35]
            sci_dict["TIME_DATA_BLOCK_WHOLE"][frame] = item[36]
            sci_dict["TIME_DATA_BLOCK_FRAC"][frame] = item[37]
            sci_dict["SDI_BIT_FIELD"][frame] = item[38]
            sci_dict["TIME_N"][frame] = item[39]
            sci_dict["RADIUS_N"][frame] = item[40]
            sci_dict["TANGENTIAL_VELOCITY_N"][frame] = item[41]
            sci_dict["RADIAL_VELOCITY_N"][frame] = item[42]
            sci_dict["TLP"][frame] = item[43]
            sci_dict["TIME_WPF"][frame] = item[44]
            sci_dict["DELTA_TIME"][frame] = item[45]
            sci_dict["TLP_INTERPOLATE"][frame] = item[46]
            sci_dict["RADIUS_INTERPOLATE"][frame] = item[47]
            sci_dict["TANGENTIAL_VELOCITY_INTERPOLATE"][frame] = item[48]
            sci_dict["RADIAL_VELOCITY_INTERPOLATE"][frame] = item[49]
            sci_dict["END_TLP"][frame] = item[50]

            # Coefficients
            sci_dict["S_COEFFS"][:, frame] = item[51:59]
            sci_dict["C_COEFFS"][:, frame] = item[59:66]

            # Remaining scalar fields
            sci_dict["SLOPE"][frame] = item[66]
            sci_dict["TOPOGRAPHY"][frame] = item[67]
            sci_dict["PHASE_COMPENSATION_STEP"][frame] = item[68]
            sci_dict["RECEIVE_WINDOW_OPENING_TIME"][frame] = item[69]
            sci_dict["ANTENNA_RELATIVE_GAIN"][frame] = item[70]

            # Echo samples: real and imaginary parts
            sci_dict["ECHO_SAMPLES_REAL"][:, frame] = item[71:738]
            sci_dict["ECHO_SAMPLES_IMAGINARY"][:, frame] = item[738:1405]

            # Trailing block metadata / geometry / housekeeping
            sci_dict["N_PRE"][frame] = item[1405]
            sci_dict["BLOCK_NR"][frame] = item[1406]
            sci_dict["BLOCK_ROWS"][frame] = item[1407]
            sci_dict["DOPPLER_BW"][frame] = item[1408]
            sci_dict["DOPPLER_CENTROID"][frame] = item[1409]
            sci_dict["AZ_TIME_SPACING"][frame] = item[1410]
            sci_dict["AZ_RES"][frame] = item[1411]
            sci_dict["T_INT"][frame] = item[1412]
            sci_dict["AVG_TAN_VELOCITY"][frame] = item[1413]
            sci_dict["RANGE_SHIFT"][frame] = item[1414]
            sci_dict["EPHEMERIS_TIME"][frame] = item[1415]
            sci_dict["GEOMETRY_EPOCH"][frame] = decode_datetime(
                item[1416:1439], AUX_DATE_FORMAT
            )
            sci_dict["SOLAR_LONGITUDE"][frame] = item[1439]
            sci_dict["ORBIT_NUMBER"][frame] = item[1440]
            sci_dict["MARS_SC_POSITION_VECTOR"][:, frame] = item[1441:1444]
            sci_dict["SPACECRAFT_ALTITUDE"][frame] = item[1444]
            sci_dict["SUB_SC_EAST_LONGITUDE"][frame] = item[1445]
            sci_dict["SUB_SC_PLANETOCENTRIC_LATITUDE"][frame] = item[1446]
            sci_dict["SUB_SC_PLANETOGRAPHIC_LATITUDE"][frame] = item[1447]
            sci_dict["MARS_SC_VELOCITY_VECTOR"][:, frame] = item[1448:1451]
            sci_dict["MARS_SC_RADIAL_VELOCITY"][frame] = item[1451]
            sci_dict["MARS_SC_TANGENTIAL_VELOCITY"][frame] = item[1452]
            sci_dict["LOCAL_TRUE_SOLAR_TIME"][frame] = item[1453]
            sci_dict["SOLAR_ZENITH_ANGLE"][frame] = item[1454]
            sci_dict["SC_PITCH_ANGLE"][frame] = item[1455]
            sci_dict["SC_YAW_ANGLE"][frame] = item[1456]
            sci_dict["SC_ROLL_ANGLE"][frame] = item[1457]
            sci_dict["MRO_SAMX_INNER_GIMBAL_ANGLE"][frame] = item[1458]
            sci_dict["MRO_SAMX_OUTER_GIMBAL_ANGLE"][frame] = item[1459]
            sci_dict["MRO_SAPX_INNER_GIMBAL_ANGLE"][frame] = item[1460]
            sci_dict["MRO_SAPX_OUTER_GIMBAL_ANGLE"][frame] = item[1461]
            sci_dict["MRO_HGA_INNER_GIMBAL_ANGLE"][frame] = item[1462]
            sci_dict["MRO_HGA_OUTER_GIMBAL_ANGLE"][frame] = item[1463]
            sci_dict["DES_TEMP"][frame] = item[1464]
            sci_dict["DES_5V"][frame] = item[1465]
            sci_dict["DES_12V"][frame] = item[1466]
            sci_dict["DES_2V5"][frame] = item[1467]
            sci_dict["RX_TEMP"][frame] = item[1468]
            sci_dict["TX_TEMP"][frame] = item[1469]
            sci_dict["TX_LEV"][frame] = item[1470]
            sci_dict["TX_CURR"][frame] = item[1471]
            sci_dict["QUALITY_CODE"][frame] = item[1472]

    # Combine real and imaginary parts into complex echo samples
    sci_dict["ECHO_SAMPLES"] = (
        sci_dict["ECHO_SAMPLES_REAL"] + 1j * sci_dict["ECHO_SAMPLES_IMAGINARY"]
    )

    return sci_dict

##############################################################################################################
#
# PDS US_RDRs
#
##############################################################################################################
def parse_sharad_usrdr_aux(aux_path: str | Path,
                           ) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD US_RDR auxiliary CSV file into a preallocated auxiliary
    data dictionary.

    This function reads a SHARAD US_RDR AUX text file where each line
    contains comma-separated auxiliary values. The number of records is
    inferred from the number of lines in the file. Data are parsed and
    stored into the arrays returned by
    :func:`make_sharad_usrdr_aux_dict`.

    The ``TIME`` field is parsed using :func:`decode_datetime` and
    ``AUX_DATE_FORMAT``. All numeric fields are cast to ``float`` or
    ``int`` as appropriate.

    Args:
        aux_path: Path to the SHARAD US_RDR auxiliary CSV file.

    Returns:
        A dictionary of NumPy arrays containing the parsed SHARAD US_RDR
        auxiliary data. Array lengths are equal to the number of records
        in the file.
    """
    with open(aux_path, "r") as f:
        content = [line.strip() for line in f.readlines()]

    n_recs = len(content)
    aux_dict: dict[str, NDArray[Any]] = make_sharad_usrdr_aux_dict(n_recs)

    for idx, line in enumerate(content):
        csvs = line.split(",")
        aux_dict["RADARGRAM_COLUMN"][idx] = int(float(csvs[0]))
        aux_dict["TIME"][idx] = decode_datetime(csvs[1], AUX_DATE_FORMAT)
        aux_dict["LATITUDE"][idx] = float(csvs[2])
        aux_dict["LONGITUDE"][idx] = float(csvs[3])
        aux_dict["MARS_RADIUS"][idx] = float(csvs[4])
        aux_dict["SPACECRAFT_RADIUS"][idx] = float(csvs[5])
        aux_dict["RADIAL_VELOCITY"][idx] = float(csvs[6])
        aux_dict["TANGENTIAL_VELOCITY"][idx] = float(csvs[7])
        aux_dict["SZA"][idx] = float(csvs[8])
        aux_dict["PHASE/1.0E16"][idx] = float(csvs[9])

    return aux_dict


def parse_sharad_usrdr_sci(sci_path: str | Path,
                           rec_len: int,
                           n_samp: int = 3600,
                           ) -> dict[str, NDArray[np.float32]]:
    """Parse the SHARAD US_RDR science (SCI) data file into a preallocated
    science data dictionary.

    This function reads a SHARAD US_RDR science file containing
    range-compressed echo data stored as 32-bit floating-point values.
    The number of records is inferred from the file size and the provided
    record length. The data are read with ``np.fromfile``, reshaped to
    ``(n_samp, n_recs)``, and stored in the ``IMAGE`` array returned by
    :func:`make_sharad_usrdr_sci_dict`.

    Args:
        sci_path: Path to the SHARAD US_RDR science file.
        rec_len: Record length in bytes.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing the preallocated and filled ``IMAGE``
        array with shape ``(n_samp, n_recs)`` and ``float32`` dtype.

    Raises:
        ValueError: If the total number of samples in the file does not
            match ``n_samp * n_recs``.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict: dict[str, NDArray[np.float32]] = make_sharad_usrdr_sci_dict(n_recs, n_samp)

    raw = np.fromfile(sci_path, dtype=np.float32, count=-1)
    expected = n_samp * n_recs
    if raw.size != expected:
        raise ValueError(
            f"Unexpected number of samples in {sci_path!s}: "
            f"got {raw.size}, expected {expected} "
            f"(n_samp={n_samp}, n_recs={n_recs})."
        )

    sci_dict["IMAGE"][:] = raw.reshape((n_samp, n_recs))
    return sci_dict

##############################################################################################################
#
# PDS SCSs
#
##############################################################################################################

def parse_sharad_usscs_sim(sim_path: str | Path,
                           rec_len: int,
                           n_samp: int = 3600,
                           ) -> dict[str, NDArray[np.float32]]:
    """Parse the SHARAD US_RDR CSIM (simulation) binary file into a
    preallocated science data dictionary.

    The CSIM file contains three products stored sequentially as
    32-bit little-endian floats:

        1. LEFT
        2. RIGHT
        3. COMBINED

    Each product consists of ``n_recs`` records with ``n_samp`` samples
    per record. The number of records is inferred from the file size,
    the provided record length, and the fixed number of products (3).

    Data are read with ``np.fromfile`` using little-endian ``float32``
    and reshaped into ``(n_samp, n_recs)`` arrays for each product.

    Args:
        sim_path: Path to the SHARAD US_RDR CSIM binary file.
        rec_len: Record length in bytes.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing three ``float32`` arrays with shape
        ``(n_samp, n_recs)``:
            - ``LEFT``
            - ``RIGHT``
            - ``COMBINED``

    Raises:
        ValueError: If the file size is not consistent with the expected
            number of records, products, and samples.
    """
    n_prods = 3  # LEFT, RIGHT, COMBINED

    file_size = os.path.getsize(sim_path)
    if file_size % (rec_len * n_prods) != 0:
        raise ValueError(
            f"File size of {sim_path!s} ({file_size} bytes) is not "
            f"compatible with rec_len={rec_len} and n_prods={n_prods}."
        )

    n_recs = file_size // rec_len // n_prods
    sci_dict: dict[str, NDArray[np.float32]] = make_sharad_usscs_sim_dict(
        n_recs, n_samp
    )

    rec_size = n_recs * n_samp  # samples per product

    csim = np.fromfile(sim_path, dtype="<f4")
    expected = rec_size * n_prods
    if csim.size != expected:
        raise ValueError(
            f"Unexpected number of samples in {sim_path!s}: "
            f"got {csim.size}, expected {expected} "
            f"(n_samp={n_samp}, n_recs={n_recs}, n_prods={n_prods})."
        )

    sci_dict["LEFT"][:] = csim[0:rec_size].reshape((n_samp, n_recs))
    sci_dict["RIGHT"][:] = csim[rec_size : 2 * rec_size].reshape((n_samp, n_recs))
    sci_dict["COMBINED"][:] = csim[2 * rec_size : 3 * rec_size].reshape(
        (n_samp, n_recs)
    )

    return sci_dict

def parse_sharad_usscs_rtrn(csc_path: str | Path,) -> dict[str, NDArray[Any]]:
    """
    Parse a SHARAD US_RDR CSC return CSV file into a preallocated
    auxiliary data dictionary.

    The CSC return file is expected to contain a single header line
    followed by data lines. Each data line must contain exactly 12
    comma-separated values, corresponding to:

        1. COLUMN
        2. SPACECRAFTLON
        3. SPACECRAFTLAT
        4. SPACECRAFTHGT
        5. NADIRHGT
        6. NADIRLINE
        7. NADIRAREOIDRAD
        8. NADIRELLIPSOIDRAD
        9. FIRSTLON
        10. FIRSTLAT
        11. FIRSTHGT
        12. FIRSTLINE

    The header is skipped, and the remaining lines are parsed and stored
    into the arrays returned by :func:`make_sharad_usscs_rtrn_dict`.

    Args:
        csc_path: Path to the SHARAD US_RDR CSC return CSV file.

    Returns:
        A dictionary of NumPy arrays containing the parsed CSC return
        data. Array lengths equal the number of data (non-header) lines.

    Raises:
        ValueError: If the file has no data lines, if any data line does
            not have exactly 12 columns, or if a numeric conversion
            fails.
    """
    with open(csc_path, "r") as f:
        content = [line.strip() for line in f.readlines()]

    if not content:
        raise ValueError(f"CSC return file {csc_path!s} is empty.")

    n_recs = len(content) - 1  # subtract header line
    if n_recs <= 0:
        raise ValueError(f"CSC return file {csc_path!s} has no data lines.")

    aux_dict: dict[str, NDArray[Any]] = make_sharad_usscs_rtrn_dict(n_recs)

    # content[0] is header; start from the first data line
    for idx, line in enumerate(content[1:], start=0):
        csvs = line.split(",")

        if len(csvs) != 12:
            raise ValueError(
                f"Malformed CSC line {idx + 2} in {csc_path!s}: "
                f"expected 12 values, got {len(csvs)} -> {line!r}"
            )

        try:
            aux_dict["COLUMN"][idx] = int(float(csvs[0]))
            aux_dict["SPACECRAFTLON"][idx] = float(csvs[1])
            aux_dict["SPACECRAFTLAT"][idx] = float(csvs[2])
            aux_dict["SPACECRAFTHGT"][idx] = float(csvs[3])
            aux_dict["NADIRHGT"][idx] = float(csvs[4])
            aux_dict["NADIRLINE"][idx] = int(float(csvs[5]))
            aux_dict["NADIRAREOIDRAD"][idx] = float(csvs[6])
            aux_dict["NADIRELLIPSOIDRAD"][idx] = float(csvs[7])
            aux_dict["FIRSTLON"][idx] = float(csvs[8])
            aux_dict["FIRSTLAT"][idx] = float(csvs[9])
            aux_dict["FIRSTHGT"][idx] = float(csvs[10])
            aux_dict["FIRSTLINE"][idx] = int(float(csvs[11]))
        except ValueError as e:
            raise ValueError(
                f"Error parsing numeric value on line {idx + 2} in {csc_path!s}: "
                f"{e}. Line content: {line!r}"
            ) from e

    return aux_dict


def parse_sharad_usscs_emap(emap_path: str | Path,
                            n_lines: int = 181,
                            ) -> dict[str, NDArray[np.float32]]:
    """
    Parse the SHARAD US SCS EMAP (echo power map) binary file.

    The EMAP file contains a single 2D image of simulated backscattered
    power from martian topography, stored as 32-bit little-endian floats
    with shape ``(n_lines, n_samples)``. The number of lines is always
    181; the number of samples is inferred from the file size.

    Args:
        emap_path: Path to the SHARAD US SCS EMAP binary file.
        n_lines: Number of lines (rows) in the image.

    Returns:
        A dictionary containing a single ``float32`` array:
            - ``ECHO_POWER_MAP``: shape ``(n_lines, n_samples)``

    Raises:
        ValueError: If the file size is not evenly divisible by
            the number of lines and bytes per sample.
    """
    bytes_per_sample = 4  # float32
    file_size = os.path.getsize(emap_path)

    if file_size % (n_lines * bytes_per_sample) != 0:
        raise ValueError(
            f"File size of {emap_path!s} ({file_size} bytes) is not "
            f"evenly divisible by n_lines={n_lines} * {bytes_per_sample} bytes."
        )

    n_samples = file_size // (n_lines * bytes_per_sample)

    emap = np.fromfile(emap_path, dtype="<f4").reshape((n_lines, n_samples))

    return {"ECHO_POWER_MAP": emap}

##############################################################################################################
#
# CO-SHARPS Decoder Products (DEC_DATA; US EDRs)
#
##############################################################################################################

def parse_sharad_usedr_sci(sci_path: str | Path,
                           rec_len: int,
                           n_samp: int = 3600,
                           ) -> dict[str, NDArray[np.float32]]:
    """Parse a SHARAD US EDR science file into a preallocated science
    data dictionary.

    This function reads a SHARAD US EDR science (USEDR) file containing
    only echo samples stored as 8-bit signed integers. The total number
    of records is inferred from the file size and the provided record
    length. The raw samples are read with ``np.fromfile``, reshaped to
    ``(n_samp, n_recs)``, cast to ``float32``, and stored in the
    ``ECHO_SAMPLES`` array returned by
    :func:`make_sharad_usedr_sci_dict`.

    Args:
        sci_path: Path to the SHARAD US EDR science file.
        rec_len: Length in bytes of a single record.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing the preallocated and filled
        ``ECHO_SAMPLES`` array with shape ``(n_samp, n_recs)``.

    Raises:
        ValueError: If the total number of samples in the file does not
            match ``n_samp * n_recs``.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict: dict[str, NDArray[Any]] = make_sharad_usedr_sci_dict(n_recs, n_samp)

    raw = np.fromfile(sci_path, dtype=np.int8)

    expected = n_samp * n_recs
    if raw.size != expected:
        raise ValueError(
            f"Unexpected number of samples in {sci_path!s}: "
            f"got {raw.size}, expected {expected} (n_samp={n_samp}, n_recs={n_recs})."
        )

    sci_dict["ECHO_SAMPLES"][:] = raw.reshape(n_recs, n_samp).T.astype(
        np.float32, copy=False
    )
    return sci_dict


def parse_sharad_usedr_aux(aux_path: str | Path,
                           ) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD US EDR auxiliary (AUX) text file into a preallocated
    auxiliary data dictionary.

    This function reads a US EDR (DEC DATA) AUX file produced by the
    US decoder. Each line is expected to contain three comma-separated
    values:

        1. Nominal time along track
        2. Receive window opening time
        3. An undocumented third field (stored as ``"JUNK"``)

    The number of records is inferred from the number of lines in the file.
    All values are cast to ``float32`` and stored in the arrays returned by
    :func:`make_sharad_usedr_aux_dict`.

    Args:
        aux_path: Path to the SHARAD US EDR AUX text file.

    Returns:
        A dictionary of NumPy arrays containing the parsed SHARAD US EDR
        auxiliary data. Array lengths equal the number of records in the
        file.

    Raises:
        ValueError: If any line in the file does not contain exactly three
            comma-separated values.
    """
    with open(aux_path, "r") as f:
        content = [line.strip() for line in f.readlines()]

    n_recs = len(content)
    aux_dict: dict[str, NDArray[Any]] = make_sharad_usedr_aux_dict(n_recs)

    for idx, line in enumerate(content):
        csvs = line.split(",")

        if len(csvs) != 3:
            raise ValueError(
                f"Malformed AUX line {idx} in {aux_path!s}: "
                f"expected 3 values, got {len(csvs)} -> {line!r}"
            )

        aux_dict["TIME"][idx] = np.float32(csvs[0])
        aux_dict["RECEIVE_WINDOW_OPENING_TIME"][idx] = np.float32(csvs[1])
        aux_dict["JUNK"][idx] = np.float32(csvs[2])

    return aux_dict


def parse_sharad_usedr_orb(orb_path: str | Path,
                           ) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD US EDR orbit (ORB) text file into a preallocated
    orbit/geometry data dictionary.

    This function reads a SHARAD US EDR ORB file, skipping empty lines and
    comment lines starting with ``#``. Each remaining line is expected to
    contain exactly 24 whitespace-separated fields corresponding to the
    US EDR orbit/geometry quantities. The values are parsed and stored in
    the arrays returned by :func:`make_sharad_usedr_orb_dict`.

    The first field is treated as a timestamp and decoded using
    :func:`decode_datetime` and ``AUX_DATE_FORMAT`` into
    ``GEOMETRY_EPOCH``. All remaining fields are cast to ``float32`` and
    stored in the appropriate geometry and attitude fields.

    Args:
        orb_path: Path to the SHARAD US EDR ORB text file.

    Returns:
        A dictionary of NumPy arrays containing the parsed SHARAD US EDR
        orbit/geometry data. Array lengths equal the number of valid
        (non-comment, non-empty) lines in the file.

    Raises:
        ValueError: If any non-comment, non-empty line does not contain
            exactly 24 whitespace-separated fields.
    """
    valid_lines: list[str] = []

    with open(orb_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments or empty lines
            if not line or line.startswith("#"):
                continue
            valid_lines.append(line)

    n_recs = len(valid_lines)
    orb_dict: dict[str, NDArray[Any]] = make_sharad_usedr_orb_dict(n_recs)

    for idx, line in enumerate(valid_lines):
        parts = line.split()
        if len(parts) != 24:
            raise ValueError(
                f"Malformed ORB line {idx} in {orb_path!s}: "
                f"expected 24 fields, got {len(parts)} -> {line!r}"
            )

        orb_dict["GEOMETRY_EPOCH"][idx] = decode_datetime(
            parts[0].strip(), AUX_DATE_FORMAT
        )
        orb_dict["SUB_SC_PLANETOCENTRIC_LATITUDE"][idx] = np.float32(parts[1])
        orb_dict["SUB_SC_EAST_LONGITUDE"][idx] = np.float32(parts[2])
        orb_dict["SPACECRAFT_RADIUS"][idx] = np.float32(parts[3])
        orb_dict["MARS_SC_TANGENTIAL_VELOCITY"][idx] = np.float32(parts[4])
        orb_dict["MARS_SC_RADIAL_VELOCITY"][idx] = np.float32(parts[5])
        orb_dict["X_MARS_SC_POSITION_VECTOR"][idx] = np.float32(parts[6])
        orb_dict["Y_MARS_SC_POSITION_VECTOR"][idx] = np.float32(parts[7])
        orb_dict["Z_MARS_SC_POSITION_VECTOR"][idx] = np.float32(parts[8])
        orb_dict["X_MARS_SC_VELOCITY_VECTOR"][idx] = np.float32(parts[9])
        orb_dict["Y_MARS_SC_VELOCITY_VECTOR"][idx] = np.float32(parts[10])
        orb_dict["Z_MARS_SC_VELOCITY_VECTOR"][idx] = np.float32(parts[11])
        orb_dict["SC_ROLL_ANGLE"][idx] = np.float32(parts[12])
        orb_dict["SC_PITCH_ANGLE"][idx] = np.float32(parts[13])
        orb_dict["SC_YAW_ANGLE"][idx] = np.float32(parts[14])
        orb_dict["MRO_HGA_INNER_GIMBAL_ANGLE"][idx] = np.float32(parts[15])
        orb_dict["MRO_HGA_OUTER_GIMBAL_ANGLE"][idx] = np.float32(parts[16])
        orb_dict["MRO_SAPX_INNER_GIMBAL_ANGLE"][idx] = np.float32(parts[17])
        orb_dict["MRO_SAPX_OUTER_GIMBAL_ANGLE"][idx] = np.float32(parts[18])
        orb_dict["MRO_SAMX_INNER_GIMBAL_ANGLE"][idx] = np.float32(parts[19])
        orb_dict["MRO_SAMX_OUTER_GIMBAL_ANGLE"][idx] = np.float32(parts[20])
        orb_dict["SOLAR_ZENITH_ANGLE"][idx] = np.float32(parts[21])
        orb_dict["MAGNETIC_FIELD"][idx] = np.float32(parts[22])
        orb_dict["MARS_SUN_DISTANCE"][idx] = np.float32(parts[23])

    return orb_dict


def parse_sharad_usedr_hdr(hdr_path: str | Path,
                           ) -> dict[str, Any]:
    """Parse a SHARAD US decoder header file into a dictionary.

    The US decoder header file is divided into sections marked by comment
    lines, e.g.:

        # Begin Obs Data
        ...
        # End Obs Data

        # Begin Mode Data
        ...
        # End Mode Data

    This function parses:

    * The observation-level (``Obs Data``) key–value pairs into the
      top-level dictionary.
    * Each ``Mode Data`` block into a separate dict, which is then
      attached to the top-level dictionary as:

          MODEXX: { ... }

      where ``XX`` is the zero-padded mode number, taken from the
      ``Mode_Number`` field if present, or from the block index
      otherwise.

    Lines starting with ``#`` (comments) or that are empty are skipped.
    Inline comments after a ``#`` on a data line are also stripped. Each
    data line is expected to have at least two whitespace-separated
    tokens: a key and a value string. Values are converted using
    ``convert_value``, except for time-like keys which are parsed using
    ``decode_datetime`` and ``AUX_DATE_FORMAT``.

    Args:
        hdr_path: Path to the US decoder header file.

    Returns:
        A dictionary containing observation-level metadata and one
        sub-dictionary per mode (e.g., ``MODE01``, ``MODE02``, ...).
    """
    hdr_data: dict[str, Any] = {}
    mode_dicts: list[dict[str, Any]] = []

    current_section: str | None = None
    current_mode: dict[str, Any] | None = None

    with open(hdr_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Comment / section markers
            if line.startswith("#"):
                if "Begin Obs Data" in line:
                    current_section = "obs"
                    continue
                if "End Obs Data" in line:
                    current_section = None
                    continue
                if "Begin Mode Data" in line:
                    current_section = "mode"
                    current_mode = {}
                    continue
                if "End Mode Data" in line:
                    if current_section == "mode" and current_mode is not None:
                        mode_dicts.append(current_mode)
                        current_mode = None
                    current_section = None
                    continue
                # Other comment line – ignore
                continue

            # Ignore lines outside known sections
            if current_section not in ("obs", "mode"):
                continue

            # Strip inline comment
            no_comment = line.split("#", 1)[0].strip()
            if not no_comment:
                continue

            parts = no_comment.split()
            if len(parts) < 2:
                continue

            key = parts[0]
            value_str = parts[1]

            # Time-like keys: use datetime parser, everything else via convert_value
            if key in ("OST_Start_UTC", "Mode_Start_UTC"):
                value = decode_datetime(value_str, AUX_DATE_FORMAT)
            else:
                value = convert_value(value_str)

            if current_section == "obs":
                hdr_data[key] = value
            elif current_section == "mode" and current_mode is not None:
                current_mode[key] = value

    # Attach mode dictionaries to the top-level header dict
    for idx, mode in enumerate(mode_dicts, start=1):
        num = mode.get("Mode_Number", idx)
        try:
            num_int = int(num)
        except (TypeError, ValueError):
            num_int = idx
        mode_key = f"MODE{num_int:02d}"
        hdr_data[mode_key] = mode

    return hdr_data

##############################################################################################################
#
# CO-SHARPS UPB Files
#
##############################################################################################################

def parse_sharad_upb_sci(sci_path: str | Path,
                         rec_len: int,
                         n_samp: int = 3600,
                         ) -> dict[str, NDArray[np.float32]]:
    """Parse a CO-SHARPS UPB (Unfocused Power) science data file.

    Thin wrapper around :func:`parse_sharad_usrdr_sci` for reading
    CO-SHARPS Unfocused Processor Bruce (UPB) ``.raw`` files, which share
    the same binary layout as US_RDR science files.

    Args:
        sci_path: Path to the UPB science file.
        rec_len: Record length in bytes.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing the filled ``IMAGE`` array with shape
        ``(n_samp, n_recs)`` and ``float32`` dtype.

    Raises:
        ValueError: If the total number of samples in the file does not
            match ``n_samp * n_recs``.
    """
    return parse_sharad_usrdr_sci(sci_path, rec_len, n_samp)

##############################################################################################################
#
# CO-SHARPS FPB Files
#
##############################################################################################################
def parse_sharad_fpb_sci(sci_path: str | Path,
                         rec_len: int,
                         n_samp: int = 3600,
                         ) -> dict[str, NDArray[np.float32]]:
    """Parse a CO-SHARPS FPB (Focused Power) science data file.

    Thin wrapper around :func:`parse_sharad_usrdr_sci` for reading
    CO-SHARPS Focused Power Bruce (FPB) ``.img`` files, which share
    the same binary layout as US_RDR science files.

    Args:
        sci_path: Path to the FPB science file.
        rec_len: Record length in bytes.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing the filled ``IMAGE`` array with shape
        ``(n_samp, n_recs)`` and ``float32`` dtype.

    Raises:
        ValueError: If the total number of samples in the file does not
            match ``n_samp * n_recs``.
    """
    return parse_sharad_usrdr_sci(sci_path, rec_len, n_samp)

def parse_sharad_fpb_sim(sim_path: str | Path,
                           rec_len: int,
                           n_samp: int = 3600,
                           ) -> dict[str, NDArray[np.float32]]:
    """Parse the SHARAD FPB SIM (simulation) binary file into a
    preallocated science data dictionary.

    The CSIM file contains one product stored as
    32-bit little-endian floats:

        1. COMBINED

    The product consists of ``n_recs`` records with ``n_samp`` samples
    per record. The number of records is inferred from the file size,
    the provided record length.

    Data are read with ``np.fromfile`` using little-endian ``float32``
    and reshaped into ``(n_samp, n_recs)`` arrays for each product.

    Args:
        sim_path: Path to the SHARAD FPB SIM binary file.
        rec_len: Record length in bytes.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing one ``float32`` arrays with shape
        ``(n_samp, n_recs)``:
            - ``COMBINED``

    Raises:
        ValueError: If the file size is not consistent with the expected
            number of records, products, and samples.
    """
    n_prods = 1  # COMBINED

    file_size = os.path.getsize(sim_path)
    if file_size % (rec_len * n_prods) != 0:
        raise ValueError(
            f"File size of {sim_path!s} ({file_size} bytes) is not "
            f"compatible with rec_len={rec_len} and n_prods={n_prods}."
        )

    n_recs = file_size // rec_len // n_prods
    sci_dict: dict[str, NDArray[np.float32]] = make_sharad_fpb_sim_dict(n_recs, n_samp)

    rec_size = n_recs * n_samp  # samples per product

    csim = np.fromfile(sim_path, dtype="<f4")
    expected = rec_size * n_prods
    if csim.size != expected:
        raise ValueError(
            f"Unexpected number of samples in {sim_path!s}: "
            f"got {csim.size}, expected {expected} "
            f"(n_samp={n_samp}, n_recs={n_recs}, n_prods={n_prods})."
        )

    sci_dict["COMBINED"][:] = csim[0:rec_size].reshape((n_samp, n_recs))
    return sci_dict


def parse_sharad_fpb_rtrn(csc_path: str | Path,) -> dict[str, NDArray[Any]]:
    """Parse a SHARAD FPB SIM return CSV file into a preallocated
    auxiliary data dictionary.

    The FPB NADIR or FIRSTRETURN file is expected to contain a single header line
    followed by data lines. Each data line must contain exactly 4
    comma-separated values, corresponding to:

        1. LONGITUDE
        2. LATITUDE
        3. ELEV_IAU2000
        4. SAMPLE

    The header is skipped, and the remaining lines are parsed and stored
    into the arrays returned by :func:`make_sharad_fpb_rtrn_dict`.

    Args:
        csc_path: Path to the SHARAD FPB SIM return CSV file.

    Returns:
        A dictionary of NumPy arrays containing the parsed CSC return
        data. Array lengths equal the number of data (non-header) lines.

    Raises:
        ValueError: If the file has no data lines, if any data line does
            not have exactly 4 columns, or if a numeric conversion
            fails.
    """
    with open(csc_path, "r") as f:
        content = [line.strip() for line in f.readlines()]

    if not content:
        raise ValueError(f"CSC return file {csc_path!s} is empty.")

    n_recs = len(content) - 1  # subtract header line
    if n_recs <= 0:
        raise ValueError(f"CSC return file {csc_path!s} has no data lines.")

    aux_dict: dict[str, NDArray[Any]] = make_sharad_fpb_rtrn_dict(n_recs)

    # content[0] is header; start from the first data line
    for idx, line in enumerate(content[1:], start=0):
        csvs = line.split(",")

        if len(csvs) != 4:
            raise ValueError(
                f"Malformed CSC line {idx + 2} in {csc_path!s}: "
                f"expected 4 values, got {len(csvs)} -> {line!r}"
            )

        try:
            aux_dict["LONGITUDE"][idx] = float(csvs[0])
            aux_dict["LATITUDE"][idx] = float(csvs[1])
            aux_dict["ELEV_IAU2000"][idx] = float(csvs[2])
            aux_dict["SAMPLE"][idx] = int(float(csvs[3]))
        except ValueError as e:
            raise ValueError(
                f"Error parsing numeric value on line {idx + 2} in {csc_path!s}: "
                f"{e}. Line content: {line!r}"
            ) from e

    return aux_dict
##############################################################################################################
#
# CO-SHARPS QDA Files
#
##############################################################################################################
def parse_sharad_qda_sci(sci_path: str | Path,
                         rec_len: int,
                         n_samp: int = 4096,
                         ) -> dict[str, NDArray[np.float32]]:
    """Parse the SHARAD QDA science (SCI) data file into a preallocated
    science data dictionary.

    This function reads a SHARAD QDA science file containing
    range-compressed echo data stored as 32-bit floating-point values.
    The number of records is inferred from the file size and the provided
    record length. The data are read with ``np.fromfile``, reshaped to
    ``(n_samp, n_rec)``, and stored in the ``IMAGE`` array returned by
    :func:`make_sharad_usrdr_sci_dict`.

    Args:
        sci_path: Path to the SHARAD QDA science file.
        rec_len: Record length in bytes.
        n_samp: Number of samples per record.

    Returns:
        A dictionary containing the preallocated and filled ``IMAGE``
        array with shape ``(n_samp, n_recs)`` and ``float32`` dtype.

    Raises:
        ValueError: If the total number of samples in the file does not
            match ``n_samp * n_recs``.
    """
    n_recs = calc_nrecs(sci_path, rec_len)

    sci_dict: dict[str, NDArray[np.float32]] = make_sharad_usrdr_sci_dict(n_recs, n_samp)

    raw = np.fromfile(sci_path, dtype=np.float32, count=-1)
    expected = n_samp * n_recs
    if raw.size != expected:
        raise ValueError(
            f"Unexpected number of samples in {sci_path!s}: "
            f"got {raw.size}, expected {expected} "
            f"(n_samp={n_samp}, n_recs={n_recs})."
        )

    sci_dict["IMAGE"][:] = raw.reshape((n_recs, n_samp)).T
    return sci_dict

####################################################################################################################
#
# Helper Functions
#
####################################################################################################################
def _load_sharad_4bit(buff: bytes | bytearray | memoryview,
                     n_samp: int = 3600,
                     ) -> NDArray[np.int8]:
    """
    Decode 4-bit packed SHARAD science samples into 8-bit signed integers.

    This function unpacks 4-bit nibbles from an 8-bit byte buffer, producing
    one output sample per nibble. The high nibble of each byte is placed in
    the even indices of the output array, and the low nibble is placed in the
    odd indices.

    Args:
        buff: Input byte buffer containing 4-bit packed samples.
        n_samp: Number of decoded samples expected in the output array.

    Returns:
        A 1D NumPy array of decoded 4-bit samples as ``np.int8`` with
        shape ``(n_samp,)``.

    Raises:
        ValueError: If the input buffer does not contain enough data to
            generate ``n_samp`` samples.
    """
    byte_array: NDArray[np.uint8] = np.frombuffer(buff, dtype=np.uint8)

    if byte_array.size * 2 < n_samp:
        raise ValueError(
            f"Input buffer too small: {byte_array.size} bytes can only "
            f"produce {byte_array.size * 2} samples, but {n_samp} were requested."
        )

    decoded_data: NDArray[np.int8] = np.zeros(n_samp, dtype=np.int8)

    decoded_data[0::2] = byte_array[: (n_samp + 1) // 2] >> 4
    decoded_data[1::2] = byte_array[: n_samp // 2] & 0x0F

    return decoded_data


def _load_sharad_6bit(buff: bytes | bytearray | memoryview,
                     n_samp: int = 3600,
                     ) -> NDArray[np.int8]:
    """
    Decode 6-bit packed SHARAD science samples into 8-bit signed integers.

    SHARAD 6-bit data are packed as 4 samples in 3 bytes (24 bits):

        Byte0: [ b7 b6 b5 b4 b3 b2 b1 b0 ]
        Byte1: [ c7 c6 c5 c4 c3 c2 c1 c0 ]
        Byte2: [ d7 d6 d5 d4 d3 d2 d1 d0 ]

    yielding four 6-bit values:

        s0 = Byte0[7:2]                     (bits 7–2)
        s1 = (Byte0[1:0] << 4) | Byte1[7:4] (bits 1–0 + 7–4)
        s2 = (Byte1[3:0] << 2) | Byte2[7:6] (bits 3–0 + 7–6)
        s3 = Byte2[5:0]                     (bits 5–0)

    This function unpacks those nibbles for as many samples as requested.

    Args:
        buff: Input byte buffer containing 6-bit packed samples.
        n_samp: Number of decoded samples expected in the output array.

    Returns:
        A 1D NumPy array of decoded 6-bit samples as ``np.int8`` with
        shape ``(n_samp,)``.

    Raises:
        ValueError: If the input buffer does not contain enough bytes to
            generate ``nSamp`` samples.
    """
    byte_array: NDArray[np.uint8] = np.frombuffer(buff, dtype=np.uint8)

    # 3 bytes → 4 samples; we need enough groups to cover nSamp
    n_groups = (n_samp + 3) // 4  # ceil(nSamp / 4)
    needed_bytes = 3 * n_groups

    if byte_array.size < needed_bytes:
        raise ValueError(
            f"Input buffer too small: {byte_array.size} bytes can only "
            f"produce {(byte_array.size // 3) * 4} samples, "
            f"but {n_samp} were requested."
        )

    # Use only the bytes we need
    b0 = byte_array[0:needed_bytes:3]
    b1 = byte_array[1:needed_bytes:3]
    b2 = byte_array[2:needed_bytes:3]

    # Decode into a temporary array of 4 * n_groups samples
    decoded_full: NDArray[Any] = np.zeros(4 * n_groups, dtype=np.int8)

    decoded_full[0::4] = (b0 >> 2).astype(np.int8)
    decoded_full[1::4] = (((b0 & 0x03) << 4) | (b1 >> 4)).astype(np.int8)
    decoded_full[2::4] = (((b1 & 0x0F) << 2) | (b2 >> 6)).astype(np.int8)
    decoded_full[3::4] = (b2 & 0x3F).astype(np.int8)

    # Return exactly nSamp samples
    return decoded_full[:n_samp]