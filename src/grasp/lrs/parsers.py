# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import struct
from typing import Any

from ..common.utils import (calc_nrecs, check_frame_length, decode_datetime,  parse_pds_lbl)
from .dictionaries import (make_lrs_wf_dict, make_lrs_sar_c_dict, make_lrs_sar_p_dict)

##############################################################################################################
#
# Wishlist
#
##############################################################################################################
# SS-HIGH and SS-LOW appear to be waveform data taken through preprocessing and possibly
# joined with other adjacent observations. Could have been used as input data to their SAR
# processor.
# TODO (low-priority; post-beta): sln-l-lrs-5-sndr-ss-high-v2.0 Support
# TODO (low-priority; post-beta): sln-l-lrs-5-sndr-ss-low-v1.0 Support
##############################################################################################################
#
# Byte Structure and Date Formats
#
##############################################################################################################
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # DATE FORMAT in LRS Science Data
EDR_SCI_BYTE_FORMAT = '>23cfH4f14c2048H'  # Byte Format for LRS Science Data
SAR_C_BYTE_FORMAT = ">23cfH4f10c1000f1000f"  # Byte Format for LRS SAR Data
SAR_P_HDR_FORMAT = "<23cfH4f10c"  # Byte Format for LRS SAR Data
##############################################################################################################
#
# Notes on Products
#
##############################################################################################################
#
# LRS waveform "EDRs" -- Rawest form of LRS data available from the DARTS
#   - High-resolution (sln-l-lrs-2-sndr-waveform-high-v1.0)
#   - Low-resolution  (sln-l-lrs-2-sndr-waveform-low-v1.0)
#
# LRS SAR "RDRs" -- SAR Processed form of LRS data available from the DARTS
#   - 5 km, complex (sln-l-lrs-5-sndr-ss-sar05-complex-v1.0)
#   - 5 km, power (sln-l-lrs-5-sndr-ss-sar05-power-v1.0)
#   - 10 km, complex (sln-l-lrs-5-sndr-ss-sar10-complex-v1.0)
#   - 10 km, power (sln-l-lrs-5-sndr-ss-sar10-power-v1.0)
#   - 40 km, power (sln-l-lrs-5-sndr-ss-sar40-power-v1.0)
##############################################################################################################
#
# Wrappers
#
##############################################################################################################
def parse_lrs_lbl(lblFile: str,) -> dict[str, Any]:
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


def parse(path: str | Path,
          prod_type: str,
          role: str,
          mode: str | None = None,
          rec_len: int | None = None,
          ) -> dict[str, Any]:
    """Read an LRS DARTS File into a Python dictionary

    Wrapper that will parse an LRS data file retrieved either from the DARTS.
    The following prodTypes and roles are currently supported

    DARTS
        - WAVEFORMS (EDRs)
            - SCIENCE, LABEL
        - SAR
            - SCIENCE, LABEL

    Args:
        path:  Path to the input PDS file
        prod_type: The product type (EDR, RDR)
        role: The file role (SCIENCE, LABEL)
        mode: The mode ("SW_WF", "SA_WF", 'SAR05KM', 'SAR05KM-C', 'SAR10KM', 'SAR10KM-C', 'SAR40KM')
        rec_len: The length of each data record (can be None for some products).

    Returns:
        Dictionary containing the parse data

    Raises:
        ValueError: If the prod_type, role, or mode is not supported.
    """
    if role == "LABEL":
        return parse_lrs_lbl(path)
    elif prod_type == "EDR":
        if role == "SCIENCE":
            if mode in ("SW_WF", "SA_WF"):
                return parse_lrs_wf_sci(path, rec_len)
            else:
                raise ValueError(f"Unknown mode for {prod_type} {role}: {mode}")
        else:
            raise ValueError(f"Unknown role for {prod_type}: {role}")
    elif prod_type == "RDR":
        if role == "SCIENCE":
            if mode in ('SAR05KM-C', 'SAR10KM-C'):
                return parse_lrs_sar_c(path, rec_len)
            elif mode in ('SAR05KM', 'SAR10KM', 'SAR40KM'):
                return parse_lrs_sar_p(path, rec_len)
            else:
                raise ValueError(f"Unknown mode for {prod_type} {role}: {mode}")
        else:
            raise ValueError(f"Unknown role for {prod_type}: {role}")
    else:
        raise ValueError(f"Unknown prodType for LRS: {prod_type}")

##############################################################################################################
#
# Parsers
#
##############################################################################################################
#
# LRS Sci Parser
#
def parse_lrs_wf_sci(path: str | Path,
                     rec_len: int,
                     ) -> dict[str, NDArray[Any]]:
    """Parses LRS EDR (WF) science records into a preallocated dictionary.

    Reads a fixed-length binary file containing LRS EDR waveform (WF) science
    records and returns a dictionary of NumPy arrays keyed by field name. Each
    record is unpacked using `EDR_SCI_BYTE_FORMAT`, decoded, and stored into
    column `frame` of the output arrays.

    Args:
        path: Path to the input LRS EDR (WF) science file.
        rec_len: Record length in bytes (i.e., the number of bytes read per
            record).

    Returns:
        A dictionary mapping field names to NumPy arrays. The number of columns
        equals the number of records in `path`, and waveform arrays are shaped
        `(n_samp, n_recs)`.

    Raises:
        ValueError: If `rec_len <= 0`, if the file size is inconsistent with
            `rec_len`, or if a record cannot be unpacked/decoded.

    Notes:
        - `OBSERVATION_TIME` is decoded using `decode_datetime` with
          `DATE_FORMAT`.
        - `TI` is decoded as UTF-8 from the unpacked byte fields `item[29:43]`
          and concatenated.
        - The waveform samples are stored in `WAVEFORM` as
          `(n_samp, n_recs)` (samples × records).
    """
    if rec_len <= 0:
        raise ValueError(f"rec_len must be positive; got {rec_len}")


    n_recs = calc_nrecs(path, rec_len)
    sci_dict = make_lrs_wf_dict(n_recs)

    record_unpacker = struct.Struct(EDR_SCI_BYTE_FORMAT).unpack

    with open(path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            check_frame_length(frame, column, rec_len)
            try:
                item = record_unpacker(column)

                sci_dict['OBSERVATION_TIME'][frame] = decode_datetime(item[0:23], DATE_FORMAT)
                sci_dict['DELAY'][frame] = item[23]
                sci_dict['START_STEP'][frame] = item[24]
                sci_dict['SUB_SPACECRAFT_LATITUDE'][frame] = item[25]
                sci_dict['SUB_SPACECRAFT_LONGITUDE'][frame] = item[26]
                sci_dict['SPACECRAFT_ALTITUDE'][frame] = item[27]
                sci_dict['RANGE0_ALTITUDE'][frame] = item[28]
                sci_dict['TI'][frame] = b"".join(item[29:43]).decode('utf8', errors="strict")
                sci_dict['WAVEFORM'][:, frame] = item[43:]
            except (struct.error, UnicodeDecodeError, ValueError) as e:
                raise ValueError(f"Error parsing frame {frame} of {path!s}") from e
    return sci_dict



def parse_lrs_sar_c(path: str | Path,
                    rec_len: int,
                    ) -> dict[str, NDArray[Any]]:
    """Parses LRS RDR (SAR-C) science records into a preallocated dictionary.

    Reads a fixed-length binary file containing LRS RDR SAR-C science records
    and returns a dictionary of NumPy arrays keyed by field name. Each record
    is unpacked using `SAR_C_BYTE_FORMAT`, decoded, and stored into column
    `frame` of the output arrays.

    Args:
        path: Path to the input LRS RDR (SAR-C) science file.
        rec_len: Record length in bytes (i.e., the number of bytes read per
            record).

    Returns:
        A dictionary mapping field names to NumPy arrays. The number of columns
        equals the number of records in `path`. SAR image arrays are shaped
        `(n_samp, n_recs)` (samples × records).

    Raises:
        ValueError: If `rec_len <= 0`, if a record has an unexpected length,
            or if a record cannot be unpacked/decoded.

    Notes:
        - `OBSERVATION_TIME` is decoded using `decode_datetime` with
          `DATE_FORMAT`.
        - `TI` is decoded as UTF-8 from the unpacked byte fields `item[29:39]`
          and concatenated.
        - SAR image samples are stored in `SAR_IMAGE_REAL` and `SAR_IMAGE_IMAG`
          as `(n_samp, n_recs)` (samples × records).
    """
    if rec_len <= 0:
        raise ValueError(f"rec_len must be positive; got {rec_len}")

    n_recs = calc_nrecs(path, rec_len)

    sci_dict = make_lrs_sar_c_dict(n_recs)
    record_unpacker = struct.Struct(SAR_C_BYTE_FORMAT).unpack
    with open(path, 'rb') as f:
        for frame in range(n_recs):
            column = f.read(rec_len)
            check_frame_length(frame, column, rec_len)
            try:
                item = record_unpacker(column)
                sci_dict['OBSERVATION_TIME'][frame] = decode_datetime(item[0:23], DATE_FORMAT)
                sci_dict['DELAY'][frame] = item[23]
                sci_dict['START_STEP'][frame] = item[24]
                sci_dict['SUB_SPACECRAFT_LATITUDE'][frame] = item[25]
                sci_dict['SUB_SPACECRAFT_LONGITUDE'][frame] = item[26]
                sci_dict['SPACECRAFT_ALTITUDE'][frame] = item[27]
                sci_dict['RANGE0_ALTITUDE'][frame] = item[28]
                sci_dict['TI'][frame] = b"".join(item[29:39]).decode('utf8', errors="strict")
                sci_dict['SAR_IMAGE_REAL'][:, frame] = item[39:1039]
                sci_dict['SAR_IMAGE_IMAG'][:, frame] = item[1039:2039]
            except (struct.error, UnicodeDecodeError, ValueError) as e:
                raise ValueError(f"Error parsing frame {frame} of {path!s}") from e
    return sci_dict


def parse_lrs_sar_p(
    path: str | Path,
    rec_len: int,
    hdr_len: int = 55,
    n_samp: int = 1000,
) -> dict[str, NDArray[Any]]:
    """Parses LRS RDR (SAR-P) science records into a preallocated dictionary.

    Reads a fixed-length binary file containing LRS RDR SAR-P science records.
    The file is interpreted as a contiguous header block (all records' header
    fields first), followed by a contiguous SAR image block.

    Args:
        path: Path to the input LRS RDR (SAR-P) science file.
        rec_len: Record length in bytes (used to compute the number of records).
        hdr_len: Header length in bytes for a single record. Defaults to 55.
        n_samp: Number of image samples per record. Defaults to 1000.

    Returns:
        A dictionary mapping field names to NumPy arrays. The number of columns
        equals the number of records in `path`. `SAR_IMAGE` is shaped
        `(n_samp, n_recs)` (samples × records).

    Raises:
        ValueError: If any length parameter is non-positive, if the file size is
            inconsistent with `rec_len`, if the header block is incomplete, if
            the SAR image block does not match `n_samp * n_recs`, or if any
            header record cannot be unpacked/decoded.

    Notes:
        - `OBSERVATION_TIME` is decoded using `decode_datetime` with
          `DATE_FORMAT`.
        - `TI` is decoded as UTF-8 from the unpacked byte fields `item[29:39]`
          and concatenated.
        - `SAR_IMAGE` is interpreted as `uint8` and reshaped to
          `(n_samp, n_recs)`.
    """
    if rec_len <= 0:
        raise ValueError(f"rec_len must be positive; got {rec_len}")
    if hdr_len <= 0:
        raise ValueError(f"hdr_len must be positive; got {hdr_len}")
    if n_samp <= 0:
        raise ValueError(f"n_samp must be positive; got {n_samp}")

    n_recs = calc_nrecs(path, rec_len)
    hdr_bytes = hdr_len * n_recs

    # SAR-P dict (not SAR-C)
    sci_dict = make_lrs_sar_p_dict(n_recs, n_samp=n_samp)

    hdr_unpacker = struct.Struct(SAR_P_HDR_FORMAT).unpack

    with open(path, "rb") as f:
        hdr = f.read(hdr_bytes)
        if len(hdr) != hdr_bytes:
            raise ValueError(
                f"Header block truncated in {path!s}: expected {hdr_bytes} bytes, "
                f"got {len(hdr)}"
            )
        sci = f.read()

    # Parse per-record header values
    for frame in range(n_recs):
        hdr_st = frame * hdr_len
        hdr_en = hdr_st + hdr_len
        hdr_column = hdr[hdr_st:hdr_en]

        try:
            item = hdr_unpacker(hdr_column)

            sci_dict["OBSERVATION_TIME"][frame] = decode_datetime(item[0:23], DATE_FORMAT)
            sci_dict["DELAY"][frame] = item[23]
            sci_dict["START_STEP"][frame] = item[24]
            sci_dict["SUB_SPACECRAFT_LATITUDE"][frame] = item[25]
            sci_dict["SUB_SPACECRAFT_LONGITUDE"][frame] = item[26]
            sci_dict["SPACECRAFT_ALTITUDE"][frame] = item[27]
            sci_dict["DISTANCE_TO_RANGE0"][frame] = item[28]
            sci_dict["TI"][frame] = b"".join(item[29:39]).decode("utf-8", errors="strict")

        except (struct.error, UnicodeDecodeError, ValueError) as e:
            raise ValueError(f"Error parsing header frame {frame} of {path!s}") from e

    # Parse SAR image block
    expected_sci_bytes = n_samp * n_recs  # uint8 => 1 byte/sample
    if len(sci) != expected_sci_bytes:
        raise ValueError(
            f"SAR image block size mismatch in {path!s}: expected {expected_sci_bytes} "
            f"bytes for uint8 image ({n_samp}×{n_recs}), got {len(sci)}"
        )

    sci_dict["SAR_IMAGE"][:, :] = np.frombuffer(sci, dtype=np.uint8).reshape(
        (n_samp, n_recs)
    )

    return sci_dict