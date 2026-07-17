# SPDX-License-Identifier: BSD-3-Clause
import importlib
import os
from pathlib import Path
from typing import Any

from ..common.utils import check_file
from .utils import identify_file, grab_record_length

##############################################################################################################
#
# MAIN READ FUNCTION
#
##############################################################################################################
def read(iFile: str | os.PathLike[str]) -> Any:
    """Read any GRaSP-supported radar sounder data file.

    This is a high-level dispatcher that:
      1) Routes ``.grsp`` files to the GRaSP-output triplet reader.
      2) Otherwise identifies the instrument/product/file role from
         the filename.
      3) Loads record format (record lengths) for non-label files.
      4) Dispatches to the appropriate instrument-specific parser.

    Args:
        iFile: Path to the input file.

    Returns:
        Parsed data object. The return type depends on the input:

        - ``.grsp`` → :class:`grasp.grasp_types.GraspOutput`
        - PDS files → dict of arrays (science/aux) or parsed label (PVL)

    Raises:
        FileNotFoundError: If the file does not exist or is not a regular file.
        PermissionError: If the file is not readable.
        ValueError: If the file cannot be identified or is not supported.
    """
    check_file(iFile)

    path = Path(os.fspath(iFile))

    # Route GRaSP-format output triplets (.grsp + .img + .csv) to the
    # dedicated reader before falling through to PDS-style sniffing.
    if path.suffix.lower() == ".grsp":
        from .load import read_grasp_output
        return read_grasp_output(path)

    fid = identify_file(path.name)

    # ----------------------------
    # SHARAD
    # ----------------------------
    if fid.instrument == "SHARAD":
        parse = getattr(importlib.import_module("grasp.sharad.parsers"), "parse")

        rec_len: int | None = None

        if fid.file_role != "LABEL":
            if fid.product_type == "EDR":
                fmt = grab_record_length("SHARAD", fid.product_type, fid.operative_mode)
            else:
                fmt = grab_record_length("SHARAD", fid.product_type)

            if fid.file_role in ("SCIENCE", "SIM"):
                rec_len = fmt.reclen
            elif fid.file_role in ("AUXILIARY", "GEOMETRY", "RTRN"):
                rec_len = fmt.auxlen

        return parse(str(path), fid.product_type, fid.file_role, rec_len)

    # ----------------------------
    # MARSIS
    # ----------------------------
    elif fid.instrument == "MARSIS":
        parse = getattr(importlib.import_module("grasp.marsis.parsers"), "parse")

        rec_len: int | None = None
        if fid.file_role in ("SCIENCE", "AUXILIARY"):
            if fid.operative_mode in ("AIS", "CAL", "RXO"):
                fmt = grab_record_length("MARSIS", fid.product_type, fid.operative_mode)
            else:
                fmt = grab_record_length("MARSIS", fid.product_type, fid.operative_mode,
                                         fid.instrument_state, fid.data_form)
            rec_len = fmt.reclen if fid.file_role == "SCIENCE" else fmt.auxlen
        elif fid.file_role != "LABEL":
            raise ValueError(f"Unknown role for {fid.instrument}: {fid.file_role}")

        return parse(str(path), fid.product_type, fid.file_role,
                      fid.operative_mode, fid.instrument_state, fid.data_form, rec_len)

    # ----------------------------
    # LRS
    # ----------------------------
    elif fid.instrument == "LRS":
        parse = getattr(importlib.import_module("grasp.lrs.parsers"), "parse")

        rec_len: int | None = None
        if fid.file_role == "SCIENCE":
            if fid.product_type == "EDR":
                fmt = grab_record_length("LRS", fid.product_type)
            elif fid.product_type == "RDR":
                fmt = grab_record_length("LRS", fid.product_type, fid.operative_mode)
            else:
                raise ValueError(f"Unknown role for {fid.instrument}: {fid.product_type}")
            rec_len = fmt.reclen
        elif fid.file_role != "LABEL":
            raise ValueError(f"Unknown role for {fid.instrument}: {fid.file_role}")

        return parse(str(path), fid.product_type, fid.file_role,
                      fid.operative_mode, rec_len)

    else:
        raise ValueError(f"Invalid instrument: {fid.instrument}")