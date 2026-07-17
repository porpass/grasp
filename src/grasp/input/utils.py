# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib
import os
from pathlib import Path
import re
from typing import Any

from .input_support import product_type_mapping
from .file_patterns import patterns
from ..grasp_types import (RecordFormat, RadarParams, FileIdentity, MetaDataInfo)

##############################################################################################################
#
# Record Length
#
##############################################################################################################
def grab_record_length(instrument: str, *key: str) -> RecordFormat:
    """Look up the binary record format for an instrument and mode.

    Searches for a match using progressively shorter key tuples,
    allowing fallback from specific to general. For example, given
    keys ``("EDR", "SS3", "TRK", "CMP")``, it tries the full tuple
    first, then ``("EDR", "SS3", "TRK")``, and so on.

    Args:
        instrument: Instrument name (e.g. ``"SHARAD"``).
        *key: Hierarchical lookup keys (product type, mode,
            instrument state, data form).

    Returns:
        Record format for the matched key.

    Raises:
        ValueError: If no matching record format is found.
    """
    inst = instrument.upper()
    if inst == "LRS":
        mod = importlib.import_module("grasp.lrs.modes")
    elif inst == "MARSIS":
        mod = importlib.import_module("grasp.marsis.modes")
    elif inst == "SHARAD":
        mod = importlib.import_module("grasp.sharad.modes")
    else:
        raise ValueError(f"Invalid instrument '{inst}'")

    record_formats = getattr(mod, "RECORD_FORMAT")

    # Try progressively shorter key tuples
    # e.g. ('EDR', 'SS3', 'TRK', 'CMP') -> ('EDR', 'SS3', 'TRK') -> ('EDR', 'SS3') -> ('EDR',)
    for length in range(len(key), 0, -1):
        candidate = key[:length]
        if candidate in record_formats:
            return record_formats[candidate]

    raise ValueError(f"No record format for {inst} {key}")


##############################################################################################################
#
# Mode-specific information
#
##############################################################################################################
def grab_mode_info(instrument: str, operative_mode: str) -> Any:
    """Retrieve system mode information for a given instrument and mode.

    Args:
        instrument: Instrument name (e.g., ``"SHARAD"``, ``"MARSIS"``, ``"LRS"``).
        operative_mode: Operative mode string (e.g., ``"SS01"``, ``"SS3"``,
            ``"SW_WF"``).

    Returns:
        Instrument-specific mode dataclass (SHARADMode, MARSISMode, or
        LRSMode) for the given operative mode.

    Raises:
        ValueError: If instrument is unknown or operative mode is not found.
    """
    if instrument == "LRS":
        mod = importlib.import_module("grasp.lrs.modes")
    elif instrument == "MARSIS":
        mod = importlib.import_module("grasp.marsis.modes")
    elif instrument == "SHARAD":
        mod = importlib.import_module("grasp.sharad.modes")
    else:
        raise ValueError(f"Invalid instrument '{instrument}'")

    subsystem_modes = getattr(mod, "SUBSYSTEM_MODES")

    if operative_mode not in subsystem_modes:
        raise ValueError(
            f"No subsystem mode info for {instrument} '{operative_mode}'"
        )

    return subsystem_modes[operative_mode]


##############################################################################################################
#
# Radar Parameters
#
##############################################################################################################
def grab_radar_params(instrument: str | FileIdentity | MetaDataInfo,
                      mode: str | None = None,
                      product_type: str | None = None,
                      data_form: str | None = None) -> RadarParams:
    """Retrieve radar parameters for a given instrument.

    Loads the base instrument parameters, then applies optional
    overrides in order:

        1. Mode-specific overrides (e.g., presum for SHARAD, prf for LRS)
        2. Product-type overrides (e.g., RDR parameters for SHARAD)
        3. Data-form overrides (e.g., RAW parameters for MARSIS)

    Later overrides take precedence over earlier ones.

    Args:
        instrument: Either an instrument name (e.g., ``"SHARAD"``,
            ``"MARSIS"``, ``"LRS"``) — in which case ``mode``,
            ``product_type``, and ``data_form`` are read from the
            remaining positional arguments — or a ``FileIdentity`` /
            ``MetaDataInfo`` object whose ``instrument``,
            ``operative_mode``, ``product_type``, and ``data_form``
            fields supply all four values. The remaining positional
            arguments are ignored in the latter form.
        mode: Operative mode string (e.g., ``"SS19"``, ``"SS3"``,
            ``"SW_WF"``). If provided, mode-specific parameters are
            merged into the base parameters.
        product_type: PDS product type (e.g., ``"EDR"``, ``"RDR"``).
            If provided, product-type-specific overrides are applied.
        data_form: Data form code (e.g., ``"CMP"``, ``"RAW"``).
            If provided, data-form-specific overrides are applied.

    Returns:
        Radar parameters with base values and any applicable overrides.

    Raises:
        ValueError: If instrument is not recognized.
    """
    if isinstance(instrument, (FileIdentity, MetaDataInfo)):
        info = instrument
        instrument = info.instrument
        mode = info.operative_mode
        product_type = info.product_type
        data_form = info.data_form

    _INSTRUMENT_MODULES = {
        "SHARAD": ("grasp.sharad.parameters", "grasp.sharad.modes"),
        "MARSIS": ("grasp.marsis.parameters", "grasp.marsis.modes"),
        "LRS":    ("grasp.lrs.parameters",    "grasp.lrs.modes"),
    }

    inst = instrument.upper()
    if inst not in _INSTRUMENT_MODULES:
        raise ValueError(f"Unknown instrument '{instrument}'")

    param_path, modes_path = _INSTRUMENT_MODULES[inst]

    # Load base parameters
    base = deepcopy(getattr(importlib.import_module(param_path), "RADAR_PARAMS"))

    # 1. Mode-specific overrides
    if mode:
        modes_mod = importlib.import_module(modes_path)
        mi = modes_mod.SUBSYSTEM_MODES.get(mode.upper())
        if mi:
            base = replace(base, **vars(mi))

    # 2. Product-type overrides
    if product_type and product_type.upper() == "RDR" and inst == "SHARAD":
        overrides = getattr(importlib.import_module(param_path),
                            "RDR_OVERRIDES", {})
        if overrides:
            base = replace(base, **overrides)

    # 3. Data-form overrides
    if data_form and data_form.upper() == "RAW" and inst == "MARSIS":
        overrides = getattr(importlib.import_module(param_path),
                            "RAW_OVERRIDES", {})
        if overrides:
            base = replace(base, **overrides)

    return base

##############################################################################################################
#
# Identify the File
#
##############################################################################################################
def identify_file(filename: str | os.PathLike[str]) -> FileIdentity:
    """Identify a radar sounder file (SHARAD, MARSIS, or LRS) from its filename.

    Args:
        filename: File path or basename.

    Returns:
        FileIdentity with parsed metadata.

    Raises:
        ValueError: If the filename does not match any known pattern or
            contains unsupported fields.
    """
    name = os.path.basename(os.fspath(filename))

    for instrument_key, pattern in patterns.items():
        m = pattern.match(name)
        if not m:
            continue

        groups = m.groups()
        ext = groups[-1].upper()
        file_role = "LABEL" if ext in ("LBL", "XML") else "Unknown"

        if instrument_key.startswith("SHARAD"):
            return _identify_sharad(name, instrument_key, groups, ext, file_role)
        elif instrument_key.startswith("MARSIS"):
            return _identify_marsis(name, instrument_key, groups, ext, file_role)
        elif instrument_key.startswith("LRS"):
            return _identify_lrs(name, instrument_key, groups, ext, file_role)

    raise ValueError(f"Could not identify file {name}")


# ----------------------------
# SHARAD
# ----------------------------
def _identify_sharad(
            name: str,
            instrument_key: str,
            groups: tuple,
            ext: str,
            file_role: str,
    ) -> FileIdentity:
    """Parse a SHARAD filename into a FileIdentity."""

    fid = FileIdentity(
        platform="MRO",
        instrument="SHARAD",
        product_type=product_type_mapping[instrument_key],
        file_role=file_role,
        filename=name,
        target="MARS",
    )

    if instrument_key in ("SHARAD_EDR", "SHARAD_RDR"):
        fid.product_id = groups[0]
        fid.ost_line = groups[1]
        fid.operative_mode = re.sub(
            r"^(SS|RO)(\d)$",
            lambda mm: f"{mm.group(1)}0{mm.group(2)}",
            groups[2].upper(),
        )
        fid.instrument_state = groups[4].upper()

        if fid.file_role != "LABEL":
            if fid.product_type == "EDR":
                suffix = (groups[5] or "").upper()
                if suffix == "S":
                    fid.file_role = "SCIENCE"
                elif suffix == "A":
                    fid.file_role = "AUXILIARY"
                else:
                    raise ValueError(f"Invalid suffix {suffix!r} for SHARAD EDR file {name}")
                fid.data_form = "CMP"

            elif fid.product_type == "RDR":
                if ext == "DAT":
                    fid.file_role = "SCIENCE"
                    fid.data_form = "UNC"
                else:
                    raise ValueError(f"Invalid extension {ext} for SHARAD RDR file {name}")

    elif instrument_key == "SHARAD_US_EDR":
        fid.product_id = f"{groups[0]}000"
        fid.data_form = "CMP"

        if fid.file_role != "LABEL":
            aux_tag = groups[2]
            header_tag = groups[4]
            hk_tag = groups[5]
            log_tag = groups[6]
            mode_tag = groups[7]
            orbit_tag1 = groups[9]
            orbit_tag2 = groups[11]

            if aux_tag:
                fid.file_role = "AUXILIARY"
                fid.operative_mode = "AUX"
                fid.data_form = "UNC"
            elif header_tag:
                fid.file_role = "HEADER"
                fid.operative_mode = "AUX"
                fid.data_form = "UNC"
            elif mode_tag:
                fid.file_role = "SCIENCE"
                fid.operative_mode = "MODE"
            elif orbit_tag2:
                raise ValueError("SHARAD US EDR Orbit Log files are not yet supported.")
            elif orbit_tag1:
                fid.file_role = "ORBIT"
                fid.operative_mode = "AUX"
                fid.data_form = "UNC"
            elif hk_tag:
                raise ValueError("SHARAD US EDR HK files are not yet supported.")
            elif log_tag:
                raise ValueError("SHARAD US EDR Log files are not yet supported.")
            else:
                raise ValueError(f"Unrecognized SHARAD US EDR file type for {name}")

    elif instrument_key == "SHARAD_US_RDR":
        fid.product_id = groups[0]
        fid.data_form = "UNC"

        if fid.file_role != "LABEL":
            suffix = groups[1].upper()
            if suffix == "RGRAM":
                fid.file_role = "SCIENCE"
            elif suffix == "GEOM":
                fid.file_role = "GEOMETRY"
            else:
                raise ValueError(f"Invalid suffix {suffix} for SHARAD US_RDR file {name}")

    elif instrument_key in ["SHARAD_UPB", "SHARAD_FPB", "SHARAD_QDA"]:
        fid.product_id = groups[0]
        fid.data_form = "UNC"
        if fid.file_role != "LABEL":
            suffix = groups[1].upper()
            if suffix in ['UNFOCPOW', "RGRAM", "MLK"]:
                fid.file_role = "SCIENCE"
            elif suffix == "GEOM":
                fid.file_role = "GEOMETRY"
            elif suffix == "ETM":
                raise NotImplementedError("QDA Geometry (ETM) reading not yet supported.")
            else:
                raise ValueError(f"Invalid suffix {suffix} for SHARAD UPB|FPB|QDA file {name}")
    elif instrument_key == "SHARAD_FPB_SIM":
        fid.product_id = groups[0]
        fid.data_form = "UNC"
        if fid.file_role != "LABEL":
            suffix = groups[1].upper()
            if suffix == "COMBINED":
                fid.operative_mode = "SCS"
                fid.file_role =  "SIM"
            elif suffix  in ("NADIR", "FIRSTRETURN"):
                fid.operative_mode = "RTRN"
                fid.file_role = "RTRN"
            else:
                raise ValueError(f"Invalid suffix {suffix} for SHARAD FPB SIM file")


    elif instrument_key == "SHARAD_US_SCS":
        fid.product_id = groups[0]
        fid.data_form = "UNC"

        if fid.file_role != "LABEL":
            suffix = groups[1].upper()
            if suffix == "SIM":
                fid.operative_mode = "SCS"
                fid.file_role = "SIM"
            elif suffix == "RTRN":
                fid.operative_mode = "RTRN"
                fid.file_role = "RTRN"
            elif suffix == "EMAP":
                fid.operative_mode = "EMAP"
                fid.file_role = "EMAP"
            else:
                raise ValueError(f"Invalid suffix {suffix} for SHARAD SCS file {name}")

    return fid


# ----------------------------
# MARSIS
# ----------------------------
_MARSIS_TARGET_FLAG = {
    "M": "MARS",
    "P": "PHOBOS",
    "T": "TRANSIT",
}


def _identify_marsis(
        name: str,
        instrument_key: str,
        groups: tuple,
        ext: str,
        file_role: str,
) -> FileIdentity:
    """Parse a MARSIS filename into a FileIdentity."""

    if instrument_key == "MARSIS_AIS_CAL_RXO":
        # groups: (e, product_id, mode, target, suffix, ext)
        fid = FileIdentity(
            platform="MEX",
            instrument="MARSIS",
            product_type="EDR",
            product_id=groups[1],
            operative_mode=groups[2].upper(),
            data_form="CMP",
            file_role=file_role,
            filename=name,
            target=_MARSIS_TARGET_FLAG[(groups[3] or "").upper()],
        )

        if fid.file_role != "LABEL":
            suffix = (groups[4] or "").upper()
            if suffix == "F":
                fid.file_role = "SCIENCE"
            elif suffix == "G":
                fid.file_role = "AUXILIARY"
            else:
                raise ValueError(f"Invalid suffix {suffix!r} for MARSIS {fid.operative_mode} file {name}")

    else:
        # Standard MARSIS SS pattern
        # groups: (e/r, product_id, mode, state, form, target, suffix, ext)
        marsis_kind = "EDR" if groups[0].lower() == "e" else "RDR"

        fid = FileIdentity(
            platform="MEX",
            instrument="MARSIS",
            product_type=product_type_mapping[f"MARSIS_{marsis_kind}"],
            product_id=groups[1],
            operative_mode=groups[2].upper(),
            instrument_state=groups[3].upper(),
            data_form=groups[4].upper(),
            file_role=file_role,
            filename=name,
            target=_MARSIS_TARGET_FLAG[(groups[5] or "").upper()],
        )

        if fid.file_role != "LABEL":
            if fid.product_type == "EDR":
                suffix = (groups[6] or "").upper()
                if suffix == "F":
                    fid.file_role = "SCIENCE"
                elif suffix == "G":
                    fid.file_role = "AUXILIARY"
                else:
                    raise ValueError(f"Invalid suffix {suffix!r} for MARSIS EDR file {name}")

            elif fid.product_type == "RDR":
                if ext == "DAT":
                    fid.file_role = "SCIENCE"
                else:
                    raise ValueError(f"Invalid extension {ext} for MARSIS RDR file {name}")

    return fid


# ----------------------------
# LRS
# ----------------------------
def _identify_lrs(
        name: str,
        instrument_key: str,
        groups: tuple,
        ext: str,
        file_role: str,
) -> FileIdentity:
    """Parse an LRS filename into a FileIdentity."""

    fid = FileIdentity(
        platform="SELENE",
        instrument="LRS",
        product_type=product_type_mapping[instrument_key],
        data_form="UNC",
        file_role=file_role,
        filename=name,
        target="MOON",
    )

    if instrument_key in ("LRS_EDR", "LRS_EDR_MALFORMED"):
        fid.operative_mode = f"{groups[0].upper()}_WF"
        fid.product_id = Path(name).stem

        if fid.file_role != "LABEL":
            if ext == "TBL":
                fid.file_role = "SCIENCE"
            else:
                raise ValueError(f"Invalid extension {ext} for LRS EDR file {name}")

    elif instrument_key == "LRS_RDR_COMPLEX":
        fid.operative_mode = f"SAR{groups[0]}KM-C"
        fid.product_id = f"{groups[1]}_{groups[2]}"
        if fid.file_role != "LABEL":
            if ext == "TBL":
                fid.file_role = "SCIENCE"
            else:
                raise ValueError(f"Invalid extension {ext} for LRS complex RDR file {name}")

    elif instrument_key == "LRS_RDR_POWER":
        fid.operative_mode = f"SAR{groups[0]}KM"
        fid.product_id = groups[1]
        if fid.file_role != "LABEL":
            if ext == "IMG":
                fid.file_role = "SCIENCE"
            else:
                raise ValueError(f"Invalid extension {ext} for LRS power RDR file {name}")
    else:
        raise ValueError(f"Invalid instrument key {instrument_key}")

    return fid


##############################################################################################################
#
# Check Supported
#
##############################################################################################################
def check_supported(
        instrument: str,
        product_type: str,
        operative_mode: str,
    ) -> bool:
    """Check whether an instrument / product type / operative mode combination is supported.

    Args:
        instrument: Instrument name (e.g., ``"SHARAD"``, ``"MARSIS"``, ``"LRS"``).
        product_type: Product type (e.g., ``"EDR"``, ``"RDR"``, ``"US EDR"``).
        operative_mode: Operative mode identifier (e.g., ``"SS04"``,
            ``"SAR10KM-C"``, ``"US_RDR"``).

    Returns:
        ``True`` if the combination is supported.

    Raises:
        ValueError: If the instrument, product type, or operative mode
            is not supported.
    """
    instrument = instrument.upper()
    operative_mode = operative_mode.upper()
    product_type = product_type.upper()

    if instrument == "LRS":
        mod = importlib.import_module("grasp.lrs.supported")
    elif instrument == "MARSIS":
        mod = importlib.import_module("grasp.marsis.supported")
    elif instrument == "SHARAD":
        mod = importlib.import_module("grasp.sharad.supported")
    else:
        raise ValueError(f"Invalid instrument {instrument}")

    supported = getattr(mod, "SUPPORTED")

    try:
        if operative_mode in supported[product_type]:
            return True
    except KeyError:
        pass

    raise ValueError(
        f"{instrument} {product_type} {operative_mode} is not supported"
    )