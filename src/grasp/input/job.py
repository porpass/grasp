# SPDX-License-Identifier: BSD-3-Clause
"""GRaSP processing job loader.

Provides a dataclass for representing a complete processing job
and a loader function that parses a TOML configuration file into
that dataclass.
"""

import tomllib
from pathlib import Path
from ..grasp_types import (
    ProcessingJob,
    ProcessingParameters,
    PreprocessingParams,
    RangeCompressionParams,
    EMISuppresionParams,
    IonoCompParams,
    SARParams,
    MLKParams,
    OutputParameters,
    PlotParameters,
    ClutterSimParams,
    FinalOutputParams
)
from ..section_aliases import LONG_TO_ATTR as SECTION_MAP


# Maps internal keys to their corresponding dataclass constructors.
STAGE_CONSTRUCTORS = {
    "preprocessing": PreprocessingParams,
    "range_compression": RangeCompressionParams,
    "emi_suppression": EMISuppresionParams,
    "iono_comp": IonoCompParams,
    "sar": SARParams,
    "mlk": MLKParams,
    "csim": ClutterSimParams,
    "output": OutputParameters,
    "plots": PlotParameters,
    "final_output": FinalOutputParams,
}


def load_processing_job(toml_path: str | Path) -> ProcessingJob:
    """Load a processing job from a TOML configuration file.

    The TOML file must contain ``[General]`` and ``[Input]`` sections.
    Processing stage sections (``[Preprocessing]``, ``[Range Compression]``,
    etc.) are optional — defaults from the corresponding dataclass are
    used for any omitted section or field.

    Section names are case-insensitive. Parameter names within each
    section must match the dataclass field names exactly.

    Args:
        toml_path: Path to the TOML configuration file.

    Returns:
        A populated ProcessingJob instance.

    Raises:
        ValueError: If a required section is missing or an
            unrecognized section name is encountered.
        TypeError: If a parameter name within a section does not
            match any field in the corresponding dataclass.
    """
    toml_path = Path(toml_path)

    with open(toml_path, 'rb') as f:
        raw = tomllib.load(f)

    # Normalize section names to lowercase
    cfg: dict[str, dict] = {}
    for section_name, section_data in raw.items():
        key = section_name.lower()
        if key not in SECTION_MAP:
            raise ValueError(
                f"Unrecognized section '{section_name}' in {toml_path}. "
                f"Valid sections: {', '.join(SECTION_MAP.keys())}"
            )
        mapped = SECTION_MAP[key]
        if mapped in cfg:
            raise ValueError(
                f"Duplicate section '{section_name}' in {toml_path}."
            )
        cfg[mapped] = section_data

    # Validate required sections
    if "general" not in cfg:
        raise ValueError(
            f"Missing required [General] section in {toml_path}."
        )
    if "input" not in cfg:
        raise ValueError(
            f"Missing required [Input] section in {toml_path}."
        )

    # Build input paths
    input_cfg = cfg["input"]
    label_file = Path(input_cfg["label_file"])
    science_file = Path(input_cfg["science_file"])
    auxiliary_file = (
        Path(input_cfg["auxiliary_file"])
        if "auxiliary_file" in input_cfg
        else None
    )

    # Build stage params
    stages = {}
    for stage_key, constructor in STAGE_CONSTRUCTORS.items():
        if stage_key in cfg:
            stages[stage_key] = constructor(**cfg[stage_key])
        # else: default from ProcessingParameters field factory

    # Build ProcessingParameters from general + stages
    parameters = ProcessingParameters(
        **cfg["general"],
        **stages,
    )

    return ProcessingJob(
        label_file=label_file,
        science_file=science_file,
        auxiliary_file=auxiliary_file,
        parameters=parameters,
    )