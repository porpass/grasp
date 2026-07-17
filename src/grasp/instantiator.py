# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

from .input.utils import identify_file
from .common.utils import check_file
from .radar_sounder import RadarSounder
from .sharad.sharad import SHARADSounder
from .marsis.marsis import MARSISSounder
from .lrs.lrs import LRSSounder
from .input.job import load_processing_job
from .input.load import load_hdf5
from .processing_defaults import resolve_instrument_defaults

def radar_sounder(x: str | Path | list[str | Path]) -> RadarSounder:
    """Factory function to create a radar sounder object.

    Args:
        x: A file path (str), a list of file paths, or a path
            to a TOML job file.

    Returns:
        A RadarSounder subclass with file paths and metadata populated.
        If a TOML file is provided, processing parameters are also set.

    Raises:
        ValueError: If files belong to different observations or
            the instrument is unsupported.
    """
    processing_parameters = None

    if isinstance(x, (str, Path)) and str(x).endswith('.h5'):
        state = load_hdf5(x)
        instrument = state["metadata"]["instrument"].upper()
        # instantiate subclass
        if instrument == "SHARAD":
            sounder = SHARADSounder()
        elif instrument == "MARSIS":
            sounder = MARSISSounder()
        elif instrument == "LRS":
            sounder = LRSSounder()
        else:
            raise ValueError(f"Unsupported instrument: {instrument}")
        sounder._load_from_state(state)
        return sounder

    if isinstance(x, (str, Path)) and str(x).endswith('.toml'):
        job = load_processing_job(x)
        x = [str(job.label_file), str(job.science_file)]
        if job.auxiliary_file is not None:
            x.append(str(job.auxiliary_file))
        processing_parameters = job.parameters

    if isinstance(x, str):
        x = [x]

    if isinstance(x, list):
        identities = []
        for filepath in x:
            check_file(filepath)
            fid = identify_file(filepath)
            fid.filepath = filepath
            identities.append(fid)

        # Validate all files belong to the same observation
        instruments = {f.instrument for f in identities}
        product_ids = {f.product_id for f in identities}
        if len(instruments) > 1:
            raise ValueError(
                f"All files must be from the same instrument, "
                f"got: {instruments}"
            )
        if len(product_ids) > 1:
            raise ValueError(
                f"All files must belong to the same observation, "
                f"got product IDs: {product_ids}"
            )

        instrument = identities[0].instrument.upper()
        file_info = identities[0]

    else:
        raise ValueError("Input must be a file path, list of file paths, "
                         "or a path to a TOML job file.")

    # Create the appropriate subclass
    if instrument == "SHARAD":
        sounder = SHARADSounder(file_info)
    elif instrument == "MARSIS":
        sounder = MARSISSounder(file_info)
    elif instrument == "LRS":
        sounder = LRSSounder(file_info)
    else:
        raise ValueError(f"Unsupported instrument: {instrument}")

    # Set file paths based on file_role
    for fid in identities:
        if fid.file_role == "LABEL":
            sounder.set_label_file(fid.filepath)
        elif fid.file_role == "SCIENCE":
            sounder.set_science_file(fid.filepath)
        elif fid.file_role == "AUXILIARY":
            sounder.set_auxiliary_file(fid.filepath)

    # Set processing parameters if loaded from TOML
    if processing_parameters is not None:
        sounder.processing_parameters = processing_parameters

    # Apply instrument-specific default parameters (fills None fields)
    if sounder.processing_parameters is not None:
        resolve_instrument_defaults(
            sounder.processing_parameters,
            instrument=instrument,
            metadata=file_info,
        )
        # Validate at construction time so support-table conflicts and
        # enum-field typos (e.g. "HANNNING") surface immediately rather
        # than after hours of processing get to the affected stage.
        sounder.validate_processing_parameters()

    return sounder