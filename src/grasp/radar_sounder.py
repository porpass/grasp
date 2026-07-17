# SPDX-License-Identifier: BSD-3-Clause
from abc import ABC, abstractmethod
from dataclasses import fields, replace
import numpy as np
from pathlib import Path
from pyproj import CRS as _CRS
from .geospatial.crs import gcs_2000_crs, normalize_crs
from typing import Any, Literal
import warnings
import importlib
from .constants import C
from .input.utils import grab_radar_params, grab_record_length
from .input.job import load_processing_job
from .processing_defaults import resolve_instrument_defaults
from .common.config import get_data_path
from .spice.utils import grab_spice_params, find_mk_files, furnish, utc2et
from .spice.observation_geometry import compute_geometry
from .common.utils import decimate_dict, decimate_dataclass, determine_observation_years
from .output.writers import save_hdf5
from .input.load import load_hdf5

from .postprocessing.multilook import multilook as _multilook
from .processing.sar.dispatch import sar_process as _sar_process

from .processing.emi.suppression import suppress_emi as _suppress_emi
from .processing.ionosphere.ionospheric_compensation import ionospheric_compensation as _iono_comp
from .schema.choices import CHOICES as _CHOICES, get_choices as _get_choices

from .simulation.clutter import simulate_clutter as _simulate_clutter

from .output.writers import (export_data as _export_data, export_images as _export_images,
                             export_segy as _export_segy, export_csim_data as _export_csim_data,
                             export_csim_images as _export_csim_images)

from .grasp_types import (
    CampbellResult,
    ContrastResult,
    FileIdentity,
    GeometryResult,
    MetaDataInfo,
    ProcessingParameters,
    RadarParams,
    RecordFormat,
    SpiceParams,
    SARResult,
    ClutterSimParams
)


class RadarSounder(ABC):
    """Base class for all radar sounder classes.

    Handles initialization of metadata, file paths, static parameters,
    and processing options.

    Subclasses must define a ``PROCESSING_SUPPORT`` class attribute
    that maps product types to per-stage boolean support flags. The
    table is loaded from ``processing_support.toml`` at class
    definition time. Example::

        PROCESSING_SUPPORT = {
            "EDR": {
                "preprocessing":     True,
                "range_compression": True,
                "emi_suppression":   True,
                "iono_comp":         True,
                "sar":               True,
                "mlk":               True,
            },
            "RDR": {
                "preprocessing":     True,
                "range_compression": False,
                "emi_suppression":   False,
                "iono_comp":         False,
                "sar":               False,
                "mlk":               True,
            },
        }

    Stages flagged ``False`` are disabled automatically with a warning
    by :meth:`validate_processing_parameters`; ``True`` stages run
    without comment.
    """

    PROCESSING_SUPPORT: dict[str, dict[str, str]] = {}
    GEOMETRY_KEYS: dict[str, dict[str, Any]] = {}

    def __init__(self, file_info=None):
        self.label_file: str | None = None
        self.science_file: str | None = None
        self.auxiliary_file: str | None = None

        self.label_data: dict[str, Any] = {}
        self.science_data: dict[str, Any] = {}
        self.auxiliary_data: dict[str, Any] = {}


        self.metadata: MetaDataInfo = MetaDataInfo()
        self.parameters: RadarParams = RadarParams()
        self.processing_parameters: ProcessingParameters | None = None
        self.clutter_sim_params: ClutterSimParams | None = None
        self.spice: SpiceParams = SpiceParams()
        self.crs: _CRS | None = None
        self.record_format: RecordFormat = RecordFormat()
        self.geometry: GeometryResult | None = None
        self.status: dict[str, str | None] = {
            "DOMAIN": None, "FORM": None, "LEVEL": None,
        }
        self.phase_shift = None
        self.emi_mask = None
        self.years: list[int] | None = None
        self.ionosphere: CampbellResult | ContrastResult | None = None
        self.sar: SARResult | None = None
        self.clutter_result = None
        self.datum = None

        if file_info is not None:
            self.set_metadata(file_info)
            self.check_supported()
            self.set_record_lengths(file_info)
            self.set_parameters(file_info.instrument, file_info.operative_mode)
            self.set_spice(file_info.platform)

    @property
    def data(self):
        md = self.metadata
        inst = md.instrument
        if inst == "SHARAD":
            if md.product_type in ('EDR', 'DEC'):
                return self.science_data['ECHO_SAMPLES']
            elif md.product_type in ('RDR', 'US_RDR'):
                return self.science_data['IMAGE']
            else:
                raise ValueError(f"Unsupported product type: {md.product_type}")
        elif inst == "LRS":
            if md.product_type == "EDR":
                return self.science_data['WAVEFORM']
            elif md.product_type == "RDR":
                return self.science_data['IMAGE']
            else:
                raise ValueError(f"Unsupported product type: {md.product_type}")
        elif inst == "MARSIS":
            raise NotImplementedError("MARSIS has multiple science arrays. Access science_data directly.")
        else:
            raise ValueError(f"Unsupported instrument {inst}")


    @data.setter
    def data(self, value):
        md = self.metadata
        inst = md.instrument
        if inst == "SHARAD":
            if md.product_type in ('EDR', 'DEC'):
                self.science_data['ECHO_SAMPLES'] = value
            elif self.metadata.product_type in ('RDR', 'US_RDR'):
                self.science_data['IMAGE'] = value
            else:
                raise ValueError(f"Unsupported product type: {md.product_type}")
        elif inst == "LRS":
            if md.product_type == "EDR":
                self.science_data['WAVEFORM'] = value
            elif md.product_type == "RDR":
                self.science_data['IMAGE'] = value
            else:
                raise ValueError(f"Unsupported product type: {md.product_type}")
        elif inst == "MARSIS":
            raise NotImplementedError("MARSIS has multiple science arrays. Access science_data directly.")
        else:
            raise ValueError(f"Unsupported instrument {inst}")
    ###################################################################################################################
    #
    # File path setters
    #
    ###################################################################################################################
    def set_label_file(self, label_file: str | Path) -> None:
        """Set the label file path.

        Args:
            label_file: Path to the label file.
        """
        self.label_file = str(label_file)

    def set_science_file(self, science_file: str | Path) -> None:
        """Set the science file path.

        Args:
            science_file: Path to the science file.
        """
        self.science_file = str(science_file)

    def set_auxiliary_file(self, auxiliary_file: str | Path) -> None:
        """Set the auxiliary file path.

        Args:
            auxiliary_file: Path to the auxiliary file.
        """
        self.auxiliary_file = str(auxiliary_file)

    def set_files(self,
                  *,
                  label_file: str | Path,
                  science_file: str | Path,
                  auxiliary_file: str | Path,
                  verbose: bool = False) -> None:
        """Set label, science, and auxiliary file paths.

        Args:
            label_file: Path to the label file.
            science_file: Path to the science file.
            auxiliary_file: Path to the auxiliary file.
            verbose: If True, print the resolved file paths.
        """
        self.set_label_file(label_file)
        self.set_science_file(science_file)
        self.set_auxiliary_file(auxiliary_file)

        if verbose:
            print("File Paths:")
            print(f"\tLabel File: {self.label_file}")
            print(f"\tScience File: {self.science_file}")
            print(f"\tAuxiliary File: {self.auxiliary_file}")

    def print_job_summary(self) -> None:
        """Print the resolved job configuration.

        Summarizes the input files, instrument identity, radar
        parameters, record format, SPICE configuration, and
        processing parameters. If the sounder has not been
        populated from a job file or input set, prints a short
        notice instead.
        """
        if self.metadata is None:
            print("(no job loaded — nothing to summarize)")
            return
        print(f"--- GRaSP :: {self.metadata.instrument} "
              f"{self.metadata.product_type} {self.metadata.operative_mode} ---")
        print(f"Label:     {self.label_file}")
        print(f"Science:   {self.science_file}")
        print(f"Auxiliary: {self.auxiliary_file}")
        self.metadata.print()
        self.parameters.print()
        self.record_format.print()
        self.spice.print()
        self.processing_parameters.print()

    ####################################################################################################################
    #
    # Configuration setters
    #
    ####################################################################################################################
    def set_metadata(self, file_info: FileIdentity) -> None:
        """Populate observation metadata from file identity.

        Args:
            file_info: Parsed file identity from ``identify_file``.
        """
        self.metadata = MetaDataInfo(
            platform=file_info.platform,
            instrument=file_info.instrument.upper(),
            product_type=file_info.product_type.upper(),
            product_id=file_info.product_id,
            operative_mode=file_info.operative_mode.upper(),
            instrument_state=file_info.instrument_state.upper(),
            data_form=file_info.data_form.upper(),
        )

    def set_record_lengths(self, file_info: FileIdentity) -> None:
        """Populate record format from file identity.

        Args:
            file_info: Parsed file identity from ``identify_file``.
        """
        self.record_format = grab_record_length(
            file_info.instrument,
            file_info.product_type,
            file_info.operative_mode,
            file_info.instrument_state,
            file_info.data_form,
        )

    def set_parameters(self,
                       instrument: str,
                       mode: str | None = None) -> None:
        """Set radar configuration parameters.

        Args:
            instrument: Radar instrument identifier.
            mode: Operative mode string.

        Raises:
            ValueError: If the instrument is invalid.
        """
        instrument = instrument.upper()
        if instrument in ["MARSIS", "LRS", "SHARAD"]:
            self.parameters = grab_radar_params(
                instrument, mode,
                self.metadata.product_type,
                self.metadata.data_form,
            )
        else:
            raise ValueError(f"Invalid instrument {instrument}")

    def set_spice(self, platform: str) -> None:
        """Populate SPICE parameters and set the target body CRS.

        Args:
            platform: Spacecraft/platform identifier.

        Raises:
            ValueError: If the target body does not have a defined CRS.
        """
        self.spice = grab_spice_params(platform)
        target = self.spice.target_str.upper()
        self.crs = gcs_2000_crs(target)

    def set_spice_param(self, key: str, value: Any) -> None:
        """Update a single SPICE parameter.

        Args:
            key: SPICE parameter name (e.g. ``"method"``, ``"ab_corr"``).
            value: New value.

        Raises:
            TypeError: If key is not a valid SpiceParams field.
        """
        self.spice = replace(self.spice, **{key: value})

    def update_spice(self, **kwargs: Any) -> None:
        """Update one or more SPICE parameters.

        Args:
            **kwargs: Key-value pairs to update (e.g.
                ``update_spice(ab_corr="LT+S", method="ELLIPSOID")``).

        Raises:
            TypeError: If any key is not a valid SpiceParams field.
        """
        self.spice = replace(self.spice, **kwargs)

    def set_processing_parameters(self,
                                  toml_path: str | Path | None = None,
                                  *,
                                  out_dir: str | Path | None = None,
                                  verbose: bool = False,
                                  **stage_overrides: dict) -> None:
        """Set processing parameters, either from a TOML job file or programmatically.

        Two modes:

        * **TOML mode** (``toml_path`` is provided): load processing
          parameters from the TOML job file. File paths and input
          settings in the TOML are ignored — only the parameter
          section is consumed. ``out_dir`` and ``stage_overrides``
          are ignored in this mode.
        * **Programmatic mode** (``toml_path`` is ``None``): build a
          fresh ``ProcessingParameters`` rooted at ``out_dir`` and
          apply any ``stage_overrides``. Example::

              sounder.set_processing_parameters(
                  out_dir="./out",
                  emi_suppression={"k": 4.0},
              )

        In both modes the result is then passed through
        :func:`resolve_instrument_defaults` so any unset (``None``)
        fields are filled from the instrument's defaults table.

        Args:
            toml_path: Path to a TOML configuration file. If
                ``None``, parameters are constructed programmatically
                from ``out_dir`` and ``stage_overrides``.
            out_dir: Output directory for processing products.
                Required in programmatic mode; ignored in TOML mode.
            verbose: If True, print the resolved parameters.
            **stage_overrides: Stage-keyed dicts of values to apply
                on top of the instrument defaults. Each key must be
                an attribute of ``ProcessingParameters`` (e.g.
                ``range_compression``, ``emi_suppression``) and each
                value must be a dict of field names to values valid
                for that stage's params dataclass. Ignored in TOML
                mode.

        Raises:
            ValueError: In programmatic mode, if ``out_dir`` is
                missing/empty, if a stage name is unknown, or if a
                stage field is unknown.
        """
        if toml_path is not None:
            job = load_processing_job(toml_path)
            params = job.parameters
        else:
            if not out_dir:
                raise ValueError(
                    "out_dir is required when toml_path is not provided."
                )
            params = ProcessingParameters(out_dir=out_dir)
            for stage_name, overrides in stage_overrides.items():
                stage = getattr(params, stage_name, None)
                if stage is None:
                    raise ValueError(
                        f"Unknown processing stage: {stage_name!r}."
                    )
                for field_name, value in overrides.items():
                    if not hasattr(stage, field_name):
                        raise ValueError(
                            f"Unknown field {field_name!r} for stage "
                            f"{stage_name!r}."
                        )
                    setattr(stage, field_name, value)

        resolve_instrument_defaults(
            params,
            instrument=self.metadata.instrument,
            metadata=self.metadata,
        )
        self.processing_parameters = params

        # Surface support-table and enum-field issues at job-load time
        # rather than per-stage at processing time — saves hours when
        # a typo would otherwise blow up at the last stage.
        self.validate_processing_parameters()

        if verbose:
            self.processing_parameters.print()

    ####################################################################################################################
    #
    # Processing parameter validation
    #
    ####################################################################################################################

    def validate_processing_parameters(self) -> None:
        """Validate processing parameters against the instrument support table.

        Three checks:

        1. Walks each enabled processing stage and disables any that
           are flagged ``False`` in ``PROCESSING_SUPPORT`` for the
           current product type, emitting a warning per disabled stage.
        2. Walks the :data:`grasp.schema.choices.CHOICES` registry and
           checks every string-enum field on every enabled stage against
           its allowed values (case-insensitive). Per-instrument
           restrictions from
           :data:`grasp.schema.choices.RESTRICTIONS` are applied when
           the current instrument/product-type is known — so e.g.
           ``chirp_type = "CALIBRATED"`` is rejected on MARSIS even
           though the global set allows it. All bad values are
           accumulated and surfaced in a single ``ValueError`` — catches
           typos like ``"HANNNING"`` before the pipeline burns hours
           getting to the affected stage.
        3. If CSIM is enabled, eagerly resolves clutter-sim params via
           :meth:`_get_csp` so configuration errors (missing
           ``dem_path``, unresolved ``target``, etc.) surface up front
           rather than after minutes of unrelated processing. When SAR
           is *not* also enabled, ``at_dist`` / ``at_step`` must be
           non-None — CSIM has no SAR output to derive them from.

        This method is called automatically by :meth:`_get_pp`.

        Raises:
            ValueError: If any enum-style field has an unrecognised
                value, or if CSIM is enabled but its configuration is
                incomplete (e.g. missing ``at_dist`` / ``at_step`` when
                SAR is not also running).
        """
        _STAGE_NAMES = {
            "preprocessing": "Preprocessing",
            "range_compression": "Range Compression",
            "emi_suppression": "EMI Suppression",
            "iono_comp": "Ionospheric Compensation",
            "sar": "SAR Processing",
            "mlk": "Multilook",
        }
        if self.processing_parameters is None:
            return

        pp = self.processing_parameters
        instrument = self.metadata.instrument
        product_type = self.metadata.product_type

        # 1. Support-table check (only if both are populated).
        support = self.PROCESSING_SUPPORT
        if support and product_type in support:
            stage_support = support[product_type]
            for stage_attr, display_name in _STAGE_NAMES.items():
                stage_params = getattr(pp, stage_attr, None)
                if stage_params is None:
                    continue
                if not getattr(stage_params, 'enabled', False):
                    continue
                if not stage_support.get(stage_attr, False):
                    warnings.warn(
                        f"{display_name} is not supported for "
                        f"{instrument} {product_type} data. "
                        f"Disabling {display_name}.",
                        UserWarning,
                    )
                    stage_params.enabled = False

        # 2. Enum-style field validation. Walks the registry and
        # accumulates every bad value before raising, so the user fixes
        # all of them at once instead of getting one error per re-run.
        # Runs independently of the support check so typos surface even
        # when PROCESSING_SUPPORT is empty or the product_type is not
        # listed.
        enum_errors: list[str] = []
        for stage_attr, field_name in _CHOICES.keys():
            stage_params = getattr(pp, stage_attr, None)
            if stage_params is None:
                continue
            # Stages that carry an `enabled` flag only validate when
            # they're enabled. Config sections without `enabled`
            # (output, plots) always validate.
            enabled_attr = getattr(stage_params, "enabled", None)
            if enabled_attr is False:
                continue
            value = getattr(stage_params, field_name, None)
            if value is None:
                continue
            valid = _get_choices(stage_attr, field_name, instrument, product_type)
            if valid is None:
                continue
            if str(value).upper() not in {v.upper() for v in valid}:
                enum_errors.append(
                    f"  - {stage_attr}.{field_name} = {value!r} "
                    f"(valid: {sorted(valid)})"
                )
        if enum_errors:
            raise ValueError(
                "Invalid processing-parameter values:\n"
                + "\n".join(enum_errors)
            )

        # Eager CSIM validation: surface config errors before the
        # pipeline burns time on unrelated stages.
        csim = getattr(pp, "csim", None)
        if csim is not None and getattr(csim, "enabled", False):
            cs = self._get_csp()  # raises on dem_path / target issues
            sar = getattr(pp, "sar", None)
            sar_enabled = sar is not None and getattr(sar, "enabled", False)
            if not sar_enabled and (cs.at_dist is None or cs.at_step is None):
                raise ValueError(
                    "CSIM requires at_dist and at_step when SAR is not "
                    "enabled (no SAR output to derive them from). Set "
                    "both in the job file's [csim] section or in the "
                    f"instrument's defaults TOML for {instrument} "
                    f"{product_type}."
                )

    ####################################################################################################################
    #
    # Reading methods
    #
    ####################################################################################################################
    def read_files(self,
                   label_file: str | Path | None = None,
                   science_file: str | Path | None = None,
                   auxiliary_file: str | Path | None = None,
                   verbose: bool = False) -> None:
        """Read label, science, and auxiliary files.

        If paths are provided, sets them first. Then reads whichever
        files have paths set.

        Args:
            label_file: Path to the label file.
            science_file: Path to the science file.
            auxiliary_file: Path to the auxiliary file.
            verbose: If True, print progress messages.
        """
        if label_file is not None:
            self.set_label_file(label_file)
        if science_file is not None:
            self.set_science_file(science_file)
        if auxiliary_file is not None:
            self.set_auxiliary_file(auxiliary_file)

        if self.label_file is not None:
            self.read_label_file(verbose=verbose)
        if self.science_file is not None:
            self.read_science_file(verbose=verbose)
        if self.auxiliary_file is not None:
            self.read_auxiliary_file(verbose=verbose)

    def read_label_file(self, verbose=False):
        """Read a label file."""
        self.check_supported()
        inst = self.metadata.instrument
        parse = self._grab_parser()
        assert self.label_file is not None
        if verbose:
            print(f"Parsing {inst.upper()} Label File: {self.label_file}")
        self.label_data = parse(self.label_file,
                                prod_type=self.metadata.product_type,
                                role='LABEL')

    def read_science_file(self, verbose=False):
        """Read a science file."""
        self.check_supported()
        md = self.metadata
        inst = md.instrument
        parse = self._grab_parser()
        assert self.science_file is not None
        if verbose:
            print(f"Parsing {inst.upper()} Science File: {self.science_file}")
        if inst == 'LRS':
            self.science_data = parse(self.science_file, prod_type=md.product_type, role="SCIENCE",
                                      mode=md.operative_mode, rec_len=self.record_format.reclen)
            self.years = determine_observation_years(self.science_data['OBSERVATION_TIME'])
            self.geometry = self._build_geometry()
        elif inst == 'MARSIS':
            self.science_data = parse(self.science_file, prod_type=md.product_type, role='SCIENCE',
                                      mode=md.operative_mode, state=md.instrument_state, form=md.data_form,
                                      rec_len=self.record_format.reclen)
            if md.product_type == "RDR":
                self.geometry = self._build_geometry()
                self.years = determine_observation_years(self.science_data['GEOMETRY_EPOCH'])
        elif inst == 'SHARAD':
            self.science_data = parse(self.science_file, prod_type=md.product_type, role='SCIENCE',
                                      rec_len=self.record_format.reclen)

            if md.product_type == "RDR":
                self.geometry = self._build_geometry()
                self.years = determine_observation_years(self.science_data['GEOMETRY_EPOCH'])
        else:
            raise ValueError(f"Unsupported instrument {inst}")

    def read_auxiliary_file(self, verbose=False):
        """Read an auxiliary file."""
        md = self.metadata
        inst = md.instrument
        if inst == 'LRS':
            if verbose:
                print(f"Auxiliary files for LRS do not exist.")
            return
        if inst in ('MARSIS', 'SHARAD'):
            if md.product_type == "RDR":
                print(f"Auxiliary files for {inst} RDRs do not exist.")
                return

        self.check_supported()
        parse = self._grab_parser()
        assert self.auxiliary_file is not None

        if verbose:
            print(f"Parsing {inst.upper()} Auxiliary File: {self.auxiliary_file}")

        if inst in ('MARSIS', 'SHARAD'): # Only MARSIS and SHARAD EDRs to this point
            self.auxiliary_data = parse(self.auxiliary_file, prod_type=md.product_type, role='AUXILIARY',
                                        rec_len=self.record_format.auxlen)
            self.years = determine_observation_years(self.auxiliary_data['GEOMETRY_EPOCH'])
            self.geometry = self._build_geometry()
        else:
            raise ValueError(f"Unsupported instrument {inst}")

    ###################################################################################################################
    #
    # Load previous saved state (HDF5)
    #
    ###################################################################################################################
    def load(self, filepath: str | Path, verbose: bool = False) -> None:
        """Load observation state from an HDF5 file.

        Args:
            filepath: Path to the HDF5 file.
            verbose: If True, print progress messages.
        """

        state = load_hdf5(filepath, verbose=verbose)
        self._load_from_state(state)

    ####################################################################################################################
    #
    # SPICE / Geometry
    #
    ####################################################################################################################
    def recompute_geometry(self, verbose: bool = False) -> None:
        """Recompute spacecraft and target geometry using SPICE.

        Args:
            verbose: If True, print progress messages.

        Raises:
            ValueError: If observation years have not been determined.
        """
        if self.years is None:
            raise ValueError("Observation years not determined. "
                             "Read files first.")

        mk_files = find_mk_files(
            self.metadata.instrument,
            years=self.years,
            mk_path=self.spice.mk_path,
        )
        for mk in mk_files:
            if verbose:
                print(f"Loading metakernel: {mk}")
            furnish(str(mk))

        if verbose:
            print("Computing geometry...")

        if self.geometry.et is None:
            et = np.asarray(utc2et(self.geometry.epoch), dtype=np.float64)
        else:
            et = self.geometry.et

        self.geometry = compute_geometry(et,
                                         self.spice.observer_str,
                                         self.spice.target_str,
                                         self.spice.fix_ref,
                                         ab_corr=self.spice.ab_corr,
                                         intercept_method_str=self.spice.method,
                                         compute_illum=True)
        if verbose:
            print("Geometry computation complete.")

    ####################################################################################################################
    #
    # Processing Methods
    #
    ####################################################################################################################
    def ionospheric_correction(self,
                              method: str | None = None,
                              n_take: int | None = None,
                              b: float | None = None,
                              n_phase: int | None = None,
                              delta: float | None = None,
                              sgn: Literal[-1, 1] | None = None,
                              window_type: str | None = None,
                              window_alpha: float | None = None,
                              L_eq: float | None = None,
                              metric: str | None = None,
                              save_data: bool | None = None,
                              save_images: bool | None = None,
                              save_state: bool | None = None,
                              verbose: bool | None = None):
        """Apply ionospheric distortion correction to the echoes.

        Parameters fall back to ``processing_parameters.iono_comp``
        if not provided, then to dataclass defaults.

        Args:
            method: Correction method (``"CAMPBELL"`` or ``"CONTRAST"``).
            n_take: Neighborhood width for chunked processing.
            b: Power-law exponent for the Campbell phase model.
            n_phase: Number of phase states for Campbell method.
            delta: Step size for Campbell E-values.
            sgn: Sign convention for phase exponential.
            window_type: Spectral window type.
            window_alpha: Tukey taper fraction.
            L_eq: Equivalent path length scale in meters.
            save_data: If True, export data after ionospheric correction.
            save_images: If True, export images after ionospheric correction.
            save_state: If True, save HDF5 state after ionospheric correction.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        ic = pp.iono_comp
        md = self.metadata
        params = self.parameters
        verbose = verbose if verbose is not None else pp.verbose

        if not ic.enabled:
            if verbose:
                print("Ionospheric Compensation is disabled. Skipping.")
            return

        method = method if method is not None else ic.method
        n_take = n_take if n_take is not None else ic.n_take
        b = b if b is not None else ic.campbell_b
        n_phase = n_phase if n_phase is not None else ic.campbell_n_phase
        delta = delta if delta is not None else ic.campbell_delta
        sgn = sgn if sgn is not None else ic.sgn
        window_type = window_type if window_type is not None else ic.window
        window_alpha = window_alpha if window_alpha is not None else ic.window_alpha
        L_eq = L_eq if L_eq is not None else ic.contrast_L_eq
        metric = metric if metric is not None else ic.metric
        save_data = save_data if save_data is not None else ic.save_data
        save_images = save_images if save_images is not None else ic.save_images
        save_state = save_state if save_state is not None else ic.save_state

        if verbose:
            print(f"Applying ionospheric correction ({method})...")

        dt = self.parameters.dt
        bw = self.parameters.bw
        f_cen = params.f_cen[0]  # The actual carrier frequency
        if md.instrument == "SHARAD":
            sgn = -1
            if n_take is None:
                n_take = 8192 // self.parameters.presum

        self.data, self.ionosphere = _iono_comp(self.data, method=method, f_cen=f_cen, dt=dt, bw=bw,
                                                n_take=n_take, wnd=window_type, alpha=window_alpha, L_eq=L_eq,
                                                campbell_b=b, n_phase=n_phase, campbell_delta=delta, sgn=sgn,
                                                metric=metric)
        method_str = method.lower()
        if save_data:
            self.export_data(level="iono", method=method_str, verbose=verbose)
        if save_images:
            self.export_images(level="iono", method=method_str, verbose=verbose)
        if save_state:
            base = self._build_base("iono", method_str, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)


    def emi_suppression(self,
                        method: str | None = None,
                        statistic: str | None = None,
                        k: float | None = None,
                        window_size: int | None = None,
                        replace_strategy: str | None = None,
                        interp_pad: int | None = None,
                        save_data: bool | None = None,
                        save_images: bool | None = None,
                        save_state: bool | None = None,
                        verbose: bool | None = None):
        """Suppress electromagnetic interference in echoes.

        Parameters fall back to ``processing_parameters.emi_suppression``
        if not provided, then to dataclass defaults.

        Args:
            method: EMI suppression method.
            statistic: Local-statistic method used to form the threshold ("MAD" or "STD").
            k: Threshold multiplier for outlier detection.
            window_size: Window size for local statistics.
            replace_strategy: Replacement strategy for ADAPTIVE.
            interp_pad: Interpolation anchor padding.
            save_data: If True, export data after EMI suppression.
            save_images: If True, export images after EMI suppression.
            save_state: If True, save HDF5 state after EMI suppression.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        emi = pp.emi_suppression
        params = self.parameters
        md = self.metadata
        verbose = verbose if verbose is not None else pp.verbose

        if not emi.enabled:
            if verbose:
                print("EMI Suppression is disabled. Skipping.")
            return

        method = method if method is not None else emi.method
        statistic = statistic if statistic is not None else emi.statistic
        k = k if k is not None else emi.k
        window_size = window_size if window_size is not None else emi.window_size
        replace_strategy = replace_strategy if replace_strategy is not None else emi.replace_strategy
        interp_pad = interp_pad if interp_pad is not None else emi.interp_pad
        save_data = save_data if save_data is not None else emi.save_data
        save_images = save_images if save_images is not None else emi.save_images
        save_state = save_state if save_state is not None else emi.save_state

        if verbose:
            print(f"Applying EMI suppression ({method})...")

        nfft = params.nfft
        dt = params.dt
        bw = params.bw
        #
        # In the RadarSounder class, anything reaching here is assumed to be in complex baseband.
        # f_cen is forced to 0.0.
        #
        f_cen = 0.0
        self.data, self.emi_mask = _suppress_emi(self.data, nfft=nfft, dt=dt, bw=bw, f_cen=f_cen, method=method,
                                                 statistic=statistic, k=k, window_size=window_size,
                                                 replace=replace_strategy, interp_pad=interp_pad, input_time=True,
                                                 output_time=True)
        method_str = method.lower()
        if save_data:
            self.export_data(level="emi", method=method_str, verbose=verbose)
        if save_images:
            self.export_images(level="emi", method=method_str, verbose=verbose)
        if save_state:
            base = self._build_base("emi", method_str, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

    def sar_process(self,
                    method: str | None = None,
                    L_n: int | None = None,
                    lamb: float | None = None,
                    os_factor: int | None = None,
                    window_type: str | None = None,
                    window_alpha: float | None = None,
                    n_mlk: int | None = None,
                    remove_doppler_centroid: bool | None = None,
                    interp_cval: float | None = None,
                    coherent: bool | None = None,
                    save_data: bool | None = None,
                    save_images: bool | None = None,
                    save_state: bool | None = None,
                    verbose: bool | None = None):
        """Apply SAR (azimuth) processing to the echoes.

            Parameters fall back to ``processing_parameters.sar``
            if not provided, then to dataclass defaults.

            Args:
                method: SAR method (``"UNFOCUSED"``, ``"RANGE-DOPPLER"``,
                    or ``"BACKSCATTER"``).
                L_n: Synthetic aperture length in samples.
                lamb: Wavelength to use in SAR Processing
                os_factor: Oversampling factor for the output grid.
                window_type: Azimuth window type.
                window_alpha: Tukey taper fraction.
                n_mlk: Number of multilook Doppler bins (backscatter only).
                remove_doppler_centroid: If True, demodulate Doppler centroid.
                interp_cval: Fill value for out-of-bounds RCMC interpolation.
                coherent: If True, return complex output.
                save_data: If True, export data after SAR processing.
                save_images: If True, export images after SAR processing.
                save_state: If True, save HDF5 state after SAR processing.
                verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        sp = pp.sar
        params = self.parameters
        verbose = verbose if verbose is not None else pp.verbose

        if not sp.enabled:
            if verbose:
                print("SAR Processing is disabled. Skipping.")
            return

        if self.geometry is None:
            raise ValueError("Geometry must be computed before SAR processing. "
                             "Call recompute_geometry() first.")

        if self.geometry.r_st is None:
            raise ValueError("Full geometry must be computed before SAR processing. "
                             "Call recompute_geometry() first.")
        geo = self.geometry
        Vt = geo.v_tangential
        Vr = geo.v_radial
        r_st = geo.r_st
        r_s = geo.r_s
        r_t = geo.r_t

        method = method if method is not None else sp.method
        L_n = L_n if L_n is not None else sp.aperture_length
        if lamb is None:
            lamb = sp.lamb if sp.lamb is not None else C / params.f_cen[0]
        os_factor = os_factor if os_factor is not None else sp.os_factor
        window_type = window_type if window_type is not None else sp.window
        window_alpha = window_alpha if window_alpha is not None else sp.window_alpha
        n_mlk = n_mlk if n_mlk is not None else sp.number_of_looks
        remove_doppler_centroid = remove_doppler_centroid if remove_doppler_centroid is not None else sp.remove_doppler_centroid
        interp_cval = interp_cval if interp_cval is not None else sp.interp_cval
        coherent = coherent if coherent is not None else sp.coherent
        save_data = save_data if save_data is not None else sp.save_data
        save_images = save_images if save_images is not None else sp.save_images
        save_state = save_state if save_state is not None else sp.save_state

        pri = params.pri if params.pri is not None else 1
        dz = params.dz
        D = params.l_a
        presum = params.presum if params.presum is not None else 1

        method = method.upper()
        if self.metadata.instrument == 'SHARAD':
            sgn = -1
        else:
            sgn = +1

        self.data, self.sar = _sar_process(self.data, method=method, presum=presum, pri=pri, dz=dz, lamb=lamb,
                                           l_n=L_n, d=D, os_factor=os_factor, number_of_looks=n_mlk,
                                           remove_doppler_centroid=remove_doppler_centroid, coherent=coherent,
                                           r_st=r_st, r_s=r_s, r_t=r_t, vt=Vt, vr=Vr, window_type=window_type,
                                           window_alpha=window_alpha, interp_cval=interp_cval, sgn=sgn)
        frames = self.sar.frames
        self._decimate_to_frames(frames, skip_keys=['ECHO_SAMPLES', 'WAVEFORM'], verbose=verbose)

        method_str = method.lower()
        if save_data:
            self.export_data(level="sar", method=method_str, verbose=verbose)
        if save_images:
            self.export_images(level="sar", method=method_str, verbose=verbose)
        if save_state:
            base = self._build_base("sar", method_str, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)


    def multilook(self,
                  n_looks: int | None = None,
                  os_factor: int | None = None,
                  window_type: str | None = None,
                  window_alpha: float | None = None,
                  coherent: bool | None = None,
                  save_data: bool | None = None,
                  save_images: bool | None = None,
                  save_state: bool | None = None,
                  verbose: bool | None = None):
        """Apply multilook averaging to focused SAR data.

        Parameters fall back to ``processing_parameters.mlk``
        if not provided, then to dataclass defaults.

        Args:
            n_looks: Number of looks (window length).
            os_factor: Oversampling factor for the output grid.
            window_type: Window type.
            window_alpha: Tukey taper fraction.
            coherent: If True, average complex data preserving phase.
            save_data: If True, export data after multilooking.
            save_images: If True, export images after multilooking.
            save_state: If True, save HDF5 state after multilooking.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        md = self.metadata
        mk = pp.mlk
        verbose = verbose if verbose is not None else pp.verbose

        if not mk.enabled:
            if verbose:
                print("Multilook is disabled. Skipping.")
            return

        if self.sar is None:
            raise ValueError("SAR processing must be run before multilooking.")

        if self.sar.method == "BACKSCATTER":
            warnings.warn("Backscatter output is already multilooked. Skipping.")
            return

        if self.sar.method == "UNFOCUSED":
            warnings.warn("Multilooking is not applicable to unfocused SAR. Skipping.")
            return

        if self.metadata.product_type != "EDR":
            raise NotImplementedError("Multilooking not ready for RDRs.")

        n_looks = n_looks if n_looks is not None else mk.number_of_looks
        os_factor = os_factor if os_factor is not None else mk.os_factor
        window_type = window_type if window_type is not None else mk.window_type
        window_alpha = window_alpha if window_alpha is not None else mk.window_alpha
        coherent = coherent if coherent is not None else mk.coherent
        save_data = save_data if save_data is not None else mk.save_data
        save_images = save_images if save_images is not None else mk.save_images
        save_state = save_state if save_state is not None else mk.save_state

        s_mlk, frames, rho_mlk =  _multilook(self.data, rho_a=self.sar.rho_a, n_looks=n_looks, os_factor=os_factor,
                                             window_type=window_type, window_alpha=window_alpha, coherent=coherent,
                                             verbose=verbose)
        self.data = s_mlk
        self.sar = replace(self.sar,
                           frames=frames,
                           rho_df=rho_mlk)
        self._decimate_to_frames(frames, skip_keys=['ECHO_SAMPLES', 'WAVEFORM'], verbose=verbose)

        if save_data:
            self.export_data(level="mlk", verbose=verbose)
        if save_images:
            self.export_images(level="mlk", verbose=verbose)
        if save_state:
            base = self._build_base("mlk", None, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

    ################################################################################################################
    #
    # Simulation Methods
    #
    ################################################################################################################
    def simulate_clutter(self,
                         dem_path: str | None = None,
                         at_step: float | None = None,
                         at_dist: float | None = None,
                         ct_step: float | None = None,
                         ct_dist: float | None = None,
                         n_center: int | None = None,
                         save_data: bool | None = None,
                         save_images: bool | None = None,
                         save_state: bool | None = None,
                         apply_curve: bool | None = None,
                         verbose: bool | None = None):
        """Simulate surface clutter from a DEM.

        Parameters fall back to clutter simulation parameters
        if not provided, then to dataclass defaults.

        Args:
            dem_path: Path to a rasterio-readable DEM file.
            at_step: Along-track facet dimension in metres.
            at_dist: Along-track half-extent of the grid in metres.
            ct_step: Cross-track facet dimension in metres.
            ct_dist: Cross-track extent from nadir in metres.
            n_center: Sample index at which the ellipsoid surface return is centred.
            save_data: If True, export data after simulation.
            save_images: If True, export images after simulation.
            save_state: If True, save HDF5 state after simulation.
            apply_curve: If True, apply the simc tone-mapping LUT to
                the cluttergram and echomap.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        cs = self._get_csp()
        verbose = verbose if verbose is not None else pp.verbose

        if not cs.enabled:
            if verbose:
                print("Clutter simulation is disabled. Skipping.")
            return

        dem_path = dem_path if dem_path is not None else cs.dem_path
        at_step = at_step if at_step is not None else cs.at_step
        at_dist = at_dist if at_dist is not None else cs.at_dist
        ct_step = ct_step if ct_step is not None else cs.ct_step
        ct_dist = ct_dist if ct_dist is not None else cs.ct_dist
        n_center = n_center if n_center is not None else cs.n_center
        save_data = save_data if save_data is not None else cs.save_data
        save_images = save_images if save_images is not None else cs.save_images
        save_state = save_state if save_state is not None else cs.save_state
        apply_curve = apply_curve if apply_curve is not None else (cs.apply_curve if cs.apply_curve is not None else True)

        if dem_path is None:
            raise ValueError("dem_path must be provided or set in processing parameters.")

        #
        # If the SAR Processor was run, update the at_dist and at_step parameters to better match the
        # resolution of the SAR processed results
        #
        if at_dist is None and self.sar is not None:
            if self.sar.rho_df is None:
                at_dist = float(np.floor(self.sar.rho_a.mean()))
            else:
                at_dist = float(np.floor(self.sar.rho_df.mean()))
            at_step = at_dist / 2

        self.clutter_result = _simulate_clutter(
            lat=self.geometry.latitude,
            lon=self.geometry.longitude,
            alt=self.geometry.altitude,
            datum=self.datum,
            dem_path=dem_path,
            target=self.spice.target_str.upper(),
            bin_size=cs.bin_size,
            n_samples=cs.n_samples,
            at_step=at_step,
            at_dist=at_dist,
            ct_step=ct_step,
            ct_dist=ct_dist,
            n_center=n_center,
        )

        if save_data:
            base = self._build_base("csim", None, None)
            _export_csim_data(
                self.clutter_result,
                self.geometry,
                out_dir=Path(pp.out_dir),
                base=base,
                instrument=self.metadata.instrument,
                bin_size=cs.bin_size,
                byte_order=pp.output.byte_order,
                verbose=verbose,
            )
        if save_images:
            base = self._build_base("csim", None, None)
            # Radar-equation column weights: compensate cluttergram
            # brightness for per-trace R^4 path loss so column ceilings
            # don't track spacecraft altitude. Normalised by the minimum
            # altitude so the closest trace gets weight 1.
            altitude = np.asarray(self.geometry.altitude, dtype=np.float64)
            alt_min = altitude.min()
            column_weights = (altitude / alt_min) ** 4 if alt_min > 0 else None

            _export_csim_images(
                self.clutter_result,
                out_dir=Path(pp.out_dir),
                base=base,
                column_weights=column_weights,
                apply_curve=apply_curve,
                invert=pp.plots.invert,
                verbose=verbose,
            )
        if save_state:
            base = self._build_base("csim", None, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

    ################################################################################################################
    #
    # Final Output
    #
    ################################################################################################################
    def finalize(self,
                 save_data: bool | None = None,
                 save_images: bool | None = None,
                 save_segy: bool | None = None,
                 save_state: bool | None = None,
                 verbose: bool | None = None):
        """Export the current state of ``self.data`` as the final product.

        Writes data files, browse images, SEG-Y, and/or HDF5 state based on
        ``processing_parameters.final_output``. Has no dependence on which
        stages ran upstream — exports whatever is currently in ``self.data``.

        The filename uses a bare prefix (no level/method suffix). The prefix
        letter reflects whether SAR processing has been applied:
        ``r_`` if ``self.sar`` has been populated (i.e. SAR or multilook ran),
        otherwise the input product type's prefix (``e_`` for EDR, ``r_`` for
        RDR).

        Args:
            save_data: If True, export data files. Falls back to
                ``processing_parameters.final_output.save_data``.
            save_images: If True, export browse images. Falls back to
                ``processing_parameters.final_output.save_images``.
            save_segy: If True, export SEG-Y. Falls back to
                ``processing_parameters.final_output.save_segy``.
            save_state: If True, save HDF5 state. Falls back to
                ``processing_parameters.final_output.save_state``.
            verbose: If True, print progress messages. Falls back to
                ``processing_parameters.verbose``.
        """
        pp = self._get_pp()
        fo = pp.final_output
        verbose = verbose if verbose is not None else pp.verbose

        save_data = save_data if save_data is not None else fo.save_data
        save_images = save_images if save_images is not None else fo.save_images
        save_segy = save_segy if save_segy is not None else fo.save_segy
        save_state = save_state if save_state is not None else fo.save_state

        if not (save_data or save_images or save_segy or save_state):
            if verbose:
                print("Finalize has nothing to export. Skipping.")
            return

        if verbose:
            print("Finalizing outputs...")

        level = "sar" if self.sar is not None else self.metadata.product_type.lower()
        base = self._build_base(level, None, None, bare=True)

        if save_data:
            self.export_data(prefix=base, verbose=verbose)
        if save_images:
            self.export_images(prefix=base, verbose=verbose)
        if save_segy:
            self.export_segy(prefix=base, verbose=verbose)
        if save_state:
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)
    ################################################################################################################
    #
    # Output Methods
    #
    ################################################################################################################
    def save(self, filepath: str | Path, verbose: bool | None = None) -> None:
        """Save the full observation state to an HDF5 file.

        Args:
            filepath: Output HDF5 file path.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        verbose = verbose if verbose is not None else pp.verbose

        state = self._build_state_dict()
        save_hdf5(filepath, state, verbose=verbose)

    def export_data(self,
                    level: str | None = None,
                    method: str | None = None,
                    output_dir: str | Path | None = None,
                    output_format: str | None = None,
                    prefix: str | None = None,
                    byte_order: str | None = None,
                    verbose: bool | None = None):
        """Export processed data to output files.

        Parameters fall back to ``processing_parameters`` if not
        provided, then to defaults. If ``level`` is None, defaults
        to the product type. If ``method`` is None, the method
        suffix is omitted from the filename.

        Args:
            level: Processing level string.
            method: Processing method string.
            output_dir: Directory to write output files.
            output_format: Output format (default ``"basic"``).
            prefix: Filename prefix.
            byte_order: Byte order for the binary file.
            verbose: If True, print progress messages.
        """
        if self.geometry is None:
            raise RuntimeError("Geometry not available. Call read_auxiliary_file() first.")

        pp = self._get_pp()
        output_dir = Path(output_dir) if output_dir is not None else Path(pp.out_dir)
        output_format = output_format if output_format is not None else pp.output.data_output_type
        byte_order = byte_order if byte_order is not None else pp.output.byte_order
        verbose = verbose if verbose is not None else pp.verbose

        if level is None:
            level = self.metadata.product_type.lower()

        base = self._build_base(level, method, prefix)

        _export_data(self.data, self.geometry, out_dir=output_dir, base=base, instrument=self.metadata.instrument,
                     dt=self.parameters.dt, pri=self.parameters.pri, presum=self.parameters.presum or 1, sar=self.sar,
                     ionosphere=self.ionosphere, output_format=output_format, byte_order=byte_order, verbose=verbose,)

    def export_images(self,
                      level: str | None = None,
                      method: str | None = None,
                      output_dir: str | Path | None = None,
                      prefix: str | None = None,
                      dem: str | Path | None = None,
                      buffer_km: float | None = None,
                      lower_percentile: float | None = None,
                      upper_percentile: float | None = None,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      power: bool | None = None,
                      invert: bool | None = None,
                      per_frame: bool | None = None,
                      verbose: bool | None = None):
        """Export radargram and browse images.

        Parameters fall back to ``processing_parameters.plots``
        if not provided, then to defaults. If ``level`` is None,
        defaults to the product type. If ``method`` is None, the
        method suffix is omitted from the filename.

        Args:
            level: Processing level string.
            method: Processing method string.
            output_dir: Directory to write output files.
            prefix: Filename prefix.
            dem: DEM source for the browse image.
            buffer_km: Half-width of the DEM swath in kilometers.
            lower_percentile: Lower percentile for image scaling.
            upper_percentile: Upper percentile for image scaling.
            vmin: Explicit minimum dB value for scaling.
            vmax: Explicit maximum dB value for scaling.
            power: If True, use ``10 * log10`` for scaling.
            invert: If True, invert the grayscale.
            per_frame: If True, compute vmin/vmax per column
                (column-wise frame normalisation).
            verbose: If True, print progress messages.
        """
        DEFAULT_DEM = {"MARS": "HRSC-MOLA", "MOON": "LOLA"}
        if self.geometry is None:
            raise RuntimeError("Geometry not available. Call read_auxiliary_file() first.")

        pp = self._get_pp()
        pl = pp.plots

        output_dir = Path(output_dir) if output_dir is not None else Path(pp.out_dir)
        dem = dem if dem is not None else DEFAULT_DEM.get(self.spice.target_str, None)
        buffer_km = buffer_km if buffer_km is not None else (pl.buffer_km if pl.buffer_km is not None else 25.0)
        lower_percentile = lower_percentile if lower_percentile is not None else pl.lower_percentile
        upper_percentile = upper_percentile if upper_percentile is not None else pl.upper_percentile
        vmin = vmin if vmin is not None else pl.vmin
        vmax = vmax if vmax is not None else pl.vmax
        invert = invert if invert is not None else pl.invert
        per_frame = per_frame if per_frame is not None else (pl.per_frame if pl.per_frame is not None else False)
        verbose = verbose if verbose is not None else pp.verbose

        if level is None:
            level = self.metadata.product_type.lower()

        base = self._build_base(level, method, prefix)

        _export_images(self.data, self.geometry, self.crs, out_dir=output_dir, base=base, dem=dem, buffer_km=buffer_km,
                       lower_percentile=lower_percentile, upper_percentile=upper_percentile, vmin=vmin, vmax=vmax,
                       power=power, invert=invert, per_frame=per_frame, verbose=verbose)

    def export_segy(self,
                    level: str | None = None,
                    method: str | None = None,
                    output_dir: str | Path | None = None,
                    prefix: str | None = None,
                    tgt_crs: str | _CRS | Path | None = None,
                    verbose: bool | None = None):
        """Export radargram to SEG-Y format.

        Parameters fall back to ``processing_parameters`` if not
        provided, then to defaults. If ``level`` is None, defaults
        to the product type. If ``method`` is None, the method
        suffix is omitted from the filename.

        Args:
            level: Processing level string.
            method: Processing method string.
            output_dir: Directory to write output file.
            prefix: Filename prefix.
            tgt_crs: Target CRS for reprojection. Accepts a
                ``pyproj.CRS``, an authority string (e.g.
                ``"ESRI:104971"``), or a ``Path`` to a WKT/PRJ
                file. If ``None``, falls back to
                ``processing_parameters.output.tgt_crs``.
            verbose: If True, print progress messages.
        TODO: Known bug. Export broken for MARSIS
        """
        if self.geometry is None:
            raise RuntimeError("Geometry not available. Call read_auxiliary_file() first.")

        pp = self._get_pp()
        output_dir = Path(output_dir) if output_dir is not None else Path(pp.out_dir)
        tgt_crs = tgt_crs if tgt_crs is not None else pp.output.tgt_crs
        verbose = verbose if verbose is not None else pp.verbose

        if level is None:
            level = self.metadata.product_type.lower()

        base = self._build_base(level, method, prefix)

        _export_segy(
            self.data,
            self.geometry.latitude,
            self.geometry.longitude,
            self.geometry.epoch[0],
            out_dir=output_dir,
            base=base,
            dt=self.parameters.dt,
            instrument=self.metadata.instrument,
            product_id=self.metadata.product_id,
            product_type=self.metadata.product_type,
            src_crs=self.crs,
            tgt_crs=normalize_crs(tgt_crs),
            verbose=verbose,
        )

    def export(self,
               level: str | None = None,
               method: str | None = None,
               save_data: bool = True,
               save_images: bool = True,
               prefix: str | None = None,
               verbose: bool | None = None,
               **kwargs):
        """Export data files and images.

        Convenience wrapper that calls :meth:`export_data` and
        :meth:`export_images`. All parameters fall back to
        ``processing_parameters`` if not provided. If ``level``
        is None, defaults to the product type. If ``method`` is
        None, the method suffix is omitted from filenames.

        Args:
            level: Processing level string.
            method: Processing method string.
            save_data: If True, export data.
            save_images: If True, export images.
            prefix: Filename prefix.
            verbose: If True, print progress messages.
        """
        if self.geometry is None:
            raise RuntimeError("Geometry not available. Call read_auxiliary_file() first.")

        if save_data:
            self.export_data(level, method,
                             prefix=prefix,
                             verbose=verbose,
                             **kwargs)
        if save_images:
            self.export_images(level, method,
                               prefix=prefix,
                               verbose=verbose,
                               **kwargs)
    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------
    @abstractmethod
    def check_supported(self):
        """Check whether the product is supported by this class."""
        pass

    @abstractmethod
    def preprocess(self, **kwargs):
        """Preprocess the data.

        Each instrument has its own preprocessing steps.

        LRS: lrs/preprocessing.py
        MARSIS: marsis/preprocessing.py
        SHARAD: sharad/preprocessing.py
        """
        pass

    @abstractmethod
    def range_compression(self, **kwargs):
        """Apply range compression to the science data."""
        pass

    ##################################################################################################################
    #
    # Helpers
    #
    ##################################################################################################################
    def _get_pp(self) -> ProcessingParameters:
        """Return processing parameters, using defaults if not set.

        If ``processing_parameters`` is None, issues a warning and
        creates a default instance with ``out_dir`` set to the
        current working directory.

        Also calls :meth:`validate_processing_parameters` to check
        for unsupported or untested processing stages.

        Returns:
            The current ProcessingParameters instance.
        """
        if self.processing_parameters is None:
            warnings.warn(
                "No ProcessingParameters provided. Using defaults.",
                UserWarning,
            )
            self.processing_parameters = ProcessingParameters(
                out_dir=Path("."),
            )
        assert self.processing_parameters is not None

        self.validate_processing_parameters()

        return self.processing_parameters

    def _get_csp(self) -> ClutterSimParams:
        """Return merged clutter simulation parameters.

        Starts from instrument-derived defaults (TOML defaults via
        resolve_instrument_defaults) and overlays any non-None
        fields from ``processing_parameters.csim``, so TOML overrides
        win while computed fields are preserved. The result is cached
        on ``clutter_sim_params``.

        Returns:
            The merged ClutterSimParams instance.
        """
        base = self._default_clutter_params()  # instrument-derived fields
        if base is None:
            raise ValueError(
                f"No clutter defaults for target '{self.spice.target_str}'."
            )
        user = self.processing_parameters.csim if self.processing_parameters else None
        if user is None:
            self.clutter_sim_params = base
            return base
        # overlay only the user-set TOML fields onto the instrument defaults
        overrides = {f.name: getattr(user, f.name)
                     for f in fields(user)
                     if getattr(user, f.name) is not None}
        merged = replace(base, **overrides)
        self.clutter_sim_params = merged
        return merged

    def _decimate_to_frames(self, indices=None, factor=None,
                            skip_keys=None, verbose=False):
        """Decimate science, auxiliary, and geometry data.

        Slices all per-trace arrays in science_data, auxiliary_data,
        and geometry.

        Args:
            indices: Array of trace indices to retain.
            factor: Decimation factor (group size).
            skip_keys: Keys in science_data to skip (already replaced
                by the calling method).
            verbose: If True, print progress messages.
        """
        if verbose:
            print("Decimating ancillary data...")

        self.science_data = decimate_dict(
            self.science_data, factor=factor, indices=indices,
            skip_keys=skip_keys or [])

        self.auxiliary_data = decimate_dict(
            self.auxiliary_data, factor=factor, indices=indices)

        if self.geometry is not None:
            self.geometry = decimate_dataclass(
                self.geometry, factor=factor, indices=indices)

        if self.datum is not None:
            if indices is not None:
                self.datum = self.datum[indices]
            elif factor is not None:
                self.datum = self.datum[::factor]

        if verbose:
            print("Decimation complete.")

    def _build_state_dict(self) -> dict[str, dict]:
        """Build a state dict from class attributes for HDF5 serialization."""
        from dataclasses import asdict

        state = {
            "science_data": self.science_data,
            "auxiliary_data": self.auxiliary_data,
        }

        for name in ("metadata", "parameters", "record_format", "spice"):
            obj = getattr(self, name)
            if obj is not None:
                state[name] = asdict(obj)

        if self.geometry is not None:
            state["geometry"] = asdict(self.geometry)

        if self.processing_parameters is not None:
            d = asdict(self.processing_parameters)
            d["out_dir"] = str(self.processing_parameters.out_dir)
            state["processing_parameters"] = d

        if self.ionosphere is not None:
            d = asdict(self.ionosphere)
            d["__type__"] = type(self.ionosphere).__name__
            state["ionosphere"] = d

        if self.sar is not None:
            state["sar"] = asdict(self.sar)

        if self.datum is not None:
            state["datum"] = {"data": self.datum}

        if self.years is not None:
            state["years"] = {"data": self.years}

        return state

    def _load_from_state(self, state: dict[str, dict]) -> None:
        """Populate class attributes from a state dict.

        Args:
            state: Dict-of-dicts as returned by :func:`load_hdf5`.
        """
        from .grasp_types import (
            MetaDataInfo, RadarParams, RecordFormat, SpiceParams,
            GeometryResult, ProcessingParameters, PreprocessingParams,
            RangeCompressionParams, EMISuppresionParams, IonoCompParams,
            SARParams as SARParamsType, MLKParams, OutputParameters,
            PlotParameters, CampbellResult, ContrastResult, SARResult,
        )

        if "metadata" in state:
            self.metadata = MetaDataInfo(**state["metadata"])

        if "parameters" in state:
            self.parameters = RadarParams(**state["parameters"])

        if "record_format" in state:
            self.record_format = RecordFormat(**state["record_format"])

        if "spice" in state:
            self.spice = SpiceParams(**state["spice"])
            # Rebuild CRS
            self.set_spice(self.metadata.platform)

        if "science_data" in state:
            self.science_data = state["science_data"]

        if "auxiliary_data" in state:
            self.auxiliary_data = state["auxiliary_data"]

        if "geometry" in state:
            self.geometry = GeometryResult(**state["geometry"])

        if "processing_parameters" in state:
            pp = state["processing_parameters"]
            self.processing_parameters = ProcessingParameters(
                out_dir=Path(pp.pop("out_dir", ".")),
                verbose=pp.pop("verbose", False),
                preprocessing=PreprocessingParams(**pp.pop("preprocessing", {})),
                range_compression=RangeCompressionParams(**pp.pop("range_compression", {})),
                emi_suppression=EMISuppresionParams(**pp.pop("emi_suppression", {})),
                iono_comp=IonoCompParams(**pp.pop("iono_comp", {})),
                sar=SARParamsType(**pp.pop("sar", {})),
                mlk=MLKParams(**pp.pop("mlk", {})),
                output=OutputParameters(**pp.pop("output", {})),
                plots=PlotParameters(**pp.pop("plots", {})),
            )

        if "ionosphere" in state:
            iono = state["ionosphere"]
            iono_type = iono.pop("__type__", None)
            if iono_type == "CampbellResult":
                self.ionosphere = CampbellResult(**iono)
            elif iono_type == "ContrastResult":
                self.ionosphere = ContrastResult(**iono)

        if "sar" in state:
            self.sar = SARResult(**state["sar"])

        # Replace with:
        if "datum" in state:
            self.datum = state["datum"]["data"]

        if "years" in state:
            self.years = state["years"]["data"]

    def _grab_parser(self):
        instrument = self.metadata.instrument.lower()
        mod = importlib.import_module(f".{instrument}.parsers", package="grasp")
        return mod.parse

    def _build_geometry(self) -> GeometryResult | None:
        md = self.metadata
        if md.product_type not in ("EDR", "RDR"):
            warnings.warn(f"Geometry build not supported for {md.product_type}.")
            return

        keys = self.GEOMETRY_KEYS.get(md.product_type)
        if keys is None:
            warnings.warn(f"No geometry keys defined for {md.product_type}.")
            return

        src = keys.get("source", "auxiliary")
        if src == "auxiliary":
            aux = self.auxiliary_data
        elif src == "science":
            aux = self.science_data
        else:
            raise ValueError(f"Unknown geometry source: {src}")

        scale = keys.get("scale", 1.0)
        epoch_key = keys.get("epoch")
        lat_key = keys.get("latitude")
        lon_key = keys.get("longitude")
        alt_key = keys.get("altitude")
        assert isinstance(lat_key, str)
        assert isinstance(lon_key, str)
        assert isinstance(alt_key, str)
        assert isinstance(epoch_key, str)
        latitude = aux[lat_key]
        longitude = aux[lon_key]
        altitude = aux[alt_key] * scale
        epoch = aux[epoch_key]

        # Optional fields — not all instruments provide these from file data
        et_key = keys.get("et")
        et = aux[et_key] if isinstance(et_key, str) else None

        sza_key = keys.get("sza")
        sza = aux[sza_key] if isinstance(sza_key, str) else None

        r_s = None
        v_s = None
        sc_radius = None
        v_radial = None
        v_tangential = None

        pos_key = keys.get("position")

        if pos_key is not None:
            if isinstance(pos_key, list):
                r_s = np.array([aux[k] for k in pos_key], dtype=np.float32) * scale
            else:
                assert isinstance(pos_key, str)
                r_s = (scale * aux[pos_key]).astype(np.float32)
            sc_radius = np.linalg.norm(r_s, axis=0)

        vel_key = keys.get("velocity")
        if vel_key is not None:
            if isinstance(vel_key, list):
                v_s = np.array([aux[k] for k in vel_key], dtype=np.float32) * scale
            else:
                assert isinstance(vel_key, str)
                v_s = (scale * aux[vel_key]).astype(np.float32)

        if keys.get("v_radial"):
            v_radial = (scale * aux[keys["v_radial"]]).astype(np.float32)
        if keys.get("v_tangential"):
            v_tangential = (scale * aux[keys["v_tangential"]]).astype(np.float32)

        return GeometryResult(
            et=et,
            epoch=epoch,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            sc_radius=sc_radius,
            r_s=r_s,
            v_s=v_s,
            v_radial=v_radial,
            v_tangential=v_tangential,
            solar_zenith_angle=sza,
        )

    def _build_prefix(self, level: str, prefix: str | None) -> str:
        """Build the filename prefix from metadata or return user-provided prefix.

        Args:
            level: Processing level string.
            prefix: User-provided prefix, or None to auto-generate.

        Returns:
            Filename prefix string.
        """
        level_prefix = {
            "raw": "e",
            "preprocess": "i",
            "rc": "i",
            "emi": "i",
            "iono": "i",
            "sar": "r",
            "mlk": "r",
            "edr": "e",
            "rdr": "r",
            "csim": "s",
        }

        if level not in level_prefix:
            raise ValueError(f"Unsupported level '{level}'. Must be one of "
                             f"{list(level_prefix.keys())}")

        if prefix is None:
            lp = level_prefix[level]
            prefix = (f"{lp}_{self.metadata.instrument}"
                      f"_{self.metadata.product_id}"
                      f"_{self.metadata.operative_mode}").lower()

        return prefix

    def _build_base(self, level: str | None, method: str | None,
                    prefix: str | None, bare: bool = False) -> str:
        """Build the output base filename.

        Args:
            level: Processing level string, or None (defaults to product type).
            method: Processing method string, or None.
            prefix: User-provided prefix, or None to auto-generate.
            bare: If True, return just the prefix with no level/method suffix.
                Used by ``finalize`` to produce a clean end-of-pipeline name.

        Returns:
            Base filename string (without extension).
        """
        if level is None:
            level = self.metadata.product_type.lower()

        prefix = self._build_prefix(level, prefix)

        if bare:
            return prefix.lower()
        if method is not None:
            return f"{prefix}_{level}_{method}".lower()
        return f"{prefix}_{level}".lower()

    def _default_clutter_params(self) -> ClutterSimParams:
        sp = self.spice
        body = sp.target_str.upper()

        if body == "MARS":
            dem_path = str(get_data_path("mola_global"))
        elif body == "MOON":
            dem_path = str(get_data_path("lola_global"))
        else:
            dem_path = None

        try:
            n_samples = self.data.shape[0]
        except (NotImplementedError, KeyError):
            n_samples = self.parameters.nfft

        return ClutterSimParams(
            dem_path=dem_path,
            target=self.spice.target_str.upper(),
            bin_size=self.parameters.dt,
            n_samples=n_samples,
        )
