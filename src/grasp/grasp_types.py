# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Any
from numpy.typing import NDArray
import numpy as np
from pathlib import Path

########################################################################################################################
# Type aliases
########################################################################################################################
Spacecraft = Literal["SELENE", "MEX", "MRO"]
RadarInstrument: TypeAlias = Literal["LRS", "MARSIS", "SHARAD"]

########################################################################################################################
# Printable mixin
########################################################################################################################
class Printable:
    """Mixin that adds a ``print()`` method to any dataclass.

    When called, recursively prints all fields to stdout with
    indentation for nested dataclasses that also inherit from
    ``Printable``.
    """

    def print(self, _depth: int = 0) -> None:
        """Print all fields to stdout.

        Args:
            _depth: Current indentation level (used internally
                for recursive printing).
        """
        indent = "\t" * _depth
        print(f"{indent}{self.__class__.__name__}:")
        for k, v in vars(self).items():
            if hasattr(v, 'print'):
                print(f"{indent}\t{k}:")
                v.print(_depth=_depth + 2)
            else:
                print(f"{indent}\t{k}: {v}")

########################################################################################################################
# Dataclasses
########################################################################################################################
@dataclass(frozen=True)
class RadarParams(Printable):
    """Static radar configuration parameters.

    Holds instrument-level constants that do not change between
    observations for a given instrument mode. Populated during
    file identification and used throughout processing.

    Attributes:
        sci_key: Science data key identifier.
        fs: Sampling frequency in Hz.
        dt: Sample spacing in seconds.
        dz: Range resolution in metres.
        tau: Pulse duration in seconds.
        bw: Bandwidth in Hz.
        f_cen: Centre frequency per channel/filter (Hz).
        f_start: Start frequency per channel/filter (Hz).
        f_end: End frequency per channel/filter (Hz).
        n_samp: Number of samples per record.
        nfft: FFT length.
        prf: Pulse repetition frequency in Hz.
        pri: Pulse repetition interval in seconds.
        latency: Instrument latency per channel/filter (seconds).
        l_a: Antenna length in metres.
        presum: Onboard presumming factor.
        bits_per_sample: Quantisation bits per sample.
        n_filt: Number of Doppler filters.
        filt_str: String labels for each Doppler filter.
        n_chan: Number of receive channels.
        chan_str: String labels for each receive channel.
    """
    sci_key: str = ""
    fs: float = 0.0
    dt: float = 0.0
    dz: float = 0.0
    tau: float = 0.0
    bw: float = 0.0
    f_cen: list[float] = field(default_factory=list)
    f_start: list[float] = field(default_factory=list)
    f_end: list[float] = field(default_factory=list)
    n_samp: int = 0
    nfft: int = 0
    prf: float = 0.0
    pri: float = 0.0
    latency: list[float] = field(default_factory=list)
    l_a: float = 0.0
    presum: int | None = None
    bits_per_sample: int | None = None
    n_filt: int | None = None
    filt_str: list[str] | None = None
    n_chan: int | None = None
    chan_str: list[str] | None = None


@dataclass(frozen=True)
class MetaDataInfo(Printable):
    """Top-level radar observation metadata.

    Identifies the source mission, instrument, and product for a
    single radar sounder observation.

    Attributes:
        platform: Mission platform (e.g. ``"MRO"``, ``"MEX"``,
            ``"SELENE"``).
        instrument: Instrument identifier (e.g. ``"SHARAD"``).
        product_type: PDS product type (e.g. ``"EDR"``, ``"RDR"``).
        product_id: Observation product identifier.
        operative_mode: Operative mode string (e.g. ``"SS3"``).
        instrument_state: Instrument state (e.g. ``"TRK"``).
        data_form: Data form code (e.g. ``"CMP"``, ``"UNC"``).
    """
    platform: str = "Unknown"
    instrument: str = "Unknown"
    product_type: str = "Unknown"
    product_id: str = "Unknown"
    operative_mode: str = "Unknown"
    instrument_state: str = "TRK"
    data_form: str = "Unknown"


@dataclass
class GraspOutput(Printable):
    """In-memory representation of a GRaSP output triplet.

    Loaded by :func:`grasp.input.load.read_grasp_output` from a
    ``.grsp`` descriptor and its companion ``.img`` / ``.csv`` files.
    Carries everything needed to re-export the data to SEG-Y or any
    other supported format (except a target CRS, which must be
    supplied by the caller).

    Attributes:
        data: Radargram array, shape ``(n_samp, n_col)``. Either
            ``float32`` or ``complex64`` depending on the descriptor.
        instrument: Radar instrument identifier from the descriptor.
        product_id: Observation product identifier.
        dt: Range sample spacing in seconds.
        dx: Along-track sample spacing in metres.
        rho_a: Azimuth resolution in metres.
        frame_index: Per-trace frame index, shape ``(n_col,)``.
        ephemeris_time: SPICE ephemeris time per trace,
            shape ``(n_col,)``.
        geometry_epoch: UTC epoch string per trace, shape ``(n_col,)``.
        latitude: Latitude per trace in degrees, shape ``(n_col,)``.
        longitude: Longitude per trace in degrees, shape ``(n_col,)``.
        altitude: Spacecraft altitude per trace in metres,
            shape ``(n_col,)``.
        sc_radius: Spacecraft radius per trace in metres,
            shape ``(n_col,)``.
        el_radius: Ellipsoid radius per trace in metres,
            shape ``(n_col,)``.
        solar_zenith_angle: Solar zenith angle per trace in degrees,
            shape ``(n_col,)``. NaN-filled if the writer didn't have
            this field.
        iono_value: Ionosphere correction value per trace,
            shape ``(n_col,)``. NaN-filled if the writer didn't have
            this field.
    """
    data: NDArray[Any]
    instrument: str
    product_id: str
    dt: float
    dx: float
    rho_a: float
    frame_index: NDArray[np.integer[Any]]
    ephemeris_time: NDArray[np.floating[Any]]
    geometry_epoch: NDArray[Any]
    latitude: NDArray[np.floating[Any]]
    longitude: NDArray[np.floating[Any]]
    altitude: NDArray[np.floating[Any]]
    sc_radius: NDArray[np.floating[Any]]
    el_radius: NDArray[np.floating[Any]]
    solar_zenith_angle: NDArray[np.floating[Any]]
    iono_value: NDArray[np.floating[Any]]


@dataclass
class ClutterSimOutput(Printable):
    """In-memory representation of a CSIM (clutter simulation) triplet.

    Loaded by :func:`grasp.input.load.read_grasp_output` when the
    ``.grsp`` descriptor's CSV companion uses the CSIM schema
    (``TRACE_INDEX, LATITUDE, LONGITUDE, NADIR_TWTT, FRET_TWTT``)
    rather than the standard per-trace state schema.

    Attributes:
        data: Cluttergram array, shape ``(n_samp, n_col)``, ``float32``.
        instrument: Radar instrument identifier from the descriptor.
        product_id: Observation product identifier.
        dt: Range sample spacing in seconds.
        trace_index: Per-trace index, shape ``(n_col,)``.
        latitude: Latitude per trace in degrees, shape ``(n_col,)``.
        longitude: Longitude per trace in degrees, shape ``(n_col,)``.
        nadir_twtt: Nadir two-way travel time per trace in seconds,
            shape ``(n_col,)``.
        fret_twtt: First-return two-way travel time per trace in
            seconds, shape ``(n_col,)``. NaN for traces with no
            valid return.
    """
    data: NDArray[Any]
    instrument: str
    product_id: str
    dt: float
    trace_index: NDArray[np.integer[Any]]
    latitude: NDArray[np.floating[Any]]
    longitude: NDArray[np.floating[Any]]
    nadir_twtt: NDArray[np.floating[Any]]
    fret_twtt: NDArray[np.floating[Any]]


@dataclass(frozen=True)
class SpiceParams(Printable):
    """SPICE geometry and reference parameters for a spacecraft.

    Provides all NAIF identifiers, method strings, and kernel paths
    needed to compute observation geometry via SPICE.

    Attributes:
        target_int: NAIF integer ID for the target body.
        target_str: NAIF string name for the target body.
        observer_int: NAIF integer ID for the observer.
        observer_str: NAIF string name for the observer.
        method: SPICE method string for sub-observer point.
        method2: SPICE method string for illumination angles.
        ab_corr: Aberration correction string.
        fix_ref: Body-fixed reference frame string.
        mk_path: Path to the metakernel directory.
    """
    target_int: int = 0
    target_str: str = ""
    observer_int: int = 0
    observer_str: str = ""
    method: str = ""
    method2: str = ""
    ab_corr: str = ""
    fix_ref: str = ""
    mk_path: str = ""


@dataclass(frozen=True)
class GeometryResult(Printable):
    """Geometric, kinematic, and illumination quantities for an observation.

    Contains all navigation and geometry data for a radar sounder
    observation. Fields populated at construction time (epoch,
    longitude, latitude, altitude) are always present; remaining
    fields are populated by the SPICE geometry computation and
    default to ``None`` until computed.

    Attributes:
        epoch: UTC epochs, shape (n_col,).
        longitude: Sub-spacecraft longitude (rad), shape (n_col,).
        latitude: Sub-spacecraft latitude (rad), shape (n_col,).
        altitude: Spacecraft altitude (m), shape (n_col,).
        et: Ephemeris times (s past J2000), shape (n_col,).
        sc_radius: Spacecraft distance from body centre (m),
            shape (n_col,).
        el_radius: Local ellipsoid radius (m), shape (n_col,).
        r_s: Spacecraft position vectors (m), shape (n_col, 3).
        v_s: Spacecraft velocity vectors (m/s), shape (n_col, 3).
        r_t: Target position vectors (m), shape (n_col, 3).
        v_t: Target velocity vectors (m/s), shape (n_col, 3).
        r_st: Spacecraft-to-target vectors (m), shape (n_col, 3).
        v_radial: Radial speed (m/s), shape (n_col,).
        v_tangential: Tangential speed (m/s), shape (n_col,).
        phase_angle: Phase angle (rad), shape (n_col,).
        solar_zenith_angle: Solar incidence angle (rad),
            shape (n_col,).
        emission_angle: Emission angle (rad), shape (n_col,).
    """
    # Should be available in any grasp support data
    epoch: NDArray[np.datetime64]
    longitude: NDArray[np.floating[Any]]
    latitude: NDArray[np.floating[Any]]
    # Maybe?
    altitude: NDArray[np.floating[Any]]
    # Populated on computation
    et: NDArray[np.floating[Any]] | None = None
    sc_radius: NDArray[np.floating[Any]] | None = None
    el_radius: NDArray[np.floating[Any]] | None = None
    r_s: NDArray[np.floating[Any]] | None = None
    v_s: NDArray[np.floating[Any]] | None = None
    r_t: NDArray[np.floating[Any]] | None = None
    v_t: NDArray[np.floating[Any]] | None = None
    r_st: NDArray[np.floating[Any]] | None = None
    v_radial: NDArray[np.floating[Any]] | None = None
    v_tangential: NDArray[np.floating[Any]] | None = None
    phase_angle: NDArray[np.floating[Any]] | None = None
    solar_zenith_angle: NDArray[np.floating[Any]] | None = None
    emission_angle: NDArray[np.floating[Any]] | None = None

@dataclass(frozen=True)
class GeodeticResult(Printable):
    """Geodetic coordinates and radii for an observation.

    Returned by the geodetic computation step, providing
    body-fixed geographic coordinates and reference radii
    for each trace.

    Attributes:
        longitude: Sub-spacecraft longitude (rad), shape (n_col,).
        latitude: Sub-spacecraft latitude (rad), shape (n_col,).
        altitude: Spacecraft altitude above the ellipsoid (m),
            shape (n_col,).
        sc_radius: Spacecraft distance from body centre (m),
            shape (n_col,).
        el_radius: Local ellipsoid radius (m), shape (n_col,).
    """
    longitude: NDArray[np.floating[Any]]
    latitude: NDArray[np.floating[Any]]
    altitude: NDArray[np.floating[Any]]
    sc_radius: NDArray[np.floating[Any]]
    el_radius: NDArray[np.floating[Any]]


@dataclass(frozen=True)
class StateVectors(Printable):
    """State vector quantities returned by ``compute_state_vectors``.

    Contains spacecraft and target position and velocity vectors
    in the body-fixed reference frame at each ephemeris time.

    Attributes:
        et: Ephemeris times (s past J2000), shape (n_col,).
        r_s: Spacecraft position vectors (m), shape (n_col, 3).
        v_s: Spacecraft velocity vectors (m/s), shape (n_col, 3).
        r_t: Target position vectors (m), shape (n_col, 3).
        v_t: Target velocity vectors (m/s), shape (n_col, 3).
        r_st: Spacecraft-to-target vectors (m), shape (n_col, 3).
    """
    et: NDArray[np.floating[Any]]
    r_s: NDArray[np.floating[Any]]
    v_s: NDArray[np.floating[Any]]
    r_t: NDArray[np.floating[Any]]
    v_t: NDArray[np.floating[Any]]
    r_st: NDArray[np.floating[Any]]


@dataclass(frozen=True)
class VelocityComponents(Printable):
    """Velocity components returned by ``decompose_velocity``.

    Decomposes the spacecraft velocity into radial and tangential
    components relative to the body centre.

    Attributes:
        v_radial: Radial speed (m/s), component of velocity along the
            position vector. Scalar or shape (n_col,).
        v_tangential: Tangential speed (m/s), component of velocity
            perpendicular to the position vector. Scalar or shape
            (n_col,).
    """
    v_radial: NDArray[np.floating[Any]]
    v_tangential: NDArray[np.floating[Any]]

@dataclass(frozen=True)
class IllumResult(Printable):
    """Illumination angles computed at a surface point for each epoch.

    Returned by the illumination angle computation, providing
    the phase, solar incidence, and emission angles at the
    sub-spacecraft point.

    Attributes:
        phase: Phase angle in radians, shape (n_col,).
        solar: Solar incidence angle in radians, shape (n_col,).
        emission: Emission angle in radians, shape (n_col,).
    """
    phase: NDArray[np.floating[Any]]
    solar: NDArray[np.floating[Any]]
    emission: NDArray[np.floating[Any]]

@dataclass(frozen=True)
class RecordFormat(Printable):
    """Binary record layout for a radar product.

    Defines the byte lengths of science and auxiliary records
    for reading raw binary data files.

    Attributes:
        reclen: Science record length in bytes.
        auxlen: Auxiliary record length in bytes, or ``None`` if the
            product has no auxiliary records.
    """
    reclen: int = 0
    auxlen: int | None = None

@dataclass(frozen=True)
class SHARADMode(Printable):
    """SHARAD operative mode parameters.

    Instrument-specific parameters that vary by operative mode,
    extracted from the label file during preprocessing.

    Attributes:
        presum: Onboard presumming factor.
        bits_per_sample: Quantisation bits per sample.
    """
    presum: int = 0
    bits_per_sample: int = 0

@dataclass(frozen=True)
class MARSISMode(Printable):
    """MARSIS operative mode parameters.

    Instrument-specific parameters that vary by operative mode,
    defining the channel and filter configuration.

    Attributes:
        n_chan: Number of receive channels.
        n_filt: Number of Doppler filters in the mode.
        filt_str: String labels for each Doppler filter.
        chan_str: String labels for each receive channel.
    """
    n_chan: int = 0
    n_filt: int = 0
    filt_str: list[str] = field(default_factory=list)
    chan_str: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class LRSMode(Printable):
    """LRS operative mode parameters.

    Instrument-specific timing parameters for the Lunar Radar
    Sounder on SELENE/Kaguya.

    Attributes:
        prf: Pulse repetition frequency in Hz.
        pri: Pulse repetition interval in seconds.
    """
    prf: float = 0.0
    pri: float = 0.0

@dataclass
class FileIdentity(Printable):
    """Metadata extracted from a radar sounder filename.

    Populated by the file identification step from the filename
    and label contents. Used to dispatch to the correct instrument
    subclass and configure readers.

    Attributes:
        platform: Mission platform identifier.
        instrument: Instrument identifier.
        product_type: PDS product type.
        product_id: Observation product identifier.
        ost_line: OST line number.
        operative_mode: Operative mode string.
        instrument_state: Instrument state.
        data_form: Data form code.
        file_role: Role of the file (e.g. ``"LABEL"``, ``"SCIENCE"``).
        filename: Original filename.
        target: Planetary target body. ``"MARS"`` for SHARAD and
            MARSIS Mars-pointing data, ``"PHOBOS"`` for MARSIS Phobos
            flybys, ``"TRANSIT"`` for MARSIS transit data, and
            ``"MOON"`` for LRS.
    """
    platform: str = "Unknown"
    instrument: str = "Unknown"
    product_type: str = "Unknown"
    product_id: str = "Unknown"
    ost_line: str = "001"
    operative_mode: str = "Unknown"
    instrument_state: str = "TRK"
    data_form: str = "Unknown"
    file_role: str = "Unknown"
    filename: str = ""
    target: str = "Unknown"

@dataclass
class CampbellResult(Printable):
    """Diagnostic output from Campbell ionospheric correction.

    Stores the per-column E-values, the complex phase correction
    applied in the frequency domain, and the bulk delay shift
    applied to align with Mars topography (Campbell 2014 / 2016
    SHARAD-empirical calibration).

    Attributes:
        e_values: Optimal E value per column, shape (n_col,).
        iono_phase: Complex phase correction applied in the frequency
            domain, shape (n_samp, n_col).
        delay_cells: Per-column bulk delay applied to ``ts_out``, in
            range cells. Positive means the corrected pulse was
            rolled up-range. Shape (n_col,).
    """
    e_values: NDArray[np.floating]
    iono_phase: NDArray[np.complexfloating]
    delay_cells: NDArray[np.floating]

@dataclass
class ContrastResult(Printable):
    """Diagnostic output from contrast ionospheric correction.

    Stores the estimated dispersion polynomial coefficients per
    column. ``a3`` and ``a4`` are derived from ``a2`` via the
    Cartacci coupling formulas (which depend on ``f_phys`` and
    ``tau_0 = 2 * L_eq / c``), but are returned here so downstream
    code does not need to know the formulas or carry the auxiliary
    constants.

    Attributes:
        a2: Estimated second-order dispersion coefficient per column,
            shape (n_col,).
        a3: Estimated third-order dispersion coefficient per column,
            shape (n_col,).
        a4: Estimated fourth-order dispersion coefficient per column,
            shape (n_col,).
        delay_cells: Per-column bulk delay applied to ``ts_out``, in
            range cells. Positive means the corrected pulse was
            rolled up-range. Derived from ``a2`` via the Cartacci
            2013 TEC formula combined with Campbell 2014's empirical
            TEC→delay calibration (Level 1). Shape (n_col,).
    """
    a2: NDArray[np.floating]
    a3: NDArray[np.floating]
    a4: NDArray[np.floating]
    delay_cells: NDArray[np.floating]

@dataclass
class MARSISIonosphereResult(Printable):
    """Ionospheric correction results for all MARSIS channels/filters.

    Aggregates individual correction results across the full set
    of MARSIS channels and Doppler filters.

    Attributes:
        method: Compensation method used (e.g. ``"CAMPBELL"``
            or ``"CONTRAST"``).
        results: Dict mapping ``"ECHO_{filt}_{chan}_DIP"`` keys to
            individual ``CampbellResult`` or ``ContrastResult``
            instances.
    """
    method: str
    results: dict[str, CampbellResult | ContrastResult]

@dataclass(frozen=True)
class SARResult(Printable):
    """Diagnostic output from SAR processing.

    Records the method, frame positions, and resolution achieved
    by the SAR processor.

    Attributes:
        method: SAR method used (e.g. ``"RANGE_DOPPLER"``,
            ``"OMEGA_K"``, ``"BACKSCATTER"``, ``"UNFOCUSED"``).
        frames: Centre frame indices into the original azimuth axis,
            shape (n_frames,).
        aperture_step: Step size between apertures in traces.
        rho_a: Azimuth resolution at each output frame (m),
            shape (n_frames,).
        rho_df: Multilooked azimuth resolution (m). Only populated
            for backscatter processing; ``None`` otherwise.
    """
    method: str
    frames: NDArray[np.intp]
    aperture_step: int
    rho_a: NDArray[np.float64]
    rho_df: NDArray[np.float64] | None = None

@dataclass
class ClutterResult:
    """Output of a surface clutter simulation.

    Contains the simulated cluttergram, echo map, and nadir/first-return
    diagnostic quantities produced by :func:`simulate_clutter`.

    Attributes:
        cluttergram: Simulated clutter power, shape
            (n_samples, n_traces).
        echomap: Peak return power per cross-track bin, shape
            (n_ct_bins, n_traces). Rows run from the left side
            (-ct_dist) to the right side (+ct_dist), with nadir
            at the centre row.
        nadir_twtt: Two-way travel time to the nadir surface per
            trace (seconds), shape (n_traces,).
        fret_twtt: First-return two-way travel time per trace
            (seconds), shape (n_traces,). ``NaN`` for traces with
            no valid return.
        nadir_ct_idx: Cross-track bin index of the nadir return
            per trace in the echomap, shape (n_traces,).
        fret_ct_idx: Cross-track bin index of the first return
            per trace in the echomap, shape (n_traces,).
    """
    cluttergram: NDArray[np.floating]
    echomap: NDArray[np.floating]
    nadir_twtt: NDArray[np.floating]
    fret_twtt: NDArray[np.floating]
    nadir_ct_idx: NDArray[np.intp]
    fret_ct_idx: NDArray[np.intp]


#######################################################################################################################
# Processing Parameters Data Classes
#######################################################################################################################
@dataclass
class ClutterSimParams(Printable):
    """Parameters for clutter simulation.

    Controls the surface clutter simulation grid and DEM input.
    Instrument-specific defaults are provided by the instrument
    subclass via ``_default_clutter_params()``.

    Attributes:
        enabled: Whether to run the clutter simulation.
        dem_path: Path to a rasterio-readable DEM file.
        target: String of target body.
        bin_size: Sampling period of the radar in seconds.
        n_samples: Number of range samples per trace.
        at_step: Along-track facet dimension in metres.
        at_dist: Along-track half-extent of the grid in metres.
        ct_step: Cross-track facet dimension in metres.
        ct_dist: Cross-track extent from nadir in metres.
        n_center: Sample index at which the ellipsoid surface return
            is centred. Defaults to ``n_samples // 2`` if ``None``.
        save_data: If True, export data after clutter simulation.
        save_state: If True, save HDF5 state after clutter simulation.
        save_images: If True, export images after clutter simulation.
        apply_curve: If True, apply the simc soft-shoulder tone-mapping
            LUT to the cluttergram BMP after the linear max-normalize.
    """
    enabled: bool | None = None
    dem_path: str | None = None
    target: str | None = None
    bin_size: float | None = None
    n_samples: int | None = None
    at_step: float | None = None
    at_dist: float | None = None
    ct_step: float | None = None
    ct_dist: float | None = None
    n_center: int | None = None
    save_data: bool | None = None
    save_state: bool | None = None
    save_images: bool | None = None
    apply_curve: bool | None = None

@dataclass
class PreprocessingParams(Printable):
    """Parameters for the preprocessing stage.

    Controls onboard presumming correction, trace alignment,
    and output options for the preprocessing step.

    Attributes:
        enabled: Whether to run preprocessing.
        target_presum: Target presumming factor. If ``None``,
            no additional presumming is applied.
        align_traces: Whether to remove receive window opening
            time offsets and centre the surface return.
        n_center: Sample index to centre the surface return on.
            Defaults to ``n_samp // 2`` if ``None``.
        save_data: If True, export data after preprocessing.
        save_state: If True, save HDF5 state after preprocessing.
        save_images: If True, export images after preprocessing.
    """
    enabled: bool | None = None
    target_presum: int | None = None
    align_traces: bool | None = None
    n_center: int | None = None
    save_data: bool | None = None
    save_state: bool | None = None
    save_images: bool | None = None

@dataclass
class RangeCompressionParams(Printable):
    """Parameters for range compression.

    Controls the chirp type, matched/mismatched filtering,
    and spectral windowing applied during range compression.

    Attributes:
        enabled: Whether to run range compression.
        chirp_type: Chirp type (e.g. ``"ideal"``, ``"calibrated"``).
        filter_type: Filter type (e.g. ``"matched"``).
        window: Spectral window type (e.g. ``"hanning"``).
        window_alpha: Tukey taper fraction in [0, 1]. Only used
            when ``window`` is ``"TUKEY"``.
        save_data: If True, export data after range compression.
        save_images: If True, export images after range compression.
        save_state: If True, save HDF5 state after range compression.
    """
    enabled: bool | None = None
    chirp_type: str | None = None
    filter_type: str | None = None
    window: str | None = None
    window_alpha: float | None = None
    save_data: bool | None = None
    save_images: bool | None = None
    save_state: bool | None = None

@dataclass
class EMISuppresionParams(Printable):
    """Parameters for EMI suppression.

    Controls the electromagnetic interference detection and
    replacement strategy applied in the spectral domain.

    Attributes:
        enabled: Whether to run EMI suppression.
        method: Detection method (``"ADAPTIVE"``).
        statistic: Local-statistic method used to form the threshold ("MAD" or "STD").
            Defaults to "MAD".
        k: Outlier threshold multiplier.
        window_size: Sliding window size for local statistics (bins).
        replace_strategy: Replacement strategy for flagged bins
            (``"INTERP"``, ``"ZERO"``, or ``"BASELINE"``).
        interp_pad: Number of bins to pad when interpolating across
            flagged samples.
        save_data: If True, export data after EMI suppression.
        save_images: If True, export images after EMI suppression.
        save_state: If True, save HDF5 state after EMI suppression.
    """
    enabled: bool | None = None
    method: str | None = None
    statistic: str | None = None
    k: float | None = None
    window_size: int | None = None
    replace_strategy: str | None = None
    interp_pad: int | None = None
    save_data: bool | None = None
    save_images: bool | None = None
    save_state: bool | None = None

@dataclass
class IonoCompParams(Printable):
    """Parameters for ionospheric compensation.

    Controls the ionospheric phase correction method, search
    parameters, and spectral windowing.

    Attributes:
        enabled: Whether to run ionospheric compensation.
        method: Compensation method (``"CONTRAST"`` or
            ``"CAMPBELL"``).
        n_take: Neighbourhood width for stacking. If ``None``,
            a default is chosen based on the method.
        campbell_b: Power-law exponent for the Campbell phase model.
        campbell_n_phase: Number of phase states to test (Campbell
            method).
        campbell_delta: Step size for E-value generation (Campbell
            method).
        sgn: Sign convention for the phase exponential (-1 or +1).
        window: Spectral window type (e.g. ``"hanning"``).
        window_alpha: Tukey taper fraction in [0, 1]. Only used
            when ``window`` is ``"TUKEY"``.
        contrast_L_eq: Equivalent path length scale in metres
            (contrast method).
        metric: Autofocus metric name (``"L1"``, ``"L4"``,
            ``"ENTROPY"``, or ``"PEAK_SNR"``). If ``None``, falls
            back to the method's built-in default (``"PEAK_SNR"``
            for Campbell, ``"L4"`` for Contrast).
        save_data: If True, export data after ionospheric
            compensation.
        save_images: If True, export images after ionospheric
            compensation.
        save_state: If True, save HDF5 state after ionospheric
            compensation.
    """
    enabled: bool | None = None
    method: str | None = None
    n_take: int | None = None
    campbell_n_phase: int | None = None
    campbell_delta: float | None = None
    campbell_b: float | None = None
    sgn: Literal[-1, 1] | None = None
    window: str | None = None
    window_alpha: float | None = None
    contrast_L_eq: float | None = None
    metric: str | None = None
    save_data: bool | None = None
    save_images: bool | None = None
    save_state: bool | None = None

@dataclass
class SARParams(Printable):
    """Parameters for SAR processing.

    Controls the SAR focusing method, aperture configuration,
    windowing, and output options.

    Attributes:
        enabled: Whether to run SAR processing.
        method: SAR method (e.g. ``"backscatter"``,
            ``"range_doppler"``, ``"omega_k"``, ``"unfocused"``).
        aperture_length: Synthetic aperture length in traces.
            If ``None``, computed automatically from geometry.
        lamb: Effective wavelength (m) used for the azimuth matched
            filter and RCMC. If ``None``, falls back to
            ``C / RadarParams.f_cen[0]`` (the nominal center
            wavelength). The optimal value is instrument-specific and
            usually differs from the nominal center: SHARAD focuses
            best near 12 m (corresponding to the high-frequency band
            edge), LRS near 60 m (the true 5 MHz center). See the
            per-instrument example TOMLs for tuned values.
        os_factor: Oversampling factor for the output grid.
        window: Spectral window type (e.g. ``"hanning"``).
        window_alpha: Tukey taper fraction in [0, 1]. Only used
            when ``window`` is ``"TUKEY"``.
        number_of_looks: Number of looks for backscatter averaging.
        remove_doppler_centroid: If True, remove the Doppler
            centroid before processing.
        interp_cval: Constant fill value for interpolation outside
            the data bounds.
        coherent: If True, average complex data preserving phase.
        sgn: Sign convention for the phase exponential
        save_data: If True, export data after SAR processing.
        save_images: If True, export images after SAR processing.
        save_state: If True, save HDF5 state after SAR processing.
    """
    enabled: bool | None = None
    method: str | None = None
    aperture_length: int | None = None
    lamb: float | None = None
    os_factor: int | None = None
    window: str | None = None
    window_alpha: float | None = None
    number_of_looks: int | None = None
    remove_doppler_centroid: bool | None = None
    interp_cval: float | None = None
    coherent: bool | None = None
    sgn: Literal[-1, 1] | None = None
    save_data: bool | None = None
    save_images: bool | None = None
    save_state: bool | None = None

@dataclass
class MLKParams(Printable):
    """Parameters for multi-look processing.

    Controls the post-SAR multilook averaging step, including
    the number of looks, oversampling, and windowing.

    Attributes:
        enabled: Whether to run multilook processing.
        number_of_looks: Number of looks (window length).
        os_factor: Oversampling factor for the output grid.
        window_type: Window type for multilook averaging.
        window_alpha: Tukey taper fraction in [0, 1]. Only used
            when ``window_type`` is ``"TUKEY"``.
        coherent: If True, average complex data preserving phase.
        save_data: If True, export data after multilooking.
        save_images: If True, export images after multilooking.
        save_state: If True, save HDF5 state after multilooking.
    """
    enabled: bool | None = None
    number_of_looks: int | None = None
    os_factor: int | None = None
    window_type: str | None = None
    window_alpha: float | None = None
    coherent: bool | None = None
    save_data: bool | None = None
    save_images: bool | None = None
    save_state: bool | None = None

@dataclass
class OutputParameters(Printable):
    """Parameters controlling data export format and options.

    Attributes:
        data_output_type: Output format type (e.g. ``"basic"``).
        byte_order: Byte order for binary output (``"big"`` or
            ``"little"``).
        tgt_crs: Target coordinate reference system string for
            georeferenced outputs. If ``None``, no reprojection
            is applied.
    """
    data_output_type: str | None = None
    byte_order: str | None = None
    tgt_crs: str | None = None

@dataclass
class FinalOutputParams(Printable):
    """Parameters for the finalize step.

    Controls which products are written by :meth:`RadarSounder.finalize`
    at the end of the processing pipeline. Operates on the current
    state of ``self.data`` with no dependence on which stages ran.

    Attributes:
        save_data: If True, export data files.
        save_images: If True, export browse images.
        save_segy: If True, export SEG-Y.
        save_state: If True, save HDF5 state file.
    """
    save_data: bool | None = None
    save_images: bool | None = None
    save_segy: bool | None = None
    save_state: bool | None = None

@dataclass
class PlotParameters(Printable):
    """Parameters controlling radargram and browse image rendering.

    Attributes:
        lower_percentile: Lower percentile for automatic dB
            clipping.
        upper_percentile: Upper percentile for automatic dB
            clipping.
        vmin: Explicit minimum dB value for scaling. Overrides
            ``lower_percentile`` when set.
        vmax: Explicit maximum dB value for scaling. Overrides
            ``upper_percentile`` when set.
        cmap: Matplotlib colormap name. If ``None``, grayscale
            is used.
        invert: If True, invert the grayscale so that strong
            returns appear dark.
        buffer_km: Buffer distance in kilometres for DEM swath
            extraction around the ground track.
        per_frame: If True, compute vmin/vmax per column and apply
            the stretch column-by-column (frame-by-frame). Useful
            when trace-to-trace brightness varies along the orbit
            (e.g., MARSIS altitude swings).
    """
    lower_percentile: float | None = None
    upper_percentile: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    cmap: str | None = None
    invert: bool | None = None
    buffer_km: float | None = None
    per_frame: bool | None = None


@dataclass
class ProcessingParameters(Printable):
    """Top-level container for all processing parameters.

    Aggregates parameters for every stage of the GRaSP processing
    pipeline. Populated from a TOML job file or constructed
    programmatically. Individual stage parameters fall back to
    their dataclass defaults when not explicitly set.

    Attributes:
        out_dir: Output directory for all data products.
        verbose: If True, print progress messages during
            processing.
        preprocessing: Preprocessing stage parameters.
        range_compression: Range compression stage parameters.
        emi_suppression: EMI suppression stage parameters.
        iono_comp: Ionospheric compensation stage parameters.
        sar: SAR processing stage parameters.
        mlk: Multilook processing stage parameters.
        csim: Clutter simulation parameters.
        output: Data export format parameters.
        plots: Image rendering parameters.
        final_output: Final output stage parameters.
    """
    # Global
    out_dir: Path
    verbose: bool | None = None
    # Stages
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    range_compression: RangeCompressionParams = field(default_factory=RangeCompressionParams)
    emi_suppression: EMISuppresionParams = field(default_factory=EMISuppresionParams)
    iono_comp: IonoCompParams = field(default_factory=IonoCompParams)
    sar: SARParams = field(default_factory=SARParams)
    mlk: MLKParams = field(default_factory=MLKParams)
    csim: ClutterSimParams = field(default_factory=ClutterSimParams)
    output: OutputParameters = field(default_factory=OutputParameters)
    plots: PlotParameters = field(default_factory=PlotParameters)
    final_output: FinalOutputParams = field(default_factory=FinalOutputParams)

    def __post_init__(self):
        if isinstance(self.out_dir, str):
            object.__setattr__(self, 'out_dir', Path(self.out_dir))

@dataclass
class ProcessingJob(Printable):
    """Complete processing job parsed from a TOML file.

    Represents a fully configured processing run, including
    input file paths and all processing parameters.

    Attributes:
        label_file: Path to the label file.
        science_file: Path to the science data file.
        auxiliary_file: Path to the auxiliary data file. ``None``
            if not applicable (e.g. some product types).
        parameters: Processing parameters for all stages.
    """
    label_file: Path
    science_file: Path
    auxiliary_file: Path | None = None
    parameters: ProcessingParameters | None = None