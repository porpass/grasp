# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from pathlib import Path

import warnings
from typing import Literal


from ..radar_sounder import RadarSounder
#
# Preprocessing Functions
#
from .preprocessing import preprocess_marsis_edr
from .datuming import remove_rwot_offset
#
# Range Compression Wrapper
#
from .range_compression import range_compression
#
# EMI Suppression Wrapper
#
from .emi import suppress_emi as _suppress_emi
#
# Ionosphere Compensation Wrapper
#
from .ionosphere import ionospheric_compensation

from .multilook import _multilook
from .writers import export_data as marsis_export_data
from .writers import export_images as marsis_export_images


class MARSISSounder(RadarSounder):
    """MARSIS Radar Sounder Class"""

    GEOMETRY_KEYS = {
        "EDR": {
            "source": "auxiliary",
            "epoch": "GEOMETRY_EPOCH",
            "et": "GEOMETRY_EPHEMERIS_TIME",
            "latitude": "SUB_SC_LATITUDE",
            "longitude": "SUB_SC_LONGITUDE",
            "altitude": "SPACECRAFT_ALTITUDE",
            "scale": 1000.0,
            "position": "TARGET_SC_POSITION_VECTOR",
            "velocity": "TARGET_SC_VELOCITY_VECTOR",
            "v_radial": "TARGET_SC_RADIAL_VELOCITY",
            "v_tangential": "TARGET_SC_TANG_VELOCITY",
            "sza": "SOLAR_ZENITH_ANGLE",
        },
        "RDR": {
            "source": "science",
            # same keys as EDR for MARSIS
            "epoch": "GEOMETRY_EPOCH",
            "et": "GEOMETRY_EPHEMERIS_TIME",
            "latitude": "SUB_SC_LATITUDE",
            "longitude": "SUB_SC_LONGITUDE",
            "altitude": "SPACECRAFT_ALTITUDE",
            "scale": 1000.0,
            "position": "TARGET_SC_POSITION_VECTOR",
            "velocity": "TARGET_SC_VELOCITY_VECTOR",
            "v_radial": "TARGET_SC_RADIAL_VELOCITY",
            "v_tangential": "TARGET_SC_TANG_VELOCITY",
            "sza": "SOLAR_ZENITH_ANGLE",
        },
    }

    def __init__(self, file_info=None):
        """Initialize the MARSISSounder.

        Delegates to ``RadarSounder.__init__`` to set up the empty file paths,
        data containers, metadata, parameters, and processing options.
        """
        super().__init__(file_info)

    # No direct call to self.data. MARSIS has a variety of science keys depending
    # on the operative mode.
    ################################################################################################################
    #
    # Input Methods
    #
    ################################################################################################################
    def check_supported(self):
        """Check whether the operative mode is supported.

        Raises:
            ValueError: If the product type is not supported.
        """
        md = self.metadata

        if md.product_type not in ("EDR", "RDR"):
            raise ValueError(f"{md.instrument} {md.product_type} "
                             f"products are not currently supported by the RadarSounder class.")
        if md.instrument_state != "TRK":
            raise ValueError(f"{md.instrument} {md.product_type} {md.instrument_state} "
                             f"products are not currently supported by the RadarSounder class.")
        if md.data_form != "CMP": # TODO (low priority; post-beta): Add RAW Support (No RAW on PDS)
            raise ValueError(f"{md.instrument} {md.product_type} {md.data_form} "
                             f"products are not currently supported by the RadarSounder class.")
        if md.operative_mode[:2] != "SS":
            raise ValueError(f"{md.instrument} {md.product_type} {md.operative_mode} "
                             f"products are not currently supported by the RadarSounder class.")

    ################################################################################################################
    #
    # Preprocessing Methods
    #
    ################################################################################################################
    def preprocess(self,
                   enabled: bool = True,
                   align_traces: bool | None = None,
                   n_center: int | None = None,
                   save_data: bool | None = None,
                   save_state: bool | None = None,
                   verbose: bool | None = None):
        """Preprocess MARSIS EDR data.

        Parameters fall back to ``processing_parameters.preprocessing``
        if not provided, then to dataclass defaults.

        Args:
            enabled: Enable preprocessing
            align_traces: If True, align traces by removing RWOT offsets.
            n_center: Sample index to center the surface return on.
            save_data: If True, export data after preprocessing.
            save_state: If True, save HDF5 state after preprocessing.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        pre = pp.preprocessing
        md = self.metadata
        enabled = enabled if enabled is not None else pre.align_traces
        verbose = verbose if verbose is not None else pp.verbose

        if not enabled:
            if verbose:
                print("Preprocessing is disabled. Skipping.")
            return

        if md.product_type not in ('EDR', 'RDR'):
            raise ValueError(f"{md.product_type} preprocessing is not supported")

        align_traces = align_traces if align_traces is not None else pre.align_traces
        n_center = n_center if n_center is not None else pre.n_center
        save_data = save_data if save_data is not None else pre.save_data
        save_state = save_state if save_state is not None else pre.save_state
        verbose = verbose if verbose is not None else pp.verbose

        if md.product_type == 'EDR':
            if verbose:
                print("Preprocessing MARSIS EDR Data")

            self.science_data = preprocess_marsis_edr(self.science_data,
                                                      md.operative_mode,
                                                      verbose=verbose,
                                                      )
        elif md.product_type == 'RDR':
            raise NotImplementedError("RDR preprocessing is not yet implemented.")

        if align_traces:
            self._align_range_window(n_center=n_center, verbose=verbose)

        if save_data:
            self.export_data(level="preprocess", verbose=verbose)
        if save_state:
            base = self._build_base("preprocess", None, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

    ################################################################################################################
    #
    # Range Processing Methods
    #
    ################################################################################################################
    def range_compression(self,
                          enabled: bool = True,
                          chirp_type: str | None = "ideal",
                          filter_type: str | None = "matched",
                          window_type: str | None = None,
                          window_alpha: float | None = None,
                          save_data: bool | None = None,
                          save_images: bool | None = None,
                          save_state: bool | None = None,
                          verbose: bool | None = None):
        """Apply range compression to MARSIS EDR data.

        Parameters fall back to ``processing_parameters.range_compression``
        if not provided, then to dataclass defaults.

        Args:
            enabled: Enable range compression
            chirp_type: Range compression method to use.
            filter_type: matched or inverse
            window_type: Spectral window type.
            window_alpha: Tukey window taper fraction.
            save_data: If True, export data after range compression.
            save_images: If True, export images after range compression.
            save_state: If True, save HDF5 state after range compression.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        md = self.metadata
        rc = pp.range_compression
        params = self.parameters
        enabled = enabled if enabled is not None else rc.enabled
        verbose = verbose if verbose is not None else pp.verbose

        if not enabled:
            if verbose:
                print("Range Compression is disabled. Skipping.")
            return

        if md.product_type != "EDR":
            warnings.warn(f"Range compression valid for EDRs only. Skipping.")
            return
        mode = md.operative_mode.upper()
        chirp_type = chirp_type.lower() if chirp_type is not None else rc.chirp_type
        filter_type = filter_type.lower() if filter_type is not None else rc.filter_type
        window_type = window_type if window_type is not None else rc.window
        window_alpha = window_alpha if window_alpha is not None else rc.window_alpha
        save_data = save_data if save_data is not None else rc.save_data
        save_images = save_images if save_images is not None else rc.save_images
        save_state = save_state if save_state is not None else rc.save_state

        self.science_data = range_compression(self.science_data,
                                              mode,
                                              filter_type=filter_type,
                                              chirp_type=chirp_type,
                                              nfft=params.nfft,
                                              dt=params.dt,
                                              tau=params.tau,
                                              bw=params.bw,
                                              window_type=window_type,
                                              alpha=window_alpha,
                                              verbose=verbose)
        if save_data:
            self.export_data(level="rc", method=chirp_type, verbose=verbose)
        if save_images:
            self.export_images(level="rc", method=chirp_type, verbose=verbose)
        if save_state:
            base = self._build_base("rc", chirp_type, None)
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
                        verbose: bool = False) -> None:
        """Suppress electromagnetic interference in MARSIS data.

        Detects and removes narrowband interference from each
        channel/filter combination. Parameters fall back to
        ``processing_parameters.emi_suppression`` if not provided,
        then to dataclass defaults.

        Args:
            method: EMI detection method (e.g., ``"adaptive"``).
            statistic: Local-statistic to use (``"MAD"`` or ``"STD"``)
            k: Detection threshold multiplier.
            window_size: Sliding window size for adaptive detection.
            replace_strategy: Replacement strategy — ``"INTERP"``,
                ``"ZERO"``, or ``"BASELINE"``.
            interp_pad: Number of bins to pad when interpolating
                across flagged samples.
            save_data: If True, export data after EMI suppression.
            save_images: If True, export images after EMI suppression.
            save_state: If True, save HDF5 state after EMI suppression.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        md = self.metadata
        emi = pp.emi_suppression
        params = self.parameters
        verbose = verbose if verbose is not None else pp.verbose

        if not emi.enabled:
            if verbose:
                print("EMI Suppression is disabled. Skipping.")
            return

        if md.product_type not in ("EDR", "RDR"):
            warnings.warn(f"EMI Suppression valid for EDRs and RDRs only. Skipping.")
            return

        mode = md.operative_mode.upper()
        method = method.lower() if method is not None else emi.method
        statistic = statistic.lower() if statistic is not None else emi.statistic
        k = k if k is not None else emi.k
        window_size = window_size if window_size is not None else emi.window_size
        replace_strategy = replace_strategy.lower() if replace_strategy is not None else emi.replace_strategy
        interp_pad = interp_pad if interp_pad is not None else emi.interp_pad
        save_data = save_data if save_data is not None else emi.save_data
        save_images = save_images if save_images is not None else emi.save_images
        save_state = save_state if save_state is not None else emi.save_state

        self.science_data = _suppress_emi(self.science_data,
                                             mode,
                                             nfft=params.nfft,
                                             dt=params.dt,
                                             bw=params.bw,
                                             method=method,
                                             statistic=statistic,
                                             k=k,
                                             window_size=window_size,
                                             replace=replace_strategy,
                                             interp_pad=interp_pad,
                                             verbose=verbose)
        if save_data:
            self.export_data(level="emi", method=method, verbose=verbose)
        if save_images:
            self.export_images(level="emi", method=method, verbose=verbose)
        if save_state:
            base = self._build_base("emi", method, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

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
        md = self.metadata
        ic = pp.iono_comp
        params = self.parameters
        verbose = verbose if verbose is not None else pp.verbose

        if not ic.enabled:
            if verbose:
                print("Ionospheric compensation is disabled. Skipping.")
            return

        if md.product_type not in ("EDR", "RDR"):
            warnings.warn("Ionospheric compensation valid for EDRs and RDRs only. Skipping.")
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
        save_data = save_data if save_data is not None else ic.save_data
        save_images = save_images if save_images is not None else ic.save_images
        save_state = save_state if save_state is not None else ic.save_state

        dcg_config = self.science_data['DCG_CONFIGURATION']
        mode = self.metadata.operative_mode

        self.science_data, self.ionosphere = ionospheric_compensation(
            self.science_data,
            mode,
            dcg_config,
            method=method,
            dt=params.dt,
            bw=params.bw,
            n_take=n_take,
            wnd=window_type,
            alpha=window_alpha,
            L_eq=L_eq,
            campbell_b=b,
            n_phase=n_phase,
            campbell_delta=delta,
            sgn=sgn,
            verbose=verbose,
        )

        if save_data:
            self.export_data(level="iono", method=method, verbose=verbose)
        if save_images:
            self.export_images(level="iono", method=method, verbose=verbose)
        if save_state:
            base = self._build_base("iono", method, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)


    def sar_process(self, verbose: bool = False) -> None:
        """SAR processing for MARSIS data.

        MARSIS CMP data is azimuth-compressed onboard. SAR processing
        is only applicable to SS3 RAW EDR data, which is not yet
        implemented.

        Args:
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        md = self.metadata
        sp = pp.sar
        verbose = verbose if verbose is not None else pp.verbose

        if not sp.enabled:
            if verbose:
                print("SAR Processing is disabled. Skipping.")
            return

        if md.product_type == "EDR" and md.data_form == "RAW":
            raise NotImplementedError("EDR RAW SAR processing is not implemented yet.")
        if verbose:
            print("MARSIS CMP data does not require SAR processing. Skipping.")

    ################################################################################################################
    #
    # Post Processing Methods (e.g., redatuming, image filters, etc.)
    #
    ################################################################################################################
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
        """Apply multilook processing to MARSIS data.

        Computes along-track resolution from auxiliary or science
        data and applies incoherent or coherent multilook averaging
        to each channel/filter combination. Parameters fall back to
        ``processing_parameters.mlk`` if not provided, then to
        dataclass defaults.

        Args:
            n_looks: Number of looks.
            os_factor: Oversampling factor.
            window_type: Spectral window type.
            window_alpha: Tukey window taper fraction.
            coherent: If True, apply coherent multilooking.
            save_data: If True, export data after multilooking.
            save_images: If True, export images after multilooking.
            save_state: If True, save HDF5 state after multilooking.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        md = self.metadata
        mlk = pp.mlk
        verbose = verbose if verbose is not None else pp.verbose

        if not mlk.enabled:
            if verbose:
                print("Multilooking is disabled. Skipping.")
            return

        if md.product_type not in ("EDR", "RDR"):
            warnings.warn("Multilooking valid for EDRs and RDRs only. Skipping.")
            return

        mode = md.operative_mode.upper()
        n_looks = n_looks if n_looks is not None else mlk.number_of_looks
        os_factor = os_factor if os_factor is not None else mlk.os_factor
        window_type = window_type if window_type is not None else mlk.window_type
        window_alpha = window_alpha if window_alpha is not None else mlk.window_alpha
        coherent = coherent if coherent is not None else mlk.coherent
        save_data = save_data if save_data is not None else mlk.save_data
        save_images = save_images if save_images is not None else mlk.save_images
        save_state = save_state if save_state is not None else mlk.save_state

        self.science_data = _multilook(
            self.science_data,
            self.auxiliary_data,
            mode,
            md.product_type,
            n_looks=n_looks,
            os_factor=os_factor,
            window_type=window_type,
            window_alpha=window_alpha,
            coherent=coherent,
            verbose=verbose,
        )

        if save_data:
            self.export_data(level="mlk", method=None, verbose=verbose)
        if save_images:
            self.export_images(level="mlk", method=None, verbose=verbose)
        if save_state:
            base = self._build_base("mlk", None, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

    ################################################################################################################
    #
    # Export Methods
    #
    ################################################################################################################
    def export_data(self,
                    level: str | None = None,
                    method: str | None = None,
                    prefix: str | None = None,
                    verbose: bool | None = None,
                    **kwargs):
        """Export MARSIS data files, one per channel/filter.

        Args:
            level: Processing level string.
            method: Processing method string.
            prefix: Filename prefix.
            verbose: If True, print progress messages.
        """
        if self.geometry is None:
            raise RuntimeError("Geometry not available.")

        pp = self._get_pp()
        md = self.metadata
        par = self.parameters
        op = pp.output
        verbose = verbose if verbose is not None else pp.verbose

        if level is None:
            level = md.product_type.lower()

        base = self._build_base(level, method, prefix)

        marsis_export_data(
            self.science_data,
            self.geometry,
            md.operative_mode,
            out_dir=pp.out_dir,
            base=base,
            instrument=md.instrument,
            dt=par.dt,
            pri=par.pri if par.pri is not None else 1.0,
            presum=par.presum if par.presum is not None else 1,
            sar=self.sar,
            ionosphere=self.ionosphere,
            output_format=op.data_output_type,
            byte_order=op.byte_order,
            verbose=verbose,
            **kwargs,
        )

    def export_images(self,
                      level: str | None = None,
                      method: str | None = None,
                      prefix: str | None = None,
                      verbose: bool | None = None,
                      **kwargs):
        """Export MARSIS radargram images, one per channel/filter.

        Args:
            level: Processing level string.
            method: Processing method string.
            prefix: Filename prefix.
            verbose: If True, print progress messages.
        """
        if self.geometry is None:
            raise RuntimeError("Geometry not available.")

        pp = self._get_pp()
        md = self.metadata
        plots = pp.plots
        verbose = verbose if verbose is not None else pp.verbose

        if level is None:
            level = md.product_type.lower()

        base = self._build_base(level, method, prefix)

        marsis_export_images(
            self.science_data,
            self.geometry,
            self.crs,
            md.operative_mode,
            out_dir=pp.out_dir,
            base=base,
            dem=kwargs.pop("dem", "HRSC-MOLA"),
            buffer_km=kwargs.pop("buffer_km",
                                 plots.buffer_km if plots.buffer_km is not None else 25.0),
            lower_percentile=plots.lower_percentile,
            upper_percentile=plots.upper_percentile,
            vmin=plots.vmin,
            vmax=plots.vmax,
            power=kwargs.pop("power", None),
            invert=plots.invert,
            per_frame=plots.per_frame if plots.per_frame is not None else False,
            verbose=verbose,
        )
    ################################################################################################################
    #
    # Internal helper methods
    #
    ################################################################################################################
    def _align_range_window(self,
                            n_center: int | None = None,
                            verbose: bool = False):
        """Remove RWOT offsets and center the surface return.

        Args:
            n_center: Sample index to center the surface return on.
            verbose: If True, print progress messages.
        """
        if self.geometry is None:
            raise RuntimeError("Geometry not available. Call read_auxiliary_file() first.")
        md = self.metadata

        if verbose:
            print("Aligning range window...")

        altitude = self.geometry.altitude
        if n_center is None:
            n_center = self.parameters.nfft//2
        self.science_data, self.datum = remove_rwot_offset(self.science_data,
                                                           md.operative_mode,
                                                           altitude,
                                                           fs=self.parameters.fs,
                                                           n_center=n_center,
                                                           verbose=verbose)

    def _pad_science_data(self, nfft: int):
        """Zero-pad echo samples to the FFT length.

        Args:
            nfft: Target number of rows (FFT length).
        """
        tmp = self.data
        n_samp, n_col = tmp.shape
        self.data = np.zeros((nfft, n_col), tmp.dtype)
        self.data[:n_samp, :] = tmp[:n_samp, :]
