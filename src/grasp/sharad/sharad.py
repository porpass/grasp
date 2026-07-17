# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import replace
import numpy as np
from pathlib import Path

from ..radar_sounder import RadarSounder
from .preprocessing import preprocess_sharad_edr_sci
from .datuming import remove_rwot_offset
from .range_compression import range_compression as _range_compression

class SHARADSounder(RadarSounder):
    """SHARAD Radar Sounder Class"""

    GEOMETRY_KEYS = {
        "EDR": {
            "source": "auxiliary",
            "epoch": "GEOMETRY_EPOCH",
            "et": "EPHEMERIS_TIME",
            "latitude": "SUB_SC_PLANETOCENTRIC_LATITUDE",
            "longitude": "SUB_SC_EAST_LONGITUDE",
            "altitude": "SPACECRAFT_ALTITUDE",
            "position": ["X_MARS_SC_POSITION_VECTOR",
                         "Y_MARS_SC_POSITION_VECTOR",
                         "Z_MARS_SC_POSITION_VECTOR"],
            "velocity": ["X_MARS_SC_VELOCITY_VECTOR",
                         "Y_MARS_SC_VELOCITY_VECTOR",
                         "Z_MARS_SC_VELOCITY_VECTOR"],
            "v_radial": "MARS_SC_RADIAL_VELOCITY",
            "v_tangential": "MARS_SC_TANGENTIAL_VELOCITY",
            "sza": "SOLAR_ZENITH_ANGLE",
        },
        "RDR": {
            "source": "science",
            "epoch": "GEOMETRY_EPOCH",
            "et": "EPHEMERIS_TIME",
            "latitude": "SUB_SC_PLANETOCENTRIC_LATITUDE",
            "longitude": "SUB_SC_EAST_LONGITUDE",
            "altitude": "SPACECRAFT_ALTITUDE",
            "scale": 1000.0,
            "position": "MARS_SC_POSITION_VECTOR",
            "velocity": "MARS_SC_VELOCITY_VECTOR",
            "v_radial": "MARS_SC_RADIAL_VELOCITY",
            "v_tangential": "MARS_SC_TANGENTIAL_VELOCITY",
            "sza": "SOLAR_ZENITH_ANGLE",
        },
    }

    def __init__(self, file_info=None):
        """Initialize the SHARADSounder.

        Delegates to ``RadarSounder.__init__`` to set up the empty file paths,
        data containers, metadata, parameters, and processing options.
        """
        super().__init__(file_info)

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
            raise ValueError(f"{md.instrument} {md.product_type} {md.operative_mode} "
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
                   target_presum: int | None = None,
                   align_traces: bool | None = None,
                   n_center: int | None = None,
                   save_data: bool | None = None,
                   save_state: bool | None = None,
                   verbose: bool | None = None):
        """Preprocess SHARAD EDR data.

        Parameters fall back to ``processing_parameters.preprocessing``
        if not provided, then to dataclass defaults.

        Args:
            target_presum: Desired total presumming factor.
            align_traces: If True, align traces by removing RWOT offsets.
            n_center: Sample index to center the surface return on.
            save_data: If True, export data after preprocessing.
            save_state: If True, save HDF5 state after preprocessing.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        pre = pp.preprocessing
        md = self.metadata
        verbose = verbose if verbose is not None else pp.verbose

        if not pre.enabled:
            if verbose:
                print("Preprocessing is disabled. Skipping.")
            return


        if md.product_type not in ('EDR', 'RDR'):
            raise ValueError(f"{md.product_type} preprocessing is not supported")

        target_presum = target_presum if target_presum is not None else pre.target_presum
        align_traces = align_traces if align_traces is not None else pre.align_traces
        n_center = n_center if n_center is not None else pre.n_center
        save_data = save_data if save_data is not None else pre.save_data
        save_state = save_state if save_state is not None else pre.save_state

        if md.product_type == 'EDR':
            if verbose:
                print("Preprocessing SHARAD EDR Data")

            presum = self.parameters.presum
            assert presum is not None, "SHARAD must have a presum value"

            self.science_data, self.auxiliary_data, presum_factor = preprocess_sharad_edr_sci(
                self.science_data,
                self.auxiliary_data,
                presum,
                apply_instrument_response=True,
                target_presum=target_presum,
                verbose=verbose,
            )

            if presum_factor is not None:
                self.parameters = replace(self.parameters, presum=target_presum)
                self._decimate_to_frames(factor=presum_factor, skip_keys='ECHO_SAMPLES', verbose=verbose)

            self._pad_science_data(nfft=self.parameters.nfft)
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
    # Range Compression
    #
    ################################################################################################################
    def range_compression(self,
                          chirp_type: str | None = None,
                          filter_type: str | None = None,
                          window_type: str | None = None,
                          window_alpha: float | None = None,
                          save_data: bool | None = None,
                          save_images: bool | None = None,
                          save_state: bool | None = None,
                          verbose: bool | None = None):
        """Apply range compression to SHARAD EDR data.

        Parameters fall back to ``processing_parameters.range_compression``
        if not provided, then to dataclass defaults.

        Args:
            chirp_type: Type of chirp to use (COMPLEX, IDEAL, CALIBRATED)
                - Only COMPLEX is currently available
            filter_type: Range compression method to use. (MATCHED or INVERSE)
            window_type: Spectral window type.
            window_alpha: Tukey window taper fraction.
            save_data: If True, export data after range compression.
            save_images: If True, export images after range compression.
            save_state: If True, save HDF5 state after range compression.
            verbose: If True, print progress messages.
        """
        pp = self._get_pp()
        rc = pp.range_compression
        params = self.parameters
        verbose = verbose if verbose is not None else pp.verbose

        if not rc.enabled:
            if verbose:
                print("Range Compression is disabled. Skipping.")
            return

        filter_type = filter_type.lower() if filter_type is not None else rc.filter_type
        chirp_type = chirp_type.lower() if chirp_type is not None else rc.chirp_type
        window_type = window_type if window_type is not None else rc.window
        window_alpha = window_alpha if window_alpha is not None else rc.window_alpha
        save_data = save_data if save_data is not None else rc.save_data
        save_images = save_images if save_images is not None else rc.save_images
        save_state = save_state if save_state is not None else rc.save_state
        f_cen = params.f_cen[0]

        self.data = _range_compression(self.data, filter_type=filter_type, chirp_type=chirp_type,
                                       nfft=params.nfft, dt=params.dt, tau=params.tau, bw=params.bw,
                                       f_cen=f_cen, window_type=window_type, alpha=window_alpha,
                                       tx_temp=self.auxiliary_data['TX_TEMP'], rx_temp=self.auxiliary_data['RX_TEMP'])
        if save_data:
            self.export_data(level="rc", method=chirp_type, verbose=verbose)
        if save_images:
            self.export_images(level="rc", method=chirp_type, verbose=verbose)
        if save_state:
            base = self._build_base("rc", chirp_type, None)
            self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

    ################################################################################################################
    #
    # EMI Suppression (handled in base class, RadarSounder)
    #
    ###############################################################################################################

    ################################################################################################################
    #
    # Ionospheric Compensation
    #
    ################################################################################################################

    ################################################################################################################
    #
    # Azimuth Processing Methods
    #
    ################################################################################################################

    ################################################################################################################
    #
    # Post Processing Methods (e.g., redatuming, image filters, etc.)
    #
    ################################################################################################################

    ################################################################################################################
    #
    # Internal helper methods
    #
    ################################################################################################################
    def _undo_baseband_shift(self):
        """Reverse the complex baseband shift applied during range compression."""
        if self.phase_shift is not None:
            self.data *= np.conj(self.phase_shift)[:, np.newaxis]
            self.phase_shift = None

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

        if verbose:
            print("Aligning range window...")

        self.data, self.datum = remove_rwot_offset(self.data,
                                                   self.science_data['RECEIVE_WINDOW_OPENING_TIME'],
                                                   self.geometry.altitude,
                                                   dt=self.parameters.dt,
                                                   n_center=n_center,
                                                   )

    def _pad_science_data(self, nfft: int):
        """Zero-pad echo samples to the FFT length.

        Args:
            nfft: Target number of rows (FFT length).
        """
        tmp = self.data
        n_samp, n_col = tmp.shape
        self.data = np.zeros((nfft, n_col), tmp.dtype)
        self.data[:n_samp, :] = tmp[:n_samp, :]