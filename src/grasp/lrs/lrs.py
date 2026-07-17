# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path
from typing import Literal

from ..radar_sounder import RadarSounder
from .preprocessing import preprocess_lrs_wf
from .datuming import remove_rwot_offset



class LRSSounder(RadarSounder):
    """Radar sounder implementation for the SELENE LRS instrument."""

    GEOMETRY_KEYS = {
        "EDR": {
            "source": "science",
            "epoch": "OBSERVATION_TIME",
            "latitude": "SUB_SPACECRAFT_LATITUDE",
            "longitude": "SUB_SPACECRAFT_LONGITUDE",
            "altitude": "SPACECRAFT_ALTITUDE",
            "scale": 1000,
        },
        "RDR": {
            "source": "science",
            "epoch": "OBSERVATION_TIME",
            "latitude": "SUB_SPACECRAFT_LATITUDE",
            "longitude": "SUB_SPACECRAFT_LONGITUDE",
            "altitude": "SPACECRAFT_ALTITUDE",
            "scale": 1000,
        },
    }

    def __init__(self, file_info=None):
        """Initialize the LRSSounder.

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
    ################################################################################################################
    #
    # Preprocessing Methods
    #
    ################################################################################################################
    def preprocess(self,
                   align_traces: bool | None = None,
                   n_center: int | None = None,
                   save_data: bool | None = None,
                   save_state: bool | None = None,
                   save_images: bool | None = None,
                   verbose: bool | None = None):
        """Preprocess LRS WF observations.

        Parameters fall back to ``processing_parameters.preprocessing``
        if not provided, then to dataclass defaults.

        Args:
            align_traces: If True, align traces by removing RWOT offsets.
            n_center: Sample index to center the surface return on.
            save_data: If True, export data after preprocessing.
            save_state: If True, save HDF5 state after preprocessing.
            save_images: If True, save images state after preprocessing.
            verbose: If True, print progress messages.
        """
        md = self.metadata
        pp = self._get_pp()
        pre = pp.preprocessing
        par = self.parameters

        verbose = verbose if verbose is not None else pp.verbose

        if not pre.enabled:
            if verbose:
                print("Preprocessing is disabled. Skipping")
            return

        if md.product_type not in ("EDR", "RDR"):
            raise ValueError(f"{md.product_type} preprocessing is not supported")

        dt = par.dt
        tau = par.tau
        bw = par.bw

        align_traces = align_traces if align_traces is not None else pre.align_traces
        n_center = n_center if n_center is not None else pre.n_center
        save_data = save_data if save_data is not None else pre.save_data
        save_state = save_state if save_state is not None else pre.save_state
        save_images = save_images if save_images is not None else pre.save_images

        if self.metadata.product_type == 'EDR':
            if verbose:
                print(f"Preprocessing LRS EDR Data")
            self.data = preprocess_lrs_wf(self.data, dt, tau, bw)

            if align_traces:
                self._align_range_window(n_center=n_center)

            if save_data:
                self.export_data(level="preprocess", verbose=verbose)
            if save_images:
                self.export_images(level="preprocess", verbose=verbose)
            if save_state:
                base = self._build_base(level="preprocess", method=None, prefix=None)
                self.save(Path(pp.out_dir) / f"{base}.h5", verbose=verbose)

        elif self.metadata.product_type == 'RDR':
            raise NotImplementedError("LRS RDR Preprocessing is not yet implemented.")
        else:
            raise ValueError(f"Unsupported product type: {self.metadata.product_type}")

    ################################################################################################################
    #
    # Range Processing Methods
    #
    ################################################################################################################
    def range_compression(self, verbose: bool = False) -> None:
        """Empty method (required); LRS data is range-compressed onboard.

        Args:
            verbose: If True, print progress messages.
        """
        if verbose:
            print("LRS data does not require range compression. Skipping.")
        pass

    def emi_suppression(self, verbose: bool = False) -> None:
        """EMI Suppression for LRS.

        Unsupported

        Args:
            verbose: If True, print progress messages.
        """

        if verbose:
            print("EMI Suppression for LRS not supported")

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
        """No-op override; the Moon has no ionosphere.

        Args:
            method: Ignored.
            n_take: Ignored.
            b: Ignored.
            n_phase: Ignored.
            delta: Ignored.
            sgn: Ignored.
            window_type: Ignored.
            window_alpha: Ignored.
            L_eq: Ignored.
            save_data: Ignored.
            save_images: Ignored.
            save_state: Ignored.
            verbose: If True, print progress messages.
        """
        if verbose:
            print("LRS data does not require ionospheric compensation. Skipping.")
        pass
    ################################################################################################################
    #
    # Post Processing Methods (e.g., redatuming, image filters, etc.)
    #
    ################################################################################################################

    ################################################################################################################
    #
    # Output Methods
    #
    ################################################################################################################

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
            raise RuntimeError("Geometry not available. Call read_science_file() first.")

        if verbose:
            print("Aligning range window...")

        altitude = self.geometry.altitude

        self.data, self.datum = remove_rwot_offset(
            self.data,
            self.science_data['DELAY'],
            altitude,
            self.parameters.dt,
            n_center=n_center,
        )