# SPDX-License-Identifier: BSD-3-Clause
"""MARSIS-specific output wrappers.

The base ``RadarSounder`` export paths assume a single radargram per
observation, which doesn't fit MARSIS — operative modes carry one or
more channels and one or more filters, each producing an independent
science array indexed in ``science_data`` by an ``ECHO_<F>_<C>_DIP``
key. These wrappers iterate the per-mode channel/filter grid and
delegate each radargram to the central
``grasp.output.writers.export_data`` / ``export_images`` functions.

Designed to be called by ``MARSISSounder.export_data`` /
``export_images`` — these functions take no ``self``.
"""
from pathlib import Path

from numpy.typing import NDArray
from pyproj import CRS

from .modes import iter_channel_filter
from ..grasp_types import (
    CampbellResult,
    ContrastResult,
    GeometryResult,
    SARResult,
)
from ..output.writers import export_data as _export_data
from ..output.writers import export_images as _export_images


def export_data(science_data: dict[str, NDArray],
                geometry: GeometryResult,
                operative_mode: str,
                *,
                out_dir: str | Path,
                base: str,
                instrument: str,
                dt: float,
                pri: float,
                presum: int = 1,
                sar: SARResult | None = None,
                ionosphere: CampbellResult | ContrastResult | None = None,
                output_format: str = "basic",
                byte_order: str = "little",
                verbose: bool = False,
                ) -> None:
    """Export MARSIS data files, one per channel/filter.

    Iterates the channel/filter grid for ``operative_mode``, builds a
    per-channel suffix, and delegates each radargram to
    :func:`grasp.output.writers.export_data`.

    Args:
        science_data: MARSIS science data dict keyed by
            ``"ECHO_<F>_<C>_DIP"``.
        geometry: Observation geometry, shared across channels.
        operative_mode: Operative mode (e.g. ``"SS3"``) used to
            look up the channel/filter grid.
        out_dir: Output directory.
        base: Base filename (without the per-channel suffix).
        instrument: Instrument identifier (e.g. ``"MARSIS"``).
        dt: Sample spacing in seconds.
        pri: Pulse repetition interval in seconds.
        presum: Along-track presumming factor.
        sar: SAR result if SAR processing has run.
        ionosphere: Ionospheric correction result if applicable.
        output_format: Output format. Only ``"basic"`` is currently
            supported.
        byte_order: Byte order for binary output.
        verbose: If True, print progress messages.
    """
    out_dir = Path(out_dir)
    for f_str, c_str, key in iter_channel_filter(operative_mode):
        if key not in science_data:
            continue
        suffix = f"_{f_str}_{c_str}".lower()
        full_base = f"{base}{suffix}"
        if verbose:
            print(f"Exporting data: {full_base}")
        _export_data(
            science_data[key],
            geometry,
            out_dir=out_dir,
            base=full_base,
            instrument=instrument,
            dt=dt,
            pri=pri,
            presum=presum,
            sar=sar,
            ionosphere=ionosphere,
            output_format=output_format,
            byte_order=byte_order,
            verbose=verbose,
        )


def export_images(science_data: dict[str, NDArray],
                  geometry: GeometryResult,
                  crs: CRS | None,
                  operative_mode: str,
                  *,
                  out_dir: str | Path,
                  base: str,
                  dem: str | Path,
                  buffer_km: float,
                  lower_percentile: float | None = None,
                  upper_percentile: float | None = None,
                  vmin: float | None = None,
                  vmax: float | None = None,
                  power: bool | None = None,
                  invert: bool = False,
                  per_frame: bool = False,
                  verbose: bool = False,
                  ) -> None:
    """Export MARSIS radargram and browse images, one per channel/filter.

    Iterates the channel/filter grid for ``operative_mode``, builds a
    per-channel suffix, and delegates each radargram to
    :func:`grasp.output.writers.export_images`.

    Args:
        science_data: MARSIS science data dict keyed by
            ``"ECHO_<F>_<C>_DIP"``.
        geometry: Observation geometry, shared across channels.
        crs: Source coordinate reference system.
        operative_mode: Operative mode (e.g. ``"SS3"``).
        out_dir: Output directory.
        base: Base filename (without the per-channel suffix).
        dem: DEM source for the browse image (``"HRSC-MOLA"``, a
            path, or ``None``).
        buffer_km: DEM swath half-width in kilometres.
        lower_percentile: Lower percentile for image scaling.
        upper_percentile: Upper percentile for image scaling.
        vmin: Explicit minimum dB value for scaling.
        vmax: Explicit maximum dB value for scaling.
        power: If True, use ``10 * log10`` scaling. If None,
            auto-detected from data dtype.
        invert: If True, invert the grayscale.
        per_frame: If True, compute vmin/vmax per column.
        verbose: If True, print progress messages.
    """
    out_dir = Path(out_dir)
    for f_str, c_str, key in iter_channel_filter(operative_mode):
        if key not in science_data:
            continue
        suffix = f"_{f_str}_{c_str}".lower()
        full_base = f"{base}{suffix}"
        if verbose:
            print(f"Exporting image: {full_base}")
        _export_images(
            science_data[key],
            geometry,
            crs,
            out_dir=out_dir,
            base=full_base,
            dem=dem,
            buffer_km=buffer_km,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            vmin=vmin,
            vmax=vmax,
            power=power,
            invert=invert,
            per_frame=per_frame,
            verbose=verbose,
        )
