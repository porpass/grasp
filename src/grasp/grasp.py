# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end GRaSP processor.

Runs a complete processing job from a TOML job file. The processor is
deliberately instrument-agnostic: every stage method on
:class:`RadarSounder` self-resolves its parameters from
``processing_parameters`` and self-gates on its own ``enabled`` flag,
and :meth:`RadarSounder._get_pp` automatically validates each enabled
stage against the instrument's ``PROCESSING_SUPPORT`` table (disabling
``unsupported`` stages and warning on ``untested`` ones). As a result
the processor needs no per-instrument branching and no manual
``enabled`` checks; it simply walks the canonical pipeline in order and
lets each subclass decide what to run.
"""

from pathlib import Path

from .instantiator import radar_sounder

from .radar_sounder import RadarSounder

from . import __description__, __version__, __author__, __funding__, __grants__, __license__



# Canonical processing order. Each entry is the public method name on the
# RadarSounder instance. Every method is safe to call unconditionally: it
# resolves its own parameters and returns early when its stage is disabled
# or unsupported for the active instrument/product type.
_PIPELINE = (
    "preprocess",
    "range_compression",
    "emi_suppression",
    "ionospheric_correction",
    "sar_process",
    "multilook",
    "simulate_clutter",
    "finalize",
)

# Human-readable labels for verbose stage announcements. Keep keys
# aligned with _PIPELINE so missing labels surface as a KeyError.
_STAGE_LABELS = {
    "preprocess":             "Preprocess",
    "range_compression":      "Range Compression",
    "emi_suppression":        "EMI Suppression",
    "ionospheric_correction": "Ionospheric Correction",
    "sar_process":            "SAR Processing",
    "multilook":              "Multilook",
    "simulate_clutter":       "Clutter Simulation",
    "finalize":               "Finalize",
}


def process_job(job_file: str | Path, *, verbose: bool | None = None) -> RadarSounder:
    """Run a full GRaSP processing job from a TOML job file.

    Instantiates a radar sounder from the job file and walks the
    canonical processing pipeline (preprocessing through multilook,
    plus an optional clutter simulation). Per-stage behavior, output,
    and state persistence are driven entirely by the job file's
    ``processing_parameters``; this function only orchestrates call
    order.

    Args:
        job_file: Path to a TOML job file.
        verbose: If provided, overrides the job file's ``verbose``
            flag for the orchestration-level messages and is passed
            down to each stage. If ``None``, each stage falls back to
            ``processing_parameters.verbose``.

    Returns:
        The processed :class:`RadarSounder` instance, for inspection
        or further processing.
    """
    rs = radar_sounder(job_file)
    pp = rs.processing_parameters
    verbose = pp.verbose if verbose is None else verbose

    _print_banner()

    if verbose:
        print_job_summary(rs)

    # ------------------------------------------------------------------ #
    #  Read input files                                                  #
    # ------------------------------------------------------------------ #
    rs.read_files(verbose=verbose)

    # ------------------------------------------------------------------ #
    #  Recompute geometry (SPICE)                                        #
    # ------------------------------------------------------------------ #
    rs.recompute_geometry(verbose=verbose)

    # ------------------------------------------------------------------ #
    #  Processing pipeline                                               #
    # ------------------------------------------------------------------ #
    for stage in _PIPELINE:
        if verbose:
            print(f"--- Stage: {_STAGE_LABELS[stage]} ---", flush=True)
        getattr(rs, stage)(verbose=verbose)

    return rs


def _print_banner(width: int = 120) -> None:
    """Print the GRaSP attribution banner.

    Always printed regardless of verbosity, so version, funding, and
    license attribution appear on every run.

    Args:
        width: Total character width to center the banner lines within.
    """
    rule = "-" * width
    lines = (
        __description__,
        f"Current Version {__version__}",
        f"Written by {__author__}",
        "Documentation: https://github.com/porpass/grasp",
        f"Funding Source(s): {__funding__}",
        f"Grant Number(s): {__grants__}",
        f"License: {__license__}",
    )
    print(rule, flush=True)
    for line in lines:
        print(line.center(width), flush=True)
    print(rule, flush=True)
    print("", flush=True)


def print_job_summary(rs: RadarSounder) -> None:
    """Print the resolved job configuration before processing.

    Delegates to :meth:`RadarSounder.print_job_summary` — kept as a
    free function so existing ``grasp.print_job_summary(rs)`` callers
    continue to work.

    Args:
        rs: The radar sounder instance, already populated with
            metadata, parameters, and processing parameters.
    """
    rs.print_job_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a GRaSP processing job.")
    parser.add_argument("job_file", help="Path to a TOML job file.")
    parser.add_argument("-v", "--verbose", action="store_true", default=None,
                        help="Override the job file's verbose flag.")
    args = parser.parse_args()

    process_job(args.job_file, verbose=args.verbose)