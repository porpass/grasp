# SPDX-License-Identifier: BSD-3-Clause
"""Unified ``grasp`` command-line entry point.

Two subcommands today:

- ``grasp run <job.toml> [-v]`` — runs a full processing job. Wraps
  :func:`grasp.grasp.process_job`.
- ``grasp export-schema --out <dir> [--instrument SHARAD] [--product EDR]``
  — writes JSON schema artifacts for consumption by external form
  generators (PORPASS). Wraps :func:`grasp.schema.export.export_schema`.

The legacy ``python -m grasp.grasp <job>`` invocation still works —
``grasp.grasp`` retains its ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Dispatch a ``grasp`` invocation. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="grasp",
        description="GRaSP: Generalized Radar Sounder Processor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a full processing job from a TOML file.")
    run.add_argument("job_file", help="Path to a TOML job file.")
    run.add_argument(
        "-v", "--verbose",
        action="store_true", default=None,
        help="Override the job file's verbose flag.",
    )

    export = sub.add_parser(
        "export-schema",
        help="Write JSON schema artifacts for external UIs (PORPASS).",
    )
    export.add_argument(
        "--out", required=True, type=Path,
        help="Target directory; created if missing.",
    )
    export.add_argument(
        "--instrument",
        help="Emit only this instrument (e.g. SHARAD, MARSIS, LRS).",
    )
    export.add_argument(
        "--product",
        help="Emit only this product type (e.g. EDR, RDR, US_RDR). "
             "Requires --instrument.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        from .grasp import process_job
        process_job(args.job_file, verbose=args.verbose)
        return 0

    if args.command == "export-schema":
        if args.product and not args.instrument:
            parser.error("--product requires --instrument")
        from .schema.export import export_schema
        written = export_schema(
            args.out,
            instrument=args.instrument,
            product_type=args.product,
        )
        for path in written:
            print(path)
        return 0

    parser.error(f"unknown command: {args.command!r}")
    return 2  # unreachable — parser.error exits


if __name__ == "__main__":
    sys.exit(main())
