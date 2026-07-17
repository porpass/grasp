# SPDX-License-Identifier: BSD-3-Clause
"""Canonical accessor for the per-instrument support matrix."""

import tomllib
from functools import cache
from pathlib import Path

_SUPPORT_PATH = Path(__file__).resolve().parent.parent / "processing_support.toml"


@cache
def load_support() -> dict[str, dict[str, dict[str, bool]]]:
    """Return the full support matrix as ``{instrument: {product_type: {stage: bool}}}``.

    Read from :file:`src/grasp/processing_support.toml`. The result is
    cached; the file is read once per process.
    """
    with _SUPPORT_PATH.open("rb") as f:
        raw = tomllib.load(f)
    # TOML sections come out as {"LRS": {"EDR": {"preprocessing": True, ...}}}
    # already — nothing to reshape.
    return raw
