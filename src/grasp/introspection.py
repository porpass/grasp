# SPDX-License-Identifier: BSD-3-Clause
"""Public introspection surface of GRaSP's processing-parameter model.

Stable public API for downstream tools (``porpass/daemon``, and any other
external process that needs to understand GRaSP's stages, fields, and
correctness-only choice sets) without importing GRaSP's internal modules
directly.

**Stability contract.** Symbols re-exported here are part of the public API
and follow the project's version policy — additions are safe, removals or
renames need a coordinated release. Everything under ``grasp.schema.*`` is
**deprecated** and will be removed in the next release; migrate to
``grasp.introspection``.

Typical use::

    from dataclasses import fields
    from grasp.introspection import (
        STAGES, GLOBALS,
        CHOICES, RESTRICTIONS, get_choices,
        WINDOW_NAMES, WINDOW_FIELDS,
        LONG_TO_ATTR, ATTR_TO_LONG,
        load_support,
        get_defaults, MetaDataInfo,
    )

    defaults = get_defaults("SHARAD", MetaDataInfo(product_type="EDR"))
    for stage_attr, cls in STAGES:
        for f in fields(cls):
            valid = get_choices(stage_attr, f.name, "SHARAD", "EDR")
            ...

Scope. This module deliberately exposes the *primitives* needed to describe
GRaSP's parameter model: the ordered stage/dataclass registry, the
correctness-only enum choices (what values the validator will accept),
per-(instrument, product-type) restrictions on those choices, the
TOML-section alias map, the support matrix, and per-(instrument,
product-type) resolved default values. It does **not** carry form-rendering
curation — display orderings, dropdown short-lists, visibility rules,
human-readable default hints — which live in the consuming tool.
"""

from .grasp_types import (
    ClutterSimParams,
    EMISuppresionParams,
    FinalOutputParams,
    IonoCompParams,
    MetaDataInfo,
    MLKParams,
    OutputParameters,
    PlotParameters,
    PreprocessingParams,
    ProcessingParameters,
    RangeCompressionParams,
    SARParams,
)
from .processing.windows import WINDOW_NAMES
from .processing_defaults import get_defaults
from .schema.choices import CHOICES, RESTRICTIONS, get_choices
from .schema.support import load_support
from .section_aliases import ATTR_TO_LONG, LONG_TO_ATTR

# Canonical pipeline order. Each entry pairs the short dataclass-attribute
# name on ``ProcessingParameters`` (used by the validator, the loader, and
# ``LONG_TO_ATTR``) with the dataclass itself (walk with
# ``dataclasses.fields(cls)``).
STAGES: tuple[tuple[str, type], ...] = (
    ("preprocessing",     PreprocessingParams),
    ("range_compression", RangeCompressionParams),
    ("emi_suppression",   EMISuppresionParams),
    ("iono_comp",         IonoCompParams),
    ("sar",               SARParams),
    ("mlk",               MLKParams),
    ("csim",              ClutterSimParams),
)

# Config sections that live at the top level of ProcessingParameters rather
# than being a pipeline stage. ``ProcessingParameters`` itself owns the
# global scalars (``out_dir``, ``verbose``) accessed via the ``"general"``
# key; downstream tools should filter its fields to those two names.
GLOBALS: tuple[tuple[str, type], ...] = (
    ("general",          ProcessingParameters),
    ("output",           OutputParameters),
    ("plots",            PlotParameters),
    ("final_output",     FinalOutputParams),
)

# (stage_attr, field_name) pairs whose choice set is a window-name selector.
# Derived by *value* equality against :data:`WINDOW_NAMES` — safe if GRaSP
# ever rebuilds or reorders that frozenset internally. Downstream tools use
# this to key their curated window-display order + alias mapping instead of
# relying on object identity (``CHOICES.get(key) is WINDOW_NAMES``), which
# was a fragile implicit contract.
WINDOW_FIELDS: frozenset[tuple[str, str]] = frozenset(
    key for key, valid in CHOICES.items() if valid == WINDOW_NAMES
)

__all__ = [
    "STAGES",
    "GLOBALS",
    "CHOICES",
    "RESTRICTIONS",
    "get_choices",
    "WINDOW_NAMES",
    "WINDOW_FIELDS",
    "LONG_TO_ATTR",
    "ATTR_TO_LONG",
    "load_support",
    "get_defaults",
    "MetaDataInfo",
]
