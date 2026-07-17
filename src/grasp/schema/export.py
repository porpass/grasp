# SPDX-License-Identifier: BSD-3-Clause
"""Build and export the per-(instrument, product_type) schema artifact.

Contract: see the ``feature/export-schema`` plan file. Each artifact is one
JSON object with a fixed shape (top-level ``schema_version``,
``grasp_version``, ``target_body``, ``inputs``, ``stages``, ``globals``)
that PORPASS consumes to render job-file forms.
"""

from __future__ import annotations

import json
import typing
from dataclasses import Field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from ..grasp_types import (
    FinalOutputParams,
    MetaDataInfo,
    OutputParameters,
    PlotParameters,
    PreprocessingParams,
    ProcessingParameters,
    RangeCompressionParams,
    EMISuppresionParams,
    IonoCompParams,
    SARParams,
    MLKParams,
    ClutterSimParams,
)
from ..processing_defaults import get_defaults
from ..section_aliases import ATTR_TO_LONG
from .choices import choice_entries, get_choices
from .defaults_notes import DEFAULT_NOTES
from .docstrings import field_help, stage_help
from .support import load_support


SCHEMA_VERSION = "1.1"

# Canonical stage order — mirrors grasp.grasp._PIPELINE but keyed by
# the parameter-dataclass attribute name rather than the RadarSounder
# method name.
_STAGES: tuple[tuple[str, type], ...] = (
    ("preprocessing",     PreprocessingParams),
    ("range_compression", RangeCompressionParams),
    ("emi_suppression",   EMISuppresionParams),
    ("iono_comp",         IonoCompParams),
    ("sar",               SARParams),
    ("mlk",               MLKParams),
    ("csim",              ClutterSimParams),
)

# Config sections that live at the top level of ProcessingParameters
# rather than being a pipeline stage. Emitted under ``"globals"``.
_GLOBALS: tuple[tuple[str, type], ...] = (
    ("general",          ProcessingParameters),
    ("output",           OutputParameters),
    ("plots",            PlotParameters),
    ("final_output",     FinalOutputParams),
)

# Human-friendly stage titles for form headers.
_STAGE_TITLES: dict[str, str] = {
    "preprocessing":     "Preprocessing",
    "range_compression": "Range Compression",
    "emi_suppression":   "EMI Suppression",
    "iono_comp":         "Ionospheric Compensation",
    "sar":               "SAR Processing",
    "mlk":               "Multilooking",
    "csim":              "Clutter Simulation",
    "general":           "General",
    "output":            "Output Parameters",
    "plots":             "Plot Parameters",
    "final_output":      "Final Output",
}

# Fields whose visibility in the form depends on the value of a sibling.
VISIBLE_WHEN: dict[tuple[str, str], dict[str, Any]] = {
    ("range_compression", "window_alpha"): {"field": "window",           "equals": "TUKEY"},
    ("sar",               "window_alpha"): {"field": "window",           "equals": "TUKEY"},
    ("mlk",               "window_alpha"): {"field": "window_type",      "equals": "TUKEY"},
    ("iono_comp",         "window_alpha"): {"field": "window",           "equals": "TUKEY"},
    ("emi_suppression",   "interp_pad"):   {"field": "replace_strategy", "equals": "INTERP"},
    ("iono_comp",         "contrast_L_eq"):    {"field": "method", "equals": "CONTRAST"},
    ("iono_comp",         "campbell_b"):       {"field": "method", "equals": "CAMPBELL"},
    ("iono_comp",         "campbell_n_phase"): {"field": "method", "equals": "CAMPBELL"},
    ("iono_comp",         "campbell_delta"):   {"field": "method", "equals": "CAMPBELL"},
    ("iono_comp",         "sgn"):              {"field": "method", "equals": "CAMPBELL"},
    ("sar",               "number_of_looks"):  {"field": "method", "equals": "BACKSCATTER"},
}

# Stage-level: selecting this value for this field disables the listed stages.
STAGE_DISABLES: dict[tuple[str, str, str], list[str]] = {
    ("sar", "method", "BACKSCATTER"): ["multilooking"],
}

# Per-(instrument, product) input-file requirements. Emitted under
# ``"inputs"`` so PORPASS knows which upload widgets to show.
_INPUT_FILES: dict[tuple[str, str], tuple[str, ...]] = {
    ("SHARAD", "EDR"):    ("label", "science", "auxiliary"),
    ("SHARAD", "RDR"):    ("label", "science", "auxiliary"),
    ("SHARAD", "US_RDR"): ("label", "science", "auxiliary"),
    ("MARSIS", "EDR"):    ("label", "science", "auxiliary"),
    ("MARSIS", "RDR"):    ("label", "science", "auxiliary"),
    ("LRS",    "EDR"):    ("label", "science"),
    ("LRS",    "RDR"):    ("label", "science"),
}

_INPUT_HELP: dict[str, str] = {
    "label":     "PDS label (PDS3/PDS4)",
    "science":   "PDS science file",
    "auxiliary": "PDS auxiliary file",
}

_TARGET_BODY: dict[str, str] = {
    "SHARAD": "MARS",
    "MARSIS": "MARS",  # Phobos/transit MARSIS observations aren't supported yet
    "LRS":    "MOON",
}


########################################################################
# Public API
########################################################################

def build_schema(instrument: str, product_type: str) -> dict[str, Any]:
    """Assemble the schema dict for one ``(instrument, product_type)`` combo.

    Args:
        instrument: ``"SHARAD"``, ``"MARSIS"``, or ``"LRS"``
            (case-insensitive).
        product_type: PDS product type, e.g. ``"EDR"``, ``"RDR"``,
            ``"US_RDR"`` (case-insensitive).

    Returns:
        A dict conforming to the schema contract. Serialise with
        :func:`json.dumps` or write via :func:`export_schema`.

    Raises:
        ValueError: If the combo is not present in the support matrix.
    """
    instrument = instrument.upper()
    product_type = product_type.upper()

    support = load_support()
    if instrument not in support:
        raise ValueError(f"Unknown instrument {instrument!r}; support matrix "
                         f"has {sorted(support)}")
    if product_type not in support[instrument]:
        raise ValueError(
            f"Unknown product_type {product_type!r} for {instrument}; "
            f"support matrix has {sorted(support[instrument])}"
        )
    stage_support = support[instrument][product_type]

    defaults_table = get_defaults(instrument, MetaDataInfo(product_type=product_type))

    stages = [
        _build_stage(stage_attr, cls, defaults_table, stage_support,
                     instrument, product_type)
        for stage_attr, cls in _STAGES
    ]
    globals_ = [
        _build_stage(stage_attr, cls, defaults_table, stage_support,
                     instrument, product_type)
        for stage_attr, cls in _GLOBALS
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "grasp_version":  _grasp_version(),
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "instrument":     instrument,
        "product":        product_type,
        "target_body":    _TARGET_BODY[instrument],
        "inputs":         _build_inputs(instrument, product_type),
        "stages":         stages,
        "globals":        globals_,
    }


def export_schema(
    target_dir: str | Path,
    instrument: str | None = None,
    product_type: str | None = None,
) -> list[Path]:
    """Write one JSON schema file per combo into ``target_dir``.

    Args:
        target_dir: Directory to write to. Created if it doesn't exist.
        instrument: If given, only write this instrument's files.
        product_type: If given (with ``instrument``), only write this one
            combo's file.

    Returns:
        List of paths written, in the order they were emitted.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    combos = _select_combos(instrument, product_type)
    written: list[Path] = []
    for inst, product in combos:
        schema = build_schema(inst, product)
        path = target_dir / f"{inst}_{product}.schema.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")
        written.append(path)
    return written


########################################################################
# Helpers
########################################################################

def _select_combos(
    instrument: str | None,
    product_type: str | None,
) -> list[tuple[str, str]]:
    support = load_support()
    if instrument is None:
        return [
            (inst, product)
            for inst in sorted(support)
            for product in sorted(support[inst])
        ]
    inst = instrument.upper()
    if inst not in support:
        raise ValueError(f"Unknown instrument {instrument!r}")
    if product_type is None:
        return [(inst, product) for product in sorted(support[inst])]
    product = product_type.upper()
    if product not in support[inst]:
        raise ValueError(
            f"Unknown product_type {product_type!r} for {instrument}"
        )
    return [(inst, product)]


def _build_stage(
    stage_attr: str,
    cls: type,
    defaults_table: dict[str, dict[str, Any]],
    stage_support: dict[str, bool],
    instrument: str,
    product_type: str,
) -> dict[str, Any]:
    """Assemble one stage or globals block."""
    long_name = ATTR_TO_LONG.get(stage_attr, stage_attr)
    supported = stage_support.get(stage_attr, True)  # globals are always "supported"
    stage_defaults = defaults_table.get(stage_attr, {})
    helps = field_help(cls)

    field_dicts = [
        _build_field(stage_attr, f, stage_defaults, helps, instrument, product_type)
        for f in _dataclass_fields(cls, stage_attr)
    ]

    out: dict[str, Any] = {
        "key":        long_name,
        "dataclass":  cls.__name__,
        "supported":  supported,
        "title":      _STAGE_TITLES.get(stage_attr, stage_attr),
        "help":       stage_help(cls),
        "fields":     field_dicts,
    }
    for (s, field_name, value), disables in STAGE_DISABLES.items():
        if s == stage_attr:
            out.setdefault("disables", []).append({
                "when":   {"field": field_name, "equals": value},
                "stages": list(disables),
            })
    return out


def _dataclass_fields(cls: type, stage_attr: str) -> list[Field]:
    """Return the dataclass fields for one stage.

    For the top-level ``general`` block we only want the scalar globals
    (``out_dir``, ``verbose``) — not the nested stage-parameter fields
    like ``preprocessing`` / ``range_compression`` / etc.
    """
    if stage_attr == "general":
        return [f for f in fields(cls) if f.name in ("out_dir", "verbose")]
    return list(fields(cls))


def _build_field(
    stage_attr: str,
    field: Field,
    stage_defaults: dict[str, Any],
    helps: dict[str, str],
    instrument: str,
    product_type: str,
) -> dict[str, Any]:
    """Assemble one field entry."""
    name = field.name
    typing_info = _field_typing_info(
        stage_attr, name, field.type, instrument, product_type
    )

    default = stage_defaults.get(name, None)
    entry: dict[str, Any] = {
        "name":    name,
        "default": default,
        "help":    helps.get(name, ""),
    }
    entry.update(typing_info)
    if default is None:
        note = DEFAULT_NOTES.get((stage_attr, name))
        if note is not None:
            entry["default_note"] = note
    visible = VISIBLE_WHEN.get((stage_attr, name))
    if visible is not None:
        entry["visible_when"] = dict(visible)
    return entry


def _field_typing_info(
    stage_attr: str,
    field_name: str,
    annotation: Any,
    instrument: str,
    product_type: str,
) -> dict[str, Any]:
    """Return the type-shaped payload for one field.

    Shape:

    - Non-enum:  ``{"type": "<bool|int|float|str>"}``
    - Enum:      ``{"type": "enum", "value_type": "<str|int>",
                    "choices": [{"value": ..., "ui": bool,
                                 "alias_of": <str, optional>}, ...]}``

    ``"enum"`` iff the field is in the CHOICES registry (respecting
    per-combo restrictions), or the annotation is ``Literal[-1, 1] | None``
    (``sgn``). The ``value_type`` key tells PORPASS whether to render/emit
    the value as a quoted string or a bare number when writing TOML back
    out — required to keep the round-trip type-correct for numeric enums.
    """
    inner = _strip_optional(annotation)

    # sgn is Literal[-1, 1] — numeric enum. Emit ints (not strings) so
    # a re-serialised job writes ``sgn = -1`` rather than ``sgn = "-1"``,
    # matching the int-typed dataclass field.
    if get_origin(inner) is Literal:
        values = sorted(get_args(inner))
        return {
            "type":       "enum",
            "value_type": "int",
            "choices":    [{"value": v, "ui": True} for v in values],
        }

    entries = choice_entries(stage_attr, field_name, instrument, product_type)
    if entries is not None:
        return {
            "type":       "enum",
            "value_type": "str",
            "choices":    entries,
        }

    # Path-like fields get "str".
    if inner in (str, Path) or _is_str_or_path(inner):
        return {"type": "str"}
    if inner is bool:
        return {"type": "bool"}
    if inner is int:
        return {"type": "int"}
    if inner is float:
        return {"type": "float"}
    return {"type": "str"}


def _strip_optional(annotation: Any) -> Any:
    """Strip ``| None`` / ``Optional[T]`` down to ``T``."""
    if get_origin(annotation) is typing.Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    # Support the PEP 604 form (``X | None``) which shows up in get_origin
    # as types.UnionType at runtime.
    if hasattr(annotation, "__args__") and type(None) in getattr(annotation, "__args__", ()):
        args = [a for a in annotation.__args__ if a is not type(None)]
        if len(args) == 1:
            return args[0]
    if isinstance(annotation, str):
        # String forward-refs (from `from __future__ import annotations`).
        # Best-effort match on the common shapes.
        stripped = annotation.replace(" ", "")
        for base in ("bool", "int", "float", "str", "Path"):
            if stripped == f"{base}|None" or stripped == base:
                return {"bool": bool, "int": int, "float": float,
                        "str": str, "Path": Path}[base]
    return annotation


def _is_str_or_path(annotation: Any) -> bool:
    if annotation in (str, Path):
        return True
    if isinstance(annotation, str):
        return annotation.strip() in ("str", "Path")
    return False


def _build_inputs(instrument: str, product_type: str) -> dict[str, dict[str, Any]]:
    files = _INPUT_FILES.get((instrument, product_type), ())
    return {
        f"{kind}_file": {"required": True, "help": _INPUT_HELP[kind]}
        for kind in files
    }


def _grasp_version() -> str:
    from .. import __version__
    return __version__
