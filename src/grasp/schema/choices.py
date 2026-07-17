# SPDX-License-Identifier: BSD-3-Clause
"""Canonical enum-field choice registry.

Single source of truth for the allowed string values of every enum-style
field on the processing-parameter dataclasses. Consulted by both the
runtime validator on :class:`RadarSounder` (so job-file typos surface at
construction time) and the schema exporter (so PORPASS renders the same
choice lists the loader accepts).

Two layers:

- :data:`CHOICES`: global ``(stage_attr, field_name) -> frozenset[str]``.
  Applies unless overridden.
- :data:`RESTRICTIONS`: per-instrument, per-product-type narrowing on top
  of the global set (e.g. MARSIS has no calibrated chirps, so
  ``chirp_type`` is restricted to ``{"IDEAL"}``).

Use :func:`get_choices` to resolve the effective set for a combo.
"""

from ..processing.windows import WINDOW_NAMES

# Curated matplotlib colormap shortlist. Extend as needed — kept small so
# PORPASS renders a manageable dropdown. Runtime does not currently enforce
# this (matplotlib itself will error on unknown names during plotting).
CMAP_NAMES: frozenset[str] = frozenset({
    "gray", "gray_r", "viridis", "plasma", "inferno", "magma", "cividis",
    "hot", "jet", "turbo", "bwr", "seismic", "coolwarm", "RdBu", "RdBu_r",
    "cubehelix",
})

# Canonical display order for window-typed fields. The 10 entries below are
# the values the web form should show, in this order. Every other value in
# WINDOW_NAMES is a hidden alias (see WINDOW_ALIASES) — the runtime validator
# still accepts them, but PORPASS doesn't advertise them as choices.
WINDOW_DISPLAY_ORDER: tuple[str, ...] = (
    "RECTANGLE",
    "HANN",
    "HAMMING",
    "BLACKMAN",
    "BLACKMAN-HARRIS",
    "NUTTALL",
    "BARTLETT",
    "COSINE",
    "FLATTOP",
    "TUKEY",
)

# Non-canonical → canonical mapping for window aliases. Derived from the
# equivalence classes in :func:`grasp.processing.windows.form_window`
# (each ``elif wt in {...}:`` block).
WINDOW_ALIASES: dict[str, str] = {
    "NONE":            "RECTANGLE",
    "HANNING":         "HANN",
    "BLACKMAN_HARRIS": "BLACKMAN-HARRIS",
    "BH":              "BLACKMAN-HARRIS",
    "SINE":            "COSINE",
    "RAISED_COSINE":   "COSINE",
}

# Global choice map. Keys are (stage attribute, field name); values are the
# uppercase-normalised allowed set. String membership tests must upper-case
# the candidate before checking, since job files may use any case.
CHOICES: dict[tuple[str, str], frozenset[str]] = {
    ("range_compression", "window"):           WINDOW_NAMES,
    ("range_compression", "chirp_type"):       frozenset({"IDEAL", "CALIBRATED"}),
    ("range_compression", "filter_type"):      frozenset({"MATCHED", "INVERSE"}),
    ("emi_suppression",   "method"):           frozenset({"ADAPTIVE"}),
    ("emi_suppression",   "statistic"):        frozenset({"MAD", "STD"}),
    ("emi_suppression",   "replace_strategy"): frozenset({"BASELINE", "INTERP", "ZERO"}),
    ("iono_comp",         "method"):           frozenset({"CAMPBELL", "CONTRAST"}),
    ("iono_comp",         "metric"):           frozenset({"L1", "L4", "ENTROPY", "PEAK_SNR"}),
    ("iono_comp",         "window"):           WINDOW_NAMES,
    ("sar",               "method"):           frozenset({"UNFOCUSED", "RANGE-DOPPLER", "BACKSCATTER"}),
    ("sar",               "window"):           WINDOW_NAMES,
    ("mlk",               "window_type"):      WINDOW_NAMES,
    ("output",            "byte_order"):       frozenset({"BIG", "LITTLE"}),
    ("output",            "data_output_type"): frozenset({"BASIC"}),
    ("plots",             "cmap"):             CMAP_NAMES,
}

# Per-(instrument, product_type) narrowing of CHOICES entries. Absent keys
# mean "use the global set". Present keys must be a subset of the matching
# CHOICES entry.
RESTRICTIONS: dict[tuple[str, str, str, str], frozenset[str]] = {
    ("MARSIS", "EDR", "range_compression", "chirp_type"): frozenset({"IDEAL"}),
    ("MARSIS", "RDR", "range_compression", "chirp_type"): frozenset({"IDEAL"}),
}


def get_choices(
    stage_attr: str,
    field_name: str,
    instrument: str | None = None,
    product_type: str | None = None,
) -> frozenset[str] | None:
    """Return the effective allowed set for one field.

    Args:
        stage_attr: Short dataclass-attribute name of the stage
            (``"range_compression"``, ``"iono_comp"``, ...).
        field_name: Field name on the stage dataclass.
        instrument: Optional instrument identifier (case-insensitive).
        product_type: Optional PDS product type (case-insensitive with
            ``_`` preserved, e.g. ``"US_RDR"``).

    Returns:
        The narrower :data:`RESTRICTIONS` frozenset when a matching
        entry exists; the global :data:`CHOICES` frozenset otherwise;
        ``None`` if the field has no bounded choice list.
    """
    global_set = CHOICES.get((stage_attr, field_name))
    if global_set is None:
        return None
    if instrument is None or product_type is None:
        return global_set
    key = (instrument.upper(), product_type.upper(), stage_attr, field_name)
    return RESTRICTIONS.get(key, global_set)


def _is_window_field(stage_attr: str, field_name: str) -> bool:
    """True iff ``(stage_attr, field_name)`` selects a window name.

    Identity check against :data:`WINDOW_NAMES` — every window field is
    registered in :data:`CHOICES` as pointing at the same frozenset
    instance.
    """
    return CHOICES.get((stage_attr, field_name)) is WINDOW_NAMES


def choice_entries(
    stage_attr: str,
    field_name: str,
    instrument: str | None = None,
    product_type: str | None = None,
) -> list[dict] | None:
    """Return the ordered choice-object list emitted by the schema exporter.

    Each entry is a dict with keys:

    - ``value``: the accepted string (uppercase, exactly as the
      validator matches).
    - ``ui``: whether the web form advertises this option (True) or
      keeps it as a hidden alias (False).
    - ``alias_of``: present only when ``ui`` is False and the value is
      a non-canonical alias of another entry in the same list.

    For window-typed fields, the ui:true entries appear first in
    :data:`WINDOW_DISPLAY_ORDER`; every other window name is emitted as
    ui:false with :data:`WINDOW_ALIASES` supplying the canonical value.

    For all other enum fields, all entries are ui:true, sorted for
    determinism, with no aliases.

    Returns:
        List of dicts, or ``None`` if the field has no bounded choice
        list.
    """
    valid = get_choices(stage_attr, field_name, instrument, product_type)
    if valid is None:
        return None

    if _is_window_field(stage_attr, field_name):
        entries: list[dict] = []
        # Canonical, ui:true, in the curated display order.
        for value in WINDOW_DISPLAY_ORDER:
            if value in valid:
                entries.append({"value": value, "ui": True})
        # Aliases + any other value we still accept but don't advertise.
        remaining = sorted(valid - set(WINDOW_DISPLAY_ORDER))
        for value in remaining:
            entry: dict = {"value": value, "ui": False}
            canonical = WINDOW_ALIASES.get(value)
            if canonical is not None and canonical in valid:
                entry["alias_of"] = canonical
            entries.append(entry)
        return entries

    return [{"value": v, "ui": True} for v in sorted(valid)]
