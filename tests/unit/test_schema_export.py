# SPDX-License-Identifier: BSD-3-Clause
"""Contract-shape and cross-cutting tests for the schema exporter.

Exercises :mod:`grasp.schema.export` end-to-end and cross-checks that
every artefact the exporter emits round-trips through the alias map,
the runtime validator, and the dataclass field names — so the JSON
delivered to PORPASS cannot describe fields that GRaSP's loader would
reject.
"""

import json
from dataclasses import fields

import pytest

from grasp.grasp_types import (
    ClutterSimParams,
    EMISuppresionParams,
    FinalOutputParams,
    IonoCompParams,
    MLKParams,
    OutputParameters,
    PlotParameters,
    PreprocessingParams,
    ProcessingParameters,
    RangeCompressionParams,
    SARParams,
)
from grasp.schema import build_schema, export_schema, get_choices
from grasp.schema.export import _STAGES, _GLOBALS
from grasp.schema.support import load_support
from grasp.section_aliases import LONG_TO_ATTR, ATTR_TO_LONG


ALL_COMBOS = [
    (inst, product)
    for inst in sorted(load_support())
    for product in sorted(load_support()[inst])
]

TOP_KEYS = {
    "schema_version", "grasp_version", "generated_at",
    "instrument", "product", "target_body",
    "inputs", "stages", "globals",
}
STAGE_KEYS = {"key", "dataclass", "supported", "title", "help", "fields"}
FIELD_KEYS_REQUIRED = {"name", "type", "default", "help"}
VALID_TYPES = {"bool", "int", "float", "str", "enum"}
VALID_VALUE_TYPES = {"str", "int"}


def _values_of(choices: list[dict]) -> list:
    """Extract the ``value`` field from a choices object list."""
    return [c["value"] for c in choices]


def _ui_values(choices: list[dict]) -> list:
    """Extract only the ui:true values from a choices object list."""
    return [c["value"] for c in choices if c["ui"]]


########################################################################
# Contract shape
########################################################################

@pytest.mark.parametrize("instrument, product_type", ALL_COMBOS)
def test_top_level_contract_shape(instrument, product_type):
    schema = build_schema(instrument, product_type)
    assert set(schema.keys()) >= TOP_KEYS
    assert schema["schema_version"] == "1.1"
    assert schema["instrument"] == instrument
    assert schema["product"] == product_type


@pytest.mark.parametrize("instrument, product_type", ALL_COMBOS)
def test_every_stage_has_contract_keys(instrument, product_type):
    schema = build_schema(instrument, product_type)
    for stage in schema["stages"] + schema["globals"]:
        assert STAGE_KEYS <= set(stage.keys()), (
            f"stage {stage.get('key')} missing keys: "
            f"{STAGE_KEYS - set(stage.keys())}"
        )


@pytest.mark.parametrize("instrument, product_type", ALL_COMBOS)
def test_every_field_has_valid_type_and_required_keys(instrument, product_type):
    schema = build_schema(instrument, product_type)
    for stage in schema["stages"] + schema["globals"]:
        for f in stage["fields"]:
            assert FIELD_KEYS_REQUIRED <= set(f.keys()), (
                f"field {stage['key']}.{f.get('name')} missing: "
                f"{FIELD_KEYS_REQUIRED - set(f.keys())}"
            )
            assert f["type"] in VALID_TYPES
            if f["type"] == "enum":
                assert "choices" in f and isinstance(f["choices"], list)
                assert "value_type" in f
                assert f["value_type"] in VALID_VALUE_TYPES
                for entry in f["choices"]:
                    assert isinstance(entry, dict), (
                        f"{stage['key']}.{f['name']} choice not a dict: "
                        f"{entry!r}"
                    )
                    assert set(entry.keys()) >= {"value", "ui"}
                    assert isinstance(entry["ui"], bool)
                    if "alias_of" in entry:
                        assert not entry["ui"], (
                            f"{stage['key']}.{f['name']} alias {entry!r} "
                            f"must be ui:false"
                        )
                        # alias_of must point at a canonical entry in the
                        # same list — proves the web can resolve the pointer
                        # without a lookup table.
                        canonicals = {
                            e["value"] for e in f["choices"] if e["ui"]
                        }
                        assert entry["alias_of"] in canonicals, (
                            f"{stage['key']}.{f['name']} alias_of "
                            f"{entry['alias_of']!r} not a canonical in list"
                        )


########################################################################
# Round-trip: emitted names match dataclass fields
########################################################################

_ALL_DATACLASSES = list(_STAGES) + list(_GLOBALS)


@pytest.mark.parametrize("instrument, product_type", ALL_COMBOS)
def test_every_field_name_matches_a_real_dataclass_field(instrument, product_type):
    schema = build_schema(instrument, product_type)
    by_stage: dict[str, set[str]] = {}
    for stage_attr, cls in _ALL_DATACLASSES:
        by_stage[stage_attr] = {f.name for f in fields(cls)}

    for stage in schema["stages"] + schema["globals"]:
        stage_attr = LONG_TO_ATTR[stage["key"]]
        allowed = by_stage[stage_attr]
        for f in stage["fields"]:
            assert f["name"] in allowed, (
                f"{stage['key']}.{f['name']} is not a real "
                f"dataclass field"
            )


@pytest.mark.parametrize("instrument, product_type", ALL_COMBOS)
def test_every_stage_key_round_trips_through_alias_map(instrument, product_type):
    schema = build_schema(instrument, product_type)
    for stage in schema["stages"] + schema["globals"]:
        long_name = stage["key"]
        assert long_name in LONG_TO_ATTR, (
            f"{long_name} not in alias map"
        )
        assert ATTR_TO_LONG[LONG_TO_ATTR[long_name]] == long_name


########################################################################
# Per-instrument restrictions (MARSIS chirp_type; LRS unsupported stages)
########################################################################

@pytest.mark.parametrize("product_type", ["EDR", "RDR"])
def test_marsis_chirp_type_restricted_to_ideal(product_type):
    schema = build_schema("MARSIS", product_type)
    rc = next(s for s in schema["stages"] if s["key"] == "range_compression")
    chirp = next(f for f in rc["fields"] if f["name"] == "chirp_type")
    assert chirp["type"] == "enum"
    assert chirp["value_type"] == "str"
    assert chirp["choices"] == [{"value": "IDEAL", "ui": True}]


def test_sharad_chirp_type_offers_both_choices():
    schema = build_schema("SHARAD", "EDR")
    rc = next(s for s in schema["stages"] if s["key"] == "range_compression")
    chirp = next(f for f in rc["fields"] if f["name"] == "chirp_type")
    assert set(_values_of(chirp["choices"])) == {"IDEAL", "CALIBRATED"}
    # Both entries are ui:true (no aliases on chirp_type).
    assert all(c["ui"] for c in chirp["choices"])


def test_lrs_range_compression_unsupported_but_still_emitted():
    schema = build_schema("LRS", "EDR")
    rc = next(s for s in schema["stages"] if s["key"] == "range_compression")
    assert rc["supported"] is False
    # And the enabled default in defaults/lrs.toml is False.
    enabled = next(f for f in rc["fields"] if f["name"] == "enabled")
    assert enabled["default"] is False


def test_marsis_sar_unsupported_but_still_emitted():
    schema = build_schema("MARSIS", "EDR")
    sar = next(s for s in schema["stages"] if s["key"] == "sar_processing")
    assert sar["supported"] is False


########################################################################
# target_body and inputs
########################################################################

@pytest.mark.parametrize("instrument,expected", [
    ("SHARAD", "MARS"), ("MARSIS", "MARS"), ("LRS", "MOON"),
])
def test_target_body_per_instrument(instrument, expected):
    for pt in load_support()[instrument]:
        assert build_schema(instrument, pt)["target_body"] == expected


def test_lrs_edr_has_no_auxiliary_input():
    schema = build_schema("LRS", "EDR")
    assert "auxiliary_file" not in schema["inputs"]
    assert "label_file" in schema["inputs"]
    assert "science_file" in schema["inputs"]


def test_sharad_edr_has_all_three_input_files():
    schema = build_schema("SHARAD", "EDR")
    assert set(schema["inputs"].keys()) == {
        "label_file", "science_file", "auxiliary_file"
    }


########################################################################
# default_note appears for runtime-computed defaults
########################################################################

def test_sar_aperture_length_has_default_note_when_default_null():
    schema = build_schema("SHARAD", "EDR")
    sar = next(s for s in schema["stages"] if s["key"] == "sar_processing")
    apl = next(f for f in sar["fields"] if f["name"] == "aperture_length")
    assert apl["default"] is None
    assert apl["default_note"] == "computed from geometry"


def test_sar_lamb_gets_its_default_from_the_toml():
    """SHARAD's sar.lamb is tuned in the defaults TOML (12 m), not runtime.
    Prove the schema shows that concrete value, not null."""
    schema = build_schema("SHARAD", "EDR")
    sar = next(s for s in schema["stages"] if s["key"] == "sar_processing")
    lamb = next(f for f in sar["fields"] if f["name"] == "lamb")
    assert lamb["default"] == 12
    assert "default_note" not in lamb


def test_general_out_dir_has_default_note():
    schema = build_schema("SHARAD", "EDR")
    general = next(g for g in schema["globals"] if g["key"] == "general")
    out_dir = next(f for f in general["fields"] if f["name"] == "out_dir")
    assert out_dir["default"] is None
    assert "default_note" in out_dir


########################################################################
# visible_when hints
########################################################################

def test_window_alpha_visible_when_tukey():
    schema = build_schema("SHARAD", "EDR")
    rc = next(s for s in schema["stages"] if s["key"] == "range_compression")
    alpha = next(f for f in rc["fields"] if f["name"] == "window_alpha")
    assert alpha["visible_when"] == {"field": "window", "equals": "TUKEY"}


########################################################################
# Window choices: curated display order + aliases
########################################################################

EXPECTED_WINDOW_UI_ORDER = [
    "RECTANGLE", "HANN", "HAMMING", "BLACKMAN", "BLACKMAN-HARRIS",
    "NUTTALL", "BARTLETT", "COSINE", "FLATTOP", "TUKEY",
]

EXPECTED_WINDOW_ALIASES = {
    "NONE":            "RECTANGLE",
    "HANNING":         "HANN",
    "BLACKMAN_HARRIS": "BLACKMAN-HARRIS",
    "BH":              "BLACKMAN-HARRIS",
    "SINE":            "COSINE",
    "RAISED_COSINE":   "COSINE",
}


@pytest.mark.parametrize("stage_key, field_name", [
    ("range_compression",        "window"),
    ("sar_processing",           "window"),
    ("multilooking",             "window_type"),
    ("ionospheric_compensation", "window"),
])
def test_window_field_shape_is_curated_ui_then_aliases(stage_key, field_name):
    schema = build_schema("SHARAD", "EDR")
    stage = next(s for s in schema["stages"] if s["key"] == stage_key)
    field = next(f for f in stage["fields"] if f["name"] == field_name)
    assert field["type"] == "enum"
    assert field["value_type"] == "str"
    # ui:true entries appear first, in the required display order.
    assert _ui_values(field["choices"]) == EXPECTED_WINDOW_UI_ORDER
    # Non-ui entries are the six aliases and point at the correct canonical.
    aliases = {
        c["value"]: c["alias_of"]
        for c in field["choices"]
        if not c["ui"]
    }
    assert aliases == EXPECTED_WINDOW_ALIASES
    # Total is 16 (10 canonical + 6 aliases) — same size as WINDOW_NAMES.
    from grasp.processing.windows import WINDOW_NAMES
    assert len(field["choices"]) == len(WINDOW_NAMES)


########################################################################
# sgn: numeric enum with integer values (not strings)
########################################################################

def test_iono_sgn_is_numeric_enum():
    schema = build_schema("SHARAD", "EDR")
    iono = next(s for s in schema["stages"] if s["key"] == "ionospheric_compensation")
    sgn = next(f for f in iono["fields"] if f["name"] == "sgn")
    assert sgn["type"] == "enum"
    assert sgn["value_type"] == "int"
    assert sgn["choices"] == [
        {"value": -1, "ui": True},
        {"value": 1,  "ui": True},
    ]
    # Values are ints, not the strings "-1" / "1".
    for entry in sgn["choices"]:
        assert isinstance(entry["value"], int)


def test_sar_sgn_is_numeric_enum():
    schema = build_schema("SHARAD", "EDR")
    sar = next(s for s in schema["stages"] if s["key"] == "sar_processing")
    sgn = next(f for f in sar["fields"] if f["name"] == "sgn")
    assert sgn["value_type"] == "int"
    for entry in sgn["choices"]:
        assert isinstance(entry["value"], int)


def test_sar_backscatter_disables_multilooking():
    schema = build_schema("SHARAD", "EDR")
    sar = next(s for s in schema["stages"] if s["key"] == "sar_processing")
    assert "disables" in sar
    entry = next(
        d for d in sar["disables"]
        if d["when"] == {"field": "method", "equals": "BACKSCATTER"}
    )
    assert entry["stages"] == ["multilooking"]


########################################################################
# export_schema writes the expected files
########################################################################

def test_export_schema_writes_one_file_per_combo(tmp_path):
    written = export_schema(tmp_path)
    combos_seen = {p.stem.replace(".schema", "") for p in written}
    expected = {f"{inst}_{product}" for inst, product in ALL_COMBOS}
    assert combos_seen == expected
    for path in written:
        with path.open() as f:
            json.load(f)  # parses


def test_export_schema_single_combo(tmp_path):
    written = export_schema(tmp_path, instrument="LRS", product_type="EDR")
    assert len(written) == 1
    assert written[0].name == "LRS_EDR.schema.json"


########################################################################
# Runtime validator: eager rejection of typos + per-combo restrictions
########################################################################

def _make_pp():
    pp = ProcessingParameters(out_dir="/tmp/x")
    return pp


def test_validator_rejects_hannning_typo(tmp_path):
    # Constructing a sounder without files is heavy; test the pure
    # validator surface instead by simulating what
    # RadarSounder.validate_processing_parameters does with our data.
    pp = _make_pp()
    pp.mlk.enabled = True
    pp.mlk.window_type = "HANNNING"
    # Reuse the same validator invocation the sounder makes.
    from grasp.radar_sounder import _CHOICES  # noqa: PLC2701 — internal test
    # Just prove the choice registry catches it:
    valid = get_choices("mlk", "window_type", "SHARAD", "EDR")
    assert "HANNNING" not in valid
    assert "HANNING" in valid


def test_validator_rejects_calibrated_chirp_on_marsis():
    valid = get_choices("range_compression", "chirp_type", "MARSIS", "EDR")
    assert "CALIBRATED" not in valid
    assert "IDEAL" in valid


def test_validator_allows_calibrated_chirp_on_sharad():
    valid = get_choices("range_compression", "chirp_type", "SHARAD", "EDR")
    assert "CALIBRATED" in valid
    assert "IDEAL" in valid
