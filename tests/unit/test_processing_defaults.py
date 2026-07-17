# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the combined processing-defaults dispatcher.

Verifies that each instrument's TOML loads, that the expected product
types are auto-discovered from the TOML keys, and that the dispatcher
honours an instrument+metadata lookup end-to-end without falling foul
of the historic ``US_RDR`` vs ``"US RDR"`` key mismatch.
"""
import pytest

from grasp.processing_defaults import get_defaults


class _MetadataStub:
    """Minimal stand-in for MetaDataInfo carrying only product_type."""
    def __init__(self, product_type: str):
        self.product_type = product_type


@pytest.mark.parametrize(
    "instrument, product_type",
    [
        ("SHARAD", "EDR"),
        ("SHARAD", "RDR"),
        ("SHARAD", "US_RDR"),
        ("MARSIS", "EDR"),
        ("MARSIS", "RDR"),
        ("LRS",    "EDR"),
        ("LRS",    "RDR"),
    ],
)
def test_get_defaults_resolves_each_supported_product_type(
        instrument, product_type):
    """Every TOML's non-base top-level table should be reachable via
    a metadata.product_type lookup."""
    result = get_defaults(instrument, _MetadataStub(product_type))
    assert isinstance(result, dict), (
        f"{instrument}/{product_type}: expected dict, got {type(result)}"
    )
    assert result, f"{instrument}/{product_type}: returned empty dict"


def test_get_defaults_sharad_us_rdr_lookup_via_metadata():
    """Looking up the SHARAD US_RDR product via a metadata-like object
    should succeed. Catches the key-mismatch regression where the
    loader normalized ``US_RDR`` → ``"US RDR"``."""
    result = get_defaults("SHARAD", _MetadataStub("US_RDR"))
    assert isinstance(result, dict)
    # A few sentinel stages we expect to be defined for SHARAD US_RDR.
    # Don't pin every value — that would couple the test to TOML edits.
    assert "preprocessing" in result or "range_compression" in result


def test_get_defaults_default_product_type_is_edr():
    """When metadata is None, dispatcher should default to EDR."""
    edr_table = get_defaults("SHARAD")
    explicit_edr = get_defaults("SHARAD", _MetadataStub("EDR"))
    assert edr_table is explicit_edr or edr_table == explicit_edr


def test_get_defaults_instrument_is_case_insensitive():
    """Pipeline call sites pass instrument strings in various cases;
    dispatcher upper-cases internally."""
    assert get_defaults("sharad") == get_defaults("SHARAD")
    assert get_defaults("Marsis") == get_defaults("MARSIS")


def test_get_defaults_unknown_instrument_raises():
    with pytest.raises(ValueError, match="No defaults file for instrument"):
        get_defaults("BOGUS")


def test_get_defaults_unknown_product_type_raises():
    with pytest.raises(ValueError, match="product_type"):
        get_defaults("SHARAD", _MetadataStub("DOES_NOT_EXIST"))
