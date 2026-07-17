# SPDX-License-Identifier: BSD-3-Clause
"""Canonical mapping between long TOML section headers and dataclass attribute names.

The TOML job files use human-friendly long names as section headers
(``[range_compression]``, ``[ionospheric_compensation]``, ``[sar_processing]``,
etc.). The corresponding attributes on
:class:`grasp.grasp_types.ProcessingParameters` use short names
(``range_compression``, ``iono_comp``, ``sar``). This module is the single
source of truth for the mapping.

Both the TOML loader (:mod:`grasp.input.job`) and the schema exporter
(:mod:`grasp.schema.export`) import from here so the two cannot drift.
"""

LONG_TO_ATTR: dict[str, str] = {
    "general":                   "general",
    "input":                     "input",
    "preprocessing":             "preprocessing",
    "range_compression":         "range_compression",
    "emi_suppression":           "emi_suppression",
    "ionospheric_compensation":  "iono_comp",
    "sar_processing":            "sar",
    "multilooking":              "mlk",
    "clutter_simulation":        "csim",
    "output_parameters":         "output",
    "plot_parameters":           "plots",
    "final_output":              "final_output",
}

ATTR_TO_LONG: dict[str, str] = {v: k for k, v in LONG_TO_ATTR.items()}
