# SPDX-License-Identifier: BSD-3-Clause
"""Machine-readable schema of GRaSP's processing parameters.

.. deprecated::
   This module is deprecated and will be removed in the next release.
   The JSON-schema assembly moves to ``porpass/daemon``; GRaSP retains
   only the introspection primitives (correctness enums, section aliases,
   support matrix) and now exposes them via :mod:`grasp.introspection`.

   Migrate as follows:

   ============================================  ============================
   Old                                            New
   ============================================  ============================
   ``from grasp.schema import get_choices``       ``from grasp.introspection import get_choices``
   ``from grasp.schema import CHOICES``           ``from grasp.introspection import CHOICES``
   ``from grasp.schema import RESTRICTIONS``      ``from grasp.introspection import RESTRICTIONS``
   ``from grasp.schema import build_schema``      (moves to daemon; see PORPASS)
   ``from grasp.schema import export_schema``     (moves to daemon; see PORPASS)
   ``from grasp.schema import CMAP_NAMES``        (moves to daemon; pure form-curation)
   ============================================  ============================
"""

from .choices import get_choices, CHOICES, RESTRICTIONS, CMAP_NAMES
from .export import build_schema, export_schema

__all__ = [
    "build_schema",
    "export_schema",
    "get_choices",
    "CHOICES",
    "RESTRICTIONS",
    "CMAP_NAMES",
]
