# SPDX-License-Identifier: BSD-3-Clause
"""
GRaSP input interface.

This package provides a unified interface for reading radar sounder data
products supported by GRaSP, including SHARAD, MARSIS, and LRS.

The primary public entry point is :func:`read`, which automatically
identifies the instrument, product type, and file role (science,
auxiliary, geometry, etc.) and dispatches to the appropriate parser.

For convenience, instrument-specific parse entry points are also exposed:
:func:`parse_sharad` and :func:`parse_marsis`.

Most users should only need :func:`read`.

Example usage:
    data = read("e_19689_ss3_trk_cmp_m_g.dat")
"""
from .job import load_processing_job as load_job
from .load import load_hdf5, read_grasp_output as read_grasp
from .read import read
from .utils import check_supported, grab_radar_params, identify_file


__all__ = [
    "check_supported",
    "grab_radar_params",
    "identify_file",
    "load_hdf5",
    "load_job",
    "read",
    "read_grasp",
]