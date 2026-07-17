# SPDX-License-Identifier: BSD-3-Clause
from ..datuming.utils import remove_rwot_offset as _remove_rwot_offset
import numpy as np
from numpy.typing import NDArray

from .parameters import RADAR_PARAMS as PARAMS

def calculate_rwot(delay: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute receive window opening time for LRS.

    Converts the raw DELAY value (in microseconds) to
    seconds.

    Args:
        delay: DELAY values from the
            science data, shape ``(n_cols,)``.

    Returns:
        Receive window opening time in seconds, shape ``(n_cols,)``.
    """
    return delay * 1e-6

def remove_rwot_offset(data: NDArray,
                       delay: NDArray,
                       altitude: NDArray[np.floating],
                       dt: float = PARAMS.dt,
                       n_center: int | None = None
                       ) -> NDArray:
    """Align range windows for LRS data.

    Computes the receive window opening time from the raw
    DELAY value, then removes the per-trace RWOT offset
    to align the surface return across all traces.

    Args:
        data: Echo data array, shape ``(n_samp, n_cols)``.
        delay: DELAY values from the science data, shape ``(n_cols,)``.
        altitude: Spacecraft altitude in meters, shape ``(n_cols,)``.
        dt: Sample spacing in seconds.
        n_center: Sample index to center the surface return on.
            If None, defaults to ``nfft // 2``.

    Returns:
        Aligned echo data, shape ``(n_samp, n_cols)``.
    """
    if n_center is None:
        n_center = PARAMS.n_samp // 2
    rwot = calculate_rwot(delay)
    data = _remove_rwot_offset(data, rwot, altitude, dt, n_center=n_center)

    return data

