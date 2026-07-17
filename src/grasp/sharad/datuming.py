# SPDX-License-Identifier: BSD-3-Clause
from ..datuming.utils import remove_rwot_offset as _remove_rwot_offset
import numpy as np
from numpy.typing import NDArray

from .parameters import RADAR_PARAMS as PARAMS

def calculate_rwot(receive_window_opening_time: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute receive window opening time for SHARAD.

    Converts the raw receive window opening time counts to
    seconds using the instrument sample spacing, PRI, and
    system latency from the SHARAD radar parameters.

    Args:
        receive_window_opening_time: Raw RWOT values from the
            science data, shape ``(n_cols,)``.

    Returns:
        Receive window opening time in seconds, shape ``(n_cols,)``.
    """
    rwot = (receive_window_opening_time
            * PARAMS.dt
            + PARAMS.pri
            - PARAMS.latency[0])
    return rwot

def remove_rwot_offset(data: NDArray,
                       receive_window_opening_time: NDArray,
                       altitude: NDArray[np.floating],
                       dt: float = PARAMS.dt,
                       n_center: int | None = None,
                       verbose: bool = False) -> NDArray:
    """Align range windows for SHARAD data.

    Computes the receive window opening time from the raw
    instrument counts, then removes the per-trace RWOT offset
    to align the surface return across all traces.

    Args:
        data: Echo data array, shape ``(n_samp, n_cols)``.
        receive_window_opening_time: Raw RWOT values from the
            science data, shape ``(n_cols,)``.
        altitude: Spacecraft altitude in meters, shape
            ``(n_cols,)``.
        dt: Sample spacing in seconds.
        n_center: Sample index to center the surface return on.
            If None, defaults to ``nfft // 2``.
        verbose: If True, print progress messages.

    Returns:
        Aligned echo data, shape ``(n_samp, n_cols)``.
    """
    if n_center is None:
        n_center = PARAMS.nfft // 2
    rwot = calculate_rwot(receive_window_opening_time)
    data = _remove_rwot_offset(data, rwot, altitude, dt, n_center=n_center)

    return data

