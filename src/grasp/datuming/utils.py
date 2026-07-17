# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from ..constants import C

def remove_rwot_offset(data: NDArray[np.complexfloating],
                       rwot: NDArray[np.floating],
                       altitude: NDArray[np.floating],
                       dt: float,
                       n_center: int | None = None,
                       ) -> NDArray[np.complexfloating]:
    """Remove receive window opening time offsets and center the surface return.

    Aligns each trace by rolling it so that the surface echo (estimated
    from the two-way travel time to the ellipsoid) is placed at the
    center of the range window. This removes trace-to-trace timing
    differences caused by varying receive window opening times.

    Args:
        data: Complex radar data with shape (n_samp, n_col). Rows are
            fast-time samples, columns are records/traces.
        rwot: Receive window opening time per trace in seconds,
            shape (n_col,).
        altitude: Spacecraft altitude above the ellipsoid per trace
            in meters, shape (n_col,).
        dt: Sample spacing in seconds.
        n_center: Sample index to center the surface return on.
            Defaults to ``n_samp // 2`` if not specified.

    Returns:
        Shifted complex data array with the same shape as the input.
    """
    n_samp, n_col = data.shape

    if n_center is None:
        n_center = n_samp // 2

    delay_to_ellip = 2 * altitude / C
    shift = ((rwot - delay_to_ellip) // dt + n_center).astype(int)
    #datum = delay_to_ellip - n_center * dt
    datum = rwot - shift*dt
    out = np.zeros_like(data)
    for k in range(n_col):
        out[:, k] = np.roll(data[:, k], shift[k])

    return out, datum