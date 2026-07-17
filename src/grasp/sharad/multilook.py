# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from ..processing.sar.multilook import multilook as _multilook


def multilook(data: NDArray,
              rho_a: float | NDArray[np.floating],
              *,
              n_looks: int = 5,
              os_factor: int = 1,
              window_type: str = "hanning",
              window_alpha: float | None = None,
              coherent: bool = False,
              ) -> tuple[NDArray, NDArray, NDArray]:
    """Apply multilook averaging to SHARAD SAR-processed data.

    Args:
        data: SAR-processed echo data, shape ``(n_samp, n_cols)``.
        rho_a: Along-track resolution in meters from SAR processing.
            Scalar or per-trace array.
        n_looks: Number of looks.
        os_factor: Oversampling factor.
        window_type: Spectral window type.
        window_alpha: Tukey window taper fraction.
        coherent: If True, apply coherent multilooking.

    Returns:
        A tuple ``(data, frames, rho_mlk)`` where:

        - ``data``: Multilooked echo data.
        - ``frames``: Output frame indices.
        - ``rho_mlk``: Along-track resolution after multilooking.

    Raises:
        ValueError: If any required parameter is invalid.
    """
    if n_looks < 1:
        n_looks = 1
    if os_factor < 1:
        os_factor = 1

    return _multilook(data, rho_a=rho_a, n_looks=n_looks, os_factor=os_factor, window_type=window_type,
                      window_alpha=window_alpha, coherent=coherent)