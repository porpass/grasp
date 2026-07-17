# SPDX-License-Identifier: BSD-3-Clause
from numpy.typing import NDArray
from typing import Literal
from ...grasp_types import CampbellResult, ContrastResult

from ...common.utils import assert_gt_0
from .campbell import campbell_method
from .contrast import contrast_method

def ionospheric_compensation(data: NDArray,
                             *,
                             dt: float,
                             bw: float,
                             f_cen: float | NDArray | None,
                             method: str = "contrast",
                             n_take: int = 128,
                             wnd: str = "hanning",
                             alpha: float = 1.0,
                             # Contrast Options
                             L_eq: float = 80e3,
                             # Campbell options
                             campbell_b: float = 1.93,
                             n_phase: int = 400,
                             campbell_delta: float = 0.0125,
                             sgn : Literal[-1, 1] = 1,
                             # Autofocus metric
                             metric: str | None = None,
                             ) -> tuple[NDArray, CampbellResult | ContrastResult]:
    """Apply ionospheric phase compensation to data.

    Applies either the Campbell or Contrast ionospheric compensation method to the echo array.

    Args:
        data: Science data (time-series). Modified in-place.
        method: Compensation method — ``"contrast"`` or
            ``"campbell"`` (case-insensitive).
        f_cen: Physical RF center frequency in Hz used by the
            dispersion power-law (Campbell) and polynomial
            (Contrast). Either a scalar or a 1D array of length
            ``n_col`` (per-column carrier).
        dt: Sample spacing in seconds.
        bw: Bandwidth in Hz.
        n_take: Stacking neighborhood size for the grid search.
        wnd: Spectral window type (e.g., ``"hanning"``).
        alpha: Tukey window taper fraction.
        L_eq: Equivalent path length in meters. Used by the
            Contrast method to compute tau_0.
        campbell_b: Power-law exponent for the Campbell phase
            model.
        n_phase: Number of phase states to test in the Campbell
            method.
        campbell_delta: Step size for generating E-values in
            the Campbell method.
        sgn: Sign convention for the phase exponential in the
            Campbell method. Must be ``-1`` or ``+1``.
        metric: Autofocus metric name (``"L1"``, ``"L4"``,
            ``"ENTROPY"``, or ``"PEAK_SNR"``). If ``None``, falls
            back to the method's default — ``"PEAK_SNR"`` for
            Campbell, ``"L4"`` for Contrast.
    Returns:
        A tuple ``(data, iono_result)`` where:

        - ``data``: Ionosphere-corrected echo data, shape
            ``(n_samp, n_cols)``, complex.
        - ``iono_result``: A ``CampbellResult`` or
            ``ContrastResult`` containing the estimated
            correction parameters.

    Raises:
        ValueError: If the method is not supported.
    """
    wnd = wnd.upper()
    method = method.upper()
    assert_gt_0(dt)
    assert_gt_0(bw)

    if method not in ("CAMPBELL", "CONTRAST", "CHAPMAN"):
        raise ValueError(f"Invalid method: {method}")
    if method == "CHAPMAN":
        raise NotImplementedError("Chapman method for ionospheric compensation not available for SHARAD yet.")

    if method == "CAMPBELL":
        ts_out, e_values, iono_phase, delay_cells = campbell_method(
            data,
            dt=dt,
            bw=bw,
            f_cen=f_cen,
            n_take=n_take,
            b=campbell_b,
            n_phase=n_phase,
            delta=campbell_delta,
            sgn=sgn,
            wnd=wnd,
            alpha=alpha,
            metric=metric,
        )
        iono_results = CampbellResult(e_values=e_values,
                                      iono_phase=iono_phase,
                                      delay_cells=delay_cells)
    else:
        ts_out, a2s, a3s, a4s, delay_cells = contrast_method(
            data,
            dt=dt,
            bw=bw,
            f_cen=f_cen,
            n_take=n_take,
            wnd=wnd,
            alpha=alpha,
            L_eq=L_eq,
            metric=metric,
        )
        iono_results = ContrastResult(a2=a2s, a3=a3s, a4=a4s, delay_cells=delay_cells)
    return ts_out, iono_results