# SPDX-License-Identifier: BSD-3-Clause
from numpy.typing import NDArray
from typing import Any, Literal
from ..grasp_types import MARSISIonosphereResult, CampbellResult, ContrastResult

from ..common.utils import assert_gt_0
from ..processing.ionosphere.campbell import campbell_method
from ..processing.ionosphere.contrast import contrast_method

from .modes import SUBSYSTEM_MODES, DCG_TO_FCEN, iter_channel_filter


def ionospheric_compensation(science_dict: dict[str, Any],
                             mode: str,
                             dcg_config: NDArray[Any],
                             *,
                             method: str = "contrast",
                             dt: float = 1 / 1.4e6,
                             bw: float = 1.0e6,
                             n_take: int = 128,
                             wnd: str = "hanning",
                             alpha: float = 1.0,
                             # Contrast Options
                             L_eq: float = 80e3,
                             # Campbell options
                             campbell_b: float = 0.9,
                             n_phase: int = 400,
                             campbell_delta: float = 0.0125,
                             sgn : Literal[-1, 1] = 1,
                             metric: str | None = None,
                             verbose: bool = False,
                             ) -> tuple[dict[str, Any], dict[str, CampbellResult | ContrastResult]]:
    """Apply ionospheric phase compensation to MARSIS EDR data.

    Loops over all channels and Doppler filters for the given
    operative mode and applies either the Campbell or Contrast
    ionospheric compensation method to each echo array. Center
    frequencies are derived per-channel from the DCG configuration
    word, allowing for along-track frequency changes.

    The estimated ionospheric correction values are stored in the
    science dictionary under ``"IONO_{filt}_{chan}_DIP"`` keys.

    Args:
        science_dict: Science data dictionary keyed by PDS field
            names. Modified in-place.
        mode: MARSIS operative mode (e.g., ``"SS3"``).
        dcg_config: DCG configuration array with shape
            ``(n_chan, n_col)``. Integer values 0–3 map to center
            frequencies via ``DCG_TO_FCEN``.
        method: Compensation method — ``"contrast"`` or
            ``"campbell"`` (case-insensitive).
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
        verbose: If True, print progress messages.

    Returns:
        The updated science data dictionary with ionosphere-corrected
        arrays and per-channel/filter correction values stored under
        ``"IONO_{filt}_{chan}_DIP"`` keys. For the Campbell method
        the stored values are the ionospheric phase corrections;
        for the Contrast method they are the estimated ``a2``
        dispersion parameters.

    Raises:
        ValueError: If the operative mode or method is not supported.
        KeyError: If expected science keys are missing.
    """
    mode = mode.upper()
    if mode not in SUBSYSTEM_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    f_cen = DCG_TO_FCEN[dcg_config]

    mode_info = SUBSYSTEM_MODES[mode]

    wnd = wnd.upper()
    method = method.upper()
    assert_gt_0(dt)
    assert_gt_0(bw)

    if method not in ("CAMPBELL", "CONTRAST"):
        raise ValueError(f"Invalid method: {method}")

    # Per-channel centre frequency lookup so the inner loop stays flat.
    f0_by_chan = {c_str: f_cen[c_idx, :]
                  for c_idx, c_str in enumerate(mode_info.chan_str)}

    if verbose:
        print("Peforming ionospheric compensation...")
    iono_results = {}
    for f_str, c_str, k in iter_channel_filter(mode):
        if k not in science_dict:
            raise KeyError(
                f"Key '{k}' not found in science data. You may need "
                f"to preprocess your MARSIS data first.")
        ts = science_dict[k]

        f_cen = f0_by_chan[c_str]

        if verbose:
            print(f"\tChannel {c_str} Filter {f_str}")

        if method == "CAMPBELL":
            ts_out, e_values, iono_phase, delay_cells = campbell_method(
                ts,
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
            iono_results[k] = CampbellResult(e_values=e_values,
                                             iono_phase=iono_phase,
                                             delay_cells=delay_cells)
        else:
            ts_out, a2s, a3s, a4s, delay_cells = contrast_method(ts,
                                                                 dt=dt,
                                                                 bw=bw,
                                                                 f_cen=f_cen,
                                                                 n_take=n_take,
                                                                 wnd=wnd,
                                                                 alpha=alpha,
                                                                 L_eq=L_eq,
                                                                 metric=metric,
                                                                 )
            iono_results[k] = ContrastResult(a2=a2s, a3=a3s, a4=a4s,
                                             delay_cells=delay_cells)
        science_dict[k] = ts_out
    result = MARSISIonosphereResult(method=method, results=iono_results)
    return science_dict, result