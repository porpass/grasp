# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Any

from .modes import SUBSYSTEM_MODES, iter_channel_filter
from ..postprocessing.multilook import multilook as _multilook

def multilook(science_dict: dict[str, Any],
              auxiliary_dict: dict[str, Any],
              mode: str,
              product_type: str,
              *,
              n_looks: int = 3,
              os_factor: int = 1,
              window_type: str = "hanning",
              window_alpha: float | None = None,
              coherent: bool = False,
              verbose: bool = False) -> dict[str, Any]:
    """Apply multilook processing to MARSIS data.

    Computes along-track resolution from auxiliary parameters and
    applies incoherent or coherent multilook averaging to each
    channel/filter combination. For EDR products, along-track
    spacing is derived from ``PROCESSING_PRF`` and
    ``VT_SCET_PAR``. For RDR products, ``DELTA_S_SCET_PAR`` is
    used directly.

    Args:
        science_dict: Science data dictionary keyed by PDS field
            names. Modified in-place.
        auxiliary_dict: Auxiliary data dictionary. Used for EDR
            products to compute along-track resolution.
        mode: MARSIS operative mode (e.g., ``"SS3"``).
        product_type: Product type — ``"EDR"`` or ``"RDR"``.
        n_looks: Number of looks.
        os_factor: Oversampling factor.
        window_type: Spectral window type.
        window_alpha: Tukey window taper fraction.
        coherent: If True, apply coherent multilooking.
        verbose: If True, print progress messages.

    Returns:
        The updated science data dictionary with multilooked
        arrays.

    Raises:
        ValueError: If the operative mode or product type is not
            supported.
        KeyError: If expected science or auxiliary keys are missing.
    """
    mode = mode.upper()
    if mode not in SUBSYSTEM_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    mode_info = SUBSYSTEM_MODES[mode]

    if n_looks < 1:
        n_looks = 1
    if os_factor < 1:
        os_factor = 1
    window_type = window_type.upper()
    # Compute rho_a
    if product_type == "EDR":
        prf = auxiliary_dict['PROCESSING_PRF']
        vt = auxiliary_dict['VT_SCET_PAR']
        pri = np.where(prf > 0, 1.0 / prf, 0.0)
        rho_a = vt * pri
    elif product_type == "RDR":
        rho_a = science_dict['DELTA_S_SCET_PAR']
    else:
        raise ValueError(f"Unsupported product type: {product_type}")

    if verbose:
        print("Performing multilooking...")

    for f_str, c_str, k in iter_channel_filter(mode):
        if k not in science_dict:
            raise KeyError(
                f"Key '{k}' not found in science data. You may need "
                f"to preprocess your MARSIS data first.")

        if verbose:
            print(f"\tChannel {c_str} Filter {f_str}")

        science_dict[k] = _multilook(
            science_dict[k],
            rho_a=rho_a,
            n_looks=n_looks,
            os_factor=os_factor,
            window_type=window_type,
            window_alpha=window_alpha,
            coherent=coherent,
            verbose=verbose,
        )

    return science_dict