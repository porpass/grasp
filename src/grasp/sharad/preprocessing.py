# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from typing import Any

from ..common.utils import decimate_dict
####################################################################################################################
#
# Preprocessing Functions
#
####################################################################################################################
def _validate_sharad_edr_sci_preprocessing_inputs(data: NDArray[np.integer[Any]],
                                                 compressionSelection: NDArray[np.integer[Any]],
                                                 onboardPresum: int,
                                                 bitResolution: int,
                                                 sdiBitField: NDArray[np.integer[Any]],
                                                 mgc: NDArray[np.integer[Any]],
                                                 ) -> None:
    """
    Validate input arrays for SHARAD EDR science preprocessing.

    This function performs type, shape, and value validation on all
    per-record metadata arrays required for SHARAD EDR science
    preprocessing. It ensures that:

    * All inputs are NumPy arrays.
    * ``data`` is a 2D array.
    * ``compressionSelection`` contains only values {0, 1}.
    * ``onboardPresum`` contains only values {2, 4, 8, 16, 28, 32}.
    * ``bitResolution`` contains only values {4, 6, 8}.
    * ``sdiBitField`` contains only non-negative values.

    This function does not return anything. It raises an exception
    immediately upon detecting invalid input.

    Args:
        data: 2D array of science data to be processed.
        compressionSelection: Per-record compression selection flags.
        onboardPresum: Per-record onboard presumming factors.
        bitResolution: Per-record bit resolutions.
        sdiBitField: Per-record SDI bitfield values.
        mgc: Per-record manual gain control values.

    Raises:
        TypeError: If any input is not a NumPy array.
        ValueError: If any array has invalid shape or contains invalid
            values.
    """
    args = {
        "data": data,
        "compressionSelection": compressionSelection,
        "sdiBitField": sdiBitField,
        "mgc": mgc,
    }
    for name, arr in args.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array, got {type(arr)}")

    if not isinstance(onboardPresum, (int, np.integer)):
        raise TypeError(
            f"onboardPresum must be an integer scalar, got {type(onboardPresum)}"
        )

    if not isinstance(bitResolution, (int, np.integer)):
        raise TypeError(
            f"bitResolution must be an integer scalar, got {type(bitResolution)}"
        )

    if data.ndim != 2:
        raise ValueError(f"data must be a 2D array, got {data.ndim}D")

    if not np.all(np.isin(compressionSelection, [0, 1])):
        raise ValueError("compressionSelection must contain only 0 or 1")

    if onboardPresum not in (1, 2, 4, 8, 16, 28, 32):
        raise ValueError("onboardPresum must be one of {1, 2, 4, 8, 16, 28, 32}")

    if not np.all(np.isin(bitResolution, [4, 6, 8])):
        raise ValueError("bitResolution must contain only {4, 6, 8}")

    if np.any(sdiBitField < 0):
        raise ValueError("sdiBitField must be non-negative")


def preprocess_sharad_edr_sci(science_data: dict[str, Any],
                              auxiliary_data: dict[str, Any],
                              onboard_presum: int,
                              apply_instrument_response: bool = True,
                              target_presum: int | None = None,
                              verbose: bool = False,
                              ) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    """Preprocess SHARAD EDR science data.

    Decompresses raw echo samples, optionally applies instrument
    response correction, and optionally applies additional coherent
    presumming to ECHO_SAMPLES only.

    Args:
        science_data: SHARAD EDR science data dictionary.
        auxiliary_data: SHARAD EDR auxiliary data dictionary.
        onboard_presum: Onboard presumming factor.
        apply_instrument_response: If True, apply instrument response
            correction after decompression.
        target_presum: Desired total presumming factor. If greater
            than onboard_presum, additional coherent presumming is
            applied. If None, no additional presumming.
        verbose: If True, print progress messages.

    Returns:
        science_data: Preprocessed science dictionary.
        auxiliary_data: Auxiliary dictionary (unchanged).
        presum_factor: Additional presumming factor applied, or
            None if no additional presumming was performed.
    """
    data = science_data['ECHO_SAMPLES']
    compression = science_data['OST_LINE_COMPRESSION_SELECTION']
    bit_res = science_data['BIT_RESOLUTION']
    unique_br = np.unique(bit_res[bit_res > 0])
    if len(unique_br) != 1:
        raise ValueError(f"Expected uniform non-zero BIT_RESOLUTION, got {unique_br}")
    bit_res = int(unique_br[0])
    sdi = science_data['SDI_BIT_FIELD']
    mgc = science_data['OST_LINE_MANUAL_GAIN_CONTROL']

    _validate_sharad_edr_sci_preprocessing_inputs(
        data, compression, onboard_presum, bit_res, sdi, mgc)

    if verbose:
        print("\tDecompressing SHARAD raw echo samples...")

    science_data['ECHO_SAMPLES'] = decompress(
        data, compression, onboard_presum, bit_res, sdi)

    if apply_instrument_response:
        if verbose:
            print("\tApplying instrument response...")
        science_data['ECHO_SAMPLES'] = instrument_response(
            science_data['ECHO_SAMPLES'], mgc)

    presum_factor = None
    if target_presum is not None and target_presum > onboard_presum:
        science_data['ECHO_SAMPLES'], presum_factor = apply_additional_presum(
            science_data['ECHO_SAMPLES'],
            onboard_presum, target_presum,
            verbose=verbose)

    return science_data, auxiliary_data, presum_factor

def decompress(data: NDArray[np.integer[Any]],
               compressionSelection: NDArray[np.integer[Any]],
               onboardPresum: int,
               bitResolution: NDArray[np.integer[Any]],
               sdiBitField: NDArray[np.integer[Any]],
               ) -> NDArray[np.floating[Any]]:
    """
    Decompress SHARAD EDR echo samples using static or dynamic compression
    parameters.

    This follows the formulas given in the SHALLOW RADAR EDR SIS.

    For *static* compression (compressionSelection == 0):

        U = C * (2**S / N)

        where:
            C : compressed data (raw echo samples)
            N : onboard presumming factor (scalar, same for all records)
            R : bit resolution per record (``bitResolution``)
            L : ceil(log2(N))
            S : L - R + 8

    For *dynamic* compression (compressionSelection == 1):

        U = C * (2**S / N)

        where:
            C   : compressed data
            N   : onboard presumming factor (scalar)
            SDI : per-record SDI bitfield (``sdiBitField``)
            S is determined from SDI as:

                S = SDI          for SDI <= 5
                S = SDI - 6      for 5 < SDI <= 16
                S = SDI - 16     for SDI > 16

    The function computes a per-record scale factor ``d[frame]`` and applies
    it across the sample dimension via broadcasting:

        decom = data * d

    Args:
        data: Raw SHARAD echo samples, integer array of shape
            ``(n_samp, n_recs)``.
        compressionSelection: Per-record compression mode (0 = static,
            1 = dynamic), shape ``(n_recs,)``.
        onboardPresum: Scalar onboard presumming factor ``N`` shared by
            all records (e.g., 2, 4, 8, 16, 28, or 32).
        bitResolution: Per-record bit resolution (4, 6, or 8),
            shape ``(n_recs,)``.
        sdiBitField: Per-record SDI bitfield values, shape ``(n_recs,)``.

    Returns:
        A floating-point NumPy array of decompressed echo samples with
        the same shape as ``data`` (``(n_samp, n_recs)``).

    Raises:
        ValueError: If an unsupported compression selection value is
            encountered.
    """
    n_samp, n_recs = data.shape
    decom = np.zeros(data.shape, dtype=float)
    d = np.zeros(n_recs, dtype=float)

    N = float(onboardPresum)

    # Fast path: uniform compression selection (expected normal case)
    u_cs = np.unique(compressionSelection)
    if len(u_cs) == 1:
        cs = int(u_cs[0])

        if cs == 0:
            # Static compression – fully vectorized across records
            L = float(np.ceil(np.log2(N)))
            R = float(bitResolution)
            S = L - R + 8.0                      # shape (n_recs,)
            d[:] = (2.0**S) / N                  # shape (n_recs,)

        elif cs == 1:
            # Dynamic compression
            u_sdi = np.unique(sdiBitField)
            if len(u_sdi) == 1:
                # Uniform SDI across all records
                sdi = int(u_sdi[0])
                if sdi <= 5:
                    S = float(sdi)
                elif 5 < sdi <= 16:
                    S = float(sdi - 6)
                else:
                    S = float(sdi - 16)
                d[:] = (2.0**S) / N
            else:
                # Per-record SDI scaling
                for frame in range(n_recs):
                    sdi = int(sdiBitField[frame])
                    if sdi <= 5:
                        S = float(sdi)
                    elif 5 < sdi <= 16:
                        S = float(sdi - 6)
                    else:  # sdi > 16
                        S = float(sdi - 16)
                    d[frame] = (2.0**S) / N
        else:
            raise ValueError(f"Unsupported compression selection: {cs}")

    else:
        # Mixed compression modes across records (should be rare, but handle it)
        for frame in range(n_recs):
            cs = int(compressionSelection[frame])

            if cs == 0:
                # Static per-frame (N is still scalar)
                L = float(np.ceil(np.log2(N)))
                R = float(bitResolution[frame])
                S = L - R + 8.0
            elif cs == 1:
                # Dynamic per-frame
                sdi = int(sdiBitField[frame])
                if sdi <= 5:
                    S = float(sdi)
                elif 5 < sdi <= 16:
                    S = float(sdi - 6)
                else:
                    S = float(sdi - 16)
            else:
                raise ValueError(f"Unsupported compression selection: {cs}")

            d[frame] = (2.0**S) / N

    # Broadcast per-record scale over samples
    decom[:] = data * d

    return decom


def instrument_response(data: NDArray[np.floating[Any]],
                        mgc: NDArray[np.floating[Any]],
                        VoltSat: float = 0.5,
                        Gain: float = 88.0,
                        ATx: float = 8.4,
                        ) -> NDArray[np.floating[Any]]:
    """
    Apply SHARAD instrument response correction to decompressed echo samples.

    This function converts raw, decompressed SHARAD echo samples from
    instrument units to calibrated voltage units using the per-record
    Manual Gain Control (MGC) and fixed instrument parameters.

    The calibration factor is computed per record as:

        calVal = VoltSat * 2 / (2**8 - 1) *
                 10 ** ((mgc - (Gain - ATx)) / 20) * 10**3

    and applied across the sample dimension via broadcasting:

        data_out = data * calVal[np.newaxis, :]

    Args:
        data: 2D array of decompressed SHARAD echo samples with shape
            ``(n_samples, n_records)``, floating-point dtype.
        mgc: 1D array of Manual Gain Control values (dB) with shape
            ``(n_records,)``.
        VoltSat: Saturation level in volts (default 0.5 V).
        Gain: Receiver maximum gain in dB (default 88 dB).
        ATx: TFE Rx path attenuation in dB (default 8.4 dB).

    Returns:
        A 2D floating-point array of calibrated echo samples with the
        same shape as ``data``.

    Raises:
        ValueError: If the length of ``mgc`` does not match the number of
            records in ``data``.
    """
    mgc = np.asarray(mgc, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape!r}")

    n_samp, n_recs = data.shape
    if mgc.shape[0] != n_recs:
        raise ValueError(
            f"mgc length {mgc.shape[0]} does not match number of records "
            f"in data ({n_recs})."
        )

    calVal = (
        VoltSat * 2.0 / (2**8 - 1)
        * 10.0 ** ((mgc - (Gain - ATx)) / 20.0)
        * 10.0**3
    )

    return data * calVal[np.newaxis, :]


def additional_presum(data: NDArray[np.complexfloating],
                      factor: int,
                      ) -> NDArray[np.complexfloating]:
    """Coherently average adjacent traces along azimuth.

    Args:
        data: Complex 2D array, shape (n_range, n_azimuth).
        factor: Number of traces to average.

    Returns:
        Presummed array, shape (n_range, n_azimuth // factor).
    """
    n_range, n_az = data.shape
    n_out = n_az // factor
    truncated = data[:, :n_out * factor]
    return truncated.reshape(n_range, n_out, factor).mean(axis=2)


def apply_additional_presum(science_data: NDArray,
                            current_presum: int,
                            target_presum: int,
                            verbose: bool = False,
                            ) -> tuple[NDArray, int]:
    """Apply additional coherent presumming to SHARAD echo samples.

    Coherently averages ECHO_SAMPLES only. All other decimation
    (science fields, auxiliary data, geometry, etc.) is handled
    by the caller via ``_decimate_to_frames``.

    Args:
        science_data: SHARAD science data
        current_presum: Current presumming factor.
        target_presum: Desired total presumming factor.
        verbose: If True, print progress messages.

    Returns:
        science_data: Presummed Science Data
        factor: The additional presumming factor applied.

    Raises:
        ValueError: If target_presum is less than current_presum or
            not an integer multiple of it.
    """
    if target_presum <= current_presum:
        raise ValueError(
            f"target_presum ({target_presum}) must be greater than "
            f"current_presum ({current_presum})")

    if target_presum % current_presum != 0:
        raise ValueError(
            f"target_presum ({target_presum}) must be an integer "
            f"multiple of current_presum ({current_presum})")

    factor = target_presum // current_presum

    if verbose:
        print(f"\tAdditional presumming: {current_presum} -> {target_presum} "
              f"(factor {factor})")

    science_data = additional_presum(science_data, factor)

    return science_data, factor