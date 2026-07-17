# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Any

from scipy.fft import ifft

from .modes import SUBSYSTEM_MODES, SYSTEM_DELAY

##############################################################################################################
#
# Preprocessing Functions
#
##############################################################################################################
def preprocess_marsis_edr(science_dict: dict[str, Any],
                          mode: str,
                          verbose: bool = False) -> dict[str, Any]:
    """Preprocess MARSIS EDR science data across all channels and filters.

    Applies decompression and AGC correction to each channel/filter
    combination in the science data dictionary. For all modes except
    SS2, both real and imaginary echo components are decompressed
    using exponent scaling before AGC correction. For SS2, only AGC
    correction is applied to the real echo data.

    Args:
        science_dict: Science data dictionary keyed by PDS field names.
            Modified in-place with preprocessed arrays replacing the
            raw values.
        mode: MARSIS operative mode (e.g., ``"SS3"``).
        verbose: If True, print progress messages.

    Returns:
        The updated science data dictionary with decompressed and
        AGC-corrected arrays.

    Raises:
        ValueError: If the operative mode is not supported.
    """
    mode = mode.upper()
    if mode not in SUBSYSTEM_MODES:
        raise ValueError(f"Operation mode {mode} is not supported.")



    REAL_EXP_KEY = "MAX_RE_EXP_{}_{}_DIP"
    IMAG_EXP_KEY = "MAX_IM_EXP_{}_{}_DIP"
    REAL_SCI_KEY = "REAL_ECHO_{}_{}_DIP"
    IMAG_SCI_KEY = "IMAG_ECHO_{}_{}_DIP"
    NEW_KEY = "ECHO_{}_{}_DIP"
    if mode == 'SS2':
        REAL_SCI_KEY = "ECHO_{}_{}_DIP"

    mode_info = SUBSYSTEM_MODES[mode]
    channels = mode_info.n_chan
    filters = mode_info.n_filt

    if verbose:
        print("Preprocessing MARSIS EDR")
    for _c in range(channels):
        _cStr = mode_info.chan_str[_c]
        agc = science_dict['AGC_SA_LEVELS_CURRENT_FRAME'][_c]
        for _f in range(filters):
            _fStr = mode_info.filt_str[_f]
            if verbose:
                print(f"\tChannel {_c+1} Filter {_f+1}")

            real_sci = REAL_SCI_KEY.format(_fStr, _cStr)
            if mode != 'SS2':
                #
                # SS1, SS3, SS4, SS5 data are all complex spectra
                #
                real_exp = REAL_EXP_KEY.format(_fStr, _cStr)
                science_dict[real_sci] = _preprocess_marsis_edr(
                    science_dict[real_sci], science_dict[real_exp], agc)

                imag_sci = IMAG_SCI_KEY.format(_fStr, _cStr)
                imag_exp = IMAG_EXP_KEY.format(_fStr, _cStr)
                science_dict[imag_sci] = _preprocess_marsis_edr(
                    science_dict[imag_sci], science_dict[imag_exp], agc)
                cmb = science_dict[real_sci] + 1j * science_dict[imag_sci]
                cmb_key = NEW_KEY.format(_fStr, _cStr)
                science_dict[cmb_key] = ifft(cmb, axis=0, workers=-1)
            else:
                #
                # SS2 is a real time series
                #
                science_dict[real_sci] = marsis_agc_correction(
                    science_dict[real_sci], agc)
    return science_dict


def _preprocess_marsis_edr(data: NDArray[np.integer],
                           exp: int | NDArray[np.integer],
                           agc_values: NDArray[np.integer],
                           ) -> NDArray[np.float32]:
    """Apply decompression and AGC correction to a single MARSIS EDR component.

    Args:
        data: Raw compressed echo data array, shape ``(n_samp, n_rec)``.
        exp: Exponent values for decompression. Scalar or 1-D array
            of length ``n_rec``.
        agc_values: AGC values for instrument response correction,
            shape ``(n_rec,)``.

    Returns:
        Decompressed and AGC-corrected data as float32.
    """
    data = decompress_marsis_edr_cmp(data, exp)
    data = marsis_agc_correction(data, agc_values)
    return data


def decompress_marsis_edr_cmp(uint8array: NDArray[np.integer],
                              maxexp: int | NDArray[np.integer],
                              ) -> NDArray[np.float32]:
    """Decompress MARSIS EDR CMP echo samples using exponent scaling.

    Encoding (per the PI's ``marsisuncompress.m``, little-endian branch):
        - MSB (bit 7) is sign: 0 → positive, 1 → negative.
        - Low 7 bits are the fractional mantissa: ``(byte & 0x7F) / 128``,
          a value in ``[0, 127/128]``. There is no implicit leading 1.
        - Decompressed value:
          ``sign * (mantissa_bits / 128) * 2**(maxexp - 127)``.

    Args:
        uint8array: Raw compressed samples. 1-D ``(n_samp,)`` or
            2-D ``(n_samp, n_rec)``.
        maxexp: Exponent values. Scalar for 1-D input, or scalar /
            1-D array of length ``n_rec`` for 2-D input.

    Returns:
        Decompressed data as float32, same shape as ``uint8array``.

    Raises:
        ValueError: If array dimensions or exponent shape are invalid.
    """
    u = np.asarray(uint8array, dtype=np.uint8)

    if u.ndim == 1:
        if not np.isscalar(maxexp):
            raise ValueError("For 1-D uint8array, maxexp must be a scalar.")
        exp_arr = np.array([float(maxexp)], dtype=np.float32)
        u2 = u[:, None]
        squeeze = True

    elif u.ndim == 2:
        u2 = u
        squeeze = False

        if np.isscalar(maxexp):
            exp_arr = np.full((u2.shape[1],), float(maxexp), dtype=np.float32)
        else:
            exp_arr = np.asarray(maxexp, dtype=np.float32)
            if exp_arr.ndim != 1 or exp_arr.size != u2.shape[1]:
                raise ValueError(
                    "For 2-D uint8array, maxexp must be a scalar or a 1-D array "
                    "with length equal to number of columns (n_rec)."
                )
    else:
        raise ValueError("uint8array must be 1-D or 2-D.")

    sign = np.where(((u2 >> 7) & 0x01) == 0, 1.0, -1.0).astype(np.float32)
    mantissa = (u2 & 0x7F).astype(np.float32) / 128.0
    scale = np.power(2.0, exp_arr - 127.0, dtype=np.float32)[None, :]

    out = sign * mantissa * scale

    if squeeze:
        return out[:, 0]
    return out


def marsis_agc_correction(data: NDArray[np.floating],
                          agc_values: NDArray[np.integer],
                          ) -> NDArray[np.float32]:
    """Apply or remove MARSIS AGC amplitude scaling.

    Decodes the AGC word and applies a per-record amplitude correction:
        - gain_code = agc_values & 0x07 (values 0–7)
        - gain_db = gain_code * 4 + 2 (values 2, 6, 10, ..., 30 dB)
        - factor = 10 ** (gain_db / 20)

    Args:
        data: Decompressed echo data, shape ``(n_samp, n_rec)``.
        agc_values: Per-record AGC values, shape ``(n_rec,)``.

    Returns:
        AGC-corrected data as float32.

    Raises:
        ValueError: If shapes are invalid or lengths mismatch.
        TypeError: If ``agc_values`` is not integer-like.
    """
    data_arr = np.asarray(data, dtype=np.float32)
    agc_arr = np.asarray(agc_values)

    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D (n_samp, n_rec); got ndim={data_arr.ndim}")

    if agc_arr.ndim != 1:
        raise ValueError(f"agc_values must be 1D (n_rec,); got ndim={agc_arr.ndim}")
    if not np.issubdtype(agc_arr.dtype, np.integer):
        raise TypeError(f"agc_values must be integer-like; got dtype={agc_arr.dtype}")

    n_samp, n_rec = data_arr.shape
    if agc_arr.size != n_rec:
        raise ValueError(f"agc_values length ({agc_arr.size}) must match n_rec ({n_rec}).")

    agc_u = agc_arr.astype(np.uint8, copy=False)
    gain_code = (agc_u & 0x07).astype(np.float32)
    gain_db = gain_code * 4.0 + 2.0

    correction = np.power(10.0, gain_db / 20.0, dtype=np.float32)

    out = data_arr * correction[None, :]
    return out