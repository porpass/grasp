# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Any

##############################################################################################################
#
# Dictionaries
#
##############################################################################################################
def make_lrs_wf_dict(n_cols: int,
                     n_samp: int = 2048,
                     ) -> dict[str, NDArray]:
    """
    Initializes an LRS EDR (WF) science dictionary.

    Preallocates NumPy arrays for each LRS EDR waveform science field. Array
    lengths are set by `n_cols` (number of records), and waveform arrays are
    shaped as `(n_samp, n_cols)` (samples × records).

    Args:
        n_cols: Number of columns (science records) in the output arrays.
        n_samp: Number of samples in each waveform record. Defaults to 2048.

    Returns:
        A dictionary mapping field names to preallocated NumPy arrays.

    Raises:
        ValueError: If `n_cols <= 0` or `n_samp <= 0`.

    Notes:
        - `OBSERVATION_TIME` is initialized to `NaT` with millisecond
          resolution (`datetime64[ms]`).
        - `TI` is initialized as a fixed-width unicode string array (`U32`).
    """
    if n_cols <= 0:
        raise ValueError(f"nCols must be positive; got {n_cols}")
    if n_samp <= 0:
        raise ValueError(f"nSamp must be positive; got {n_samp}")
    sci_dict: dict[str, NDArray[Any]] = {
        'OBSERVATION_TIME': np.full(n_cols, np.datetime64('NaT', 'ms')),
        'DELAY': np.zeros(n_cols, dtype=np.float32),
        'START_STEP': np.zeros(n_cols, dtype=np.int16),
        'SUB_SPACECRAFT_LATITUDE': np.zeros(n_cols, dtype=np.float32),
        'SUB_SPACECRAFT_LONGITUDE': np.zeros(n_cols, dtype=np.float32),
        'SPACECRAFT_ALTITUDE': np.zeros(n_cols, dtype=np.float32),
        'RANGE0_ALTITUDE': np.zeros(n_cols, dtype=np.float32),
        'TI': np.full(n_cols, "", dtype="U32"),
        'WAVEFORM': np.zeros((n_samp, n_cols), dtype=np.uint16),
    }

    return sci_dict


def make_lrs_sar_c_dict(n_cols: int,
                        n_samp: int = 1000,
                        ) -> dict[str, NDArray[Any]]:
    """
    Initializes an LRS RDR (SAR-C) science dictionary.

    Preallocates NumPy arrays for each LRS RDR SAR-C science field. Array
    lengths are set by `n_cols` (number of records), and SAR image arrays are
    shaped as `(n_samp, n_cols)` (lines/samples × records).

    Args:
        n_cols: Number of columns (science records) in the output arrays.
        n_samp: Number of samples in each SAR image record. Defaults to 1000.

    Returns:
        A dictionary mapping field names to preallocated NumPy arrays.

    Raises:
        ValueError: If `n_cols <= 0` or `n_samp <= 0`.

    Notes:
        - `OBSERVATION_TIME` is initialized to `NaT` with millisecond resolution
          (`datetime64[ms]`).
        - `TI` is initialized as a fixed-width unicode string array (`U32`).
    """
    if n_cols <= 0:
        raise ValueError(f"nCols must be positive; got {n_cols}")
    if n_samp <= 0:
        raise ValueError(f"nSamp must be positive; got {n_samp}")

    sci_dict: dict[str, NDArray[Any]] = {
                'OBSERVATION_TIME': np.full(n_cols, np.datetime64('NaT', 'ms')),
                'DELAY': np.zeros(n_cols, dtype=np.float32),
                'START_STEP': np.zeros(n_cols, dtype=np.uint16),
                'SUB_SPACECRAFT_LATITUDE': np.zeros(n_cols, dtype=np.float32),
                'SUB_SPACECRAFT_LONGITUDE': np.zeros(n_cols, dtype=np.float32),
                'SPACECRAFT_ALTITUDE': np.zeros(n_cols, dtype=np.float32),
                'RANGE0_ALTITUDE': np.zeros(n_cols, dtype=np.float32),
                'TI': np.full(n_cols, "", dtype="U32"),
                'SAR_IMAGE_REAL': np.zeros([n_samp, n_cols], dtype=np.float32),
                'SAR_IMAGE_IMAG': np.zeros([n_samp, n_cols], dtype=np.float32),
                }
    return sci_dict


def make_lrs_sar_p_dict(n_cols: int,
                        n_samp: int = 1000,
                        ) -> dict[str, NDArray[Any]]:
    """
    Initializes an LRS RDR (SAR-P) science dictionary.

    Preallocates NumPy arrays for each LRS RDR SAR-P science field. Array
    lengths are set by `n_cols` (number of records), and SAR image arrays are
    shaped as `(n_samp, n_cols)` (samples × records).

    Args:
        n_cols: Number of columns (science records) in the output arrays.
        n_samp: Number of samples in each SAR image record. Defaults to 1000.

    Returns:
        A dictionary mapping field names to preallocated NumPy arrays.

    Raises:
        ValueError: If `n_cols <= 0` or `n_samp <= 0`.

    Notes:
        - `OBSERVATION_TIME` is initialized to `NaT` with millisecond resolution
          (`datetime64[ms]`).
        - `TI` is initialized as a fixed-width unicode string array (`U32`);
          adjust width if the LRS spec requires otherwise.
        - `SAR_IMAGE` is stored as `uint8`, consistent with SAR-P magnitude
          products.
    """
    if n_cols <= 0:
        raise ValueError(f"n_cols must be positive; got {n_cols}")
    if n_samp <= 0:
        raise ValueError(f"n_samp must be positive; got {n_samp}")

    sci_dict: dict[str, NDArray[Any]] = {
        "OBSERVATION_TIME": np.full(n_cols, np.datetime64("NaT", "ms")),
        "DELAY": np.zeros(n_cols, dtype=np.float32),
        "START_STEP": np.zeros(n_cols, dtype=np.uint16),
        "SUB_SPACECRAFT_LATITUDE": np.zeros(n_cols, dtype=np.float32),
        "SUB_SPACECRAFT_LONGITUDE": np.zeros(n_cols, dtype=np.float32),
        "SPACECRAFT_ALTITUDE": np.zeros(n_cols, dtype=np.float32),
        "DISTANCE_TO_RANGE0": np.zeros(n_cols, dtype=np.float32),
        "TI": np.full(n_cols, "", dtype="U32"),
        "SAR_IMAGE": np.zeros((n_samp, n_cols), dtype=np.uint8),
    }
    return sci_dict