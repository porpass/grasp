# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

def bytscl(data: NDArray,
           top: int = 255,
           min_val: float | None = None,
           max_val: float | None = None) -> NDArray[np.uint8]:
    """
    Replicates the IDL BYTSCL function in Python using NumPy.
    Scales input data to the range [0, top] and converts to uint8.

    Args:
        data (np.ndarray): The input array (integer or float).
        top (int, optional): The maximum value of the output range (default 255).
        min_val (Number, optional): The minimum value of the input range.
                                     If None, uses the minimum of the data.
        max_val (Number, optional): The maximum value of the input range.
                                     If None, uses the maximum of the data.

    Returns:
        np.ndarray: The scaled data as an 8-bit unsigned integer array (uint8).
    """
    # Determine min/max if not provided
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)

    # Handle division by zero if range is 0
    if max_val == min_val:
        return np.full(data.shape, 0, dtype=np.uint8)

    # Apply the appropriate IDL formula based on data type
    if np.issubdtype(data.dtype, np.floating):
        # Floating-point formula
        scaled_data = (top + 0.9999) * (data - min_val) / (max_val - min_val)
    else:
        # Integer formula
        scaled_data = (top + 1) * (data - min_val - 1) / (max_val - min_val)

    # Clip values to the output range [0, top] and convert to uint8
    scaled_data = np.clip(scaled_data, 0, top)
    return scaled_data.astype(np.uint8)

