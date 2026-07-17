# SPDX-License-Identifier: BSD-3-Clause
from numpy.typing import NDArray
from scipy.fft import fft, ifft

def range_compress(data: NDArray,
                   h_f: NDArray,
                   window: NDArray,
                   input_time: bool = True,
                   output_time: bool = True) -> NDArray:
    """Apply a windowed frequency-domain filter to echo data.

    Optionally transforms the input to the frequency domain,
    applies the windowed filter ``h_f * window``, and optionally
    transforms back to the time domain. ``h_f`` may be 1D (one
    filter shared across all columns) or 2D (one filter per
    column — used when the reference chirp varies along-track,
    e.g. temperature-calibrated SHARAD chirps).

    Args:
        data: Echo data. Shape ``(n_samp,)`` for a single record or
            ``(n_samp, n_cols)`` for a record block. Time or
            frequency domain depending on ``input_time``.
        h_f: Frequency-domain filter. Shape ``(n_samp,)`` for a
            shared filter, or ``(n_samp, n_cols)`` for a per-column
            filter.
        window: Spectral window, shape ``(n_samp,)``.
        input_time: True if input data is in the time domain.
        output_time: True if output data should be in the time domain.

    Returns:
        Filtered data. Matches the input rank: shape ``(n_samp,)``
        for 1D input, ``(n_samp, n_cols)`` for 2D input.

    Raises:
        ValueError: If ``data`` is not 1D or 2D, ``h_f`` is not 1D
            or 2D, if their shapes do not match, or if ``window``
            length does not match ``n_samp``.
    """
    if data.ndim == 1:
        was_1d = True
        data = data[:, None]
    elif data.ndim == 2:
        was_1d = False
    else:
        raise ValueError(f'data must be 1D or 2D, got shape {data.shape!r}')

    n_samp, n_cols = data.shape
    if h_f.ndim == 1:
        if h_f.shape[0] != n_samp:
            raise ValueError('h_f length must equal n_samp')
    elif h_f.ndim == 2:
        if h_f.shape != (n_samp, n_cols):
            raise ValueError(
                f'h_f shape must be ({n_samp},) or ({n_samp}, {n_cols}); '
                f'got {h_f.shape}'
            )
    else:
        raise ValueError('h_f must be 1D or 2D')
    if len(window) != n_samp:
        raise ValueError('window length must equal n_samp')
    if input_time:
        data = fft(data, axis=0, workers=-1)
    if h_f.ndim == 1:
        data = data * (h_f * window)[:, None]
    else:
        data = data * h_f * window[:, None]
    if output_time:
        data = ifft(data, axis=0, workers=-1)

    if was_1d:
        data = data[:, 0]
    return data

