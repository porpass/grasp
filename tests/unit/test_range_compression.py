# SPDX-License-Identifier: BSD-3-Clause
"""Tests for range_compress in grasp.processing.range_compression.compress.

range_compress is the workhorse of pulse compression: it applies a
windowed frequency-domain filter to echo data. The tests here cover:

  - shape preservation and validation errors
  - identity behaviour when filter and window are unity
  - the input_time / output_time domain-flag combinations
  - the scientifically meaningful test: a matched filter applied to a
    chirp embedded at a known delay must produce a peak at that delay.
"""
import numpy as np
import pytest

from grasp.processing.range_compression.compress import range_compress
from grasp.processing.range_compression.chirps import create_complex_baseband_chirp


# --------------------------------------------------------------------- #
#  Shape and validation                                                 #
# --------------------------------------------------------------------- #

def _unity_filter_and_window(n):
    return (np.ones(n, dtype=np.complex64),
            np.ones(n, dtype=np.float32))


@pytest.mark.parametrize("n_samp, n_cols", [(64, 1), (256, 4), (1024, 16)])
def test_range_compress_preserves_shape(n_samp, n_cols):
    """Output shape equals input shape regardless of domain flags."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samp, n_cols)).astype(np.complex64)
    h_f, window = _unity_filter_and_window(n_samp)
    out = range_compress(data, h_f, window)
    assert out.shape == data.shape


def test_range_compress_rejects_filter_length_mismatch():
    data = np.zeros((128, 2), dtype=np.complex64)
    h_f = np.ones(127, dtype=np.complex64)
    window = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match="h_f length"):
        range_compress(data, h_f, window)


def test_range_compress_rejects_window_length_mismatch():
    data = np.zeros((128, 2), dtype=np.complex64)
    h_f = np.ones(128, dtype=np.complex64)
    window = np.ones(127, dtype=np.float32)
    with pytest.raises(ValueError, match="window length"):
        range_compress(data, h_f, window)


def test_range_compress_rejects_h_f_wrong_dimensionality():
    data = np.zeros((128, 2), dtype=np.complex64)
    h_f = np.ones((128, 2, 1), dtype=np.complex64)
    window = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match="h_f must be 1D or 2D"):
        range_compress(data, h_f, window)


def test_range_compress_rejects_2d_h_f_wrong_shape():
    data = np.zeros((128, 4), dtype=np.complex64)
    h_f = np.ones((128, 3), dtype=np.complex64)  # wrong n_cols
    window = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match=r"h_f shape must be"):
        range_compress(data, h_f, window)


# --------------------------------------------------------------------- #
#  2D h_f — per-column filter (e.g. temperature-varying SHARAD chirps)  #
# --------------------------------------------------------------------- #

def test_range_compress_2d_h_f_matches_1d_when_columns_identical():
    """If every column of a 2D h_f is the same as a 1D h_f, the
    outputs must match."""
    rng = np.random.default_rng(7)
    n_samp, n_cols = 256, 5
    data = (rng.standard_normal((n_samp, n_cols))
            + 1j * rng.standard_normal((n_samp, n_cols))).astype(np.complex64)
    h_f_1d = (rng.standard_normal(n_samp)
              + 1j * rng.standard_normal(n_samp)).astype(np.complex64)
    h_f_2d = np.broadcast_to(h_f_1d[:, None], (n_samp, n_cols)).copy()
    window = np.ones(n_samp, dtype=np.float32)

    out_1d = range_compress(data, h_f_1d, window)
    out_2d = range_compress(data, h_f_2d, window)
    np.testing.assert_allclose(out_2d, out_1d, atol=1e-4)


def test_range_compress_2d_h_f_distinct_columns_apply_independently():
    """Each column of data is filtered by the corresponding column of
    h_f. Columns must not bleed into one another."""
    n_samp, n_cols = 128, 3
    # Distinct unit-magnitude filters per column
    phases = np.linspace(0, np.pi, n_cols)
    h_f = np.exp(1j * phases)[None, :].astype(np.complex64)
    h_f = np.broadcast_to(h_f, (n_samp, n_cols)).copy()
    window = np.ones(n_samp, dtype=np.float32)
    data = np.ones((n_samp, n_cols), dtype=np.complex64)

    out = range_compress(data, h_f, window,
                         input_time=False, output_time=False)
    # output[k, c] == data[k, c] * h_f[k, c] (window is unity, no FFT)
    np.testing.assert_allclose(out, h_f, atol=1e-5)


@pytest.mark.parametrize("delay_samples", [200, 500])
def test_matched_filter_per_column_peaks_at_known_delay(delay_samples):
    """Each column gets matched-filtered by its own chirp; both peaks
    land at the expected sample."""
    nfft = 4096
    dt = 1e-8
    tau = 5e-7
    bw = 50e6
    n_cols = 2

    chirp, _, _ = create_complex_baseband_chirp(nfft, dt, tau, bw)
    n_tau = int(round(tau / dt))
    if n_tau % 2 == 0:
        n_tau += 1
    assert delay_samples + n_tau < nfft

    received = np.zeros((nfft, n_cols), dtype=np.complex64)
    for c in range(n_cols):
        received[delay_samples:delay_samples + n_tau, c] = chirp[:n_tau]

    # Build a 2D matched filter: same filter per column, but exercising
    # the 2D code path.
    h_f_1d = np.conj(np.fft.fft(chirp))
    h_f_2d = np.broadcast_to(h_f_1d[:, None], (nfft, n_cols)).copy().astype(np.complex64)
    window = np.ones(nfft, dtype=np.float32)

    compressed = range_compress(received, h_f_2d, window)
    for c in range(n_cols):
        peak = int(np.argmax(np.abs(compressed[:, c])))
        assert abs(peak - delay_samples) <= 2


# --------------------------------------------------------------------- #
#  Identity behaviour                                                   #
# --------------------------------------------------------------------- #

def test_range_compress_identity_round_trips_time_domain():
    """With h_f=1 and window=1, time→time round-trips the input."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((512, 3)).astype(np.complex64) \
        + 1j * rng.standard_normal((512, 3)).astype(np.complex64)
    h_f, window = _unity_filter_and_window(512)
    out = range_compress(data, h_f, window, input_time=True, output_time=True)
    np.testing.assert_allclose(out, data, atol=1e-4)


def test_range_compress_input_time_false_treats_data_as_freq_domain():
    """If input is already in frequency domain, FFT step is skipped."""
    rng = np.random.default_rng(2)
    data_time = rng.standard_normal((128, 1)).astype(np.complex64)
    data_freq = np.fft.fft(data_time, axis=0)
    h_f, window = _unity_filter_and_window(128)

    via_time = range_compress(data_time, h_f, window,
                              input_time=True,  output_time=True)
    via_freq = range_compress(data_freq, h_f, window,
                              input_time=False, output_time=True)
    np.testing.assert_allclose(via_time, via_freq, atol=1e-4)


def test_range_compress_output_time_false_returns_freq_domain():
    """output_time=False leaves the result in the frequency domain."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((128, 1)).astype(np.complex64)
    h_f, window = _unity_filter_and_window(128)

    out_freq = range_compress(data, h_f, window,
                              input_time=True, output_time=False)
    expected = np.fft.fft(data, axis=0)  # since h_f=window=1
    np.testing.assert_allclose(out_freq, expected, atol=1e-4)


# --------------------------------------------------------------------- #
#  Matched-filter correctness on a synthetic chirp                      #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("delay_samples", [200, 500, 1000, 2000])
def test_matched_filter_peak_at_known_delay(delay_samples):
    """Matched filtering a delayed chirp produces a peak at the delay.

    Constructs a complex LFM chirp, embeds it at a known delay in an
    otherwise empty trace, builds the matched filter h_f = conj(FFT(chirp)),
    and asserts that |compressed|.argmax() lands at delay_samples ± 2.
    """
    nfft = 4096
    dt = 1e-8         # 10 ns sample spacing
    tau = 5e-7        # 500 ns chirp
    bw = 50e6         # 50 MHz bandwidth

    chirp, _, _ = create_complex_baseband_chirp(nfft, dt, tau, bw)
    n_tau = int(round(tau / dt))
    if n_tau % 2 == 0:
        n_tau += 1
    assert delay_samples + n_tau < nfft, "delay too large for nfft"

    # Build received signal: chirp leading edge at delay_samples.
    received = np.zeros((nfft, 1), dtype=np.complex64)
    received[delay_samples:delay_samples + n_tau, 0] = chirp[:n_tau]

    # Matched filter and unit window.
    h_f = np.conj(np.fft.fft(chirp))
    window = np.ones(nfft, dtype=np.float32)

    compressed = range_compress(received, h_f, window)

    peak_idx = int(np.argmax(np.abs(compressed[:, 0])))
    assert abs(peak_idx - delay_samples) <= 2, \
        f"peak at sample {peak_idx}, expected {delay_samples}"


def test_matched_filter_separates_two_targets():
    """Two delayed chirp returns produce two distinguishable peaks."""
    nfft = 4096
    dt = 1e-8
    tau = 5e-7
    bw = 50e6
    d1, d2 = 400, 1500

    chirp, _, _ = create_complex_baseband_chirp(nfft, dt, tau, bw)
    n_tau = int(round(tau / dt))
    if n_tau % 2 == 0:
        n_tau += 1

    received = np.zeros((nfft, 1), dtype=np.complex64)
    received[d1:d1 + n_tau, 0] += chirp[:n_tau]
    received[d2:d2 + n_tau, 0] += 0.5 * chirp[:n_tau]  # weaker second return

    h_f = np.conj(np.fft.fft(chirp))
    window = np.ones(nfft, dtype=np.float32)
    compressed = np.abs(range_compress(received, h_f, window)[:, 0])

    # Each candidate region should contain a local max near the expected delay.
    half = max(n_tau // 2, 10)
    peak1 = d1 + np.argmax(compressed[max(0, d1 - half):d1 + half]) - half
    peak2 = d2 + np.argmax(compressed[max(0, d2 - half):d2 + half]) - half
    assert abs(peak1 - d1) <= 2
    assert abs(peak2 - d2) <= 2
    # Stronger target should give a larger peak.
    assert compressed[d1] > compressed[d2]
