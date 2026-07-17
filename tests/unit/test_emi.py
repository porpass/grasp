# SPDX-License-Identifier: BSD-3-Clause
"""Tests for EMI suppression in grasp.processing.emi.

Two functions under test:
    - threshold_emi: moving mean/std threshold; clips values above
      ``mean + threshold_k * std`` down to ``mean + value_k * std``.
    - adaptive_spectral_notch: MAD/STD-based outlier flagging with
      INTERP / BASELINE / ZERO replacement strategies.

The scientifically meaningful tests inject a strong tone into Gaussian
noise spectra and assert the function flags + suppresses it without
flagging the bulk of the noise.
"""
import numpy as np
import pytest

from grasp.processing.emi.threshold import threshold_emi
from grasp.processing.emi.adaptive import adaptive_spectral_notch


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #

def _white_spectrum(n_samp=1024, n_cols=1, seed=0):
    """Complex Gaussian white-noise spectrum, std ≈ 1."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n_samp, n_cols))
    imag = rng.standard_normal((n_samp, n_cols))
    return (real + 1j * imag).astype(np.complex128)


def _baseband_freqs(n_samp, dt=1.0):
    """Monotonic fftshifted frequency vector in [-0.5/dt, 0.5/dt]."""
    return np.fft.fftshift(np.fft.fftfreq(n_samp, d=dt))


# --------------------------------------------------------------------- #
#  threshold_emi — shape, behaviour, and validation                     #
# --------------------------------------------------------------------- #

def test_threshold_preserves_shape_and_mask_dtype():
    spectra = _white_spectrum(512, 3)
    freqs = _baseband_freqs(512)
    out, mask = threshold_emi(spectra, freqs, bw=0.8,
                              threshold_k=3.0, value_k=1.0)
    assert out.shape == spectra.shape
    assert mask.shape == spectra.shape
    assert mask.dtype == np.bool_


def test_threshold_suppresses_injected_tone():
    """A spike >> noise floor is both flagged and reduced in magnitude."""
    n_samp = 1024
    spectra = _white_spectrum(n_samp, 1, seed=42)
    freqs = _baseband_freqs(n_samp)
    bw = 0.8  # in-band: |f| <= 0.4

    tone_idx = n_samp // 2 + 10
    spectra[tone_idx, 0] += 50.0
    original_mag = float(np.abs(spectra[tone_idx, 0]))

    out, mask = threshold_emi(spectra, freqs, bw=bw,
                              threshold_k=3.0, value_k=1.0, window_size=64)

    assert mask[tone_idx, 0], "tone bin should be flagged"
    suppressed = float(np.abs(out[tone_idx, 0]))
    assert suppressed < original_mag / 5, \
        f"tone should be substantially reduced: was {original_mag}, now {suppressed}"


def test_threshold_preserves_phase_at_clipped_bin():
    """Clipping only the magnitude leaves the original phase intact."""
    n_samp = 1024
    spectra = _white_spectrum(n_samp, 1, seed=7)
    freqs = _baseband_freqs(n_samp)

    tone_idx = n_samp // 2 + 10
    spectra[tone_idx, 0] = 100.0 * np.exp(1j * 0.3)

    out, mask = threshold_emi(spectra, freqs, bw=0.8,
                              threshold_k=3.0, value_k=1.0)

    assert mask[tone_idx, 0]
    assert np.angle(out[tone_idx, 0]) == pytest.approx(0.3, abs=1e-6)


def test_threshold_zeros_out_of_band():
    """threshold_emi initialises output with zeros, so out-of-band bins
    are returned as exactly zero — this is documented behaviour worth
    pinning to avoid silent regressions.
    """
    n_samp = 1024
    spectra = _white_spectrum(n_samp, 1, seed=3)
    freqs = _baseband_freqs(n_samp)
    bw = 0.4  # narrow band, |f| <= 0.2 → most bins are out-of-band

    out, mask = threshold_emi(spectra, freqs, bw=bw,
                              threshold_k=3.0, value_k=1.0)

    out_of_band = np.abs(freqs) > bw / 2
    np.testing.assert_array_equal(out[out_of_band, 0], 0)
    np.testing.assert_array_equal(mask[out_of_band, 0], False)


@pytest.mark.parametrize(
    "kwargs, exc_msg",
    [
        ({"spectra": np.zeros(100, dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5,
          "threshold_k": 3, "value_k": 1},                  "must be 2-D"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.0,
          "threshold_k": 3, "value_k": 1},                  "bw must be positive"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5,
          "threshold_k": 3, "value_k": 1, "window_size": 0}, "window_size must be positive"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 50),  "bw": 0.5,
          "threshold_k": 3, "value_k": 1},                  "freqs length"),
    ],
)
def test_threshold_validation_errors(kwargs, exc_msg):
    with pytest.raises(ValueError, match=exc_msg):
        threshold_emi(**kwargs)


# --------------------------------------------------------------------- #
#  adaptive_spectral_notch — shape, behaviour, and validation           #
# --------------------------------------------------------------------- #

def test_adaptive_preserves_shape_and_mask_dtype():
    spectra = _white_spectrum(512, 3)
    freqs = _baseband_freqs(512)
    out, mask = adaptive_spectral_notch(spectra, freqs, bw=0.8)
    assert out.shape == spectra.shape
    assert mask.shape == spectra.shape
    assert mask.dtype == np.bool_


@pytest.mark.parametrize("replace", ["INTERP", "BASELINE", "ZERO"])
def test_adaptive_suppresses_tone_with_each_strategy(replace):
    """All three replacement strategies must reduce the injected tone."""
    n_samp = 1024
    spectra = _white_spectrum(n_samp, 1, seed=11)
    freqs = _baseband_freqs(n_samp)

    tone_idx = n_samp // 2 + 50
    spectra[tone_idx, 0] += 100.0
    original_mag = float(np.abs(spectra[tone_idx, 0]))

    out, mask = adaptive_spectral_notch(spectra, freqs, bw=0.8,
                                        replace=replace)

    assert mask[tone_idx, 0], f"{replace}: tone not flagged"
    suppressed = float(np.abs(out[tone_idx, 0]))
    assert suppressed < original_mag / 5, \
        f"{replace}: weak suppression ({original_mag} -> {suppressed})"


def test_adaptive_zero_actually_zeros_flagged_bins():
    """With replace='ZERO', every flagged bin must be exactly 0."""
    n_samp = 1024
    spectra = _white_spectrum(n_samp, 1, seed=12)
    freqs = _baseband_freqs(n_samp)
    spectra[n_samp // 2 + 100, 0] += 100.0

    out, mask = adaptive_spectral_notch(spectra, freqs, bw=0.8,
                                        replace="ZERO")
    np.testing.assert_array_equal(out[mask], 0)


def test_adaptive_preserves_in_band_non_flagged_bins():
    """In-band non-flagged bins should round-trip through the
    magnitude+phase reconstruction unchanged."""
    n_samp = 512
    bw = 0.8
    spectra = _white_spectrum(n_samp, 1, seed=13)
    freqs = _baseband_freqs(n_samp)
    out, mask = adaptive_spectral_notch(spectra, freqs, bw=bw)

    in_band = np.abs(freqs) <= bw / 2
    in_band_2d = in_band[:, None] & np.ones_like(mask, dtype=bool)
    keep = in_band_2d & ~mask
    np.testing.assert_allclose(out[keep], spectra[keep], atol=1e-10)


def test_adaptive_zeros_out_of_band():
    """Out-of-band bins are zeroed, matching threshold_emi."""
    n_samp = 1024
    bw = 0.4  # narrow band → most bins are out-of-band
    spectra = _white_spectrum(n_samp, 1, seed=21)
    freqs = _baseband_freqs(n_samp)

    out, mask = adaptive_spectral_notch(spectra, freqs, bw=bw)

    out_of_band = np.abs(freqs) > bw / 2
    np.testing.assert_array_equal(out[out_of_band, 0], 0)
    np.testing.assert_array_equal(mask[out_of_band, 0], False)


def test_adaptive_clean_noise_yields_low_false_positive_rate():
    """k=6 with MAD on clean noise should flag essentially nothing."""
    n_samp = 4096
    spectra = _white_spectrum(n_samp, 1, seed=17)
    freqs = _baseband_freqs(n_samp)
    _, mask = adaptive_spectral_notch(spectra, freqs, bw=0.8, k=6.0)
    fp_rate = float(mask[:, 0].sum()) / n_samp
    assert fp_rate < 0.01, f"too many false positives on clean noise: {fp_rate:.3%}"


@pytest.mark.parametrize("statistic", ["MAD", "STD"])
def test_adaptive_both_methods_suppress_tone(statistic):
    """Both MAD- and STD-based statistics should catch a strong tone."""
    n_samp = 1024
    spectra = _white_spectrum(n_samp, 1, seed=19)
    freqs = _baseband_freqs(n_samp)
    tone_idx = n_samp // 2 + 30
    spectra[tone_idx, 0] += 100.0

    out, mask = adaptive_spectral_notch(spectra, freqs, bw=0.8,
                                        statistic=statistic)
    assert mask[tone_idx, 0]
    assert np.abs(out[tone_idx, 0]) < 0.2 * np.abs(spectra[tone_idx, 0])


@pytest.mark.parametrize(
    "kwargs, exc_msg",
    [
        ({"spectra": np.zeros(100, dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5},         "must be 2-D"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.0},         "bw must be positive"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5,
          "window_size": 128},                                  "positive odd"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5,
          "interp_pad": -1},                                    "interp_pad must be"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5,
          "statistic": "FOO"},                                     "Unsupported statistic"),
        ({"spectra": np.zeros((100, 1), dtype=complex),
          "freqs": np.linspace(-1, 1, 100), "bw": 0.5,
          "replace": "FOO"},                                    "Unsupported replace"),
    ],
)
def test_adaptive_validation_errors(kwargs, exc_msg):
    with pytest.raises(ValueError, match=exc_msg):
        adaptive_spectral_notch(**kwargs)
