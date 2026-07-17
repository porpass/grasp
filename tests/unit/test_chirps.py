# SPDX-License-Identifier: BSD-3-Clause
"""Tests for chirp constructors in grasp.processing.range_compression.chirps.

create_complex_baseband_chirp and create_real_baseband_chirp are pure
math with no external dependencies and are ideal unit-test targets.
create_sharad_calibrated_chirp needs calibration files; we cover only
its error path here.
"""
import numpy as np
import pytest

from grasp.processing.range_compression.chirps import (
    create_complex_baseband_chirp,
    create_real_baseband_chirp,
    create_sharad_calibrated_chirp,
)

# (label, n, dt, tau, bw) — a few realistic configurations
CHIRP_CONFIGS = [
    ("SHARAD_like", 4096, 0.0375e-6, 85e-6,  10e6),    # 85 us, 10 MHz BW
    ("MARSIS_like", 2048, 1/2.8e6,   250e-6, 1e6),     # 250 us, 1 MHz BW, 2.8 MHz fs
    ("toy",         256,  1e-9,      100e-9, 50e6),    # tiny test case
]


# --------------------------------------------------------------------- #
#  create_complex_baseband_chirp                                        #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label, n, dt, tau, bw", CHIRP_CONFIGS, ids=[c[0] for c in CHIRP_CONFIGS]
)
def test_complex_chirp_shapes_and_dtypes(label, n, dt, tau, bw):
    """Output arrays have the contracted shapes and dtypes."""
    chirp, t_tau, f_tau = create_complex_baseband_chirp(n, dt, tau, bw)
    assert chirp.shape == (n,)
    assert chirp.dtype == np.complex64
    assert t_tau.shape == f_tau.shape
    assert t_tau.dtype == np.float32
    assert f_tau.dtype == np.float32


@pytest.mark.parametrize(
    "label, n, dt, tau, bw", CHIRP_CONFIGS, ids=[c[0] for c in CHIRP_CONFIGS]
)
def test_complex_chirp_pulse_length_is_odd(label, n, dt, tau, bw):
    """The non-zero pulse region is forced to odd length so t=0 is exact."""
    _, t_tau, _ = create_complex_baseband_chirp(n, dt, tau, bw)
    assert t_tau.size % 2 == 1


@pytest.mark.parametrize(
    "label, n, dt, tau, bw", CHIRP_CONFIGS, ids=[c[0] for c in CHIRP_CONFIGS]
)
def test_complex_chirp_constant_magnitude_in_pulse(label, n, dt, tau, bw):
    """|chirp| == 1 inside the pulse and 0 outside."""
    chirp, t_tau, _ = create_complex_baseband_chirp(n, dt, tau, bw)
    n_tau = t_tau.size
    np.testing.assert_allclose(np.abs(chirp[:n_tau]), 1.0, atol=1e-5)
    np.testing.assert_array_equal(chirp[n_tau:], 0)


@pytest.mark.parametrize(
    "label, n, dt, tau, bw", CHIRP_CONFIGS, ids=[c[0] for c in CHIRP_CONFIGS]
)
def test_complex_chirp_time_vector_centered(label, n, dt, tau, bw):
    """t_tau is uniformly spaced by dt and centered about 0."""
    _, t_tau, _ = create_complex_baseband_chirp(n, dt, tau, bw)
    n_tau = t_tau.size
    # Center sample is exactly 0
    assert t_tau[n_tau // 2] == pytest.approx(0.0, abs=1e-12)
    # Symmetric about 0
    np.testing.assert_allclose(t_tau, -t_tau[::-1], atol=1e-12)
    # Uniform spacing. Accumulated float32 round-off in `arange * dt`
    # means a small fraction of diffs land one ULP off, so rtol must
    # exceed float32 epsilon by a generous margin.
    np.testing.assert_allclose(np.diff(t_tau), dt, rtol=1e-3)


@pytest.mark.parametrize(
    "label, n, dt, tau, bw", CHIRP_CONFIGS, ids=[c[0] for c in CHIRP_CONFIGS]
)
def test_complex_chirp_instantaneous_freq_matches_K_t(label, n, dt, tau, bw):
    """f_tau == K * t_tau where K = bw / tau (definition of LFM)."""
    _, t_tau, f_tau = create_complex_baseband_chirp(n, dt, tau, bw)
    K = bw / tau
    np.testing.assert_allclose(f_tau, K * t_tau, rtol=1e-5, atol=1.0)


@pytest.mark.parametrize(
    "label, n, dt, tau, bw", CHIRP_CONFIGS, ids=[c[0] for c in CHIRP_CONFIGS]
)
def test_complex_chirp_phase_derivative_recovers_K(label, n, dt, tau, bw):
    """Phase derivative of the synthesized chirp recovers the FM rate K."""
    chirp, t_tau, _ = create_complex_baseband_chirp(n, dt, tau, bw)
    n_tau = t_tau.size
    phase = np.unwrap(np.angle(chirp[:n_tau].astype(np.complex128)))
    # Instantaneous frequency in Hz = dφ/dt / (2π)
    inst_f = np.diff(phase) / (2.0 * np.pi * dt)
    # Drop edge samples to avoid wrap artifacts
    inst_f = inst_f[5:-5]
    t_mid = (0.5 * (t_tau[:-1] + t_tau[1:]))[5:-5]
    K = bw / tau
    np.testing.assert_allclose(inst_f, K * t_mid, rtol=1e-2)


def test_complex_chirp_downchirp_is_conjugate_of_upchirp():
    """chirp_direction=-1 produces the conjugate of chirp_direction=+1."""
    up, _, _ = create_complex_baseband_chirp(1024, 1e-8, 1e-6, 50e6, chirp_direction=1)
    dn, _, _ = create_complex_baseband_chirp(1024, 1e-8, 1e-6, 50e6, chirp_direction=-1)
    np.testing.assert_allclose(dn, np.conj(up), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, exc_msg",
    [
        ({"n": 0,    "dt": 1e-8, "tau": 1e-6, "bw": 1e7},          "n must be positive"),
        ({"n": -10,  "dt": 1e-8, "tau": 1e-6, "bw": 1e7},          "n must be positive"),
        ({"n": 1024, "dt": 0.0,  "tau": 1e-6, "bw": 1e7},          "dt must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 0.0,  "bw": 1e7},          "tau must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 1e-6, "bw": 0.0},          "bw must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 1e-6, "bw": 1e7, "chirp_direction": 2}, "sgn must be -1 or 1"),
        ({"n": 16,   "dt": 1e-9, "tau": 1e-6, "bw": 1e7},          "duration exceeds"),
    ],
)
def test_complex_chirp_validation_errors(kwargs, exc_msg):
    """Each invalid input raises a ValueError with the documented message."""
    with pytest.raises(ValueError, match=exc_msg):
        create_complex_baseband_chirp(**kwargs)


# --------------------------------------------------------------------- #
#  create_real_baseband_chirp                                           #
# --------------------------------------------------------------------- #

# (label, n, dt, tau, f_start, f_end)
REAL_CHIRP_CONFIGS = [
    ("sweep_up",   4096, 1e-8,  1e-6,  5e6,  25e6),
    ("sweep_down", 4096, 1e-8,  1e-6, 25e6,   5e6),
    ("toy",         256, 1e-9, 50e-9, 10e6,  60e6),
]


@pytest.mark.parametrize(
    "label, n, dt, tau, f_start, f_end",
    REAL_CHIRP_CONFIGS, ids=[c[0] for c in REAL_CHIRP_CONFIGS],
)
def test_real_chirp_shapes_and_dtypes(label, n, dt, tau, f_start, f_end):
    chirp, t_tau, f_tau = create_real_baseband_chirp(n, dt, tau, f_start, f_end)
    assert chirp.shape == (n,)
    assert chirp.dtype == np.float32
    assert t_tau.shape == f_tau.shape
    assert t_tau.dtype == np.float32
    assert f_tau.dtype == np.float32


@pytest.mark.parametrize(
    "label, n, dt, tau, f_start, f_end",
    REAL_CHIRP_CONFIGS, ids=[c[0] for c in REAL_CHIRP_CONFIGS],
)
def test_real_chirp_is_causal_and_zero_padded(label, n, dt, tau, f_start, f_end):
    """t_tau starts at 0 and chirp is zero after the pulse."""
    chirp, t_tau, _ = create_real_baseband_chirp(n, dt, tau, f_start, f_end)
    assert t_tau[0] == 0.0
    np.testing.assert_array_equal(chirp[t_tau.size:], 0.0)


@pytest.mark.parametrize(
    "label, n, dt, tau, f_start, f_end",
    REAL_CHIRP_CONFIGS, ids=[c[0] for c in REAL_CHIRP_CONFIGS],
)
def test_real_chirp_values_in_unit_interval(label, n, dt, tau, f_start, f_end):
    """A real LFM is a sinusoid, so values must lie in [-1, 1]."""
    chirp, _, _ = create_real_baseband_chirp(n, dt, tau, f_start, f_end)
    assert chirp.min() >= -1.0 - 1e-5
    assert chirp.max() <= 1.0 + 1e-5


@pytest.mark.parametrize(
    "label, n, dt, tau, f_start, f_end",
    REAL_CHIRP_CONFIGS, ids=[c[0] for c in REAL_CHIRP_CONFIGS],
)
def test_real_chirp_freq_sweep_endpoints(label, n, dt, tau, f_start, f_end):
    """f_tau[0] == f_start; f_tau[-1] is close to f_end."""
    _, _, f_tau = create_real_baseband_chirp(n, dt, tau, f_start, f_end)
    assert f_tau[0] == pytest.approx(f_start, rel=1e-5)
    # Final sample lands at t = (n_tau-1)*dt rather than exactly tau,
    # so the end frequency is within one sample step of f_end.
    sample_step = abs(f_end - f_start) / tau * dt
    assert f_tau[-1] == pytest.approx(f_end, abs=2 * sample_step)


@pytest.mark.parametrize(
    "kwargs, exc_msg",
    [
        ({"n": 0,    "dt": 1e-8, "tau": 1e-6, "f_start": 1e6,  "f_end": 2e6}, "n must be positive"),
        ({"n": 1024, "dt": 0.0,  "tau": 1e-6, "f_start": 1e6,  "f_end": 2e6}, "dt must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 0.0,  "f_start": 1e6,  "f_end": 2e6}, "tau must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 1e-6, "f_start": 0.0,  "f_end": 2e6}, "f_start must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 1e-6, "f_start": 1e6,  "f_end": 0.0}, "f_end must be positive"),
        ({"n": 1024, "dt": 1e-8, "tau": 1e-6, "f_start": 1e6,  "f_end": 1e6}, "must be different"),
        ({"n": 16,   "dt": 1e-9, "tau": 1e-6, "f_start": 1e6,  "f_end": 2e6}, "duration exceeds"),
    ],
)
def test_real_chirp_validation_errors(kwargs, exc_msg):
    """Each invalid input raises a ValueError with the documented message."""
    with pytest.raises(ValueError, match=exc_msg):
        create_real_baseband_chirp(**kwargs)


# --------------------------------------------------------------------- #
#  create_sharad_calibrated_chirp — error path only                     #
# --------------------------------------------------------------------- #

def test_calibrated_chirp_missing_file_raises(tmp_path):
    """Pointing the loader at an empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Calibration file not found"):
        create_sharad_calibrated_chirp(tx_temp=0.0, rx_temp=0.0, calib_root=tmp_path)


def test_calibrated_chirp_loads_from_bundled_default():
    """With no calib_root, the bundled calibration files are loaded."""
    spectrum, f = create_sharad_calibrated_chirp(tx_temp=0.0, rx_temp=0.0)
    assert spectrum.shape == (2048,)
    assert spectrum.dtype == np.complex64
    assert f.shape == (2048,)
    assert f.dtype == np.float32
    # Spectrum should not be all zeros; magnitudes should be finite.
    assert np.any(np.abs(spectrum) > 0)
    assert np.all(np.isfinite(np.abs(spectrum)))


@pytest.mark.parametrize("tx_temp", [-20, -15, -10, -5, 0, 20, 40, 60])
@pytest.mark.parametrize("rx_temp", [-20, 0, 20, 40, 60])
def test_calibrated_chirp_every_bin_loads(tx_temp, rx_temp):
    """All 40 TX/RX temperature combinations resolve to a real file."""
    spectrum, f = create_sharad_calibrated_chirp(tx_temp=tx_temp, rx_temp=rx_temp)
    assert spectrum.shape == (2048,)
    assert f.shape == (2048,)
