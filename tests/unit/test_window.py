# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the window functions in grasp.processing.windows.

These tests assert universal properties that should hold for every
symmetric window we ship:
    - the output length matches the requested n
    - the output is float32
    - the output contains only finite values
    - the window is symmetric about its center
    - n=1 returns a single-element array of 1.0
    - n <= 0 raises ValueError

A separate set of tests exercises form_window (the dispatcher) and
the special-case Tukey window (which takes an alpha parameter).
"""
import numpy as np
import pytest

from grasp.processing.windows import (
    form_window,
    hann_window,
    hamming_window,
    blackman_window,
    blackman_harris_window,
    nuttall_window,
    flattop_window,
    bartlett_window,
    cosine_window,
    tukey_window,
)

# (canonical name accepted by form_window, generator function)
# flattop is included for completeness even though form_window does
# not currently expose it.
ALL_WINDOWS = [
    ("HANN",            hann_window),
    ("HAMMING",         hamming_window),
    ("BLACKMAN",        blackman_window),
    ("BLACKMAN_HARRIS", blackman_harris_window),
    ("NUTTALL",         nuttall_window),
    ("BARTLETT",        bartlett_window),
    ("COSINE",          cosine_window),
    ("FLATTOP",         flattop_window),
]

# Mix odd and even, small and large.
WINDOW_LENGTHS = [16, 17, 128, 129, 1024]


# --------------------------------------------------------------------- #
#  Universal properties — every window, every length                    #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS])
@pytest.mark.parametrize("n", WINDOW_LENGTHS)
def test_length_matches_request(name, fn, n):
    """Output array length equals the requested n."""
    w = fn(n)
    assert w.shape == (n,)


@pytest.mark.parametrize("name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS])
@pytest.mark.parametrize("n", WINDOW_LENGTHS)
def test_dtype_is_float32(name, fn, n):
    """All windows return float32 arrays per the module contract."""
    w = fn(n)
    assert w.dtype == np.float32


@pytest.mark.parametrize("name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS])
@pytest.mark.parametrize("n", WINDOW_LENGTHS)
def test_values_are_finite(name, fn, n):
    """No NaN or inf values anywhere in the window."""
    w = fn(n)
    assert np.all(np.isfinite(w)), f"{name} produced non-finite values"


@pytest.mark.parametrize("name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS])
@pytest.mark.parametrize("n", WINDOW_LENGTHS)
def test_window_is_symmetric(name, fn, n):
    """w[i] == w[-1-i] for all i (symmetric / aperiodic form).

    Tolerance scales with the window's value magnitudes; flat-top has
    values up to ~4.6 and inherits float32 round-off proportional to
    that range, while Hann/Hamming/etc. sit inside [0, 1].
    """
    w = fn(n)
    # Tolerance scaled to the window's dynamic range. Flat-top peaks
    # near 4.6 and inherits float32 round-off proportional to that.
    tol = 1e-5 * max(float(np.max(np.abs(w))), 1.0)
    np.testing.assert_allclose(w, w[::-1], atol=tol,
                               err_msg=f"{name} is not symmetric")


@pytest.mark.parametrize("name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS])
def test_n1_returns_unity(name, fn):
    """All windows return [1.0] when n=1 (documented edge case)."""
    np.testing.assert_array_equal(fn(1), np.array([1.0], dtype=np.float32))


@pytest.mark.parametrize("name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS])
@pytest.mark.parametrize("bad_n", [0, -1, -100])
def test_rejects_zero_or_negative_n(name, fn, bad_n):
    """All windows raise ValueError for non-positive n."""
    with pytest.raises(ValueError):
        fn(bad_n)


# --------------------------------------------------------------------- #
#  form_window dispatcher                                               #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, fn", ALL_WINDOWS, ids=[w[0] for w in ALL_WINDOWS]
)
def test_form_window_dispatches_correctly(name, fn):
    """form_window returns values identical to the underlying function."""
    n = 128
    np.testing.assert_allclose(form_window(n, name), fn(n))


def test_form_window_is_case_insensitive():
    """The dispatcher accepts mixed-case names."""
    n = 128
    expected = hann_window(n)
    for spelling in ("hann", "Hann", "HANN", "hAnN"):
        np.testing.assert_allclose(form_window(n, spelling), expected)


@pytest.mark.parametrize(
    "synonym, canonical",
    [
        ("HANNING",         "HANN"),
        ("BLACKMAN-HARRIS", "BLACKMAN_HARRIS"),
        ("BH",              "BLACKMAN_HARRIS"),
        ("SINE",            "COSINE"),
        ("RAISED_COSINE",   "COSINE"),
        ("NONE",            "RECTANGLE"),
    ],
)
def test_form_window_synonyms(synonym, canonical):
    """Documented synonyms produce identical output."""
    n = 64
    np.testing.assert_allclose(form_window(n, synonym),
                               form_window(n, canonical))


def test_form_window_rectangle_is_all_ones():
    """RECTANGLE / NONE returns a window of all 1.0 values."""
    n = 64
    np.testing.assert_array_equal(form_window(n, "RECTANGLE"),
                                  np.ones(n, dtype=np.float32))


def test_form_window_rejects_unknown_type():
    """An unknown window name raises ValueError."""
    with pytest.raises(ValueError, match="not recognized"):
        form_window(128, "NOTAWINDOW")


# --------------------------------------------------------------------- #
#  Tukey window — special signature                                     #
# --------------------------------------------------------------------- #

def test_tukey_alpha_zero_is_rectangle():
    """Tukey with alpha=0 reduces to a rectangular window."""
    n = 128
    np.testing.assert_allclose(tukey_window(n, alpha=0.0),
                               np.ones(n, dtype=np.float32))


def test_tukey_alpha_one_is_hann():
    """Tukey with alpha=1 reduces exactly to a Hann window."""
    n = 128
    np.testing.assert_allclose(tukey_window(n, alpha=1.0),
                               hann_window(n),
                               atol=1e-6)


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.1, -1.0, 2.0])
def test_tukey_rejects_alpha_outside_unit_interval(bad_alpha):
    """alpha must be in [0, 1]."""
    with pytest.raises(ValueError):
        tukey_window(128, alpha=bad_alpha)
