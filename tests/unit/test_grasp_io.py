# SPDX-License-Identifier: BSD-3-Clause
"""Round-trip tests for the GRaSP-format output triplet.

Exercises write_grasp_output and read_grasp_output back-to-back to
ensure the descriptor, binary radargram, and per-trace CSV survive a
write/read cycle for both real (float32) and complex (complex64)
data, in both byte orders, and that moving the triplet into a fresh
directory continues to read cleanly via the same-directory companion
lookup.
"""
import shutil

import numpy as np
import pytest

from grasp.grasp_types import ClutterResult, ClutterSimOutput, GeometryResult, GraspOutput
from grasp.input.load import read_grasp_output
from grasp.output.writers import export_csim_data, write_grasp_output


N_SAMP = 32
N_COL = 8


def _make_metadata():
    """Build a minimal per-trace metadata payload."""
    return dict(
        ephemeris_time=np.linspace(0.0, 1.0, N_COL),
        geometry_epoch=np.array([f"2025-01-01T00:00:0{i}.000" for i in range(N_COL)]),
        latitude=np.linspace(-10.0, 10.0, N_COL),
        longitude=np.linspace(100.0, 110.0, N_COL),
        altitude=np.linspace(2.5e5, 2.7e5, N_COL),
        sc_radius=np.linspace(3.5e6, 3.5e6 + 100.0, N_COL),
        el_radius=np.full(N_COL, 3.396e6),
    )


@pytest.mark.parametrize("byte_order", ["big", "little"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_grasp_output_roundtrip(tmp_path, byte_order, complex_data):
    """Round-trip a synthetic radargram through the GRaSP triplet."""
    if complex_data:
        rng = np.random.default_rng(0)
        data = (rng.standard_normal((N_SAMP, N_COL))
                + 1j * rng.standard_normal((N_SAMP, N_COL))).astype(np.complex64)
    else:
        data = np.arange(N_SAMP * N_COL, dtype=np.float32).reshape(N_SAMP, N_COL)

    md = _make_metadata()

    base = "synthetic_obs"
    write_grasp_output(
        data=data,
        output_dir=tmp_path,
        base=base,
        instrument="SHARAD",
        dt=1.0 / 26.667e6,
        dx=30.0,
        rho_a=5.0,
        ephemeris_time=md["ephemeris_time"],
        geometry_epoch=md["geometry_epoch"],
        latitude=md["latitude"],
        longitude=md["longitude"],
        altitude=md["altitude"],
        sc_radius=md["sc_radius"],
        el_radius=md["el_radius"],
        byte_order=byte_order,
    )

    grsp_path = tmp_path / f"{base}.grsp"
    out = read_grasp_output(grsp_path)

    # Binary payload
    assert out.data.shape == (N_SAMP, N_COL)
    if complex_data:
        assert out.data.dtype == np.complex64
        np.testing.assert_allclose(out.data, data)
    else:
        assert out.data.dtype == np.float32
        np.testing.assert_array_equal(out.data, data)

    # Layout metadata
    assert out.instrument == "SHARAD"
    assert out.dx == pytest.approx(30.0)
    assert out.rho_a == pytest.approx(5.0)
    assert out.dt == pytest.approx(1.0 / 26.667e6)

    # Per-trace metadata. The writer formats numeric CSV columns with
    # 3 decimal places (see write_grasp_csv), so the tolerance has to
    # match that quantisation rather than full float precision.
    csv_atol = 5e-4
    np.testing.assert_allclose(out.ephemeris_time, md["ephemeris_time"], atol=csv_atol)
    np.testing.assert_allclose(out.latitude, md["latitude"], atol=csv_atol)
    np.testing.assert_allclose(out.longitude, md["longitude"], atol=csv_atol)
    np.testing.assert_allclose(out.altitude, md["altitude"], atol=csv_atol)
    np.testing.assert_allclose(out.sc_radius, md["sc_radius"], atol=csv_atol)
    np.testing.assert_allclose(out.el_radius, md["el_radius"], atol=csv_atol)
    # Epoch strings round-trip exactly.
    assert list(out.geometry_epoch) == list(md["geometry_epoch"])
    # Optional columns were not supplied → writer NaN-fills, reader preserves NaN.
    assert np.all(np.isnan(out.solar_zenith_angle))
    assert np.all(np.isnan(out.iono_value))
    # frame_index is just 0..N_COL-1
    np.testing.assert_array_equal(out.frame_index, np.arange(N_COL))


def test_grasp_output_portable_when_moved(tmp_path):
    """Moving the triplet into a new directory still reads correctly via
    the same-directory companion-lookup fallback (the descriptor's
    absolute paths become stale once moved)."""
    md = _make_metadata()
    data = np.arange(N_SAMP * N_COL, dtype=np.float32).reshape(N_SAMP, N_COL)

    src = tmp_path / "src"
    src.mkdir()
    write_grasp_output(
        data=data,
        output_dir=src,
        base="obs",
        instrument="SHARAD",
        dt=1.0 / 26.667e6,
        dx=30.0,
        rho_a=5.0,
        **md,
    )

    dst = tmp_path / "dst"
    dst.mkdir()
    for suffix in (".grsp", ".img", ".csv"):
        shutil.copy(src / f"obs{suffix}", dst / f"obs{suffix}")

    out = read_grasp_output(dst / "obs.grsp")
    np.testing.assert_array_equal(out.data, data)


def test_grasp_output_missing_companion_raises(tmp_path):
    """If the .img companion is deleted, the reader raises FileNotFoundError
    with a message naming both candidate paths it tried."""
    md = _make_metadata()
    data = np.arange(N_SAMP * N_COL, dtype=np.float32).reshape(N_SAMP, N_COL)

    write_grasp_output(
        data=data,
        output_dir=tmp_path,
        base="obs",
        instrument="SHARAD",
        dt=1.0 / 26.667e6,
        dx=30.0,
        rho_a=5.0,
        **md,
    )

    (tmp_path / "obs.img").unlink()

    with pytest.raises(FileNotFoundError, match=r"\.img companion"):
        read_grasp_output(tmp_path / "obs.grsp")


def test_grasp_output_dispatcher_routes_grsp(tmp_path):
    """grasp.read() recognises .grsp and routes to read_grasp_output."""
    import grasp

    md = _make_metadata()
    data = np.arange(N_SAMP * N_COL, dtype=np.float32).reshape(N_SAMP, N_COL)

    write_grasp_output(
        data=data,
        output_dir=tmp_path,
        base="obs",
        instrument="SHARAD",
        dt=1.0 / 26.667e6,
        dx=30.0,
        rho_a=5.0,
        **md,
    )

    out = grasp.read(tmp_path / "obs.grsp")
    assert isinstance(out, type(read_grasp_output(tmp_path / "obs.grsp")))
    np.testing.assert_array_equal(out.data, data)


def _make_clutter_result():
    """Synthesise a minimal ClutterResult for round-trip testing."""
    rng = np.random.default_rng(0)
    cluttergram = rng.standard_normal((N_SAMP, N_COL)).astype(np.float32)
    echomap = rng.standard_normal((5, N_COL)).astype(np.float32)
    nadir_twtt = np.linspace(1e-4, 2e-4, N_COL)
    fret_twtt = np.linspace(0.9e-4, 1.9e-4, N_COL)
    fret_twtt[-1] = np.nan  # exercise NaN handling
    nadir_ct_idx = np.full(N_COL, 2, dtype=np.intp)
    fret_ct_idx = np.full(N_COL, 1, dtype=np.intp)
    return ClutterResult(
        cluttergram=cluttergram,
        echomap=echomap,
        nadir_twtt=nadir_twtt,
        fret_twtt=fret_twtt,
        nadir_ct_idx=nadir_ct_idx,
        fret_ct_idx=fret_ct_idx,
    )


def _make_geometry_result():
    """Synthesise a minimal GeometryResult for round-trip testing."""
    return GeometryResult(
        epoch=np.array([f"2025-01-01T00:00:0{i}.000" for i in range(N_COL)],
                       dtype="datetime64[ms]"),
        longitude=np.linspace(100.0, 110.0, N_COL),
        latitude=np.linspace(-10.0, 10.0, N_COL),
        altitude=np.linspace(2.5e5, 2.7e5, N_COL),
    )


def test_csim_output_roundtrip(tmp_path):
    """CSIM .grsp triplets round-trip through read_grasp_output as
    ClutterSimOutput (distinct from the science-data GraspOutput)."""
    clutter = _make_clutter_result()
    geometry = _make_geometry_result()
    base = "synthetic_obs_csim"

    export_csim_data(
        clutter,
        geometry,
        out_dir=tmp_path,
        base=base,
        instrument="SHARAD",
        bin_size=1.0 / 26.667e6,
    )

    # Filename does NOT double-suffix _csim_csim.
    grsp_path = tmp_path / f"{base}.grsp"
    assert grsp_path.is_file()
    assert not (tmp_path / f"{base}_csim.grsp").is_file()

    out = read_grasp_output(grsp_path)
    assert isinstance(out, ClutterSimOutput)
    assert not isinstance(out, GraspOutput)

    # Binary payload (cluttergram is float32, no precision loss).
    assert out.data.shape == (N_SAMP, N_COL)
    assert out.data.dtype == np.float32
    np.testing.assert_array_equal(out.data, clutter.cluttergram)

    # CSV is .6f for lat/lon, .12e for twtt — tolerances reflect that.
    np.testing.assert_allclose(out.latitude, geometry.latitude, atol=5e-7)
    np.testing.assert_allclose(out.longitude, geometry.longitude, atol=5e-7)
    np.testing.assert_allclose(out.nadir_twtt, clutter.nadir_twtt, rtol=1e-10)
    np.testing.assert_allclose(out.fret_twtt[:-1], clutter.fret_twtt[:-1], rtol=1e-10)
    assert np.isnan(out.fret_twtt[-1])

    # Trace index is 0..N_COL-1 and the descriptor product_id matches.
    # (The writer uppercases PRODUCT_ID.)
    np.testing.assert_array_equal(out.trace_index, np.arange(N_COL))
    assert out.product_id == base.upper()
    assert out.instrument == "SHARAD"
    assert out.dt == pytest.approx(1.0 / 26.667e6)


def test_csim_dispatcher_routes_grsp(tmp_path):
    """grasp.read() routes a CSIM .grsp to a ClutterSimOutput."""
    import grasp

    clutter = _make_clutter_result()
    geometry = _make_geometry_result()
    base = "obs_csim"

    export_csim_data(
        clutter,
        geometry,
        out_dir=tmp_path,
        base=base,
        instrument="SHARAD",
        bin_size=1.0 / 26.667e6,
    )

    out = grasp.read(tmp_path / f"{base}.grsp")
    assert isinstance(out, ClutterSimOutput)
