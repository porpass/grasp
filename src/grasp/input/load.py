# SPDX-License-Identifier: BSD-3-Clause
import csv
import h5py
import numpy as np
from pathlib import Path

from ..grasp_types import ClutterSimOutput, GraspOutput


def load_hdf5(filepath: str | Path,
              verbose: bool = False,
              ) -> dict[str, dict]:
    """Load observation state from an HDF5 file.

    Reconstructs the dict-of-dicts structure written by
    :func:`save_hdf5`. Datasets are returned as numpy arrays,
    attributes are returned as their native Python types.

    Args:
        filepath: Input HDF5 file path.
        verbose: If True, print progress messages.

    Returns:
        Dict-of-dicts representing the observation state.
    """
    filepath = Path(filepath)

    if verbose:
        print(f"Loading state from: {filepath}")

    state = {}
    with h5py.File(str(filepath), "r") as f:
        for group_name in f:
            state[group_name] = _read_group(f[group_name], verbose=verbose)

    if verbose:
        print("Load complete.")

    return state


def _read_group(grp: h5py.Group,
                verbose: bool = False,
                ) -> dict:
    """Read an HDF5 group into a dict.

    Datasets become numpy arrays. Sub-groups become nested dicts
    (recursive). Attributes become native Python types.

    Args:
        grp: HDF5 group to read.
        verbose: If True, print dataset/attribute names.

    Returns:
        Dict of key-value pairs from the group.
    """
    data = {}

    # Read attributes
    for key, value in grp.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value == "__NONE__":
            data[key] = None
        elif isinstance(value, str):
            # Try to recover booleans
            if value == "True":
                data[key] = True
            elif value == "False":
                data[key] = False
            else:
                data[key] = value
        else:
            data[key] = value
        if verbose:
            print(f"\tAttr:    {grp.name}/{key} = {data[key]}")

    # Read datasets and sub-groups
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Group):
            data[key] = _read_group(item, verbose=verbose)
        elif isinstance(item, h5py.Dataset):
            arr = item[()]
            # Recover string arrays
            if arr.dtype.kind == 'O' or arr.dtype.kind == 'S':
                str_arr = np.array([
                    s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in arr.flat
                ]).reshape(arr.shape)
                # Try to recover datetime64
                try:
                    dt_arr = str_arr.astype("datetime64[ms]")
                    data[key] = dt_arr
                except (ValueError, TypeError):
                    data[key] = str_arr
            else:
                data[key] = arr
            if verbose:
                print(f"\tDataset: {grp.name}/{key} {data[key].shape} {data[key].dtype}")

    return data


##############################################################################################################
#
# GRaSP-format output reader (.grsp + .img + .csv triplet)
#
##############################################################################################################
_REQUIRED_GRSP_KEYS = (
    "IMG_FILE", "CSV_FILE", "INSTRUMENT", "PRODUCT_ID",
    "DTYPE", "N_SAMP", "N_COL", "BYTE_ORDER", "DT", "DX", "RHO_A",
)


def _parse_grsp_descriptor(grsp_path: Path) -> dict[str, str]:
    """Parse a ``.grsp`` text descriptor into a key-value dict."""
    out: dict[str, str] = {}
    with open(grsp_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip().upper()] = value.strip()

    missing = [k for k in _REQUIRED_GRSP_KEYS if k not in out]
    if missing:
        raise ValueError(
            f"Missing required key(s) in {grsp_path}: {', '.join(missing)}"
        )
    return out


def _resolve_companion(grsp_path: Path,
                       recorded_path: Path,
                       suffix: str,
                       ) -> Path:
    """Locate an ``.img`` / ``.csv`` companion for a ``.grsp`` descriptor.

    Search order:
        1. Same directory as the ``.grsp``, matching basename.
        2. Absolute path recorded in the descriptor.

    Raises:
        FileNotFoundError: If neither candidate exists.
    """
    candidate = grsp_path.with_suffix(suffix)
    if candidate.is_file():
        return candidate
    if recorded_path.is_file():
        return recorded_path
    raise FileNotFoundError(
        f"{suffix} companion for {grsp_path} not found. Looked for:\n"
        f"  - {candidate}\n  - {recorded_path}"
    )


def _read_grasp_img(img_path: Path,
                    dtype: str,
                    n_samp: int,
                    n_col: int,
                    byte_order: str,
                    ) -> np.ndarray:
    """Load the binary radargram declared by a ``.grsp`` descriptor."""
    bo = ">" if byte_order.upper() == "BIG" else "<"

    if dtype.upper() == "FLOAT32":
        arr = np.fromfile(str(img_path), dtype=f"{bo}f4")
        expected = n_samp * n_col
        if arr.size != expected:
            raise ValueError(
                f"{img_path}: expected {expected} samples for "
                f"FLOAT32 ({n_samp} x {n_col}), got {arr.size}."
            )
        return arr.reshape((n_samp, n_col)).astype(np.float32, copy=False)

    if dtype.upper() == "COMPLEX64":
        interleaved = np.fromfile(str(img_path), dtype=f"{bo}f4")
        expected = 2 * n_samp * n_col
        if interleaved.size != expected:
            raise ValueError(
                f"{img_path}: expected {expected} interleaved samples "
                f"for COMPLEX64 ({n_samp} x {n_col} pairs), got "
                f"{interleaved.size}."
            )
        interleaved = interleaved.reshape((2 * n_samp, n_col))
        real = interleaved[0::2, :].astype(np.float32, copy=False)
        imag = interleaved[1::2, :].astype(np.float32, copy=False)
        return (real + 1j * imag).astype(np.complex64, copy=False)

    raise ValueError(f"Unsupported DTYPE: {dtype!r}")


def _read_grasp_csv(csv_path: Path, n_col: int) -> dict[str, np.ndarray]:
    """Parse a GRaSP per-trace ``.csv`` into a dict of arrays.

    Columns are looked up by header name (case-insensitive) so the
    parse is robust to future column additions or reordering.
    """
    str_cols = {"GEOMETRY_EPOCH"}
    int_cols = {"FRAME_INDEX", "TRACE_INDEX"}

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().upper() for h in header]
        rows = list(reader)

    if len(rows) != n_col:
        raise ValueError(
            f"{csv_path}: descriptor declared N_COL={n_col}, "
            f"but CSV has {len(rows)} data rows."
        )

    out: dict[str, np.ndarray] = {}
    for j, col_name in enumerate(header):
        raw = [row[j].strip() for row in rows]
        if col_name in str_cols:
            out[col_name] = np.asarray(raw, dtype=object)
        elif col_name in int_cols:
            out[col_name] = np.asarray(raw, dtype=np.int64)
        else:
            out[col_name] = np.asarray(
                [float(v) if v != "" else np.nan for v in raw],
                dtype=np.float64,
            )
    return out


def read_grasp_output(grsp_path: str | Path) -> GraspOutput | ClutterSimOutput:
    """Read a GRaSP-format output triplet (``.grsp`` + ``.img`` + ``.csv``).

    Parses the ``.grsp`` descriptor and loads the binary radargram and
    per-trace metadata it declares. Companion files (``.img`` and
    ``.csv``) are first sought next to the descriptor using its
    basename; if not found there, the absolute paths recorded inside
    the descriptor are honoured. This makes the triplet portable —
    moving the three files into a new directory works without editing.

    Two CSV schemas are supported:

    * Standard per-trace state (``EPHEMERIS_TIME, GEOMETRY_EPOCH,
      LATITUDE, …``) → returned as :class:`GraspOutput`.
    * CSIM clutter simulation (``TRACE_INDEX, LATITUDE, LONGITUDE,
      NADIR_TWTT, FRET_TWTT``) → returned as :class:`ClutterSimOutput`.

    The schema is detected from the CSV header.

    Args:
        grsp_path: Path to the ``.grsp`` descriptor.

    Returns:
        A :class:`GraspOutput` for science radargram triplets, or a
        :class:`ClutterSimOutput` for CSIM triplets.

    Raises:
        FileNotFoundError: If the descriptor or either companion
            cannot be located.
        ValueError: If the descriptor is missing required keys, the
            DTYPE is unrecognised, the binary / CSV files do not
            match the declared shape, or the CSV header matches
            neither schema.
    """
    grsp_path = Path(grsp_path)
    if not grsp_path.is_file():
        raise FileNotFoundError(f"GRaSP descriptor not found: {grsp_path}")

    meta = _parse_grsp_descriptor(grsp_path)

    img_path = _resolve_companion(grsp_path, Path(meta["IMG_FILE"]), ".img")
    csv_path = _resolve_companion(grsp_path, Path(meta["CSV_FILE"]), ".csv")

    n_samp = int(meta["N_SAMP"])
    n_col = int(meta["N_COL"])

    data = _read_grasp_img(
        img_path,
        dtype=meta["DTYPE"],
        n_samp=n_samp,
        n_col=n_col,
        byte_order=meta["BYTE_ORDER"],
    )

    cols = _read_grasp_csv(csv_path, n_col=n_col)

    if "EPHEMERIS_TIME" in cols:
        return GraspOutput(
            data=data,
            instrument=meta["INSTRUMENT"],
            product_id=meta["PRODUCT_ID"],
            dt=float(meta["DT"]),
            dx=float(meta["DX"]),
            rho_a=float(meta["RHO_A"]),
            frame_index=cols.get("FRAME_INDEX",
                                 np.arange(n_col, dtype=np.int64)),
            ephemeris_time=cols["EPHEMERIS_TIME"],
            geometry_epoch=cols["GEOMETRY_EPOCH"],
            latitude=cols["LATITUDE"],
            longitude=cols["LONGITUDE"],
            altitude=cols["ALTITUDE"],
            sc_radius=cols["SC_RADIUS"],
            el_radius=cols["EL_RADIUS"],
            solar_zenith_angle=cols["SOLAR_ZENITH_ANGLE"],
            iono_value=cols["IONO_VALUE"],
        )

    if "NADIR_TWTT" in cols:
        return ClutterSimOutput(
            data=data,
            instrument=meta["INSTRUMENT"],
            product_id=meta["PRODUCT_ID"],
            dt=float(meta["DT"]),
            trace_index=cols.get("TRACE_INDEX",
                                 np.arange(n_col, dtype=np.int64)),
            latitude=cols["LATITUDE"],
            longitude=cols["LONGITUDE"],
            nadir_twtt=cols["NADIR_TWTT"],
            fret_twtt=cols["FRET_TWTT"],
        )

    raise ValueError(
        f"{csv_path}: CSV header matches neither the standard GRaSP "
        f"schema (expected EPHEMERIS_TIME) nor the CSIM schema "
        f"(expected NADIR_TWTT). Got columns: {sorted(cols)}."
    )