# SPDX-License-Identifier: BSD-3-Clause
import errno
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from pds4_tools import pds4_read
from pvl import load

PVLScalar: TypeAlias = None | bool | int | float | str
PVLJSON: TypeAlias = PVLScalar | list["PVLJSON"] | dict[str, "PVLJSON"]


def _repair_seconds_overflow(date_string: str, fmt: str) -> datetime | None:
    """Attempt to parse ``date_string`` after clamping an overflowed seconds
    field (value ``60``) to ``0`` and carrying one minute forward.

    Args:
        date_string: The raw datetime string that failed normal parsing.
        fmt: The ``strptime`` format string originally used.

    Returns:
        A corrected ``datetime`` object, or ``None`` if the string does not
        contain a seconds-60 overflow or if the repaired string still fails
        to parse.
    """
    # Replace the first occurrence of ':60' (with optional fractional part)
    # that looks like a seconds field.  The regex targets ':60' followed by
    # either a non-digit boundary, a decimal point, or end-of-string so that
    # minute values of '60' are not incorrectly matched (minutes appear after
    # the hours colon and before the seconds colon, so the pattern is
    # unambiguous in ISO-8601-like strings).
    import re
    pattern = r':60(\.\d+)?(?=\D|$)'
    if not re.search(pattern, date_string):
        return None

    repaired_string = re.sub(pattern, r':00\1', date_string, count=1)
    try:
        return datetime.strptime(repaired_string, fmt) + timedelta(minutes=1)
    except ValueError:
        return None


def decode_datetime(items: str | Sequence[bytes], fmt: str,) -> np.datetime64:
    """
    Decode a datetime string or sequence of byte characters into a NumPy
    ``datetime64[ms]`` value.

    If parsing fails after all repair attempts, returns ``NaT`` at millisecond
    resolution.

    Some PDS label files contain timestamps where the seconds field is ``60``
    (e.g. ``2007-12-21T05:04:60.000``).  These are treated as a carry event:
    the seconds are clamped to ``0`` and one minute is added via
    ``datetime.timedelta`` arithmetic before retrying the parse.  This handles
    day and month rollovers correctly (e.g. ``23:59:60`` → ``00:00:00`` of the
    following day).

    Args:
        items: Datetime as a string, or a sequence of UTF-8 byte characters.
        fmt: ``datetime.strptime``-compatible format string.

    Returns:
        Parsed datetime as ``numpy.datetime64`` with millisecond resolution,
        or ``NaT`` if parsing fails.

    Raises:
        TypeError: If ``items`` is not a string or a sequence of bytes.
    """
    if isinstance(items, str):
        date_string = items
    elif isinstance(items, Sequence):
        try:
            date_string = "".join(x.decode("utf-8") for x in items).strip()
        except Exception as exc:
            raise TypeError(
                "items must be a string or a sequence of UTF-8 byte characters"
            ) from exc
    else:
        raise TypeError(f"items must be a string or a sequence of bytes, got {type(items)}")

    try:
        return np.datetime64(datetime.strptime(date_string, fmt), "ms")
    except ValueError:
        # Attempt to repair a seconds value of 60, which datetime rejects but
        # occasionally appears in PDS label files as a carry artefact.
        repaired = _repair_seconds_overflow(date_string, fmt)
        if repaired is not None:
            return np.datetime64(repaired, "ms")
        return np.datetime64("NaT", "ms")


def convert_value(text: str) -> int | float | str | None:
    """
    Convert a string value to a scalar using best-effort rules.

    Rules:
      * ``(null)``, ``null``, ``NULL`` -> ``None``
      * else try ``int``, then ``float``
      * else return the original string

    Args:
        text: Input string value to convert.

    Returns:
        Converted scalar value.
    """
    text = text.strip()
    if text in ("(null)", "null", "NULL"):
        return None

    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            continue

    return text


def convert_pvl_to_dict(obj: Any) -> PVLJSON:
    """
    Recursively convert PVL-parsed objects into pure Python containers.

    Args:
        obj: PVL-parsed object or value.

    Returns:
        A JSON-like nested structure of dict/list/scalars.
    """
    if isinstance(obj, Mapping):
        return {str(k): convert_pvl_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [convert_pvl_to_dict(v) for v in obj]

    return cast(PVLJSON, obj)


def parse_pds_lbl(lbl_path: str | Path) -> dict[str, PVLJSON]:
    """
    Parse a PDS3 ``.LBL`` file into a pure-Python nested dictionary.

    Args:
        lbl_path: Path to the PDS3 label file.

    Returns:
        Parsed label as a nested dict/list/scalar structure.

    Raises:
        TypeError: If the PVL parse result cannot be converted to a dict.
    """
    lbl = load(lbl_path)
    result = convert_pvl_to_dict(lbl)

    if not isinstance(result, dict):
        raise TypeError(f"Expected PVL label to convert to dict, got {type(result)}")

    return result


def parse_pds_xml(xml_path: str | Path) -> dict[str, Any]:
    """
    Parse a PDS4 XML label into a Python dictionary.

    Args:
        xml_path: Path to the PDS4 XML label.

    Returns:
        Dictionary representation of the PDS4 label.

    Raises:
        TypeError: If the label conversion does not return a dict.
    """
    lbl = pds4_read(xml_path)
    result = lbl.label.to_dict()

    if not isinstance(result, dict):
        raise TypeError(f"Expected PDS4 label to convert to dict, got {type(result)}")

    return result


def check_file(path: str | Path) -> None:
    """
    Validate that a path exists, is a regular file, and is readable.

    Args:
        path: Path to validate.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory.
        OSError: If the path exists but is not a regular file.
        PermissionError: If the file exists but is not readable.
    """
    path_str = os.fspath(path)

    if not os.path.exists(path_str):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_str)

    if os.path.isdir(path_str):
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path_str)

    if not os.path.isfile(path_str):
        raise OSError(errno.EINVAL, "Path exists but is not a regular file", path_str)

    if not os.access(path_str, os.R_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path_str)


def check_dir(path: str | Path) -> None:
    """
    Validate that a path exists, is a directory, and is readable.

    Args:
        path: Directory path to validate.

    Raises:
        NotADirectoryError: If the path does not exist or is not a directory.
        PermissionError: If the directory exists but is not readable.
    """
    path_str = os.fspath(path)

    if not os.path.isdir(path_str):
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path_str)

    if not os.access(path_str, os.R_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path_str)


def calc_nrecs(path: str | Path,
               rec_len: int) -> int:
    """
    Calculate the number of fixed-length records in a binary file.

    This function validates that the given file exists and that its size
    is evenly divisible by the specified record length. It then returns
    the total number of records in the file.

    Args:
        path: Path to the input file.
        rec_len: Record length in bytes. Must be a positive integer.

    Returns:
        The number of records in the file.

    Raises:
        FileNotFoundError: If the file does not exist or is not a regular file.
        PermissionError: If the file exists but is not readable.
        ValueError: If ``rec_len`` is non-positive or the file size is not
            evenly divisible by ``rec_len``.
    """
    # Reuse your existing validation logic
    check_file(path)

    path_str = os.fspath(path)
    file_size = os.path.getsize(path_str)

    if rec_len <= 0:
        raise ValueError(f"rec_len must be positive, got {rec_len}")

    if file_size % rec_len != 0:
        raise ValueError(
            f"File size ({file_size} bytes) is not divisible by rec_len "
            f"({rec_len} bytes): {path_str}"
        )

    return file_size // rec_len


def check_frame_length(frame: int,
                        column: bytes,
                        rec_len: int,
                        ) -> None:
    """Validate that a record buffer has the expected length.

    Args:
        frame: Zero-based frame index, used in the error message.
        column: Raw record bytes read from the file.
        rec_len: Expected record length in bytes.

    Raises:
        EOFError: If ``len(column) != rec_len``.
    """
    if len(column) != rec_len:
        raise EOFError(
            f"Incomplete record at frame {frame}: expected {rec_len} bytes, got {len(column)} bytes"
        )


def decimate_dict(data_dict: dict[str, Any],
                  factor: int | None = None,
                  indices: NDArray[np.intp] | None = None,
                  skip_keys: list[str] | None = None,
                  ) -> dict[str, Any]:
    """Decimate a data dictionary by selecting specific traces.

    Supports two modes:
        Factor-based: For each group of ``factor`` traces, selects
            the center trace.
        Index-based: Selects traces at the given indices directly.

    Handles 1D arrays, 2D arrays (trace axis last), and nested
    dicts. Keys in skip_keys are set to None in the output.

    Args:
        data_dict: Dictionary of arrays indexed by trace number.
        factor: Decimation factor (group size). Mutually exclusive
            with indices.
        indices: Array of trace indices to select. Mutually
            exclusive with factor.
        skip_keys: Keys to skip (set to None in output).

    Returns:
        New dictionary with decimated arrays.

    Raises:
        ValueError: If both or neither of factor and indices are
            provided.
    """
    if (factor is None) == (indices is None):
        raise ValueError("Exactly one of 'factor' or 'indices' "
                         "must be provided.")

    if skip_keys is None:
        skip_keys = []

    # Build index array from factor if needed
    if factor is not None:
        n_traces = None
        for val in data_dict.values():
            if isinstance(val, np.ndarray):
                n_traces = val.shape[-1]
                break
        if n_traces is None:
            return dict(data_dict)
        n_out = n_traces // factor
        indices = np.arange(n_out) * factor + factor // 2

    out = {}
    for key, val in data_dict.items():
        if key in skip_keys:
            out[key] = val
            continue

        if isinstance(val, dict):
            out[key] = decimate_dict(val, indices=indices,
                                     skip_keys=skip_keys)
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                out[key] = val[indices]
            elif val.ndim == 2:
                out[key] = val[:, indices]
            else:
                out[key] = val
        else:
            out[key] = val

    return out

def decimate_dataclass(dc: Any,
                       factor: int | None = None,
                       indices: NDArray[np.intp] | None = None,
                       ) -> Any:
    """Decimate a dataclass by selecting specific traces.

    Supports two modes:
        Factor-based: For each group of ``factor`` traces, selects
            the center trace.
        Index-based: Selects traces at the given indices directly.

    Handles 1D and 2D ndarray fields (trace axis last). Non-array
    fields are passed through unchanged.

    Args:
        dc: Frozen dataclass instance with ndarray fields.
        factor: Decimation factor (group size). Mutually exclusive
            with indices.
        indices: Array of trace indices to select. Mutually
            exclusive with factor.

    Returns:
        New dataclass instance with decimated arrays.

    Raises:
        ValueError: If both or neither of factor and indices are
            provided.
    """
    from dataclasses import fields, replace

    if (factor is None) == (indices is None):
        raise ValueError("Exactly one of 'factor' or 'indices' "
                         "must be provided.")

    # Build index array from factor if needed
    if factor is not None:
        n_traces = None
        for f in fields(dc):
            val = getattr(dc, f.name)
            if isinstance(val, np.ndarray):
                n_traces = val.shape[-1]
                break
        if n_traces is None:
            return dc
        n_out = n_traces // factor
        indices = np.arange(n_out) * factor + factor // 2

    updates = {}
    for f in fields(dc):
        val = getattr(dc, f.name)
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                updates[f.name] = val[indices]
            elif val.ndim == 2:
                updates[f.name] = val[:, indices]
            else:
                updates[f.name] = val
        # Non-array fields are left unchanged (not included in updates)

    return replace(dc, **updates)

def determine_observation_years(geometry_epoch: NDArray) -> list[int]:
    """Determine unique observation years from geometry epochs.

    Args:
        geometry_epoch: Array of numpy datetime64 values.

    Returns:
        Sorted list of unique years spanned by the observation.
    """
    valid = geometry_epoch[~np.isnat(geometry_epoch)]
    years = np.unique(valid.astype('datetime64[Y]').astype(int) + 1970)
    return sorted(years.tolist())

def assert_gt_0(x):
    if x <= 0.0:
        raise ValueError('Value must be positive')

def assert_str(x):
    assert isinstance(x, str)