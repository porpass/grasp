# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import replace
import importlib
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import re
import spiceypy as sp
from typing import cast, Iterable

from ..grasp_types import Spacecraft, SpiceParams
from ..common.config import get_spice_path

PLATFORM_MK_KEY = {
    "MRO": "mro_mk",
    "MEX": "mex_mk",
    "SELENE": "selene_mk",
}

INSTRUMENT_PLATFORM = {
    "SHARAD": "MRO",
    "MARSIS": "MEX",
    "LRS": "SELENE",
}


# MK filename patterns per instrument
_MK_PATTERNS = {
    "SHARAD": re.compile(r"mro_(\d{4})_v(\d+)\.tm", re.IGNORECASE),
    "MARSIS": re.compile(r"mex_v(\d+)\.tm", re.IGNORECASE),
    "LRS": re.compile(r"sel_v(\d+)\.tm", re.IGNORECASE),
}


def print_mk_paths(instrument: str | None = None) -> None:
    """Print the configured SPICE metakernel directory per instrument.

    Looks up each instrument's metakernel directory via
    :func:`grasp.common.config.get_spice_path` and prints it. If the
    path has not been configured (no config entry, no environment
    variable), prints ``(not configured)`` instead of raising — this
    is intended as a diagnostic that should never crash.

    Args:
        instrument: Optional instrument name (``"SHARAD"``,
            ``"MARSIS"``, or ``"LRS"``, case-insensitive). If
            ``None`` (default), prints the configured path for every
            supported instrument.

    Raises:
        ValueError: If ``instrument`` is provided but not one of the
            three supported names.
    """
    if instrument is None:
        targets = list(INSTRUMENT_PLATFORM)
    else:
        inst = instrument.upper()
        if inst not in INSTRUMENT_PLATFORM:
            raise ValueError(
                f"Unknown instrument {instrument!r}; expected one of "
                f"{', '.join(INSTRUMENT_PLATFORM)}"
            )
        targets = [inst]

    for inst in targets:
        platform = INSTRUMENT_PLATFORM[inst]
        key = PLATFORM_MK_KEY[platform]
        try:
            path = get_spice_path(key)
            print(f"{inst:<7} ({platform:<6}): {path}")
        except FileNotFoundError:
            print(f"{inst:<7} ({platform:<6}): (not configured)")


def find_mk_files(instrument: str,
                  years: list[int] | None = None,
                  mk_path: str | Path | None = None,
                  ) -> list[Path]:
    """Find the appropriate SPICE metakernel file(s).

    For SHARAD, returns the highest-version MK file for each requested
    year. For MARSIS and LRS, returns the single highest-version MK
    file (years argument is ignored).

    Args:
        instrument: Instrument name ("SHARAD", "MARSIS", or "LRS").
        years: List of observation years. Required for SHARAD,
            ignored for MARSIS and LRS.
        mk_path: Path to the directory containing MK files. If
            ``None`` (default), falls back to the configured
            metakernel directory for ``instrument`` resolved via
            :func:`grasp.common.config.get_spice_path`.

    Returns:
        List of Paths to the MK file(s) to load.

    Raises:
        ValueError: If instrument is unknown or no matching MK file
            is found.
        FileNotFoundError: If the resolved MK directory does not
            exist, or if ``mk_path`` is ``None`` and no metakernel
            path has been configured for the instrument.
    """
    inst = instrument.upper()
    pattern = _MK_PATTERNS.get(inst)
    if pattern is None:
        raise ValueError(f"Unknown instrument: {instrument}")

    if mk_path is None:
        platform = INSTRUMENT_PLATFORM[inst]
        mk_path = get_spice_path(PLATFORM_MK_KEY[platform])

    mk_path = Path(mk_path)
    if not mk_path.is_dir():
        raise FileNotFoundError(f"MK directory not found: {mk_path}")

    if inst == "SHARAD":
        if years is None or len(years) == 0:
            raise ValueError("years is required for SHARAD MK file selection")

        mk_files = []
        for year in years:
            best_version = -1
            best_file = None
            for f in mk_path.iterdir():
                m = pattern.match(f.name)
                if m and int(m.group(1)) == year:
                    version = int(m.group(2))
                    if version > best_version:
                        best_version = version
                        best_file = f
            if best_file is None:
                raise ValueError(
                    f"No MK file found for SHARAD year {year} in {mk_path}")
            mk_files.append(best_file)

        return mk_files

    else:
        # MARSIS or LRS — find highest version regardless of year
        best_version = -1
        best_file = None
        for f in mk_path.iterdir():
            m = pattern.match(f.name)
            if m:
                version = int(m.group(1))
                if version > best_version:
                    best_version = version
                    best_file = f

        if best_file is None:
            raise ValueError(
                f"No MK file found for {inst} in {mk_path}")

        return [best_file]


def furnish(path: str | Path | Iterable[str | Path]):
    """Load one or more SPICE kernels into the kernel pool.

    Wraps spiceypy.furnsh to accept a single path or an iterable of
    paths. ``Path`` objects are coerced to strings before being
    handed to spiceypy, which expects bytes-encodable values.

    Args:
        path: Path(s) to the SPICE kernel(s) or meta-kernel(s)
            to load.
    """
    if isinstance(path, (str, Path)):
        sp.furnsh(str(path))
    else:
        sp.furnsh([str(p) for p in path])


def unload(path: str | Path | Iterable[str | Path]):
    """Unload one or more SPICE kernels from the kernel pool.

    Wraps spiceypy.unload to accept a single path or an iterable of
    paths. ``Path`` objects are coerced to strings before being
    handed to spiceypy, which expects bytes-encodable values.

    Args:
        path: Path(s) to the SPICE kernel(s) or meta-kernel(s)
            to unload.
    """
    if isinstance(path, (str, Path)):
        sp.unload(str(path))
    else:
        sp.unload([str(p) for p in path])


def grab_spice_params(spacecraft: str) -> SpiceParams:
    """Returns SPICE parameters for a given spacecraft.

    Loads the instrument-specific SPICE constants and injects the
    metakernel path from the GRaSP configuration.

    Args:
        spacecraft: Spacecraft identifier. Must be one of
            ``"SELENE"``, ``"MEX"``, or ``"MRO"``.

    Returns:
        SPICE parameters for the spacecraft.

    Raises:
        ValueError: If ``spacecraft`` is not a supported key.
    """
    key = _cast_spacecraft(spacecraft)
    if key == "SELENE":
        mod = importlib.import_module("grasp.lrs.spice")
    elif key == "MEX":
        mod = importlib.import_module("grasp.marsis.spice")
    elif key == "MRO":
        mod = importlib.import_module("grasp.sharad.spice")
    else:
        raise ValueError(f"Unknown SPACECRAFT {key}")

    params = getattr(mod, "SPICE_PARAMS")
    mk_path = str(get_spice_path(PLATFORM_MK_KEY[key]))

    return replace(params, mk_path=mk_path)


def sc_clock_to_et_str(sclk: str, instr: int) -> float:
    """
    Convert a spacecraft clock string to ephemeris time (ET).

    This function converts a spacecraft clock value expressed as a
    SPICE-formatted clock string into ephemeris time (seconds past J2000)
    using ``spiceypy.scs2e``.

    The appropriate spacecraft clock (SCLK) kernel for ``instr`` must
    already be loaded into the SPICE kernel pool.

    Args:
        sclk:
            Spacecraft clock string (e.g., ``"1/123456789"``) in a format
            accepted by SPICE.
        instr:
            NAIF spacecraft or instrument ID associated with the clock.

    Returns:
        Ephemeris time (ET) in seconds past J2000.

    Raises:
        spiceypy.utils.exceptions.SpiceyError:
            If the spacecraft clock string cannot be parsed or required
            kernels are not loaded.
    """
    return sp.scs2e(instr, sclk)



def sc_clock_to_et_ticks(ticks: float | int, instr: int) -> float:
    """
    Convert spacecraft clock ticks to ephemeris time (ET).

    This function converts a numeric spacecraft clock tick value into
    ephemeris time (seconds past J2000) using ``spiceypy.sct2e``.

    The appropriate spacecraft clock (SCLK) kernel for ``instr`` must
    already be loaded into the SPICE kernel pool.

    Args:
        ticks:
            Spacecraft clock ticks as a numeric value.
        instr:
            NAIF spacecraft or instrument ID associated with the clock.

    Returns:
        Ephemeris time (ET) in seconds past J2000.

    Raises:
        spiceypy.utils.exceptions.SpiceyError:
            If the tick value cannot be converted or required kernels
            are not loaded.
    """
    return sp.sct2e(instr, ticks)



def sc_clock_to_et(scl: str | float | int,
                   instr: int,
                   ) -> float | NDArray:
    """
    Convert a spacecraft clock value to ephemeris time (ET).

    This function supports both spacecraft clock strings (e.g. ``"1/123456789"``)
    and numeric spacecraft clock ticks.

    Args:
        scl:
            Spacecraft clock value. May be:
            - A clock string accepted by ``sp.scs2e``.
            - A numeric tick count accepted by ``sp.sct2e``.
        instr:
            NAIF instrument or spacecraft ID. May be an int or a value
            coercible to int.

    Returns:
        Ephemeris time (ET) in seconds past J2000.

    Raises:
        TypeError:
            If ``scl`` is not a string or a float
        ValueError:
            If SPICE fails to interpret the clock value.
    """
    if isinstance(scl, str):
        return sc_clock_to_et_str(scl, instr)
    elif isinstance(scl, (int, float)):
        return sc_clock_to_et_ticks(float(scl), instr)
    else:
        raise TypeError(f"SCLK must be a string or numeric: {scl}")



def et2utc(et: float | list[float] | NDArray) -> str | NDArray:
    """Format ephemeris time(s) as fixed-width timestamp string(s).

    Wraps spiceypy.timout to format ET values as calendar date strings.

    Args:
        et: Ephemeris time(s) in seconds past J2000, as a float,
            list of floats, or numpy array.

    Returns:
        Timestamp string(s) formatted as ``YYYY-MM-DD HR:MN:SC.###``.
        Returns a string for scalar input, or a numpy array of strings
        for iterable input.

    Raises:
        SpiceyError: If a SPICE kernel with time data has not been
            loaded.
    """
    if isinstance(et, list):
        et = np.asarray(et)
    return sp.timout(et, "YYYY-MM-DD HR:MN:SC.###", 23)

def utc2et(utc: str | list[str] | NDArray) -> float | NDArray:
    """Convert UTC time(s) to SPICE ephemeris time (ET).

    Wraps spiceypy.utc2et to accept scalar or array-like UTC times.
    Numpy datetime64 values are converted to ISO format strings before
    being passed to SPICE.

    Args:
        utc: UTC time(s) as a string, list of strings, or numpy
            datetime64 array.

    Returns:
        Ephemeris time(s) in seconds past J2000. Returns a float for
        scalar input, or a numpy array for array-like input.

    Raises:
        SpiceyError: If a SPICE kernel with leap second data has not
            been loaded.
    """
    if isinstance(utc, list):
        utc = np.asarray(utc)
    if isinstance(utc, np.ndarray):
        if np.issubdtype(utc.dtype, np.datetime64):
            utc = np.datetime_as_string(utc)
        return np.array([sp.utc2et(t) for t in utc.flat]).reshape(utc.shape)
    return sp.utc2et(utc)

def _cast_spacecraft(spacecraft: str) -> Spacecraft:
    """Validates and converts an arbitrary string to a supported spacecraft key.

    Args:
        spacecraft: Spacecraft string.

    Returns:
        A validated spacecraft key ("SELENE", "MEX", or "MRO").

    Raises:
        ValueError: If `spacecraft` is not one of the supported keys.
    """
    if spacecraft in ("SELENE", "MEX", "MRO"):
        return cast(Spacecraft, spacecraft)
    raise ValueError(f"Invalid SPICE spacecraft: {spacecraft}")

def get_radii(body: str) -> tuple[int, NDArray[np.floating]]:
    """Retrieve the triaxial radii of a body from the SPICE kernel pool.

    Args:
        body: NAIF body name (e.g., "MARS", "MOON", "EUROPA").

    Returns:
        A tuple ``(dim, radii)`` where ``dim`` is the number of values
        returned and ``radii`` is a 1D array of length 3 containing
        the equatorial A, equatorial B, and polar C radii in kilometres.
    """
    return sp.bodvrd(body, "RADII", 3)

