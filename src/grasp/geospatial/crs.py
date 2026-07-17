from pyproj import CRS
from pathlib import Path

_GCS_2000_CODES = {
    "MARS":   104971, # This is actually Mars_2000_(Sphere)
    "MOON":   104903, # GCS_Moon_2000
    "PHOBOS": 104907, # GCS_Phobos_2000
}

def gcs_2000_crs(body: str) -> CRS:
    """Return the GCS_*_2000 CRS for a planetary body.

    Args:
        body: Body name (case-insensitive). Currently supported:
            ``"MARS"``, ``"MOON"``, ``"PHOBOS"``.

    Returns:
        The ``GCS_<body>_2000`` CRS via its ESRI authority code.

    Raises:
        ValueError: If ``body`` is not a supported planetary body.
    """
    key = body.upper()
    if key not in _GCS_2000_CODES:
        raise ValueError(
            f"No GCS_2000 CRS defined for body {body!r}; "
            f"supported: {sorted(_GCS_2000_CODES)}"
        )
    return CRS.from_user_input(f"ESRI:{_GCS_2000_CODES[key]}")


def geocent_crs(body: str) -> CRS:
    """Body-fixed geocentric XYZ using the _GCS_2000 CRS spheres"""
    gcs = gcs_2000_crs(body)
    ell = gcs.ellipsoid
    a = ell.semi_major_metre
    b = ell.semi_minor_metre
    return CRS.from_proj4(f"+proj=geocent +a={a} +b={b} +no_defs")

def normalize_crs(value):
    if value is None or isinstance(value, CRS):
        return value
    if isinstance(value, Path) or (isinstance(value, str) and Path(value).is_file()):
        return CRS.from_string(Path(value).read_text())
    if isinstance(value, str):
        return CRS.from_user_input(value)
    raise TypeError()

MARS_LLE =   gcs_2000_crs("MARS")
MOON_LLE =   gcs_2000_crs("MOON")
PHOBOS_LLE = gcs_2000_crs("PHOBOS")

MARS_XYZ =   geocent_crs("MARS")
MOON_XYZ =   geocent_crs("MOON")
PHOBOS_XYZ = geocent_crs("PHOBOS")