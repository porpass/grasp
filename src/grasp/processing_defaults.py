import tomllib
from pathlib import Path
from typing import Any

_DEFAULTS_DIR = Path(__file__).parent / "defaults"
_cache: dict[str, dict[str, dict[str, Any]]] = {}


def merge_defaults(base: dict, override: dict) -> dict:
    """Field-by-field per-stage merge. Override wins, base fills gaps."""
    out = {}
    for k in set(base) | set(override):
        if k in base and k in override:
            out[k] = {**base[k], **override[k]}
        else:
            out[k] = dict(base.get(k, override.get(k, {})))
    return out


def _load(instrument: str) -> dict[str, dict[str, Any]]:
    if instrument in _cache:
        return _cache[instrument]
    path = _DEFAULTS_DIR / f"{instrument.lower()}.toml"
    if not path.is_file():
        raise ValueError(f"No defaults file for instrument {instrument!r} at {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    base = raw.get("base", {})
    table = {
        pt_key: merge_defaults(base, raw[pt_key])  # was: pt_key.replace("_", " ")
        for pt_key in raw
        if pt_key != "base"
    }
    _cache[instrument] = table
    return table


def get_defaults(instrument: str, metadata: Any = None) -> dict[str, dict[str, Any]]:
    """Return per-stage defaults for an instrument and product type."""
    table = _load(instrument.upper())
    product_type = metadata.product_type if metadata is not None else "EDR"
    if product_type not in table:
        raise ValueError(
            f"No {instrument} defaults for product_type={product_type!r}; "
            f"expected one of {list(table)}."
        )
    return table[product_type]


def resolve_instrument_defaults(params, instrument: str, metadata=None):
    """Fill in any None fields in params with instrument defaults."""
    table = get_defaults(instrument, metadata)
    for stage_name, stage_defaults in table.items():
        if stage_name == "general":
            for field_name, default_value in stage_defaults.items():
                if getattr(params, field_name, None) is None:
                    setattr(params, field_name, default_value)
            continue
        stage = getattr(params, stage_name, None)
        if stage is None:
            continue
        for field_name, default_value in stage_defaults.items():
            if getattr(stage, field_name, None) is None:
                setattr(stage, field_name, default_value)
    return params