# SPDX-License-Identifier: BSD-3-Clause
"""GRaSP configuration loader.

Resolves data paths using the following priority chain:

    1. Explicit ``override`` argument passed to :func:`get_data_path`
    2. Environment variable ``GRASP_<KEY>_PATH`` (e.g., ``GRASP_MOLA_PATH``)
    3. ``src/config.toml`` in a source checkout — see the note below
    4. Raises :class:`FileNotFoundError` if none of the above are set

.. note::

   The ``config.toml`` tier only works when GRaSP is installed from a
   source checkout (e.g., ``pip install -e .``). For a regular
   ``pip install`` the discovery path lands inside site-packages, where
   no user-editable ``config.toml`` exists. Installed users should use
   the ``GRASP_<KEY>_PATH`` / ``GRASP_SPICE_<KEY>_PATH`` environment
   variables (tier 2), or pass ``override=`` explicitly.

The template lives at ``src/config.example.toml``. In a source checkout,
copy it to ``src/config.toml`` (same directory) and fill in the paths
that apply to your setup. The real ``config.toml`` is gitignored and
should never be committed.

Example ``config.toml``::

    [data_paths]
    mola = "/path/to/MGS/mola/pds/"

Example usage::

    from grasp.common.config import get_data_path

    mola_root = get_data_path("mola")
    mola_root = get_data_path("mola", override="/tmp/mola")  # explicit override
"""

import os
import tomllib
from pathlib import Path
from functools import lru_cache

# Config-file discovery root: resolves to ``src/`` in a source checkout
# (grasp/common/config.py -> common/ -> grasp/ -> src/). The user's
# ``config.toml`` is expected to live at ``src/config.toml``, alongside
# the tracked ``src/config.example.toml`` template. On an installed
# (non-editable) install this resolves inside site-packages/ and the
# resulting ``config.toml`` path will not exist — the env-var and
# override tiers handle installed users; see the module docstring.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config.toml"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load and cache the TOML configuration file.

    Returns:
        Parsed configuration dictionary, or an empty dictionary
        if the configuration file does not exist.
    """
    if not _CONFIG_PATH.is_file():
        return {}
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def get_data_path(key: str,
                  override: str | Path | None = None,
                  ) -> Path:
    """Resolve a data path by key using the priority chain.

    The lookup order is:

        1. ``override`` (if provided)
        2. Environment variable ``GRASP_<KEY>_PATH``
        3. ``config.toml`` entry under ``[data_paths]``

    Args:
        key: Data path identifier (e.g., ``"mola"``). Case-insensitive
            for config file and environment variable lookup.
        override: Explicit path that takes highest priority.

    Returns:
        Resolved :class:`~pathlib.Path` to the data directory.

    Raises:
        FileNotFoundError: If no path is configured for ``key``
            through any source in the priority chain.
    """
    # 1. Explicit override
    if override is not None:
        return Path(override)

    # 2. Environment variable (e.g., GRASP_MOLA_PATH)
    env_var = f"GRASP_{key.upper()}_PATH"
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)

    # 3. Config file
    config = _load_config()
    config_val = config.get("data_paths", {}).get(key.lower())
    if config_val:
        return Path(config_val)

    raise FileNotFoundError(
        f"No data path configured for '{key}'. Set it by:\n"
        f"  1. Passing the path directly via the 'override' argument\n"
        f"  2. Setting the {env_var} environment variable\n"
        f"  3. Adding '{key.lower()} = \"/path/to/data\"' under "
        f"[data_paths] in {_CONFIG_PATH}"
    )

def get_spice_path(key: str,
                   override: str | Path | None = None,
                   ) -> Path:
    """Resolve a SPICE kernel path by key.

    Lookup order:
        1. ``override`` (if provided)
        2. Environment variable ``GRASP_SPICE_<KEY>_PATH``
        3. ``config.toml`` entry under ``[spice]``

    Args:
        key: SPICE path identifier (e.g., ``"mro_mk"``).
        override: Explicit path that takes highest priority.

    Returns:
        Resolved path to the SPICE kernel directory.

    Raises:
        FileNotFoundError: If no path is configured.
    """
    if override is not None:
        return Path(override)

    env_var = f"GRASP_SPICE_{key.upper()}_PATH"
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)

    config = _load_config()
    config_val = config.get("spice", {}).get(key.lower())
    if config_val:
        return Path(config_val)

    raise FileNotFoundError(
        f"No SPICE path configured for '{key}'. Set it by:\n"
        f"  1. Passing the path directly via the 'override' argument\n"
        f"  2. Setting the {env_var} environment variable\n"
        f"  3. Adding '{key.lower()} = \"/path/to/kernels\"' under "
        f"[spice] in {_CONFIG_PATH}"
    )