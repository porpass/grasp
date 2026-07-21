# GRaSP — Generalized Radar Sounder Processor

[![CI](https://github.com/porpass/grasp/actions/workflows/ci.yml/badge.svg)](https://github.com/porpass/grasp/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

GRaSP is a Python library for processing planetary radar sounder data. It runs as a standalone package on your
own machine, and also serves as the processing backend for the [PORPASS](https://porpass.psi.edu) web application.

GRaSP is in active development (alpha). Interfaces may change before the first stable release.

## Project Links

- [Contributing Guidelines](CONTRIBUTING.md) — how to contribute
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md) — how to report a vulnerability
- [Changelog](CHANGELOG.md)
- [Citation](CITATION.cff) — how to cite GRaSP
- [License](LICENSE) — BSD-3-Clause
- API documentation — built with MkDocs from the [`docs/`](docs/) directory (see [Documentation](#documentation) below)

## Supported Instruments

- **SHARAD** — Shallow Radar (Mars Reconnaissance Orbiter)
  - Supported Data Types
    - EDR, RDR, US_RDR, SCS
- **MARSIS** — Mars Advanced Radar for Subsurface and Ionosphere Sounding (Mars Express)
  - Supported Data Types
    - EDR, RDR
- **LRS** — Lunar Radar Sounder (SELENE/Kaguya)
  - Supported Data Types
    - SW, SA, SAR Products

## Package Structure

```
grasp/
├── src/
│   ├── config.example.toml     # Template for local path configuration
│   └── grasp/
│       ├── __init__.py
│       ├── grasp.py            # Package-level entry points
│       ├── radar_sounder.py    # RadarSounder base class
│       ├── cli.py              # `grasp` command-line interface
│       ├── instantiator.py
│       ├── introspection.py
│       ├── constants.py
│       ├── grasp_types.py
│       ├── processing_defaults.py
│       ├── section_aliases.py
│       ├── processing_support.toml
│       ├── common/             # Shared utilities
│       ├── datuming/           # Datuming utilities
│       ├── defaults/           # Default parameter sets (TOML)
│       ├── geospatial/         # Geospatial utilities
│       ├── input/              # Data ingestion
│       ├── output/             # Data and image export
│       ├── postprocessing/     # Postprocessing utilities
│       ├── processing/         # Core processing pipeline
│       │   ├── emi/            # EMI / interference suppression
│       │   ├── ionosphere/     # Ionospheric compensation
│       │   ├── range_compression/
│       │   └── sar/            # SAR focusing (omega-k, range-Doppler, unfocused)
│       ├── schema/             # Schema definitions
│       ├── simulation/         # Simulation utilities
│       ├── spice/              # SPICE kernel management and utilities
│       ├── lrs/                # LRS-specific wrappers and class
│       ├── marsis/             # MARSIS-specific wrappers and class
│       └── sharad/             # SHARAD-specific wrappers and class
├── examples/
├── tests/
├── docs/
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── environment.yml
└── mkdocs.yml
```

## Prerequisites

- Python >= 3.11 (tested on 3.11 and 3.12)

## Note on SPICE Kernels

Some features of GRaSP require SPICE kernels to run properly. It is advised that users wishing to install GRaSP locally
download the complete SPICE kernels for each of the instruments.
- [LRS](https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-spice-6-v1.0/)
  - End of Mission. No longer updating.
- [MARSIS](https://naif.jpl.nasa.gov/pub/naif/pds/data/mex-e_m-spice-6-v2.0/)
  - Active mission. Will require updating for newer observations.
- [SHARAD](https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/)
  - Active mission. Will require updating for newer observations.

## Installation

### Note on Configuration File

GRaSP resolves data-file paths (MOLA / LOLA DEMs, SPICE metakernels, etc.)
through a three-tier lookup, checked in order:

1. An explicit `override=` argument to `get_data_path()` / `get_spice_path()`.
2. An environment variable named `GRASP_<KEY>_PATH` — e.g. `GRASP_MOLA_PATH`
   for the MOLA DEM root, `GRASP_SPICE_MRO_MK_PATH` for the MRO metakernel
   directory. **This is the recommended path for `pip install`-ed users.**
3. A `config.toml` file at `src/config.toml`, alongside the template.
   **This tier only works from an editable / source-checkout install** —
   installed users should use env vars (tier 2).

A template lives at [`src/config.example.toml`](src/config.example.toml).
Copy it to `src/config.toml` (same directory) and fill in the paths that
apply to your setup (leave keys unset for data products you don't have —
the code raises a `FileNotFoundError` with the exact env-var / config-key
to set when it needs one). To utilize the full functionality of GRaSP,
ensure each of the data products below is on your local machine and
accessible:

- Mars — MOLA MEGDR archive, or one or more of the MOLA/HRSC-blended
  global DEMs.
- Moon — LRO LOLA global DEM (LDEM 118 m).
- SPICE — full path to each of the local SPICE metakernel directories for
  the missions above.

### PIP

GRaSP is not yet on PyPI, but can still be installed with pip from a source checkout.

```bash
git clone https://github.com/porpass/grasp.git
cd grasp
pip install -e .
```

The `-e` flag installs the package in editable mode, so changes to the source code are reflected immediately without
reinstalling.

Optional dependency groups can be added with extras. Install any combination you need:

```bash
pip install -e ".[dev]"          # interactive/notebook tooling
pip install -e ".[docs]"         # documentation toolchain
pip install -e ".[test]"         # test runner
pip install -e ".[dev,docs,test]"  # everything
```

> **Note:** pip does not install the geospatial C libraries (`geos`, `libgdal`, `proj`). If you do not already
> have them available, use the conda installation below, which resolves them for you.

### Conda

#### 1. Create the conda environment

Clone the repository and create the environment from the provided `environment.yml`:

```bash
git clone https://github.com/porpass/grasp.git
cd grasp
conda env create -f environment.yml
```

This installs Python, all runtime dependencies, development tools, test tools, and documentation tools in a single
step. The geospatial stack (rasterio, shapely, geopandas, pyogrio) is left unpinned in `environment.yml` to let conda's
solver resolve compatible versions of their shared C libraries (`geos`, `libgdal`, `proj`).

> **Note:** Some conda-forge packages use underscores instead of hyphens in their names (e.g., `pds4_tools` and `pandas_stubs`). The `environment.yml` in this repository uses the correct conda-forge names.

#### 2. Activate the environment

```bash
conda activate grasp
```

#### 3. Install GRaSP in editable mode

```bash
pip install -e .
```

The `-e` flag installs the package in editable mode, so changes to the source code are reflected immediately without
reinstalling.

#### 4. Verify the installation

```bash
python -c "import grasp; print(grasp.__file__)"
grasp --help
```

Installing GRaSP also provides a `grasp` command-line entry point (see `grasp --help` for available commands).

## Testing

GRaSP uses [pytest](https://docs.pytest.org/). Install the test extra (or use the conda environment, which includes it)
and run the suite from the repository root:

```bash
pip install -e ".[test]"
pytest
```

Tests live in the [`tests/`](tests/) directory. The continuous integration workflow
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the suite on each push and pull request.

## Dependencies

### Runtime

| Package | Description |
|---------|-------------|
| numpy | Array computation |
| scipy | Scientific computing |
| matplotlib | Plotting and visualization |
| rasterio | Raster data access |
| geopandas | Geospatial DataFrames |
| pyogrio | Fast vector I/O backend for geopandas |
| shapely | Geometric operations |
| pyproj | Cartographic projections |
| spiceypy | NAIF SPICE toolkit (Python) |
| pvl | PVL (Parameter Value Language) parser |
| pds4-tools | PDS4 data product reader |
| segyio | SEG-Y seismic data I/O |
| bitstring | Binary data parsing |
| pandas | Tabular data analysis |
| h5py | HDF5 file I/O (class state save/load) |
| tqdm | Progress bars |
| antimeridian | Antimeridian-aware geometry handling |

### Development (`[dev]`)

| Package | Description |
|---------|-------------|
| ipython | Enhanced interactive shell |
| jupyterlab | Jupyter notebook environment |
| notebook | Classic notebook interface |
| pandas-stubs | Type stubs for pandas |

### Test (`[test]`)

| Package | Description |
|---------|-------------|
| pytest | Test runner |

### Documentation (`[docs]`)

| Package | Description |
|---------|-------------|
| mkdocs | Static site generator for docs |
| mkdocs-material | Material theme for MkDocs |
| mkdocs-material-extensions | Extension support for the Material theme |
| mkdocstrings | Auto-generate docs from docstrings |
| mkdocstrings-python | Python handler for mkdocstrings |
| mkdocs-autorefs | Automatic cross-references between docs pages |

## Documentation

GRaSP uses [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme and [mkdocstrings](https://mkdocstrings.github.io/) for API documentation generated from
Google-style docstrings.

Install the docs extra (or use the conda environment), then serve the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

To build a static copy:

```bash
mkdocs build
```

## Troubleshooting

### Conda solver conflicts

If `conda env create` fails with solver errors, it is usually because pinned versions of geospatial packages require incompatible versions of shared C libraries like `geos` or `libgdal`. The `environment.yml` in this repository leaves the geospatial stack unpinned to avoid this. If you still encounter issues, try updating conda itself:

```bash
conda update -n base conda
```

Then rebuild the environment from scratch:

```bash
conda deactivate
conda env remove -n grasp
conda env create -f environment.yml
conda activate grasp
pip install -e .
```

## Contributing

Contributions are welcome. Please read the [Contributing Guidelines](CONTRIBUTING.md) and
[Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or pull request.

## Citing GRaSP

If you use GRaSP in your work, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

This project is licensed under the [BSD-3-Clause](https://opensource.org/license/bsd-3-clause). See [LICENSE](LICENSE) for the full text.

---
<sub>Portions of this documentation were drafted with assistance from Claude Opus 4.8 (Anthropic), July 2026.</sub> 
