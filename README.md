# GRaSP — Generalized Radar Sounder Processor

[![CI](https://github.com/mr-perry/grasp/actions/workflows/ci.yml/badge.svg)](https://github.com/mr-perry/grasp/actions/workflows/ci.yml)

GRaSP is a Python package for processing planetary radar sounder data. It serves as the processing backend for the
[PORPASS](https://porpass.psi.edu) web application, handling jobs submitted through that interface.

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
├── src/grasp
│   ├── common/          # Shared utilities
│   ├── datuming/        # Datuming utilities
│   ├── geospatial/      # Geospatial utilities
│   ├── input/           # Data ingestion
│   ├── lrs/             # LRS-specific wrappers and class
│   ├── marsis/          # MARSIS-specific wrappers and class
│   ├── output/          # Data and image export
│   ├── postprocessing/  # Postprocessing utilities
│   ├── processing/      # Core processing pipeline
│   │   ├── emi/
│   │   ├── ionosphere/
│   │   ├── range_compression/
│   │   └── sar/
│   ├── sharad/          # SHARAD-specific wrappers and class
│   ├── simulation/      # Simulation utilities
│   └── spice/           # SPICE kernel management and utilities
├── examples/
├── tests/
├── docs/
├── pyproject.toml
├── environment.yml
└── mkdocs.yml
```

## Prerequisites

- Python >= 3.11

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

- Mars — MOLA MEGDR archive, and one or more of the MOLA/HRSC-blended
  global DEMs.
- Moon — LRO LOLA global DEM (LDEM 118 m).
- SPICE — full path to each of the local SPICE metakernel directories for
  the missions above.
### PIP

GRaSP is not yet on PyPI, but can still be installed via PIP.

#### 1. Download or clone the GRaSP Repository from GitHub
#### 2. Change directories to where you just downloaded GRaSP
#### 3. Via command line, pip install -e .

 
### Conda

#### 1. Create the conda environment

Clone the repository and create the environment from the provided `environment.yml`:

```bash
git clone https://github.com/mr-perry/grasp.git
cd grasp
conda env create -f environment.yml
```

This installs Python, all runtime dependencies, development tools, and documentation tools in a single step. The
geospatial stack (rasterio, shapely, geopandas, pyogrio) is left unpinned in `environment.yml` to let conda's solver
resolve compatible versions of their shared C libraries (`geos`, `libgdal`, `proj`).

> **Note:** Some conda-forge packages use underscores instead of hyphens in their names (e.g., `pds4_tools` and `pandas_stubs`). The `environment.yml` in this repository uses the correct conda-forge names.

### 2. Activate the environment

```bash
conda activate grasp
```

### 3. Install GRaSP in editable mode

```bash
pip install -e .
```

The `-e` flag installs the package in editable mode, so changes to the source code are reflected immediately without
reinstalling.

### 4. Verify the installation

```bash
python -c "import grasp; print(grasp.__file__)"
```

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
| tqdm | Progress bars |
| antimeridian | Antimeridian-aware geometry handling |

### Development

| Package | Description |
|---------|-------------|
| ipython | Enhanced interactive shell |
| jupyterlab | Jupyter notebook environment |
| notebook | Classic notebook interface |

### Documentation

| Package | Description |
|---------|-------------|
| mkdocs | Static site generator for docs |
| mkdocs-material | Material theme for MkDocs |
| mkdocstrings | Auto-generate docs from docstrings |
| mkdocstrings-python | Python handler for mkdocstrings |

## Documentation

GRaSP uses [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme and [mkdocstrings](https://mkdocstrings.github.io/) for API documentation generated from
Google-style docstrings.

To serve the docs locally:

```bash
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

```bash
conda deactivate
conda env remove -n grasp
conda env create -f environment.yml
conda activate grasp
pip install -e .
```

## License

This project is licensed under the [BSD-3-Clause](https://opensource.org/license/bsd-3-clause). See [LICENSE](LICENSE) for the full text.
