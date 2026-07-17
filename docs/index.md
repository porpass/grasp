# GRaSP

Generalized Radar Sounder Processor

## Overview

GRaSP is a Python package for processing radar sounder data from
SHARAD (MRO), MARSIS (Mars Express), and LRS (SELENE/Kaguya).

## Modules

- **external-data** — Important notes about external data requirements. 
- **core** — Top-level radar sounder base class, factory, typed
  containers, constants, and the high-level processing entrypoint.
- **common** — Shared configuration and utility helpers.
- **input** — Data input, file identification, job loading, and
  configuration handling.
- **sharad** — SHARAD instrument support.
- **marsis** — MARSIS instrument support.
- **lrs** — LRS instrument support.
- **geospatial** — DEM swath extraction and MOLA/LOLA reference data.
- **spice** — SPICE kernel handling and observation geometry.
- **datuming** — Datum reference and elevation correction utilities.
- **processing** — Range compression, EMI suppression, ionospheric
  correction, and SAR focusing.
- **postprocessing** — Multilooking and other post-SAR operations.
- **simulation** — Clutter simulation and forward modeling.
- **output** — Image rendering, radargram writing, and SEG-Y/HDF5 export.
