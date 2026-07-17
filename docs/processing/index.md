# Processing

GRaSP organizes signal processing into modular stages, each backed by a
subpackage:

- **Helpers** — shared windows and utilities used across every stage.
  See [Helpers](helpers.md).
- **Range Compression** — chirp generation, matched-filter construction,
  and compression. See [Range Compression](range_compression.md).
- **EMI** — electromagnetic interference suppression methods.
  See [EMI](emi.md).
- **Ionosphere** — ionospheric distortion compensation for Mars-orbiting
  sounders. See [Ionosphere](ionosphere.md).
- **SAR** — focused and unfocused synthetic-aperture processors plus the
  dispatch and geometry helpers that drive them. See [SAR](sar.md).
