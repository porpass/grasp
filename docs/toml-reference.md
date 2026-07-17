# TOML Configuration Reference

A GRaSP processing job is configured via a TOML file passed to
`grasp.process_job(...)`. Each section in the file maps to a
dataclass in `grasp.grasp_types`. Field names within a section must
match the dataclass field names exactly (case-sensitive). TOML has
no `null`; **leave a field out** to keep its default — never write
`None` or `null`.

Stages with `enabled = false` are skipped at runtime even if other
fields in their section are set. Stages without an `enabled` flag
always run.

## Section ↔ Dataclass Map

| TOML section | Dataclass | What it controls |
|---|---|---|
| `[general]` | `ProcessingParameters` | Output directory, global verbosity |
| `[input]` | (paths) | Label / science / auxiliary file locations |
| `[preprocessing]` | `PreprocessingParams` | Onboard-presum correction, trace alignment |
| `[range_compression]` | `RangeCompressionParams` | Pulse compression, chirp choice, window |
| `[emi_suppression]` | `EMISuppresionParams` | Spectral interference detection and removal |
| `[ionospheric_compensation]` | `IonoCompParams` | Mars-only ionospheric phase correction |
| `[sar_processing]` | `SARParams` | Azimuth focusing method and aperture |
| `[multilooking]` | `MLKParams` | Post-SAR multilook averaging |
| `[clutter_simulation]` | `ClutterSimParams` | Geometric clutter simulation from a DEM |
| `[output_parameters]` | `OutputParameters` | Data export format and reprojection |
| `[plot_parameters]` | `PlotParameters` | Radargram and browse image rendering |
| `[final_output]` | `FinalOutputParams` | End-of-pipeline product export |

A note on instrument support: not every stage applies to every
instrument. See the per-instrument example TOMLs for which sections
are useful where.

---

## `[general]` — top-level options

| Field | Type | Default | Description |
|---|---|---|---|
| `out_dir` | `str` (path) | **required** | Directory for all data products. Created if missing. |
| `verbose` | `bool` | `false` | If true, prints per-stage progress messages. |

---

## `[input]` — file paths

Not backed by a dataclass; these go directly into the
`ProcessingJob`.

| Field | Type | Required | Description |
|---|---|---|---|
| `label_file` | `str` (path) | yes | PDS label file. |
| `science_file` | `str` (path) | yes | Science data file. |
| `auxiliary_file` | `str` (path) | depends | Auxiliary data file. Required for SHARAD EDR; not used for some MARSIS products. |

---

## `[preprocessing]` — `PreprocessingParams`

Runs decompression and applies the SHARAD instrument response.
(optionally) trace alignment that centres the surface return.
(optionally) increases the data presum

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Run the preprocessing stage. |
| `target_presum` | `int` \| omit | omit | Target presumming factor. Omit for no additional presumming. |
| `align_traces` | `bool` | `true` | Remove receive-window opening-time offsets and centre the surface return on `n_center`. |
| `n_center` | `int` \| omit | `n_samp // 2` | Sample index for the centred surface return. |
| `save_data` | `bool` | `false` | Export data after this stage. |
| `save_images` | `bool` | `false` | Export browse images after this stage. |
| `save_state` | `bool` | `false` | Save full HDF5 state snapshot after this stage. |

---

## `[range_compression]` — `RangeCompressionParams`

Applies a matched/inverse filter to the data

| Field | Type | Default     | Valid values                        | Description                                                                                                               |
|---|---|-------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `enabled` | `bool` | `false`     | —                                   | Run the range-compression stage.                                                                                          |
| `chirp_type` | `str` | `"ideal"`   | `ideal`, `calibrated`               | Reference chirp source. `ideal` is the synthetic LFM; `calibrated` is the temperature-binned PDS reference (SHARAD only). |
| `filter_type` | `str` | `"matched"` | `matched`, `inverse`                | Matched filter is the standard; inverse de-correlates the chirp spectrum.                                                 |
| `window` | `str` | `"hanning"` | (see [window types](#window-types)) | Spectral window applied to the filter.                                                                                    |
| `window_alpha` | `float` \| omit | omit        | `[0, 1]`                            | Tukey taper fraction. Only used when `window = "tukey"`.                                                                  |
| `save_data` | `bool` | `false`     | —                                   | Export data after this stage.                                                                                             |
| `save_images` | `bool` | `false`     | —                                   | Export browse images after this stage.                                                                                    |
| `save_state` | `bool` | `false`     | —                                   | Save HDF5 state snapshot after this stage.                                                                                |

---

## `[emi_suppression]` — `EMISuppresionParams`

Detects narrow-band interference in the range spectrum and replaces
the affected bins.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` | — | Run the EMI-suppression stage. |
| `method` | `str` | `"ADAPTIVE"` | `ADAPTIVE`, `THRESHOLD` | `ADAPTIVE` uses local statistics per bin; `THRESHOLD` uses a global multiplier. |
| `threshold_k` | `float` | `1.5` | `> 0` | Outlier multiplier. For ADAPTIVE/MAD: `threshold = median + k · 1.4826 · MAD`. For THRESHOLD: `threshold = global_median · k`. |
| `window_size` | `int` | `129` | odd `> 0` | Sliding-window length (bins) used to compute the local baseline. Forced odd internally. |
| `value_k` | `float` | `1.0` | `> 0` | Replacement-magnitude scale factor (THRESHOLD method only). |
| `replace_strategy` | `str` | `"INTERP"` | `INTERP`, `ZERO`, `BASELINE` | How flagged bins are replaced. `INTERP` linearly interpolates magnitude across flagged runs; `BASELINE` uses the local-baseline value; `ZERO` zeros the bin. |
| `interp_pad` | `int` | `2` | `≥ 1` | Number of anchor bins on each side of a flagged run used by `INTERP`. |
| `save_data` | `bool` | `false` | — | Export data after this stage. |
| `save_images` | `bool` | `false` | — | Export browse images after this stage. |
| `save_state` | `bool` | `false` | — | Save HDF5 state snapshot after this stage. |

---

## `[ionospheric_compensation]` — `IonoCompParams`

Mars-only stage that corrects the dispersive phase error introduced
by Mars' ionosphere. Not applicable to LRS.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` | — | Run the ionosphere stage. |
| `method` | `str` | `"CONTRAST"` | `CONTRAST`, `CAMPBELL` | `CONTRAST` is the Mouginot et al. contrast-maximisation method; `CAMPBELL` is the Campbell et al. autofocus method. |
| `n_take` | `int` \| omit | method-dependent | `> 0` | Stacking neighbourhood width (columns). |
| `campbell_b` | `float` | `1.93` | `> 0` | Power-law exponent for the Campbell phase model. |
| `campbell_n_phase` | `int` | `400` | `> 0` | Number of phase states sampled (CAMPBELL only). |
| `campbell_delta` | `float` | `0.0125` | `> 0` | E-value step size (CAMPBELL only). |
| `contrast_L_eq` | `float` | `80e3` | `> 0` | Equivalent path length (m). Used to compute `tau_0` (CONTRAST only). |
| `sgn` | `int` | `1` | `-1`, `1` | Sign convention for the phase exponential. |
| `window` | `str` | `"hanning"` | (see [window types](#window-types)) | Spectral window. |
| `window_alpha` | `float` \| omit | omit | `[0, 1]` | Tukey taper fraction (used only when `window = "tukey"`). |
| `save_data` | `bool` | `false` | — | Export data after this stage. |
| `save_images` | `bool` | `false` | — | Export browse images after this stage. |
| `save_state` | `bool` | `false` | — | Save HDF5 state snapshot after this stage. |

---

## `[sar_processing]` — `SARParams`

Synthetic-aperture focusing in azimuth. The selected method
determines how range-compressed echoes are coherently combined.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` | — | Run the SAR stage. |
| `method` | `str` | `"backscatter"` | `unfocused`, `range-doppler`, `backscatter` | `unfocused` is a fast non-coherent aperture; `range-doppler` is the standard focused algorithm; `backscatter` is range-Doppler with built-in multilook. |
| `aperture_length` | `int` \| omit | computed | `> 0` | Synthetic aperture length in input columns. Omit to compute from geometry. Exceeding the unaliased limit triggers a warning (see `grasp.max_unaliased_aperture`). |
| `lamb` | `float` \| omit | omit | `> 0` | Effective wavelength (m) for the azimuth matched filter and RCMC. If omitted, falls back to `C / f_cen[0]` (nominal center wavelength). The empirical optimum is instrument-specific and usually differs from the nominal center: SHARAD focuses best near `12.0` (high-frequency band edge), LRS near `60.0` (true 5 MHz center). See per-instrument example TOMLs for tuned values. Verify against a focused point target if changing instruments or modes. |
| `os_factor` | `int` | `1` | `≥ 1` | Output oversampling factor relative to single-look resolution. |
| `window` | `str` | `"hanning"` | (see [window types](#window-types)) | Azimuth Doppler-domain window. Stronger edge attenuation (Blackman-Harris, Nuttall, Kaiser) suppresses any Doppler-aliased contribution from over-aperture configurations. |
| `window_alpha` | `float` \| omit | omit | `[0, 1]` | Tukey taper fraction (used only when `window = "tukey"`). |
| `number_of_looks` | `int` | `5` | `> 0` | Number of looks used by the `backscatter` method. Ignored by other methods (set in `[multilooking]` for `range-doppler`). |
| `remove_doppler_centroid` | `bool` | `false` | — | If true, demodulate the Doppler centroid before processing. |
| `interp_cval` | `float` | `0.0` | — | Fill value for RCMC interpolation outside the data bounds. |
| `coherent` | `bool` | `false` | — | If true, return complex output (preserves phase). |
| `save_data` | `bool` | `false` | — | Export data after this stage. |
| `save_images` | `bool` | `false` | — | Export browse images after this stage. |
| `save_state` | `bool` | `false` | — | Save HDF5 state snapshot after this stage. |

---

## `[multilooking]` — `MLKParams`

Post-SAR averaging that trades azimuth resolution for SNR / speckle
reduction. Only applicable to `method = "range-doppler"`;
`backscatter` already multilooks.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` | — | Run the multilook stage. |
| `number_of_looks` | `int` | `5` | odd `> 0` | Window length in columns. Forced odd internally. |
| `os_factor` | `int` | `1` | `≥ 1` | Output oversampling factor. |
| `window_type` | `str` | `"hanning"` | (see [window types](#window-types)) | Multilook averaging window. |
| `window_alpha` | `float` \| omit | omit | `[0, 1]` | Tukey taper fraction (used only when `window_type = "tukey"`). |
| `coherent` | `bool` | `false` | — | If true, average complex data; if false, detect first then average power. |
| `save_data` | `bool` | `false` | — | Export data after this stage. |
| `save_images` | `bool` | `false` | — | Export browse images after this stage. |
| `save_state` | `bool` | `false` | — | Save HDF5 state snapshot after this stage. |

---

## `[clutter_simulation]` — `ClutterSimParams`

Synthesises surface-clutter returns from a planetary DEM for
comparison against the radargram.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `true` | — | Run the clutter simulation. |
| `dem_path` | `str` (path) \| omit | omit | — | DEM raster path. Omit to auto-select per body (MOLA for Mars, LOLA for Moon). |
| `at_step` | `float` \| omit | computed | `> 0` | Along-track facet dimension (m). |
| `at_dist` | `float` \| omit | computed | `> 0` | Along-track half-extent of the simulation grid (m). |
| `ct_step` | `float` \| omit | computed | `> 0` | Cross-track facet dimension (m). |
| `ct_dist` | `float` \| omit | computed | `> 0` | Cross-track extent from nadir (m). |
| `n_center` | `int` \| omit | `n_samples // 2` | `≥ 0` | Sample index of the ellipsoid surface return. |
| `body_a` | `float` \| omit | from SPICE | `> 0` | Semi-major axis of the target body (m). Normally derived from SPICE. |
| `body_b` | `float` \| omit | from SPICE | `> 0` | Semi-minor (polar) axis of the target body (m). |
| `bin_size` | `float` \| omit | from instrument | `> 0` | Range bin size in seconds. Normally derived from instrument parameters. |
| `n_samples` | `int` \| omit | from instrument | `> 0` | Number of range samples per trace. Normally derived from instrument parameters. |
| `lower_percentile` | `float` \| omit | omit | `[0, 100]` | Lower dB clip percentile for cluttergram images. |
| `upper_percentile` | `float` \| omit | omit | `[0, 100]` | Upper dB clip percentile for cluttergram images. |
| `vmin` | `float` \| omit | omit | — | Explicit minimum dB. Overrides `lower_percentile`. |
| `vmax` | `float` \| omit | omit | — | Explicit maximum dB. Overrides `upper_percentile`. |
| `echomap_lower_percentile` | `float` \| omit | omit | `[0, 100]` | Lower dB clip percentile for the echomap diagnostic. |
| `echomap_upper_percentile` | `float` \| omit | omit | `[0, 100]` | Upper dB clip percentile for the echomap diagnostic. |
| `save_data` | `bool` | `false` | — | Export cluttergram data. |
| `save_images` | `bool` | `false` | — | Export cluttergram images. |
| `save_state` | `bool` | `false` | — | Save HDF5 state snapshot after this stage. |

---

## `[output_parameters]` — `OutputParameters`

Controls how stage data is written when `save_data = true` anywhere
upstream.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `data_output_type` | `str` | `"basic"` | `basic` | Output format type. Only `basic` is currently implemented. |
| `byte_order` | `str` | `"big"` | `big`, `little` | Byte order for binary output. |
| `tgt_crs` | `str` \| omit | omit | PROJ4 / WKT / file path | Target CRS for georeferenced SEG-Y output. Omit to skip reprojection. |

---

## `[plot_parameters]` — `PlotParameters`

Image rendering for radargrams and browse images.

| Field | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `lower_percentile` | `float` | `50` | `[0, 100]` | Lower clip percentile for automatic dB scaling. Ignored when `vmin` is set. |
| `upper_percentile` | `float` | `99` | `[0, 100]` | Upper clip percentile for automatic dB scaling. Ignored when `vmax` is set. |
| `vmin` | `float` \| omit | omit | — | Explicit minimum dB value. |
| `vmax` | `float` \| omit | omit | — | Explicit maximum dB value. |
| `cmap` | `str` \| omit | omit | Matplotlib colormap name | Colormap. Omit for grayscale. |
| `invert` | `bool` | `false` | — | Invert grayscale so strong returns appear dark. |
| `buffer_km` | `float` \| omit | omit | `> 0` | DEM swath half-width (km) for browse images. |

---

## `[final_output]` — `FinalOutputParams`

End-of-pipeline product export from `RadarSounder.finalize()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `save_data` | `bool` | `true` | Export final data files. |
| `save_images` | `bool` | `true` | Export final browse images. |
| `save_segy` | `bool` | `false` | Export SEG-Y (with `[output_parameters].tgt_crs` for reprojection). |
| `save_state` | `bool` | `false` | Save full HDF5 state snapshot. |

---

## Window types

Accepted by every `window` / `window_type` field (case-insensitive):

| Name | Notes |
|---|---|
| `rectangle` (`none`) | No window. PSLR ≈ −13 dB. |
| `hann` (`hanning`) | Smooth raised cosine. PSLR ≈ −32 dB. |
| `hamming` | Slightly tighter mainlobe than Hann, higher sidelobe floor. |
| `blackman` | 3-term cosine, deeper sidelobes. |
| `blackman-harris` (`blackman_harris`, `bh`) | 4-term, PSLR ≈ −92 dB. Strong edge attenuation; recommended for SAR Doppler windowing if exceeding the unaliased aperture limit. |
| `nuttall` | 4-term, similar deep sidelobes. |
| `flattop` | Very wide mainlobe, used for amplitude calibration. Not for matched filtering. |
| `bartlett` | Triangular. |
| `cosine` (`sine`, `raised_cosine`) | Half-sine taper. |
| `tukey` | Tapered cosine; controlled by `window_alpha`. |

---

## Stages by instrument

A condensed view of which stages have a supported/untested/unsupported
tier per instrument. Stages marked `unsupported` will be auto-disabled
with a warning; `untested` runs but prints a caution. Consult
`RadarSounder.PROCESSING_SUPPORT` for the authoritative list.

| Stage | SHARAD EDR | SHARAD RDR | MARSIS EDR | MARSIS RDR | LRS EDR     | LRS RDR |
|---|---|---|---|---|-------------|---|
| preprocessing | supported | supported | supported | supported | supported   | supported |
| range_compression | supported | unsupported | supported | unsupported | unsupported | unsupported |
| emi_suppression | supported | untested | supported | untested | unsupported | unsupported |
| ionospheric_compensation | supported | untested | supported | unsupported | unsupported | unsupported |
| sar_processing | supported | unsupported | unsupported | unsupported | supported | unsupported |
| multilooking | supported | supported | supported | supported | supported   | supported |
| clutter_simulation | supported | supported | supported | supported | supported   | supported |

*(The table above reflects current PROCESSING_SUPPORT entries; update
when subclasses change.)*
