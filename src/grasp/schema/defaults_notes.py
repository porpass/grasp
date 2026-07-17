# SPDX-License-Identifier: BSD-3-Clause
"""Human-readable descriptions of runtime-computed defaults.

Some fields on the processing dataclasses don't get a value from the
per-instrument TOML — they're filled in at runtime by the sounder
(e.g. ``sar.lamb = C / f_cen[0]``, ``preprocessing.n_center = n_samp // 2``).
The schema exporter still needs to tell PORPASS *something* for those
fields; it emits ``"default": null`` plus a ``"default_note"`` string
from this table so the form can show a placeholder hint like
"computed from geometry".
"""

DEFAULT_NOTES: dict[tuple[str, str], str] = {
    ("general",       "out_dir"):         "supplied by user at job-load time",

    ("preprocessing", "n_center"):        "n_samp // 2",

    ("iono_comp",     "metric"):          "PEAK_SNR (Campbell) / L4 (Contrast)",
    ("iono_comp",     "n_take"):          "8192 // presum (SHARAD); method default otherwise",

    ("sar",           "lamb"):            "C / RadarParams.f_cen[0]",
    ("sar",           "aperture_length"): "computed from geometry",
    ("sar",           "sgn"):             "+1 (SHARAD mix-up); -1 elsewhere",

    ("csim",          "target"):          "instrument SPICE target",
    ("csim",          "bin_size"):        "RadarParams.dt",
    ("csim",          "n_samples"):       "instrument-derived (n_samp or nfft)",
    ("csim",          "n_center"):        "n_samples // 2",
    ("csim",          "at_step"):         "computed from geometry when SAR is enabled",
    ("csim",          "at_dist"):         "computed from geometry when SAR is enabled",
    ("csim",          "dem_path"):        "supplied by user at job-load time",
}
