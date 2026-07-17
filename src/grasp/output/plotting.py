# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def plot_radargram(ts: NDArray, x_size: float = 10) -> None:
    """Display a radargram in a matplotlib figure.

    The amplitude is converted to dB relative to the mean amplitude of the
    first 50 rows, then displayed with greyscale clipping between the 50th
    and 99th percentiles.

    Args:
        ts: 2-D array of complex or real samples with shape
            ``(n_samp, n_col)``.
        x_size: Figure width in inches. The height is scaled to preserve the
            ``n_col / n_samp`` aspect ratio.
    """
    n_samp, n_col = ts.shape
    toPlot = 20 * np.log10(np.abs(ts) / np.mean(np.abs(ts[:50, :])))
    vmin = np.percentile(toPlot, 50)
    vmax = np.percentile(toPlot, 99)

    x_size = 10
    y_size = x_size * n_col / n_samp

    fig, ax = plt.subplots(1, 1, figsize=(x_size, y_size))
    ax.imshow(toPlot,
              cmap="gray",
              vmin=vmin,
              vmax=vmax,
              extent=(0, n_col, n_samp, 0),
              origin="upper",
              aspect="equal")
    plt.tight_layout()
    plt.show()

def plot_dem_swath(swath: dict,
                   aspect: float = 10.0,
                   ) -> None:
    """Plot a DEM swath and centerline elevation profile.

    Args:
        swath: Dictionary returned by :func:`extract_dem_swath`.
        aspect: Aspect ratio for the swath plot (along-track units
            per cross-track unit).
    """
    track_len = swath["along_track_km"][-1] - swath["along_track_km"][0]
    swath_width = abs(swath["cross_track_km"][0] - swath["cross_track_km"][-1])
    data_ratio = swath_width / track_len

    fig_width = 14
    fig_height_top = max(fig_width * data_ratio * aspect, 2.0)
    fig_height_bottom = 4

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height_top + fig_height_bottom),
        sharex=True,
        gridspec_kw={"height_ratios": [fig_height_top, fig_height_bottom]},
    )

    ax1.pcolormesh(
        swath["along_track_km"],
        swath["cross_track_km"],
        swath["elevation"],
        shading="auto",
        cmap="terrain",
    )
    ax1.set_ylabel("Cross-track (km)")
    ax1.set_title("DEM Swath")
    ax1.axhline(0, color="red", linewidth=0.5, linestyle="--", label="Ground track")
    ax1.legend(loc="upper right")

    center_idx = swath["cross_track_km"].size // 2
    ax2.plot(swath["along_track_km"], swath["elevation"][center_idx, :])
    ax2.set_xlabel("Along-track (km)")
    ax2.set_ylabel("Elevation")
    ax2.set_title("Centerline Elevation Profile")

    plt.tight_layout()
    plt.show()