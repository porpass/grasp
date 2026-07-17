# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from ..utils import select_iband

def adaptive_spectral_notch(spectra: NDArray[np.complexfloating],
                            freqs: NDArray[np.floating],
                            bw: float,
                            *,
                            f_cen: float = 0.0,
                            window_size: int = 129,
                            k: float = 6.0,
                            statistic: str = "MAD",
                            replace: str = "INTERP",
                            interp_pad: int = 2,
                            mask: NDArray[np.bool_] | None = None,
                           ) -> tuple[NDArray[np.complexfloating], NDArray[np.bool_]]:
    """
    Applies adaptive bin-level spectral notching to suppress narrowband EMI.

    This function identifies frequency bins with anomalously high magnitude
    (relative to a local baseline) inside the fundamental band (|f| <= bw/2),
    and replaces those bins either by:
      - a local baseline estimate (BASELINE),
      - linear interpolation across neighboring bins (INTERP), or
      - zeroing the bins (ZERO).

    Phase is preserved for BASELINE/INTERP replacements by reconstructing the
    complex spectrum as `new_mag * exp(1j * original_phase)`.

    Args:
        spectra: Complex spectra of shape `(n_samp, n_cols)`.
        freqs: Frequency vector (Hz) of length `n_samp`.
        bw: Bandwidth (Hz). Only bins with `abs(freqs) <= bw/2` are processed.
        f_cen: Center frequency of the spectra (Hz)
        window_size: Odd window length (bins) for local statistics. Defaults to 129.
        k: Outlier threshold multiplier.
            For MAD: threshold = median + k*(1.4826*MAD).
            For STD: threshold = mean + k*std. Defaults to 6.0.
        statistic: Local-statistic method used to form the threshold ("MAD" or "STD").
            Defaults to "MAD".
        replace: Replacement strategy for flagged bins:
            - "BASELINE": set magnitude to local baseline,
            - "INTERP": interpolate magnitude across flagged runs,
            - "ZERO": set magnitude to 0 (hard notch).
            Defaults to "INTERP".
        interp_pad: Number of bins on each side of a flagged run used as
            interpolation anchors (INTERP only). Defaults to 2.
        mask: Optional precomputed boolean mask of shape `(n_samp, n_cols)`.
            If provided, it is used directly (still restricted to the in-band
            region).

    Returns:
        A tuple `(spectra_out, emi_mask)` where:

        - `spectra_out` is the EMI-suppressed complex spectra with the same
            shape as `spectra`,
        - `emi_mask` is a boolean array of the same shape indicating which
            bins were notched/replaced.

    Raises:
        ValueError: If shapes are inconsistent, if `bw <= 0`, if `window_size`
            is not positive and odd, or if the selected band contains no bins.

    Notes:
        - This is designed for narrow spikes (1–few bins). For broad EMI,
          consider floor equalization or time-frequency methods.
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 2-D; got shape {spectra.shape}")
    n_samp, n_cols = spectra.shape

    if freqs.ndim != 1 or freqs.shape[0] != n_samp:
        raise ValueError(
            f"freqs must be 1-D of length spectra.shape[0]; got {freqs.shape} vs {n_samp}"
        )
    if bw <= 0:
        raise ValueError(f"bw must be positive; got {bw}")
    if window_size <= 0 or (window_size % 2) == 0:
        raise ValueError(f"window_size must be a positive odd integer; got {window_size}")
    if interp_pad < 0:
        raise ValueError(f"interp_pad must be >= 0; got {interp_pad}")

    statistic = statistic.upper()
    replace = replace.upper()

    iband = select_iband(freqs, f_cen, bw)

    out = np.zeros_like(spectra)
    emi_mask = np.zeros((n_samp, n_cols), dtype=bool)

    # Work in-band only
    band = spectra[iband, :]
    mag = np.abs(band)
    phase = np.angle(band)

    if mask is not None:
        if mask.shape != (n_samp, n_cols):
            raise ValueError(f"mask must have shape {spectra.shape}; got {mask.shape}")
        band_mask = mask[iband, :].astype(bool, copy=False)
    else:
        band_mask = np.zeros_like(mag, dtype=bool)

        half = window_size // 2
        # Pad along frequency axis for rolling stats
        mag_pad = np.pad(mag, ((half, half), (0, 0)), mode="reflect")

        if statistic == "MAD":
            # Moving median and median absolute deviation (MAD)
            for j in range(n_cols):
                # Build sliding windows using striding
                x = mag_pad[:, j]
                # windows shape: (n_band, window_size)
                windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
                med = np.median(windows, axis=1)
                mad = np.median(np.abs(windows - med[:, None]), axis=1)
                sigma = 1.4826 * mad
                thr = med + k * sigma
                band_mask[:, j] = mag[:, j] > thr
                if replace == "BASELINE":
                    mag[:, j] = np.where(band_mask[:, j], med, mag[:, j])
        elif statistic == "STD":
            for j in range(n_cols):
                x = mag_pad[:, j]
                windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
                mu = windows.mean(axis=1)
                sd = windows.std(axis=1, ddof=0)
                thr = mu + k * sd
                band_mask[:, j] = mag[:, j] > thr
                if replace == "BASELINE":
                    mag[:, j] = np.where(band_mask[:, j], mu, mag[:, j])
        else:
            raise ValueError(f"Unsupported statistic: {statistic} (use 'MAD' or 'STD')")

    # Apply replacements
    if replace == "ZERO":
        mag[band_mask] = 0.0

    elif replace == "INTERP":
        # Interpolate magnitude across flagged runs, per column.
        n_band = mag.shape[0]
        x = np.arange(n_band, dtype=float)

        for j in range(n_cols):
            mcol = mag[:, j]
            msk = band_mask[:, j]

            if not np.any(msk):
                continue

            # Identify contiguous True runs
            idx = np.flatnonzero(msk)
            # Run boundaries
            splits = np.where(np.diff(idx) > 1)[0] + 1
            runs = np.split(idx, splits)

            for run in runs:
                r0, r1 = int(run[0]), int(run[-1])

                left = max(0, r0 - interp_pad - 1)
                right = min(n_band - 1, r1 + interp_pad + 1)

                # Need distinct anchors on each side; otherwise fallback to baseline.
                if left >= r0 or right <= r1:
                    # Fallback: replace with local median of a small neighborhood.
                    lo = max(0, r0 - 8)
                    hi = min(n_band, r1 + 9)
                    baseline = float(np.median(mcol[lo:hi]))
                    mcol[r0 : r1 + 1] = baseline
                    continue

                # Linear interpolation between anchors
                y0 = float(mcol[left])
                y1 = float(mcol[right])
                mcol[r0 : r1 + 1] = np.interp(x[r0 : r1 + 1], [left, right], [y0, y1])

            mag[:, j] = mcol

    elif replace == "BASELINE":
        # BASELINE replacement is already applied during mask creation when mask is None.
        # If mask was provided, apply a robust baseline here.
        if mask is not None:
            half = window_size // 2
            mag_pad = np.pad(mag, ((half, half), (0, 0)), mode="reflect")
            for j in range(n_cols):
                xj = mag_pad[:, j]
                windows = np.lib.stride_tricks.sliding_window_view(xj, window_size)
                med = np.median(windows, axis=1)
                mag[:, j] = np.where(band_mask[:, j], med, mag[:, j])

    else:
        raise ValueError(f"Unsupported replace: {replace} (use 'BASELINE', 'INTERP', or 'ZERO')")

    # Reconstruct complex band and write back
    band_out = mag * np.exp(1j * phase)
    out[iband, :] = band_out

    emi_mask[iband, :] = band_mask
    return out, emi_mask