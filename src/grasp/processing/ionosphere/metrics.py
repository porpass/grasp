import numpy as np
# TODO Investigate best parameters (windows and such) for each method
#   - L4 should be paired with a window that greatly reduces sidelobes
#   - SNR should work pretty well with any
#
# All metric functions operate along axis=0 (fast-time): scalar return
# for 1D input, 1D per-column return for 2D input. The ionosphere
# autofocus loops scalarize per-column metrics by summing as needed.


# Minimization Routines
def l1(ts):
    """Per-column L1 norm. Scalar for 1D input."""
    return np.sum(np.abs(ts), axis=0)


def entropy(ts):
    """Per-column power-spectrum entropy. Scalar for 1D input."""
    p = np.abs(ts)**2
    p = p / p.sum(axis=0, keepdims=True).clip(min=1e-30)
    return -np.sum(p * np.log(p + 1e-30), axis=0)


# Maximization Routines
def l4(ts):
    """Per-column L4 norm. Scalar for 1D input."""
    return np.sum(np.abs(ts)**4, axis=0)


def peak_snr(ts):
    """Per-column peak signal-to-noise ratio.

    For each column, compute ``|ts|**2``, divide the peak power by the
    median of the first 50 samples (the noise-floor estimate), and
    return the result. Mirrors the PDS ``focus_array`` IDL metric with
    two tweaks: noise region widened from 20 to 50 samples, and the
    noise estimator switched from mean to median for robustness against
    bright early-time samples.

    Args:
        ts: 1D ``(n_samp,)`` or 2D ``(n_samp, n_col)`` complex time series.

    Returns:
        Scalar for 1D input, 1D array of length ``n_col`` for 2D input.
        Higher is better.
    """
    power = np.abs(ts)**2
    if power.ndim == 1:
        noise = max(float(np.median(power[:50])), 1e-30)
        return float(np.max(power) / noise)
    noise = np.maximum(np.median(power[:50], axis=0), 1e-30)
    return np.max(power, axis=0) / noise


METRICS = {
    "L1":       (l1, "minimize"),
    "L4":       (l4, "maximize"),
    "ENTROPY":  (entropy, "minimize"),
    "PEAK_SNR": (peak_snr, "maximize"),
}
