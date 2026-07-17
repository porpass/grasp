# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Any
from scipy.fft import rfft, rfftfreq

##############################################################################################################
#
# Preprocessing Functions
#
##############################################################################################################

# Number of leading samples overwritten to suppress the FCMC DC
# leakage. Empirically determined: a smaller value (e.g. 3) leaves
# a visible step in the radargram from secondary mixer settling.
_DC_SUPPRESS_LEAD_SAMPLES = 10

def _suppress_dc_spike(data: NDArray, n_lead: int = _DC_SUPPRESS_LEAD_SAMPLES) -> NDArray:
    """Suppress the LRS FCMC DC artifact at the start of each trace.

    After demean and RVP removal, a sample-0 DC spike persists from
    residual mixer leakage. Overwrite the first n_lead samples with
    the value at sample n_lead, which sits below the spike but above
    the first real-signal sample. The replacement is silent (no
    warning, no config) — it's an instrument-known artifact that
    every LRS observation has.

    Args:
        data: Range-compressed waveform, shape (n_range, n_col).
        n_lead: Number of leading samples to overwrite. Default 3.

    Returns:
        Data with the DC spike suppressed in-place.
    """
    if data.shape[0] <= n_lead + 1:
        return data
    data[:n_lead, :] = data[n_lead, :]
    return data


def preprocess_lrs_wf(data: NDArray[np.integer],
                      dt: float,
                      tau: float,
                      bw: float,
                      ) -> NDArray[np.complexfloating]:
    """
    Apply preprocessing to LRS WF (EDR) science data.

    This function performs the following operations in order:
        1. Demean LRS data.
        2. RFFT to complete range compression.
        3. Residual Video Phase (RVP) Correction.

    Args:
        data: Raw LRS data array.
        dt: Sampling interval of the data array in seconds.
        tau: Emitted chirp duration in seconds.
        bw: Bandwidth of the emitted chirp in Hz.

    Returns:
        Preprocessed echo data after range compression and RVP correction,
        as a complex-valued array in the frequency domain.
    """
    n = data.shape[0]
    data = demean_lrs_wf(data)
    freq = rfftfreq(n, d=dt)
    data = rfft(data, axis=0, workers=-1)
    data = rvp_correction(data, dt, tau, bw, freq)
    data = _suppress_dc_spike(data)
    return data



def demean_lrs_wf(data: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    """Demean raw LRS waveform data.

    Subtracts the mean of the input array, removing any DC offset prior
    to range compression.

    Args:
        data: Raw LRS integer data array of shape (n_samples, n_pulses).

    Returns:
        Mean-subtracted data array of the same shape, cast to float64.
    """
    return data - data.mean()


def rvp_correction(data: NDArray,
                   dt: float,
                   tau: float,
                   bw: float,
                   f: NDArray,
                   ) -> NDArray[np.complexfloating[Any, Any]]:
    """Apply Residual Video Phase (RVP) correction to range-compressed data.

    Constructs and applies a matched filter in the frequency domain to remove
    the quadratic phase error introduced by the FMCW ranging process. The
    correction filter is given by H_rvp = exp(-i * pi * f^2 / k_r), where
    k_r = bw / tau is the chirp rate.

    Args:
        data: Range-compressed data array of shape (n_freq, n_pulses),
            as returned by rfft along the range axis.
        dt: Sampling interval in seconds, used to compute the frequency axis.
        tau: Emitted chirp duration in seconds.
        bw: Bandwidth of the emitted chirp in Hz.
        f: Frequency Array (Hz)

    Returns:
        RVP-corrected complex data array of the same shape as the input.
    """
    k_r = bw / tau
    theta_rvp = np.pi * f ** 2 / k_r
    H_rvp = np.exp(-1j * theta_rvp)
    return data * H_rvp[:, None]

