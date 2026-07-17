# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import RadarParams

RADAR_PARAMS = RadarParams(
    sci_key = "ECHO_{}_{}_DIP",  # Default Science Key Name
    fs = 1.4e6,  # Sampling Rate [Hz]
    dt = 1 / 1.4e6,  # Sampling Time [s]
    dz = 107.0687, # Range spatial sampling [m]
    tau = 250e-6,  # Chirp Length [s]
    bw = 1e6,  # Bandwidth [Hz]
    f_cen = [1.8e6, 3.0e6, 4.0e6, 5.0e6],  # Center Frequency [Hz]
    f_start = [2.3e6, 2.5e6, 3.5e6, 4.5e6],  # Starting Frequency [Hz]
    f_end = [1.3e6, 3.5e6, 4.5e6, 5.5e6],  # Ending Frequency [Hz]
    n_samp = int(512),  # Number of Samples in Science Data
    nfft = int(512),  # Fast-time FFT Length
    prf = 127.2668685913086,  # Native Pulse Repetition Frequency [Hz]
    pri = 1 / 127.2668685913086,  # Native Pulse Repetition Interval [s]
    latency = [5.31e-6, 5.04e-6, 4.67e-6, 4.49e-6],
    l_a = 40,  # Need to look this up...I think it is 40 m
)

RAW_OVERRIDES = {
    'fs': 2.8e6,
    'dt': 3.57e-7,
    'dz': 53.534,
    'n_samp': 980,
    'nfft': 1024,
    'n_filt': 1,
    'filt_str': ['ZERO'],
}