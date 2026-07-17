# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import RadarParams

RADAR_PARAMS = RadarParams(
    sci_key = "ECHO_SAMPLES",  # Default Science Key Name
    fs = 6.25e6,  # Sampling Rate [Hz];
    dt = 3.0517578125e-07,  # Sampling Time [s]
    dz = 45.77636718750, # Range spatial sampling [m]
    tau = 200e-6,  # Chirp Length [s]
    bw = 2e6,  # Bandwidth [Hz]
    f_cen = [5e6],  # Center Frequency [Hz]
    f_start = [4e6],  # Starting Frequency [Hz]
    f_end = [6e6],  # Ending Frequency [Hz]
    n_samp = int(1025),  # Number of Samples in Science Data
    nfft = int(2048),  # Fast-time FFT Length
    prf = 20,  # Native Pulse Repetition Frequency [Hz]
    pri = 1 / 20,  # Native Pulse Repetition Interval [s]
    latency = [0.0],  # LRS does not have any latency
    l_a = 30,  # Antenna Length [m]
)