# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import RadarParams

RADAR_PARAMS = RadarParams(
    sci_key = "ECHO_SAMPLES",  # Default Science Key Name
    fs = (80e6 / 3),  # Sampling Rate [Hz]
    dt = 1 / (80e6 / 3),  # Sampling Time [s]
    dz = 5.6211, # Range spatial sampling (m)
    tau = 85.05e-6,  # Chirp Length [s]
    bw = 10e6,  # Bandwidth [Hz]
    f_cen = [20e6],  # Center Frequency [Hz]
    f_start = [25e6],  # Starting Frequency [Hz]
    f_end = [15e6],  # Ending Frequency [Hz]
    n_samp = int(3600),  # Number of Samples in Science Data
    nfft = int(4096),  # Fast-time FFT Length
    prf = 700.28,  # Native Pulse Repetition Frequency [Hz]
    pri = 1 / 700.28,  # Native Pulse Repetition Interval [s]
    latency = [12.5e-6],  # Latency due to the electronics [s]
    l_a = 10,  # Antenna Length [m]
)

RDR_OVERRIDES = {
    'dt': 75e-9,
    'dz': 11.24,
    'fs': 1 / 75.e-9,
    'n_samp': int(667),
    'nfft': int(1024),
    'latency': [0.0], # TODO (medium priority; peri-beta): Need to verify this
}