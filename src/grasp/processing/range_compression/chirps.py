# SPDX-License-Identifier: BSD-3-Clause
from importlib.resources import files
from pathlib import Path
import os

import numpy as np
from numpy.typing import NDArray


def create_complex_baseband_chirp(n: int,
                                  dt: float,
                                  tau: float,
                                  bw: float,
                                  *,
                                  chirp_direction: int = 1,
                                  ) -> tuple[NDArray[np.complex64], NDArray[np.float32], NDArray[np.float32]]:
    """
    Create an ideal complex baseband linear FM (LFM) chirp and its frequencies.

    The chirp is centered about t=0 and zero-padded to length `n`. The number of
    pulse samples is forced to be odd so the time vector is symmetric about t=0
    with an exact center sample.

    Frequencies returned are the *instantaneous frequency* of the analytic LFM
    chirp: f(t) = K * t, where K = bw / tau.

    Args:
        n: Total number of samples in the output array.
        dt: Sample spacing in seconds.
        tau: Chirp duration in seconds.
        bw: Chirp bandwidth in Hz.
        chirp_direction: Chirp direction (1 upchirp, -1 downchirp).
            Defaults to an upchirp

    Returns:
        A tuple of:
            chirp: Complex baseband chirp of shape `(n,)`, zero-padded outside the pulse.
            t_tau: Time vector (seconds) for the non-zero pulse portion, shape `(n_tau,)`, centered about 0.
            f_tau: Instantaneous frequency (Hz) for the pulse portion, shape `(n_tau,)`.

    Raises:
        ValueError: If inputs are non-positive or if the chirp duration exceeds `n`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if bw <= 0.0:
        raise ValueError("bw must be positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if abs(chirp_direction) != 1:
        raise ValueError("sgn must be -1 or 1")

    n_tau = int(round(tau / dt))
    if n_tau % 2 == 0:
        n_tau += 1
    if n_tau > n:
        raise ValueError("chirp duration exceeds output array length")
    sgn = chirp_direction
    chirp = np.zeros(n, dtype=np.complex64)
    k = bw / tau  # FM rate (Hz/s)
    t_tau = (np.arange(n_tau, dtype=np.float32) - (n_tau // 2)) * dt
    f_tau = (k * t_tau).astype(np.float32) # Instantaneous Freq. (Hz)
    chirp[:n_tau] = np.exp(sgn*1j * np.pi * k * t_tau ** 2).astype(dtype=np.complex64, copy=False)
    return chirp, t_tau, f_tau


def create_real_baseband_chirp(n: int,
                               dt: float,
                               tau: float,
                               f_start: float,
                               f_end: float,
                               ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Create a causal real-valued linear FM (LFM) chirp and its instantaneous frequencies.

    The chirp is generated for t starting at 0 with instantaneous frequency
    f(t) = f_start + k*t, where k = (f_end - f_start) / tau. The pulse is
    zero-padded to length `n`.

    Args:
        n: Total number of samples in the output array.
        dt: Sample spacing in seconds.
        tau: Chirp duration in seconds.
        f_start: Start frequency at t=0 in Hz.
        f_end: End frequency at t=tau in Hz (approximately, depending on sampling).

    Returns:
        A tuple of:
            chirp: Real chirp of shape `(n,)`, zero-padded outside the pulse.
            t_tau: Time vector (seconds) for the pulse portion, shape `(n_tau,)`, starting at 0.
            f_tau: Instantaneous frequency (Hz) for the pulse portion, shape `(n_tau,)`.

    Raises:
        ValueError: If inputs are invalid or if the chirp duration exceeds `n`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if f_start <= 0.0:
        raise ValueError("f_start must be positive")
    if f_end <= 0.0:
        raise ValueError("f_end must be positive")
    if f_start == f_end:
        raise ValueError("f_start and f_end must be different")

    n_tau = int(round(tau / dt))
    if n_tau % 2 == 0:
        n_tau += 1
    if n_tau > n:
        raise ValueError("chirp duration exceeds output array length")

    t_tau = np.arange(n_tau, dtype=np.float32) * dt
    k = (f_end - f_start) / tau  # Hz/s
    f_tau = (f_start + k * t_tau).astype(np.float32, copy=False)

    chirp = np.zeros(n, dtype=np.float32)
    phase = 2.0 * np.pi * (f_start * t_tau + 0.5 * k * t_tau**2)
    chirp[:n_tau] = np.sin(phase).astype(np.float32, copy=False)

    return chirp, t_tau, f_tau


def create_sharad_calibrated_chirp(tx_temp: float,
                                   rx_temp: float,
                                   calib_root: str | Path | None = None,
                                   ) -> tuple[NDArray[np.complex64], NDArray[np.float32]]:
    """
    Load a SHARAD temperature-calibrated reference chirp spectrum.

    The calibration file is selected by choosing the nearest available transmit (TX)
    and receive (RX) temperature bins.

    Args:
        tx_temp: Transmit temperature (°C).
        rx_temp: Receive temperature (°C).
        calib_root: Directory containing SHARAD calibration chirp files.
            If ``None`` (default), the loader uses the bundled calibration
            files at ``grasp/sharad/calibrated_chirps/``.

    Returns:
        A tuple of:
            spectrum: Complex spectrum array of shape `(2048,)`, dtype `np.complex64`.
            f: Frequency vector in Hz of shape `(2048,)`, centered about DC.

    Raises:
        FileNotFoundError: If the selected calibration file does not exist.
        ValueError: If the calibration file size is unexpected.
    """
    calib_name = "reference_chirp"
    ext = ".dat"

    dt = 0.0375e-6
    npos = 2048
    nfft = 4096


    tx_bins = np.array([-20, -15, -10, -5, 0, 20, 40, 60], dtype=np.float64)
    rx_bins = np.array([-20, 0, 20, 40, 60], dtype=np.float64)

    tx_names = ["m20tx", "m15tx", "m10tx", "m05tx", "p00tx", "p20tx", "p40tx", "p60tx"]
    rx_names = ["m20rx", "p00rx", "p20rx", "p40rx", "p60rx"]

    tx_idx = int(np.argmin(np.abs(tx_bins - tx_temp)))
    rx_idx = int(np.argmin(np.abs(rx_bins - rx_temp)))

    calib_tx = tx_names[tx_idx]
    calib_rx = rx_names[rx_idx]

    if calib_root is None:
        calib_root = files("grasp.sharad") / "calibrated_chirps"

    root = os.fspath(calib_root)
    filename = f"{calib_name}_{calib_tx}_{calib_rx}{ext}"
    path = os.path.join(root, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    tmp = np.fromfile(path, dtype="<f4")
    expected = 2 * npos
    if tmp.size != expected:
        raise ValueError(
            f"Unexpected calibration file size: got {tmp.size}, expected {expected}"
        )

    real = tmp[:npos]
    imag = tmp[npos:]
    spectrum = (real + 1j * imag).astype(np.complex64, copy=False)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
    idx0 = nfft // 4
    idx1 = idx0 + nfft // 2
    f = f[idx0:idx1].astype(np.float32, copy=False)
    if spectrum.shape != f.shape:
        raise ValueError("Spectrum and frequency vector lengths do not match")

    return spectrum, f


