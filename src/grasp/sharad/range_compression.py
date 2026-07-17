# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray

from scipy.fft import fftshift

from ..processing.windows import form_window_bandlimited
from ..processing.utils import (
    check_monotonically_increasing,
    select_iband,
    to_complex_baseband,
)
from ..processing.range_compression.chirps import (
    create_complex_baseband_chirp,
    create_sharad_calibrated_chirp as _load_calibrated_spectrum,
)
from ..processing.range_compression.filters import create_filter
from ..processing.range_compression.compress import range_compress as _range_compress

from ..common.utils import assert_gt_0
from .parameters import RADAR_PARAMS as PARAMS


# Bin centres for nearest-neighbour selection
_TX_BINS = np.array([-20, -15, -10, -5, 0, 20, 40, 60], dtype=np.float64)
_RX_BINS = np.array([-20, 0, 20, 40, 60], dtype=np.float64)

def _calibrated_h_f(tx_temp: NDArray,
                    rx_temp: NDArray,
                    *,
                    filter_type: str,
                    dt: float = PARAMS.dt,
                    nfft: int = PARAMS.nfft,
                    bw: float = PARAMS.bw,
                    eps: float | None = 1e-12,
                    ) -> tuple[NDArray[np.complex64], NDArray[np.floating]]:
    """Build the matched/inverse filter directly from the PDS calibrated
    chirp spectrum, skipping the IFFT→FFT roundtrip that
    ``_make_chirp`` + ``create_filter`` would do.

    The PDS file already contains the chirp's spectrum at baseband, so
    the matched filter is simply its conjugate at the corresponding
    in-band buffer indices. The inverse filter is ``1 / S(f)`` with
    ``eps`` stabilisation. The result is 1D when every frame maps
    to the same ``(TX, RX)`` calibration bin, or 2D ``(nfft,
    n_frames)`` when bins vary along-track. Unique bin pairs are
    loaded once and cached.

    Args:
        tx_temp: TX antenna temperature per frame (°C).
        rx_temp: RX antenna temperature per frame (°C).
        filter_type: ``"MATCHED"`` or ``"INVERSE"`` (case-insensitive).
        dt: Sample spacing in seconds. Must match the calibration
            files' native ``dt``.
        nfft: FFT length of the output buffer.
        bw: Signal bandwidth in Hz. Only spectrum bins within
            ``|f| <= bw/2`` are used.
        eps: Magnitude threshold for inverse filter stabilisation.
            Bins with ``|S(f)| < eps`` are zeroed in the INVERSE path.

    Returns:
        A tuple ``(h_f, f_buf)`` where ``h_f`` is the frequency-domain
        filter (1D ``(nfft,)`` or 2D ``(nfft, n_frames)``) and
        ``f_buf`` is the natural-FFT-order frequency vector of length
        ``nfft``.

    Raises:
        ValueError: If shapes mismatch, the filter type is unknown,
            or the in-band bin counts of the PDS spectrum and the
            output buffer differ.
    """
    tx_temp = np.atleast_1d(np.asarray(tx_temp, dtype=np.float64))
    rx_temp = np.atleast_1d(np.asarray(rx_temp, dtype=np.float64))
    if tx_temp.shape != rx_temp.shape:
        raise ValueError(
            f"tx_temp and rx_temp must have matching shape; "
            f"got {tx_temp.shape} vs {rx_temp.shape}"
        )
    ft = filter_type.upper()
    if ft not in ("MATCHED", "INVERSE"):
        raise ValueError(f"filter_type {filter_type!r} not recognized")

    tx_idx = np.argmin(np.abs(_TX_BINS[:, None] - tx_temp[None, :]), axis=0)
    rx_idx = np.argmin(np.abs(_RX_BINS[:, None] - rx_temp[None, :]), axis=0)

    f_buf = np.fft.fftfreq(nfft, d=dt)
    iband_buf = select_iband(f_buf, 0.0, bw)
    iband_buf_sorted = iband_buf[np.argsort(f_buf[iband_buf])]

    def _spectrum_to_h_f(spectrum: NDArray[np.complex64],
                         f_spec: NDArray[np.floating],
                         ) -> NDArray[np.complex64]:
        iband_spec = select_iband(f_spec, 0.0, bw)
        if iband_spec.size != iband_buf_sorted.size:
            raise ValueError(
                f"In-band bin count mismatch: spectrum has {iband_spec.size}, "
                f"output buffer has {iband_buf_sorted.size}. Check that "
                f"dt={dt}, nfft={nfft}, bw={bw} are consistent with the "
                f"calibration file's native grid."
            )
        #
        # Note: Empirically the matched filter for the PDS calibrated chirp requires
        #       flipping the in-band spectrum and skipping the conjugate. The PDS
        #       documentation states the chirps are complex spectra at baseband in
        #       increasing frequency order, which would normally call for plain
        #       conj(S) without a flip — but that does not produce sharp range
        #       compression in practice. Reason for the discrepancy is not yet
        #       pinned down; revisit when convenient.
        #
        S = np.zeros(nfft, dtype=np.complex64)
        S[iband_buf_sorted] = np.flip(spectrum[iband_spec])
        if ft == "MATCHED":
            return S
        # INVERSE
        h = np.zeros(nfft, dtype=np.complex64)
        mask = np.abs(S) >= (eps if eps is not None else 0.0)
        h[mask] = (1.0 / S[mask]).astype(np.complex64, copy=False)
        return h

    # Fast path: one filter covers the whole observation.
    if np.all(tx_idx == tx_idx[0]) and np.all(rx_idx == rx_idx[0]):
        spectrum, f_spec = _load_calibrated_spectrum(
            float(_TX_BINS[tx_idx[0]]), float(_RX_BINS[rx_idx[0]])
        )
        return _spectrum_to_h_f(spectrum, f_spec), f_buf

    # Per-frame path: build a 2D filter, loading each unique bin pair once.
    cache: dict[tuple[int, int], NDArray[np.complex64]] = {}
    h_fs = np.empty((nfft, tx_temp.size), dtype=np.complex64)
    for col, (ti, ri) in enumerate(zip(tx_idx, rx_idx)):
        key = (int(ti), int(ri))
        if key not in cache:
            spectrum, f_spec = _load_calibrated_spectrum(
                float(_TX_BINS[ti]), float(_RX_BINS[ri])
            )
            cache[key] = _spectrum_to_h_f(spectrum, f_spec)
        h_fs[:, col] = cache[key]
    return h_fs, f_buf


def _make_chirp(ct: str = "ideal",
                nfft: int = PARAMS.nfft,
                dt: float = PARAMS.dt,
                tau: float = PARAMS.tau,
                bw: float = PARAMS.bw,
                chirp_direction: int = -1,
                ) -> NDArray[np.complexfloating]:
    """Create a synthetic reference chirp for range compression.

    Calibrated chirps are handled separately by ``_calibrated_h_f``;
    this helper is only used for the synthetic path.

    Args:
        ct: Chirp type. Only ``"complex"`` is supported.
        nfft: FFT length.
        dt: Sample spacing in seconds.
        tau: Chirp duration in seconds.
        bw: Bandwidth in Hz.
        chirp_direction: ``+1`` for up-chirp, ``-1`` for down-chirp.

    Returns:
        Complex baseband chirp of length ``nfft``.

    Raises:
        NotImplementedError: If ``ct`` is not ``"complex"``.
    """
    ct = ct.lower()
    if ct == "ideal":
        chirp, _, _ = create_complex_baseband_chirp(
            nfft, dt, tau, bw, chirp_direction=chirp_direction)
    else:
        raise NotImplementedError(
            f"Chirp Type {ct} not understood for the synthetic path. "
            f"Calibrated chirps are dispatched separately."
        )
    return chirp


def range_compression(science_data: NDArray,
                      *,
                      filter_type: str = "MATCHED",
                      chirp_type: str = "IDEAL",
                      nfft: int = PARAMS.nfft,
                      dt: float = PARAMS.dt,
                      tau: float = PARAMS.tau,
                      bw: float = PARAMS.bw,
                      f_cen: float = PARAMS.fs - PARAMS.f_cen[0],
                      window_type: str = "hanning",
                      alpha: float = 1.0,
                      eps: float = 1e-12,
                      tx_temp: NDArray | None = None,
                      rx_temp: NDArray | None = None,
                      ) -> NDArray:
    """Apply range compression to SHARAD EDR echo data.

    Shifts the input to complex baseband, builds a matched or
    inverse filter from a reference chirp, applies a spectral
    window, and returns the range-compressed time-domain data
    along with the phase shift used for the baseband conversion.

    Args:
        science_data: Echo data array, shape ``(n_samp, n_cols)``.
        filter_type: ``"MATCHED"`` or ``"INVERSE"``.
        chirp_type: Chirp type. Only ``"COMPLEX"`` is currently
            supported.
        nfft: FFT length.
        dt: Sample spacing in seconds.
        tau: Chirp duration in seconds.
        bw: Bandwidth in Hz.
        f_cen: Center frequency in Hz for the baseband shift.
            Defaults to SHARAD's offset frequency of
            ``(80/3 - 20) MHz ≈ 6.667 MHz``.
        window_type: Spectral window type.
        alpha: Tukey window taper fraction.
        eps: Regularization threshold for inverse filtering.
            Bins with ``|S(f)| < eps`` are zeroed.
        tx_temp: Temperature of the antenna at transmission
        rx_temp: Temperature of the antenna at reception

    Returns:
        ``data``: Range-compressed echo data, shape
            ``(nfft, n_cols)``, complex.

    Raises:
        ValueError: If any required parameter is invalid.
    """
    ft = filter_type.upper()
    ct = chirp_type.upper()
    wt = window_type.lower()
    assert_gt_0(nfft)
    assert_gt_0(dt)
    assert_gt_0(tau)

    # Build filter
    if ct == "CALIBRATED":
        # Bypass the chirp->IFFT->FFT roundtrip for calibrated chirps:
        # build h_f directly from the PDS spectrum at the in-band buffer
        # indices. h_f is 1D when all frames share a (TX, RX) bin, or 2D
        # of shape (nfft, n_frames) when bins vary along-track.
        if tx_temp is None or rx_temp is None:
            raise ValueError(
                "Calibrated chirps require both tx_temp and rx_temp"
            )
        h_f, f = _calibrated_h_f(
            tx_temp, rx_temp,
            filter_type=ft, dt=dt, nfft=nfft, bw=bw, eps=eps,
        )
    else:
        #
        # SHARAD's chirp
        #
        # Since we are electing to keep the negative frequencies in the spectra (see note below), the default SHARAD
        # chirp is an !!!UPCHIRP!!!. This is opposite of the actual emitted chirp (DOWNCHIRP)
        #
        chirp = _make_chirp(ct=ct, nfft=nfft, dt=dt, tau=tau, bw=bw,
                            chirp_direction=-1)
        h_f, f = create_filter(filter_type=ft, chirp=chirp, dt=dt, bw=bw,
                               f_cen=f_cen, eps=eps)

    # Build window
    if not check_monotonically_increasing(f):
        f = fftshift(f)
    wnd = form_window_bandlimited(f, bw, window_type=wt, f_cen=0.0, alpha=alpha, standard_order=True)
    #
    # Shift data to Complex Baseband
    #
    # For SHARAD data, we default to retaining the NEGATIVE frequencies. This has the benefit of partially
    # undoing the full spectral aliasing caused by the undersampling. This means that the low frequencies
    # in the resulting spectra are the emitted lows (recevied highs) and the high frequencies are the emitted
    # highs (received lows).
    #
    # One draw back is that the phase, thus Doppler history are CONJUGATED!
    #
    science_data, phase_shift = to_complex_baseband(science_data, nfft=nfft, dt=dt, f_cen=f_cen, bw=bw,
                                                    shift_direction=+1, output_time=False)
    science_data = _range_compress(science_data, h_f=h_f, window=wnd, input_time=False, output_time=True)
    #science_data = science_data * np.conj(phase_shift)[:, np.newaxis] #TODO Matt you turned this off for testing!
    return science_data