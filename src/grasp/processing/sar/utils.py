# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def determine_aperture_bounds(L_n: int,
                               n_col: int,
                               step: int,
                               ) -> Tuple[NDArray[np.integer], list]:
    """Determine source and destination index bounds for each synthetic aperture.

    For each output frame, computes the clipped source range and corresponding
    destination range within a zero-padded aperture buffer, accounting for
    edge effects at the beginning and end of the observation.

    Args:
        L_n (int): Synthetic aperture length in samples.
        n_col (int): Total number of columns in the radar observation.
        step (int): Step size between output frames in samples.

    Returns:
        tuple:
            - **frames** (*NDArray[np.integer]*): Output frame center indices.
              Shape (N,) where N = ceil(n_col / step).
            - **aperture_bounds** (*list of tuple*): Per-frame index bounds, each
              a tuple of (src_start, src_end, dest_start, dest_end) where:
                - src_start, src_end: slice into the input observation array.
                - dest_start, dest_end: slice into the aperture buffer of length L_n.
    """
    aperture_bounds = []
    frames = np.arange(0, n_col, step)
    for frame in frames:
        min_dopp_bin = int(frame - L_n // 2)
        max_dopp_bin = int(frame + L_n // 2 + 1)

        src_start  = max(0, min_dopp_bin)
        src_end    = min(n_col, max_dopp_bin)

        dest_start = max(0, -min_dopp_bin)
        dest_end   = L_n - max(0, max_dopp_bin - n_col)

        aperture_bounds.append((src_start, src_end, dest_start, dest_end))

    return frames, aperture_bounds


def determine_output_frames(n_col: int,
                             step: int) -> int:
    """Determine the number of output frames for a given step size.

    Computes the ceiling division of n_col by step, equivalent to
    ``math.ceil(n_col / step)``.

    Args:
        n_col (int): Total number of columns in the radar observation.
        step (int): Step size between output frames in samples.

    Returns:
        int: Number of output frames.
    """
    return int((n_col + step - 1) // step)


def determine_aperture_step(rho: int | NDArray[np.integer],
                            del_x: int | NDArray[np.integer],
                            os_factor: int) -> int:
    """Determine the aperture step size in samples.

    Computes the step size as the ratio of the mean along-track resolution
    to the mean sample spacing, scaled by the oversampling factor.

    Args:
        rho (int or NDArray[np.integer]): Along-track resolution in meters.
        del_x (int or NDArray[np.integer]): Along-track sample spacing in meters.
        os_factor (int): Oversampling factor.

    Returns:
        int: Aperture step size in samples.

    Raises:
        ValueError: If ``os_factor`` is less than 1.
    """
    if os_factor < 1:
        raise ValueError('Oversampling factor must be >= 1.')
    rho = np.atleast_1d(rho)
    del_x = np.atleast_1d(del_x)
    return np.round((rho.mean()) / del_x.mean() / os_factor).astype(int)


def determine_aperture_resolution(L_df: float | NDArray[np.floating],
                                   lamb: float,
                                   R: float | NDArray[np.floating],
                                   *,
                                   brd: float = 1.0) -> float | NDArray[np.floating]:
    """Determine the along-track SAR resolution.

    Computes the synthetic aperture along-track resolution using the
    standard range-Doppler formulation.

    Args:
        L_df (float or NDArray[np.floating]): Doppler bandwidth-limited
            aperture length in meters.
        lamb (float): Radar wavelength in meters.
        R (float or NDArray[np.floating]): Range to target in meters.
        brd (float, optional): Broadening factor to account for aperture
            weighting. Defaults to 1.0 (rectangular window).

    Returns:
        float or NDArray[np.floating]: Along-track resolution in meters.
        Scalar if inputs are scalar, array if ``L_df`` or ``R`` are arrays.
    """
    return brd * lamb * R / (2 * L_df)


def aperture_range(R_s_ap: NDArray[np.floating],
                   G_0: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute platform-to-target range across a synthetic aperture.

    The fundamental geometric quantity used by RCMC and azimuth
    phase compensation in both time-domain (backscatter) and
    Doppler-domain (range-Doppler) SAR formulations.

    Args:
        R_s_ap: Platform position vectors over the aperture, shape (3, n_ap).
        G_0: Target ground position vector, shape (3,).

    Returns:
        H: Range from each aperture frame to the target, shape (n_ap,).
    """
    return np.linalg.norm(R_s_ap - G_0[:, None], axis=0)


def max_unaliased_aperture(lamb: float,
                           vt: float,
                           pri_eff: float,
                           R0: float,
                           del_x: float,
                           ) -> int:
    """Maximum synthetic aperture length (in samples) before Doppler aliasing.

    Solves the geometric Doppler-bandwidth identity
    ``X = λ R₀ / √((4 Vₜ Δη)² − λ²)`` for the half-aperture extent and
    returns the full-aperture sample count ``floor(2 X / del_x)``.

    Args:
        lamb: Wavelength (m).
        vt: Along-track velocity (m/s); typically the mean over the aperture.
        pri_eff: Effective PRI after presumming (s).
        R0: Reference slant range to target (m); typically the mean over
            the aperture.
        del_x: Along-track sample spacing (m); typically ``vt * pri_eff``.

    Returns:
        Maximum aperture length in input samples before Doppler aliasing.

    Raises:
        ValueError: If ``(4 Vₜ Δη)² ≤ λ²`` — the PRF cannot resolve any
            unambiguous synthetic aperture for the chosen geometry.
    """
    discriminant = (4.0 * vt * pri_eff) ** 2 - lamb ** 2
    if discriminant <= 0.0:
        raise ValueError(
            f"PRF too low to form an unaliased synthetic aperture: "
            f"4*Vt*pri_eff = {4 * vt * pri_eff:.3f} m <= lambda = {lamb:.3f} m. "
            f"Reduce presum or increase PRF."
        )
    X_max = lamb * R0 / np.sqrt(discriminant)
    return int(2.0 * X_max / del_x)


def check_aperture(L_n_requested: int | None,
                   n_col: int,
                   lamb: float,
                   vt: float,
                   pri_eff: float,
                   R0: float,
                   del_x: float,
                   ) -> tuple[int, list[str]]:
    """Validate the requested aperture against physical limits.

    Two limits are evaluated:

    * **Column-count limit** (hard clamp): an aperture cannot be longer
      than the number of input pulses available. Exceeding this is
      clamped silently in the returned length and surfaced as a warning
      message.
    * **Doppler-aliasing limit** (soft warning, no clamp): from
      ``max_unaliased_aperture``. Exceeding it does *not* clamp the
      aperture, because a Doppler-domain matched filter with strong
      edge tapering (Blackman-Harris, Nuttall, Kaiser with large β)
      can still produce usable focus by suppressing the aliased
      contribution at the band edges. Long-mode SHARAD processing
      relies on exactly this behaviour. The caller receives a warning
      so the user is aware the run is operating in the
      window-suppressed regime.

    Args:
        L_n_requested: User-requested aperture length. If ``None``, the
            return value defaults to ``min(n_col, L_n_alias_max)``.
        n_col: Number of input pulses available.
        lamb: Wavelength (m).
        vt: Along-track velocity (m/s); typically the mean over the aperture.
        pri_eff: Effective PRI after presumming (s).
        R0: Reference slant range to target (m).
        del_x: Along-track sample spacing (m).

    Returns:
        A tuple ``(L_n, messages)`` where ``L_n`` is the effective aperture
        length and ``messages`` is a list of human-readable warnings
        suitable for ``warnings.warn`` or verbose printing. The list is
        empty when the request is within both limits.

    Raises:
        ValueError: If the geometry cannot support any unaliased synthetic
            aperture (propagated from ``max_unaliased_aperture``).
    """
    L_n_alias_max = max_unaliased_aperture(lamb, vt, pri_eff, R0, del_x)
    msgs: list[str] = []

    if L_n_requested is None:
        return min(n_col, L_n_alias_max), msgs

    L_n = int(L_n_requested)
    if L_n > n_col:
        msgs.append(
            f"Requested aperture ({L_n_requested} samples) exceeds input "
            f"columns ({n_col}); clamping to {n_col}."
        )
        L_n = n_col
    if L_n > L_n_alias_max:
        msgs.append(
            f"Aperture {L_n} samples exceeds the unaliased Doppler limit "
            f"({L_n_alias_max}) for this geometry. Processing will rely on "
            f"Doppler-window suppression of aliased signal at the band "
            f"edges. For best results, use a window with strong edge "
            f"attenuation (Blackman-Harris, Nuttall, Kaiser beta >= 6). "
            f"To eliminate aliasing, reduce presum to allow a longer "
            f"unaliased aperture."
        )
        # Soft-limit: do not clamp.
    return L_n, msgs