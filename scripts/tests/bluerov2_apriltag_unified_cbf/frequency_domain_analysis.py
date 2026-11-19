"""Frequency-domain analysis for :class:`BodyVelocityTargetTracker`.

The tracker itself is static: it has no internal states, so its small-signal
frequency response is constant with frequency.  This module linearizes the
tracker about a specified operating point and optionally adds a first-order
vehicle velocity model to show the bandwidth seen by the closed-loop command.

Run, for example:

    python frequency_domain_analysis.py --plot tracker_bode.png

The input ordering for the analytical transfer matrix is
``[dx, dy, dz, dvx, dvy, dvz, dpsi, dleader_r]`` and the output ordering is
``[du, dv, dw, dr]``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


INPUT_NAMES = ("dx", "dy", "dz", "dvx", "dvy", "dvz", "dpsi", "dleader_r")
OUTPUT_NAMES = ("du", "dv", "dw", "dr")


def linearized_tracker_matrix(
    desired_rel_camera=(6.0, 0.0, 0.0),
    kp=(0.27, 0.40, 0.32),
    yaw_gain=0.75,
    bearing_yaw_gain=0.08,
    camera_offset_body=(0.22, 0.0, 0.0),
    operating_point=None,
):
    """Return the real small-signal matrix ``Dq = K Dx``.

    ``operating_point`` is ``(tag_position, relative_heading)``.  The
    default is the nominal stand-off, zero bearing, and zero relative yaw.
    The returned matrix describes the unsaturated controller; saturation makes
    the local response piecewise and should be analyzed at each operating
    point.
    """
    desired = np.asarray(desired_rel_camera, dtype=float)
    gains = np.asarray(kp, dtype=float)
    offset = np.asarray(camera_offset_body, dtype=float)
    if desired.shape != (3,) or gains.shape != (3,) or offset.shape != (3,):
        raise ValueError("desired_rel_camera, kp and camera_offset_body must be 3-vectors")
    if operating_point is None:
        position = desired
        relative_heading = 0.0
    else:
        position = np.asarray(operating_point[0], dtype=float)
        relative_heading = float(operating_point[1])
    if position.shape != (3,) or position[0] <= 0.0:
        raise ValueError("operating-point tag position must be a 3-vector with x > 0")

    x, y = position[:2]
    rho2 = x * x + y * y
    if rho2 <= 1e-12:
        raise ValueError("operating-point bearing is undefined at the camera origin")
    # d atan2(y, x) = [-y, x] / (x^2 + y^2).
    dbearing = np.array([-y / rho2, x / rho2])

    # d(relative yaw) is already represented by dpsi in the input vector.
    d_r = np.zeros(8)
    d_r[1] = bearing_yaw_gain * dbearing[1]
    d_r[6] = float(yaw_gain)
    d_r[7] = 1.0

    p_body = offset + position
    r0 = float(yaw_gain * relative_heading + bearing_yaw_gain * math.atan2(y, x))
    K = np.zeros((4, 8))

    # u = vt_x + kpx*(x-desired_x) + r*p_body_y
    K[0, 0] = gains[0]
    K[0, 1] = r0
    K[0] += p_body[1] * d_r
    K[0, 3] += 1.0

    # v = vt_y + kpy*(y-desired_y) - r*p_body_x
    K[1, 0] = -r0
    K[1, 1] += gains[1]
    K[1] -= p_body[0] * d_r
    K[1, 4] += 1.0

    K[2, 2] = gains[2]
    K[2, 5] = 1.0
    K[3] = d_r
    return K


def first_order_velocity_response(frequencies_hz, time_constants=(0.8, 0.8, 0.8, 0.5)):
    """Return diagonal actuator/vehicle response ``H(j*2*pi*f)``."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    tau = np.asarray(time_constants, dtype=float)
    if tau.shape != (4,) or np.any(tau <= 0.0):
        raise ValueError("time_constants must contain four positive values")
    s = 1j * 2.0 * math.pi * frequencies_hz[:, None]
    return 1.0 / (1.0 + s * tau[None, :])


def analytical_frequency_response(
    frequencies_hz,
    tracker_matrix,
    time_constants=None,
):
    """Return ``(frequencies, H)`` where H maps inputs to actual velocities."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    if frequencies_hz.ndim != 1 or np.any(frequencies_hz < 0.0):
        raise ValueError("frequencies_hz must be a one-dimensional non-negative array")
    K = np.asarray(tracker_matrix, dtype=float)
    if K.shape != (4, 8):
        raise ValueError("tracker_matrix must have shape (4, 8)")
    if time_constants is None:
        actuator = np.ones((frequencies_hz.size, 4), dtype=complex)
    else:
        actuator = first_order_velocity_response(frequencies_hz, time_constants)
    # H[f] has output, input axes.
    return frequencies_hz, actuator[:, :, None] * K[None, :, :]


def estimate_fft_response(input_signal, output_signal, sample_time, window=True):
    """Estimate an empirical MIMO response from synchronized log arrays.

    Signals have shape ``(samples, channels)``.  The result contains positive
    frequencies, cross-power based ``H = S_yx / S_xx`` for each input/output
    pair, and coherence.  This assumes one input is excited at a time; for
    simultaneous inputs use the analytical model or a MIMO spectral estimator.
    """
    x = np.asarray(input_signal, dtype=float)
    y = np.asarray(output_signal, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("input_signal and output_signal must be 2-D with equal sample counts")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    n = x.shape[0]
    taper = np.hanning(n) if window else np.ones(n)
    X = np.fft.rfft((x - x.mean(axis=0)) * taper[:, None], axis=0)
    Y = np.fft.rfft((y - y.mean(axis=0)) * taper[:, None], axis=0)
    frequencies = np.fft.rfftfreq(n, sample_time)
    sxx = np.abs(X) ** 2
    syy = np.abs(Y) ** 2
    cross = Y[:, :, None] * np.conj(X[:, None, :])
    response = cross / np.maximum(sxx[:, None, :], np.finfo(float).eps)
    coherence = np.abs(cross) ** 2 / np.maximum(syy[:, :, None] * sxx[:, None, :], np.finfo(float).eps)
    return frequencies, response, np.clip(coherence.real, 0.0, 1.0)


def _plot_response(frequencies, response, output_path):
    import matplotlib.pyplot as plt

    positive = frequencies > 0.0
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        for input_index, input_name in enumerate(INPUT_NAMES):
            transfer = response[:, output_index, input_index]
            axes[0].semilogx(
                frequencies[positive],
                20.0 * np.log10(np.maximum(np.abs(transfer[positive]), 1e-12)),
                label=f"{output_name}/{input_name}",
            )
            axes[1].semilogx(
                frequencies[positive],
                np.unwrap(np.angle(transfer[positive])) * 180.0 / math.pi,
            )
    axes[0].set_ylabel("Magnitude (dB)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[0].grid(True, which="both")
    axes[1].grid(True, which="both")
    axes[0].legend(ncol=4, fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="write a Bode plot (requires matplotlib)")
    parser.add_argument("--fmin", type=float, default=0.01)
    parser.add_argument("--fmax", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=300)
    args = parser.parse_args()
    if args.fmin <= 0.0 or args.fmax <= args.fmin or args.points < 2:
        parser.error("require 0 < fmin < fmax and points >= 2")

    K = linearized_tracker_matrix()
    frequencies = np.geomspace(args.fmin, args.fmax, args.points)
    frequencies, response = analytical_frequency_response(
        frequencies, K, time_constants=(0.8, 0.8, 0.8, 0.5)
    )
    print("Inputs: ", ", ".join(INPUT_NAMES))
    print("Outputs:", ", ".join(OUTPUT_NAMES))
    print("Small-signal tracker matrix (output = K * input):")
    print(K)
    plot_path = f"results_apriltag/{args.plot}"
    if args.plot:
        _plot_response(frequencies, response, plot_path)
        print(f"Saved Bode plot to {plot_path}")


if __name__ == "__main__":
    main()
