"""Closed-loop NED test for target tracking with camera-FOV CBF safety.

Compares PID and Fossen-inspired 3D ALOS tracking with/without the FOV CBF.
The leader moves independently. The controlled vehicle translates in NED and
executes prescribed yaw/pitch turns; a body-fixed forward camera observes the
leader in FRD. The CBF modifies only the body-FRD translational velocity.

Run:
    pip install -r requirements.txt
    python test_perception_safety.py
    python test_perception_safety.py --no-show
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
if __name__ == "__main__":
    import os
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scripts.target_tracking import TargetTrackingController, TargetVelocityEstimator
from scripts.cbf_perception import FOVCBFVelocityFilter


# ------------------------------ Configuration ----------------------------
DT = 0.1
SIM_TIME = 50.0
N_STEPS = int(round(SIM_TIME / DT))

# Camera full angles. Unequal horizontal and vertical FOV are intentional.
HORIZONTAL_FOV_DEG = 50.0
VERTICAL_FOV_DEG = 36.0
FOV_MARGIN_DEG = (3.0, 2.0)  # (horizontal, vertical) margin on each edge
EFFECTIVE_HALF_H_DEG = HORIZONTAL_FOV_DEG / 2 - FOV_MARGIN_DEG[0]
EFFECTIVE_HALF_V_DEG = VERTICAL_FOV_DEG / 2 - FOV_MARGIN_DEG[1]
MIN_CAMERA_DEPTH = 0.5

INITIAL_REL_FRD = np.array([12.0, 4.3, 2.5], dtype=float)
# Deliberately near/beyond the physical right FOV edge. The perception CBF
# must prioritize visibility over this nominal tracking setpoint.
DESIRED_REL_FRD = np.array([5.0, 5.7, 2.5], dtype=float)
MAX_SPEED = 2.8
VELOCITY_TAU = 0.06
MAX_ACCELERATION = 20.0
TARGET_ESTIMATOR_TAU = 0.10
POSITION_NOISE_STD_M = 0.0
RANDOM_SEED = 11

# FOV CBF parameters
FOV_CBF_ALPHA = 0.7
FOV_VELOCITY_FILTER_TAU = 0.02
FOV_QP_MAX_SPEED = MAX_SPEED

# Scenario checks (practical numerical checks, not formal proofs)
TAIL_FRACTION = 0.20
TAIL_RMS_LIMIT_M = 3.0
FINAL_ERROR_LIMIT_M = 4.0
MAX_ERROR_LIMIT_M = 50.0

OUTPUT_DIR = Path(__file__).resolve().parent / "perception_safety_results"


def rotation_body_to_ned(yaw: float, pitch: float) -> np.ndarray:
    """FRD-to-NED rotation, 3-2-1 yaw-pitch-roll with roll=0."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([
        [cp * cy, -sy, sp * cy],
        [cp * sy,  cy, sp * sy],
        [-sp,    0.0, cp],
    ], dtype=float)


def _smoothstep_cos(t: float, start: float, duration: float) -> tuple[float, float]:
    """Raised-cosine transition 0->1 and its time derivative."""
    s = (t - start) / duration
    if s <= 0.0:
        return 0.0, 0.0
    if s >= 1.0:
        return 1.0, 0.0
    value = 0.5 * (1.0 - math.cos(math.pi * s))
    derivative = 0.5 * math.pi / duration * math.sin(math.pi * s)
    return value, derivative


def controlled_attitude(t: float) -> tuple[float, float, np.ndarray]:
    """Prescribed yaw/pitch maneuver and body angular rates.

    This exogenous turn profile tests perception safety during body rotations.
    A production system may instead supply its measured IMU body rates.
    """
    turn1, turn1_dot = _smoothstep_cos(t, 12.0, 8.0)
    turn2, turn2_dot = _smoothstep_cos(t, 52.0, 8.0)
    yaw = (
        0.72 * turn1 - 0.72 * turn2
        + 0.07 * math.sin(0.055 * t)
    )
    yaw_dot = (
        0.72 * turn1_dot - 0.72 * turn2_dot
        + 0.07 * 0.055 * math.cos(0.055 * t)
    )

    pitch1, pitch1_dot = _smoothstep_cos(t, 33.0, 8.0)
    pitch2, pitch2_dot = _smoothstep_cos(t, 62.0, 8.0)
    pitch = (
        0.25 * pitch1 - 0.25 * pitch2
        + 0.035 * math.sin(0.065 * t)
    )
    pitch_dot = (
        0.25 * pitch1_dot - 0.25 * pitch2_dot
        + 0.035 * 0.065 * math.cos(0.065 * t)
    )

    # Body angular velocity for ZYX Euler angles with roll=0.
    omega_body = np.array([
        -yaw_dot * math.sin(pitch),
        pitch_dot,
        yaw_dot * math.cos(pitch),
    ])
    return yaw, pitch, omega_body


@dataclass
class LeaderVehicleNED:
    position_ned: np.ndarray

    def velocity_ned(self, t: float) -> np.ndarray:
        # Independent smooth maneuver: horizontal course changes and modest
        # vertical motion. It does not use controlled-vehicle state.
        return np.array([
            0.5 * math.cos(0.012 * t),
            0.1 * math.sin(0.012 * t) + 0.010,
            0.055 * math.sin(0.055 * t),
        ])

    def step(self, t: float, dt: float) -> None:
        self.position_ned = self.position_ned + self.velocity_ned(t) * dt


@dataclass
class ControlledVehicleNED:
    position_ned: np.ndarray
    velocity_tau: float = VELOCITY_TAU
    max_acceleration: float = MAX_ACCELERATION

    def __post_init__(self):
        self.velocity_ned = np.zeros(3, dtype=float)

    def velocity_body(self, R_body_to_ned: np.ndarray) -> np.ndarray:
        return R_body_to_ned.T @ self.velocity_ned

    def step(self, velocity_command_body: np.ndarray,
             R_body_to_ned: np.ndarray, dt: float) -> None:
        command_ned = R_body_to_ned @ np.asarray(velocity_command_body, dtype=float)
        accel_ned = (command_ned - self.velocity_ned) / self.velocity_tau
        accel_norm = float(np.linalg.norm(accel_ned))
        if accel_norm > self.max_acceleration:
            accel_ned *= self.max_acceleration / accel_norm
        self.velocity_ned += accel_ned * dt
        self.position_ned += self.velocity_ned * dt


def leader_initial_position() -> np.ndarray:
    return np.array([45.0, 0.0, -4.0], dtype=float)


def initial_vehicle_position() -> np.ndarray:
    R0 = rotation_body_to_ned(*controlled_attitude(0.0)[:2])
    return leader_initial_position() - R0 @ INITIAL_REL_FRD


def make_tracker(mode: str) -> TargetTrackingController:
    if mode == "pid":
        return TargetTrackingController(
            controller="pid",
            kp=(0.85, 0.85, 0.85),
            ki=(0.035, 0.035, 0.035),
            kd=(0.18, 0.18, 0.18),
            max_speed=MAX_SPEED,
            integral_limit=4.0,
            derivative_filter=0.25,
        )
    if mode == "alos":
        return TargetTrackingController(
            controller="alos",
            delta_h=4.0,
            delta_v=3.0,
            gamma_h=0.035,
            gamma_v=0.035,
            k_along=0.35,
            beta_inner=0.35,
            beta_outer=0.65,
            alpha_inner=0.35,
            alpha_outer=0.65,
            max_speed=MAX_SPEED,
        )
    raise ValueError("mode must be 'pid' or 'alos'")


@dataclass
class RunResult:
    label: str
    mode: str
    use_cbf: bool
    time: np.ndarray
    leader_ned: np.ndarray
    vehicle_ned: np.ndarray
    desired_vehicle_ned: np.ndarray
    relative_frd: np.ndarray
    tracking_error_frd: np.ndarray
    nominal_velocity_frd: np.ndarray
    safe_velocity_frd: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    horizontal_margin_deg: np.ndarray
    vertical_margin_deg: np.ndarray
    inside_physical_fov: np.ndarray
    inside_margined_fov: np.ndarray
    cbf_feasible: np.ndarray
    cbf_correction: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    metrics: dict


def simulate(mode: str, use_cbf: bool) -> RunResult:
    rng = np.random.default_rng(RANDOM_SEED)
    leader = LeaderVehicleNED(leader_initial_position())
    vehicle = ControlledVehicleNED(initial_vehicle_position())
    target_estimator = TargetVelocityEstimator(tau=TARGET_ESTIMATOR_TAU)
    tracker = make_tracker(mode)

    fov_filter = FOVCBFVelocityFilter(
        horizontal_fov_deg=HORIZONTAL_FOV_DEG,
        vertical_fov_deg=VERTICAL_FOV_DEG,
        fov_margin_deg=FOV_MARGIN_DEG,
        alpha=FOV_CBF_ALPHA,
        min_depth=MIN_CAMERA_DEPTH,
        barrier_buffer_m=0.25,
        max_speed=FOV_QP_MAX_SPEED,
        velocity_weights=(1.0, 1.0, 1.0),
        velocity_filter_tau=FOV_VELOCITY_FILTER_TAU,
    )

    camera_was_available = True
    last_nominal = np.zeros(3, dtype=float)
    last_safe = np.zeros(3, dtype=float)

    times = np.arange(N_STEPS, dtype=float) * DT
    leader_log = np.zeros((N_STEPS, 3))
    vehicle_log = np.zeros((N_STEPS, 3))
    desired_locus_log = np.zeros((N_STEPS, 3))
    relative_log = np.zeros((N_STEPS, 3))
    error_log = np.zeros((N_STEPS, 3))
    nominal_log = np.zeros((N_STEPS, 3))
    safe_log = np.zeros((N_STEPS, 3))
    az_log = np.zeros(N_STEPS)
    el_log = np.zeros(N_STEPS)
    hmargin_log = np.zeros(N_STEPS)
    vmargin_log = np.zeros(N_STEPS)
    physical_fov_log = np.zeros(N_STEPS, dtype=bool)
    marginal_fov_log = np.zeros(N_STEPS, dtype=bool)
    feasible_log = np.ones(N_STEPS, dtype=bool)
    correction_log = np.zeros(N_STEPS)
    yaw_log = np.zeros(N_STEPS)
    pitch_log = np.zeros(N_STEPS)

    for k, t in enumerate(times):
        yaw, pitch, omega_body = controlled_attitude(t)
        R_b_n = rotation_body_to_ned(yaw, pitch)
        v_actual_body = vehicle.velocity_body(R_b_n)

        # Camera provides the leader's relative pose only while the target is
        # inside the physical camera frustum. The simulator uses ground truth
        # only to model whether a detection exists and to score the trajectory.
        rel_true = R_b_n.T @ (leader.position_ned - vehicle.position_ned)
        physical_state = fov_filter.fov_state(rel_true)
        measurement_available = physical_state["inside_physical_fov"]
        adaptive_before = None

        if measurement_available:
            rel_meas = rel_true.copy()
            if POSITION_NOISE_STD_M > 0:
                rel_meas += rng.normal(0.0, POSITION_NOISE_STD_M, size=3)

            v_target_est = target_estimator.update(
                target_rel_body=rel_meas,
                vehicle_velocity_body=v_actual_body,
                body_rates_frd=omega_body,
                dt=DT,
            )

            if v_target_est is None:
                v_nom = np.zeros(3)
            else:
                adaptive_before = (tracker.beta_hat, tracker.alpha_hat)
                v_nom = np.asarray(tracker.compute(
                    target_rel_body=rel_meas,
                    desired_rel_body=DESIRED_REL_FRD,
                    target_velocity_body=v_target_est,
                    body_rates_frd=omega_body,
                    dt=DT,
                ), dtype=float)
        else:
            # Simulate loss of camera pose: no new target pose enters the
            # tracking stack. Hold the last command as a simple baseline
            # behavior; reset motion-estimator/controller memory on loss.
            if camera_was_available:
                target_estimator.reset()
                tracker.reset()
            v_nom = last_nominal.copy()

        if use_cbf and measurement_available:
            v_safe_tuple, info = fov_filter.filter_velocity(
                v_nom=v_nom,
                leader_rel_body=rel_meas,
                vehicle_velocity_body=v_actual_body,
                body_rates_frd=omega_body,
                dt=DT,
                return_info=True,
            )
            v_safe = np.asarray(v_safe_tuple, dtype=float)
            feasible_log[k] = info["feasible"]
            correction_log[k] = np.linalg.norm(v_safe - v_nom)
            if mode == "alos" and adaptive_before is not None and correction_log[k] > 1e-3:
                # Do not learn a commanded FOV-avoidance maneuver as a
                # persistent crab/attack-angle disturbance.
                tracker.beta_hat, tracker.alpha_hat = adaptive_before
        elif use_cbf and not measurement_available:
            # The CBF cannot guarantee visibility after the camera has already
            # lost the leader. Hold the most recent command and flag this event.
            v_safe = last_safe.copy()
            feasible_log[k] = False
        else:
            v_safe = v_nom.copy()
            info = physical_state

        if measurement_available:
            last_nominal = v_nom.copy()
            last_safe = v_safe.copy()
        camera_was_available = measurement_available
        fov_state = fov_filter.fov_state(rel_true)
        leader_log[k] = leader.position_ned
        vehicle_log[k] = vehicle.position_ned
        desired_locus_log[k] = leader.position_ned - R_b_n @ DESIRED_REL_FRD
        relative_log[k] = rel_true
        error_log[k] = rel_true - DESIRED_REL_FRD
        nominal_log[k] = v_nom
        safe_log[k] = v_safe
        az_log[k] = fov_state["azimuth_deg"]
        el_log[k] = fov_state["elevation_deg"]
        hmargin_log[k] = fov_state["horizontal_margin_deg"]
        vmargin_log[k] = fov_state["vertical_margin_deg"]
        physical_fov_log[k] = fov_state["inside_physical_fov"]
        marginal_fov_log[k] = fov_state["inside_margined_fov"]
        yaw_log[k] = yaw
        pitch_log[k] = pitch

        # Propagate translational NED kinematics and independent leader.
        vehicle.step(v_safe, R_b_n, DT)
        leader.step(t, DT)

    error_norm = np.linalg.norm(error_log, axis=1)
    tail_start = int((1.0 - TAIL_FRACTION) * N_STEPS)
    tail_rms = float(np.sqrt(np.mean(np.sum(error_log[tail_start:]**2, axis=1))))
    final_error = float(error_norm[-1])
    peak_error = float(np.max(error_norm))
    bounded = bool(np.isfinite(error_log).all() and peak_error < MAX_ERROR_LIMIT_M)
    converged = bool(bounded and tail_rms < TAIL_RMS_LIMIT_M and final_error < FINAL_ERROR_LIMIT_M)

    first_fov_loss = np.flatnonzero(~physical_fov_log)
    first_margin_loss = np.flatnonzero(~marginal_fov_log)
    metrics = {
        "tail_rms_m": tail_rms,
        "final_error_m": final_error,
        "peak_error_m": peak_error,
        "bounded": bounded,
        "tracking_converged": converged,
        "physical_fov_violation_samples": int(np.count_nonzero(~physical_fov_log)),
        "margined_fov_violation_samples": int(np.count_nonzero(~marginal_fov_log)),
        "first_physical_fov_loss_s": float(times[first_fov_loss[0]]) if len(first_fov_loss) else None,
        "minimum_horizontal_margin_deg": float(np.min(hmargin_log)),
        "minimum_vertical_margin_deg": float(np.min(vmargin_log)),
        "cbf_active_samples": int(np.count_nonzero(correction_log > 1e-3)),
        "cbf_infeasible_samples": int(np.count_nonzero(~feasible_log)),
        "mean_cbf_correction_mps": float(np.mean(correction_log)),
        "max_cbf_correction_mps": float(np.max(correction_log)),
    }
    label = f"{mode.upper()} + FOV-CBF" if use_cbf else f"{mode.upper()} (no FOV-CBF)"
    return RunResult(
        label, mode, use_cbf, times, leader_log, vehicle_log, desired_locus_log,
        relative_log, error_log, nominal_log, safe_log, az_log, el_log,
        hmargin_log, vmargin_log, physical_fov_log, marginal_fov_log,
        feasible_log, correction_log, yaw_log, pitch_log, metrics,
    )


def _draw_camera_frustum(ax, position, R_b_n, horizontal_half_deg,
                         vertical_half_deg, depth=12.0, color="#08519c",
                         alpha=0.16):
    """Draw a body-forward rectangular camera frustum wireframe in NED."""
    th = math.tan(math.radians(horizontal_half_deg))
    tv = math.tan(math.radians(vertical_half_deg))
    # Frustum corners in body FRD at far depth; vertical FOV uses sqrt(x^2+y^2).
    corners_body = np.array([
        [depth, -depth * th, -math.hypot(depth, depth * th) * tv],
        [depth,  depth * th, -math.hypot(depth, depth * th) * tv],
        [depth,  depth * th,  math.hypot(depth, depth * th) * tv],
        [depth, -depth * th,  math.hypot(depth, depth * th) * tv],
    ])
    corners_ned = position + (R_b_n @ corners_body.T).T
    p = np.asarray(position)
    for c in corners_ned:
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                color=color, alpha=0.75, linewidth=0.8)
    loop = np.vstack((corners_ned, corners_ned[0]))
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color,
            alpha=0.75, linewidth=0.9)
    poly = Poly3DCollection([corners_ned], alpha=alpha, facecolor=color,
                            edgecolor="none")
    ax.add_collection3d(poly)


def plot_results(results: list[RunResult], show: bool) -> list[Path]:
    paths = []
    style = {
        ("pid", False): ("#9ecae1", ":", 1.4),
        ("pid", True): ("#08519c", "-", 2.4),
        ("alos", False): ("#fdae6b", ":", 1.4),
        ("alos", True): ("#a63603", "-", 2.4),
    }
    first = results[0]

    # 1) NED 3D trajectories and camera frusta.
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(first.leader_ned[:, 0], first.leader_ned[:, 1], first.leader_ned[:, 2],
            color="black", linewidth=2.2, label="Autonomous leader trajectory")
    for result in results:
        color, ls, lw = style[(result.mode, result.use_cbf)]
        ax.plot(result.vehicle_ned[:, 0], result.vehicle_ned[:, 1], result.vehicle_ned[:, 2],
                color=color, linestyle=ls, linewidth=lw, label=result.label)
    ax.plot(first.desired_vehicle_ned[:, 0], first.desired_vehicle_ned[:, 1],
            first.desired_vehicle_ned[:, 2], color="gray", linestyle="--",
            linewidth=1.1, alpha=0.8, label="Desired follower locus")

    # Camera frusta are shown for the protected PID run at representative times.
    pid_cbf = next(r for r in results if r.mode == "pid" and r.use_cbf)
    for t_snapshot in (0.0, 20.0, 40.0, 65.0, 85.0):
        idx = min(N_STEPS - 1, int(round(t_snapshot / DT)))
        yaw, pitch, _ = controlled_attitude(float(pid_cbf.time[idx]))
        R_b_n = rotation_body_to_ned(yaw, pitch)
        _draw_camera_frustum(
            ax, pid_cbf.vehicle_ned[idx], R_b_n,
            EFFECTIVE_HALF_H_DEG, EFFECTIVE_HALF_V_DEG,
            depth=13.0, color="#3182bd", alpha=0.055,
        )
        ax.scatter(*pid_cbf.vehicle_ned[idx], color="#08519c", s=12)

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Down (m)")
    ax.set_title(
        "NED closed-loop leader tracking with camera-FOV CBF\n"
        "Blue frusta: effective camera visibility region at selected times"
    )
    ax.invert_zaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = OUTPUT_DIR / "fov_cbf_ned_trajectories.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    paths.append(p)

    # 2) FRD tracking errors.
    fig_err, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    component_labels = ["Forward error $e_x$ (m)", "Right error $e_y$ (m)", "Down error $e_z$ (m)"]
    for j, ax in enumerate(axes):
        for result in results:
            color, ls, lw = style[(result.mode, result.use_cbf)]
            ax.plot(result.time, result.tracking_error_frd[:, j],
                    color=color, linestyle=ls, linewidth=lw, label=result.label)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylabel(component_labels[j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig_err.suptitle("Leader-relative tracking errors in body FRD")
    fig_err.tight_layout()
    p = OUTPUT_DIR / "fov_cbf_frd_tracking_errors.png"
    fig_err.savefig(p, dpi=190, bbox_inches="tight")
    paths.append(p)

    # 3) Perception-safety margins and azimuth/elevation.
    fig_fov, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for result in results:
        color, ls, lw = style[(result.mode, result.use_cbf)]
        axes[0].plot(result.time, result.azimuth_deg, color=color,
                     linestyle=ls, linewidth=lw, label=result.label)
        axes[1].plot(result.time, result.elevation_deg, color=color,
                     linestyle=ls, linewidth=lw, label=result.label)
    axes[0].axhline(EFFECTIVE_HALF_H_DEG, color="black", linestyle="--", linewidth=1.0,
                    label="Effective horizontal FOV edge")
    axes[0].axhline(-EFFECTIVE_HALF_H_DEG, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(EFFECTIVE_HALF_V_DEG, color="black", linestyle="--", linewidth=1.0,
                    label="Effective vertical FOV edge")
    axes[1].axhline(-EFFECTIVE_HALF_V_DEG, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Leader azimuth (deg)")
    axes[1].set_ylabel("Leader elevation (deg)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
    fig_fov.suptitle(
        f"Camera visibility angles (full FOV: {HORIZONTAL_FOV_DEG:.0f}° H × "
        f"{VERTICAL_FOV_DEG:.0f}° V; effective half-angles: "
        f"±{EFFECTIVE_HALF_H_DEG:.0f}° / ±{EFFECTIVE_HALF_V_DEG:.0f}°)"
    )
    fig_fov.tight_layout()
    p = OUTPUT_DIR / "fov_cbf_visibility_angles.png"
    fig_fov.savefig(p, dpi=190, bbox_inches="tight")
    paths.append(p)

    # 4) CBF correction and FOV margins for protected cases.
    fig_diag, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for result in results:
        if not result.use_cbf:
            continue
        color = style[(result.mode, True)][0]
        axes[0].plot(result.time, result.horizontal_margin_deg,
                     color=color, linestyle="-", linewidth=1.4,
                     label=f"{result.mode.upper()} horizontal margin")
        axes[0].plot(result.time, result.vertical_margin_deg,
                     color=color, linestyle="--", linewidth=1.4,
                     label=f"{result.mode.upper()} vertical margin")
        axes[1].plot(result.time, result.cbf_correction,
                     color=color, linewidth=1.4,
                     label=f"{result.mode.upper()} CBF correction")
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Margin to effective FOV edge (deg)")
    axes[1].set_ylabel(r"$\|v_{safe}-v_{nom}\|$ (m/s)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
    fig_diag.suptitle("Perception CBF safety-filter diagnostics")
    fig_diag.tight_layout()
    p = OUTPUT_DIR / "fov_cbf_safety_diagnostics.png"
    fig_diag.savefig(p, dpi=190, bbox_inches="tight")
    paths.append(p)

    print("Generated plots:")
    for path in paths:
        print(f"  {path}")
    if show:
        plt.show()
    else:
        plt.close("all")
    return paths


def save_metrics(results: list[RunResult]) -> Path:
    path = OUTPUT_DIR / "fov_cbf_simulation_metrics.csv"
    fields = [
        "case", "tail_rms_m", "final_error_m", "peak_error_m",
        "physical_fov_violation_samples", "margined_fov_violation_samples",
        "first_physical_fov_loss_s", "minimum_horizontal_margin_deg",
        "minimum_vertical_margin_deg", "cbf_active_samples",
        "cbf_infeasible_samples", "mean_cbf_correction_mps",
        "max_cbf_correction_mps", "tracking_converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {"case": r.label}
            row.update({k: r.metrics[k] for k in fields[1:]})
            writer.writerow(row)
    print(f"Saved metrics: {path}")
    return path


def print_summary(results: list[RunResult]) -> bool:
    print("\nCamera-FOV perception-safety closed-loop results")
    print("=" * 132)
    print(
        f"{'Case':<20} {'Tail RMS':>9} {'Final err':>10} "
        f"{'FOV lost':>10} {'Margin lost':>12} {'Min H marg':>11} "
        f"{'Min V marg':>11} {'CBF active':>11} {'QP infeas':>10}"
    )
    print("-" * 132)
    for r in results:
        m = r.metrics
        print(
            f"{r.label:<20} {m['tail_rms_m']:>9.3f} {m['final_error_m']:>10.3f} "
            f"{m['physical_fov_violation_samples']:>10d} "
            f"{m['margined_fov_violation_samples']:>12d} "
            f"{m['minimum_horizontal_margin_deg']:>11.2f} "
            f"{m['minimum_vertical_margin_deg']:>11.2f} "
            f"{m['cbf_active_samples']:>11d} {m['cbf_infeasible_samples']:>10d}"
        )
    print("-" * 132)

    protected = [r for r in results if r.use_cbf]
    baseline = [r for r in results if not r.use_cbf]
    safety_ok = all(
        r.metrics["physical_fov_violation_samples"] == 0
        and r.metrics["margined_fov_violation_samples"] == 0
        and r.metrics["cbf_infeasible_samples"] == 0
        for r in protected
    )
    baseline_challenged = all(
        r.metrics["physical_fov_violation_samples"] > 0 for r in baseline
    )
    tracking_ok = all(r.metrics["tracking_converged"] for r in protected)
    print("Protected cases remain inside effective FOV:", "PASS" if safety_ok else "CHECK / FAIL")
    print("Unfiltered cases demonstrate FOV loss:", "PASS" if baseline_challenged else "CHECK / FAIL")
    print("Protected tracking recovery:", "PASS" if tracking_ok else "CHECK / FAIL")
    print("Note: this is a finite-step kinematic simulation, not a formal stability proof.")
    print("Note: CBF invariance requires an initially safe state and an adequate relative-motion estimate.")
    return safety_ok and baseline_challenged and tracking_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows")
    args = parser.parse_args()

    results = [
        simulate("pid", False),
        simulate("pid", True),
        simulate("alos", False),
        simulate("alos", True),
    ]
    plot_results(results, show=not args.no_show)
    save_metrics(results)
    passed = print_summary(results)
    if not passed:
        raise SystemExit("Perception-safety scenario did not meet all configured checks")


if __name__ == "__main__":
    main()
