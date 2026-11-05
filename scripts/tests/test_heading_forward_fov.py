"""Closed-loop NED simulation with heading-forward underwater motion + FOV CBF.

Compared cases: PID/ALOS with and without camera-FOV CBF. The controlled
vehicle is surge-dominant: it turns its yaw/pitch toward the commanded course,
tracks speed along body +x, and damps body sway/heave rather than translating
sideways to follow a Cartesian velocity command.

Run:
  pip install -r requirements.txt
  python test_heading_forward_fov.py --no-show
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

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

# ----------------------------- simulation setup -------------------------
DT = 0.1
SIM_TIME = 150.0
N_STEPS = int(round(SIM_TIME / DT))

HORIZONTAL_FOV_DEG = 80.0
VERTICAL_FOV_DEG = 64.0
FOV_MARGIN_DEG = (10.0, 10.0)
EFFECTIVE_HALF_H_DEG = HORIZONTAL_FOV_DEG / 2.0 - FOV_MARGIN_DEG[0]
EFFECTIVE_HALF_V_DEG = VERTICAL_FOV_DEG / 2.0 - FOV_MARGIN_DEG[1]
MIN_CAMERA_DEPTH = 0.5

INITIAL_REL_FRD = np.array([8.0, 4.0, 1.6], dtype=float)
# Nominal tracking point is deliberately outside the physical horizontal FOV.
DESIRED_REL_FRD = np.array([5.0, 5.7, 1.6], dtype=float)
MAX_SPEED = 2.8
TARGET_ESTIMATOR_TAU = 0.18
POSITION_NOISE_STD_M = 0.0
RANDOM_SEED = 17

# Vehicle motion: camera +x is the vehicle's surge / heading axis.
SURGE_TAU = 0.55
MAX_SURGE_ACCEL = 0.65          # m/s^2
MAX_SURGE_SPEED = MAX_SPEED
YAW_TAU = 0.75                  # course-following response
PITCH_TAU = 0.85                # pitch response
MAX_YAW_RATE = math.radians(22.0)   # rad/s
MAX_PITCH_RATE = math.radians(10.0) # rad/s
YAW_RATE_TAU = 0.20             # s
PITCH_RATE_TAU = 0.25            # s
SWAY_DECAY_TAU = 0.18            # s; no routine sway command
HEAVE_DECAY_TAU = 0.30           # s; pitch follows vertical course
MAX_TURN_SURGE_FACTOR = 0.92     # reduce surge slightly during sharp turns
SWAY_DISTURBANCE_ACCEL = 0.025   # small cross-flow disturbance (m/s^2)
HEAVE_DISTURBANCE_ACCEL = 0.010  # small vertical-flow disturbance (m/s^2)

FOV_CBF_ALPHA = 0.1
FOV_CBF_BARRIER_BUFFER_M = 0.45
FOV_CBF_VELOCITY_FILTER_TAU = 0.10
FOV_QP_MAX_SPEED = MAX_SPEED

TAIL_FRACTION = 0.20
TAIL_RMS_LIMIT_M = 5.0
FINAL_ERROR_LIMIT_M = 6.0
MAX_ERROR_LIMIT_M = 60.0
OUTPUT_DIR = Path(__file__).resolve().parent / "heading_forward_fov_results"


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_body_to_ned(yaw: float, pitch: float) -> np.ndarray:
    """FRD-to-NED rotation, 3-2-1 yaw-pitch-roll with roll=0."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([
        [cp * cy, -sy, sp * cy],
        [cp * sy,  cy, sp * sy],
        [-sp,     0.0, cp],
    ], dtype=float)


def body_rates_from_euler_rates(yaw_rate: float, pitch_rate: float,
                                pitch: float) -> np.ndarray:
    """Body FRD angular rates for roll=0, yaw-pitch sequence."""
    return np.array([
        -yaw_rate * math.sin(pitch),
        pitch_rate,
        yaw_rate * math.cos(pitch),
    ], dtype=float)


@dataclass
class LeaderVehicleNED:
    position_ned: np.ndarray

    def velocity_ned(self, t: float) -> np.ndarray:
        # Smooth independent leader maneuver; positive down is NED D.
        return np.array([
            0.72 * math.cos(0.012 * t),
            0.46 * math.sin(0.012 * t) + 0.10,
            0.045 * math.sin(0.045 * t),
        ], dtype=float)

    def step(self, t: float, dt: float) -> None:
        self.position_ned = self.position_ned + self.velocity_ned(t) * dt


@dataclass
class HeadingForwardVehicleNED:
    """Surge-dominant vehicle with rate-limited yaw/pitch course alignment.

    The guidance command is a body-FRD desired velocity vector. Its direction
    is converted into a NED course and pitch reference. The vehicle then moves
    primarily along body +x (surge); body sway and heave are damped to zero.
    """
    position_ned: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    surge_tau: float = SURGE_TAU
    max_surge_accel: float = MAX_SURGE_ACCEL
    max_surge_speed: float = MAX_SURGE_SPEED
    yaw_tau: float = YAW_TAU
    pitch_tau: float = PITCH_TAU
    max_yaw_rate: float = MAX_YAW_RATE
    max_pitch_rate: float = MAX_PITCH_RATE
    yaw_rate_tau: float = YAW_RATE_TAU
    pitch_rate_tau: float = PITCH_RATE_TAU
    sway_decay_tau: float = SWAY_DECAY_TAU
    heave_decay_tau: float = HEAVE_DECAY_TAU

    def __post_init__(self):
        self.surge = 0.0
        self.sway = 0.0
        self.heave = 0.0
        self.yaw_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_ref = self.yaw
        self.pitch_ref = self.pitch
        self.course_error = 0.0
        self.pitch_course_error = 0.0

    def rotation(self) -> np.ndarray:
        return rotation_body_to_ned(self.yaw, self.pitch)

    def velocity_body(self) -> np.ndarray:
        return np.array([self.surge, self.sway, self.heave], dtype=float)

    def velocity_ned(self) -> np.ndarray:
        return self.rotation() @ self.velocity_body()

    def body_rates_frd(self) -> np.ndarray:
        return body_rates_from_euler_rates(self.yaw_rate, self.pitch_rate, self.pitch)

    def step(self, velocity_command_body: np.ndarray, dt: float, t: float = 0.0) -> None:
        """Turn toward commanded course, accelerate in surge, suppress sway."""
        cmd = np.asarray(velocity_command_body, dtype=float)
        if cmd.shape != (3,) or not np.all(np.isfinite(cmd)):
            raise ValueError("velocity_command_body must be a finite 3-vector")

        # Map current-body command into NED, then point vehicle +x along it.
        command_ned = self.rotation() @ cmd
        command_horizontal = math.hypot(command_ned[0], command_ned[1])
        command_speed = float(np.linalg.norm(command_ned))

        if command_speed > 1e-4:
            self.yaw_ref = math.atan2(command_ned[1], command_ned[0])
            self.pitch_ref = math.atan2(-command_ned[2], max(command_horizontal, 1e-6))
        else:
            # Keep current heading when no meaningful motion is requested.
            self.yaw_ref = self.yaw
            self.pitch_ref = self.pitch

        yaw_error = wrap_pi(self.yaw_ref - self.yaw)
        pitch_error = wrap_pi(self.pitch_ref - self.pitch)
        self.course_error = yaw_error
        self.pitch_course_error = pitch_error

        yaw_rate_cmd = float(np.clip(yaw_error / self.yaw_tau,
                                     -self.max_yaw_rate, self.max_yaw_rate))
        pitch_rate_cmd = float(np.clip(pitch_error / self.pitch_tau,
                                       -self.max_pitch_rate, self.max_pitch_rate))
        self.yaw_rate += (yaw_rate_cmd - self.yaw_rate) * min(1.0, dt / self.yaw_rate_tau)
        self.pitch_rate += (pitch_rate_cmd - self.pitch_rate) * min(1.0, dt / self.pitch_rate_tau)
        self.yaw += self.yaw_rate * dt
        self.pitch += self.pitch_rate * dt
        self.pitch = float(np.clip(self.pitch, math.radians(-25), math.radians(25)))

        # A small turn-speed penalty avoids pretending full surge is available
        # through arbitrarily sharp heading changes.
        turn_factor = 1.0 - MAX_TURN_SURGE_FACTOR * min(
            1.0, abs(self.yaw_rate) / max(self.max_yaw_rate, 1e-6)
        )
        surge_target = min(command_speed, self.max_surge_speed) * (0.75 + 0.25 * turn_factor)
        surge_accel = (surge_target - self.surge) / self.surge_tau
        surge_accel = float(np.clip(surge_accel,
                                    -self.max_surge_accel, self.max_surge_accel))
        self.surge = max(0.0, self.surge + surge_accel * dt)

        # No routine lateral/heave command. Small cross-flow disturbances are
        # included so the sway damping is exercised rather than identically zero.
        sway_dot = (-self.sway / self.sway_decay_tau
                    + SWAY_DISTURBANCE_ACCEL * math.sin(0.37 * t))
        heave_dot = (-self.heave / self.heave_decay_tau
                     + HEAVE_DISTURBANCE_ACCEL * math.sin(0.23 * t))
        self.sway += sway_dot * dt
        self.heave += heave_dot * dt

        # Integrate NED position using current body-forward motion.
        self.position_ned += self.velocity_ned() * dt


def leader_initial_position() -> np.ndarray:
    return np.array([45.0, 0.0, -4.0], dtype=float)


def initial_vehicle_position() -> np.ndarray:
    R0 = rotation_body_to_ned(0.0, 0.0)
    return leader_initial_position() - R0 @ INITIAL_REL_FRD


def make_tracker(mode: str) -> TargetTrackingController:
    if mode == "pid":
        return TargetTrackingController(
            controller="pid", kp=(0.85, 0.85, 0.85),
            ki=(0.025, 0.025, 0.025), kd=(0.22, 0.22, 0.22),
            max_speed=MAX_SPEED, integral_limit=3.0,
            derivative_filter=0.22,
        )
    if mode == "alos":
        return TargetTrackingController(
            controller="alos", delta_h=4.0, delta_v=3.0,
            gamma_h=0.025, gamma_v=0.025, k_along=0.30,
            beta_inner=0.30, beta_outer=0.60,
            alpha_inner=0.30, alpha_outer=0.60,
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
    relative_frd: np.ndarray
    tracking_error_frd: np.ndarray
    nominal_velocity_frd: np.ndarray
    safe_velocity_frd: np.ndarray
    actual_velocity_body: np.ndarray
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
    yaw_rate: np.ndarray
    course_error_deg: np.ndarray
    heading_motion_alignment_deg: np.ndarray
    metrics: dict


def simulate(mode: str, use_cbf: bool) -> RunResult:
    rng = np.random.default_rng(RANDOM_SEED)
    leader = LeaderVehicleNED(leader_initial_position())
    vehicle = HeadingForwardVehicleNED(initial_vehicle_position())
    target_estimator = TargetVelocityEstimator(tau=TARGET_ESTIMATOR_TAU)
    tracker = make_tracker(mode)

    fov_filter = FOVCBFVelocityFilter(
        horizontal_fov_deg=HORIZONTAL_FOV_DEG,
        vertical_fov_deg=VERTICAL_FOV_DEG,
        fov_margin_deg=FOV_MARGIN_DEG,
        alpha=FOV_CBF_ALPHA,
        min_depth=MIN_CAMERA_DEPTH,
        barrier_buffer_m=FOV_CBF_BARRIER_BUFFER_M,
        max_speed=FOV_QP_MAX_SPEED,
        velocity_weights=(1.0, 1.15, 1.15),
        velocity_filter_tau=FOV_CBF_VELOCITY_FILTER_TAU,
    )

    camera_was_available = True
    last_nominal = np.zeros(3)
    last_safe = np.zeros(3)
    times = np.arange(N_STEPS, dtype=float) * DT

    logs = {name: np.zeros((N_STEPS, 3)) for name in (
        "leader", "vehicle", "relative", "error", "nominal", "safe", "vbody"
    )}
    scalar = {name: np.zeros(N_STEPS) for name in (
        "az", "el", "hm", "vm", "yaw", "pitch", "yaw_rate", "course_error", "alignment", "correction"
    )}
    flags = {name: np.zeros(N_STEPS, dtype=bool) for name in (
        "physical", "margined", "feasible"
    )}
    flags["feasible"][:] = True

    for k, t in enumerate(times):
        R_b_n = vehicle.rotation()
        omega_body = vehicle.body_rates_frd()
        v_actual_body = vehicle.velocity_body()

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
            if camera_was_available:
                target_estimator.reset()
                tracker.reset()
            # With no pose, hold latest command but continue heading-forward.
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
            flags["feasible"][k] = info["feasible"]
            scalar["correction"][k] = np.linalg.norm(v_safe - v_nom)
            if mode == "alos" and adaptive_before is not None and scalar["correction"][k] > 1e-3:
                tracker.beta_hat, tracker.alpha_hat = adaptive_before
        elif use_cbf and not measurement_available:
            v_safe = last_safe.copy()
            flags["feasible"][k] = False
        else:
            v_safe = v_nom.copy()

        if measurement_available:
            last_nominal = v_nom.copy()
            last_safe = v_safe.copy()
        camera_was_available = measurement_available

        state = fov_filter.fov_state(rel_true)
        logs["leader"][k] = leader.position_ned
        logs["vehicle"][k] = vehicle.position_ned
        logs["relative"][k] = rel_true
        logs["error"][k] = rel_true - DESIRED_REL_FRD
        logs["nominal"][k] = v_nom
        logs["safe"][k] = v_safe
        logs["vbody"][k] = v_actual_body
        scalar["az"][k] = state["azimuth_deg"]
        scalar["el"][k] = state["elevation_deg"]
        scalar["hm"][k] = state["horizontal_margin_deg"]
        scalar["vm"][k] = state["vertical_margin_deg"]
        scalar["yaw"][k] = math.degrees(vehicle.yaw)
        scalar["pitch"][k] = math.degrees(vehicle.pitch)
        scalar["yaw_rate"][k] = math.degrees(vehicle.yaw_rate)
        scalar["course_error"][k] = math.degrees(vehicle.course_error)
        # Angle between camera optical axis (+body x) and actual NED velocity.
        v_ned_now = vehicle.velocity_ned()
        speed_now = float(np.linalg.norm(v_ned_now))
        if speed_now > 1e-5:
            forward_ned = R_b_n[:, 0]
            cosine = float(np.clip(np.dot(forward_ned, v_ned_now / speed_now), -1.0, 1.0))
            scalar["alignment"][k] = math.degrees(math.acos(cosine))
        else:
            scalar["alignment"][k] = 0.0
        flags["physical"][k] = state["inside_physical_fov"]
        flags["margined"][k] = state["inside_margined_fov"]

        # Apply heading-forward vehicle and independent leader dynamics.
        vehicle.step(v_safe, DT, t=t)
        leader.step(t, DT)

    error_norm = np.linalg.norm(logs["error"], axis=1)
    tail_start = int((1.0 - TAIL_FRACTION) * N_STEPS)
    tail_rms = float(np.sqrt(np.mean(np.sum(logs["error"][tail_start:]**2, axis=1))))
    final_error = float(error_norm[-1])
    peak_error = float(np.max(error_norm))
    body_speed = np.linalg.norm(logs["vbody"], axis=1)
    sway_fraction = np.divide(
        np.abs(logs["vbody"][:, 1]), np.maximum(body_speed, 1e-8)
    )
    course_error_abs = np.abs(scalar["course_error"])
    physical_losses = int(np.count_nonzero(~flags["physical"]))
    marginal_losses = int(np.count_nonzero(~flags["margined"]))
    bounded = bool(np.isfinite(logs["error"]).all() and peak_error < MAX_ERROR_LIMIT_M)
    converged = bool(bounded and tail_rms < TAIL_RMS_LIMIT_M and final_error < FINAL_ERROR_LIMIT_M)
    metrics = {
        "tail_rms_m": tail_rms,
        "final_error_m": final_error,
        "peak_error_m": peak_error,
        "physical_fov_violation_samples": physical_losses,
        "margined_fov_violation_samples": marginal_losses,
        "first_physical_fov_loss_s": float(times[np.flatnonzero(~flags["physical"])[0]]) if physical_losses else None,
        "minimum_horizontal_margin_deg": float(np.min(scalar["hm"])),
        "minimum_vertical_margin_deg": float(np.min(scalar["vm"])),
        "cbf_active_samples": int(np.count_nonzero(scalar["correction"] > 1e-3)),
        "cbf_infeasible_samples": int(np.count_nonzero(~flags["feasible"])),
        "mean_cbf_correction_mps": float(np.mean(scalar["correction"])),
        "max_cbf_correction_mps": float(np.max(scalar["correction"])),
        "mean_abs_course_error_deg": float(np.mean(course_error_abs)),
        "max_abs_course_error_deg": float(np.max(course_error_abs)),
        "rms_sway_speed_mps": float(np.sqrt(np.mean(logs["vbody"][:, 1]**2))),
        "max_abs_sway_speed_mps": float(np.max(np.abs(logs["vbody"][:, 1]))),
        "mean_sway_fraction_pct": float(100.0 * np.mean(sway_fraction)),
        "max_yaw_rate_deg_s": float(np.max(np.abs(scalar["yaw_rate"]))),
        "mean_camera_motion_alignment_error_deg": float(np.mean(scalar["alignment"])),
        "max_camera_motion_alignment_error_deg": float(np.max(scalar["alignment"])),
        "tracking_converged": converged,
    }
    label = f"{mode.upper()} + FOV-CBF" if use_cbf else f"{mode.upper()} (no FOV-CBF)"
    return RunResult(
        label, mode, use_cbf, times,
        logs["leader"], logs["vehicle"], logs["relative"], logs["error"],
        logs["nominal"], logs["safe"], logs["vbody"],
        scalar["az"], scalar["el"], scalar["hm"], scalar["vm"],
        flags["physical"], flags["margined"], flags["feasible"],
        scalar["correction"], scalar["yaw"], scalar["pitch"],
        scalar["yaw_rate"], scalar["course_error"], scalar["alignment"], metrics,
    )


def draw_camera_frustum(ax, position, yaw, pitch, h_half_deg, v_half_deg,
                         depth=13.0, color="#08519c", alpha=0.10):
    th = math.tan(math.radians(h_half_deg))
    tv = math.tan(math.radians(v_half_deg))
    rho = math.hypot(depth, depth * th)
    corners_body = np.array([
        [depth, -depth*th, -rho*tv], [depth, depth*th, -rho*tv],
        [depth, depth*th, rho*tv], [depth, -depth*th, rho*tv],
    ])
    R = rotation_body_to_ned(yaw, pitch)
    corners = position + (R @ corners_body.T).T
    p = np.asarray(position)
    for c in corners:
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                color=color, alpha=0.65, linewidth=0.8)
    loop = np.vstack((corners, corners[0]))
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, alpha=0.7, linewidth=0.8)
    ax.add_collection3d(Poly3DCollection([corners], alpha=alpha,
                                         facecolor=color, edgecolor="none"))


def plot_results(results: list[RunResult], show: bool) -> list[Path]:
    paths = []
    style = {
        ("pid", False): ("#9ecae1", ":", 1.4),
        ("pid", True): ("#08519c", "-", 2.4),
        ("alos", False): ("#fdae6b", ":", 1.4),
        ("alos", True): ("#a63603", "-", 2.4),
    }
    first = results[0]

    # 1. NED trajectories and camera frusta.
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*first.leader_ned.T, color="black", linewidth=2.2, label="Autonomous leader")
    for r in results:
        c, ls, lw = style[(r.mode, r.use_cbf)]
        ax.plot(*r.vehicle_ned.T, color=c, linestyle=ls, linewidth=lw, label=r.label)
    protected = next(r for r in results if r.mode == "pid" and r.use_cbf)
    for ts in (0, 18, 36, 54, 72, 88):
        idx = min(N_STEPS-1, int(ts / DT))
        draw_camera_frustum(ax, protected.vehicle_ned[idx],
                            math.radians(protected.yaw[idx]),
                            math.radians(protected.pitch[idx]),
                            EFFECTIVE_HALF_H_DEG, EFFECTIVE_HALF_V_DEG,
                            color="#3182bd", alpha=0.035)
        ax.scatter(*protected.vehicle_ned[idx], color="#08519c", s=13)
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Down (m)")
    ax.set_title("Heading-forward NED trajectories with camera-FOV CBF\nCamera frusta follow the controlled vehicle heading")
    ax.invert_zaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = OUTPUT_DIR / "heading_forward_ned_trajectories.png"
    fig.savefig(p, dpi=190, bbox_inches="tight")
    paths.append(p)

    # 2. FRD tracking errors.
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    labels = ["Forward error $e_x$ (m)", "Right error $e_y$ (m)", "Down error $e_z$ (m)"]
    for j, ax in enumerate(axes):
        for r in results:
            c, ls, lw = style[(r.mode, r.use_cbf)]
            ax.plot(r.time, r.tracking_error_frd[:, j], color=c, linestyle=ls,
                    linewidth=lw, label=r.label)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylabel(labels[j]); ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Leader-relative tracking errors in instantaneous body FRD")
    fig.tight_layout()
    p = OUTPUT_DIR / "heading_forward_frd_tracking_errors.png"
    fig.savefig(p, dpi=190, bbox_inches="tight"); paths.append(p)

    # 3. Camera visibility angles.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for r in results:
        c, ls, lw = style[(r.mode, r.use_cbf)]
        axes[0].plot(r.time, r.azimuth_deg, color=c, linestyle=ls, linewidth=lw, label=r.label)
        axes[1].plot(r.time, r.elevation_deg, color=c, linestyle=ls, linewidth=lw, label=r.label)
    for ax, bound, name in ((axes[0], EFFECTIVE_HALF_H_DEG, "horizontal"),
                            (axes[1], EFFECTIVE_HALF_V_DEG, "vertical")):
        ax.axhline(bound, color="black", linestyle="--", linewidth=1.0, label=f"Effective {name} edge")
        ax.axhline(-bound, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=8)
    axes[0].set_ylabel("Leader azimuth (deg)")
    axes[1].set_ylabel("Leader elevation (deg)"); axes[1].set_xlabel("Time (s)")
    fig.suptitle("Camera visibility angles with heading-forward motion")
    fig.tight_layout()
    p = OUTPUT_DIR / "heading_forward_visibility_angles.png"
    fig.savefig(p, dpi=190, bbox_inches="tight"); paths.append(p)

    # 4. Heading/course and sway diagnostics.
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for r in results:
        c, ls, lw = style[(r.mode, r.use_cbf)]
        axes[0].plot(r.time, r.course_error_deg, color=c, linestyle=ls, linewidth=lw, label=r.label)
        axes[1].plot(r.time, r.actual_velocity_body[:, 1], color=c, linestyle=ls, linewidth=lw, label=r.label)
        axes[2].plot(r.time, r.actual_velocity_body[:, 0], color=c, linestyle=ls, linewidth=lw, label=r.label)
        axes[3].plot(r.time, r.heading_motion_alignment_deg, color=c, linestyle=ls, linewidth=lw, label=r.label)
    axes[0].set_ylabel("Yaw command tracking error (deg)")
    axes[1].set_ylabel("Body sway velocity v (m/s)")
    axes[2].set_ylabel("Body surge velocity u (m/s)")
    axes[3].set_ylabel("Camera axis vs. motion (deg)")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=8)
    fig.suptitle("Heading-forward motion: camera aligned with travel direction; sway damped")
    fig.tight_layout()
    p = OUTPUT_DIR / "heading_forward_motion_diagnostics.png"
    fig.savefig(p, dpi=190, bbox_inches="tight"); paths.append(p)

    # 5. FOV CBF corrections and guard margins.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for r in results:
        if not r.use_cbf: continue
        c = style[(r.mode, True)][0]
        axes[0].plot(r.time, r.horizontal_margin_deg, color=c, linewidth=1.4,
                     label=f"{r.mode.upper()} horizontal margin")
        axes[0].plot(r.time, r.vertical_margin_deg, color=c, linestyle="--", linewidth=1.4,
                     label=f"{r.mode.upper()} vertical margin")
        axes[1].plot(r.time, r.cbf_correction, color=c, linewidth=1.4,
                     label=f"{r.mode.upper()} CBF correction")
    axes[0].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Margin to effective FOV edge (deg)")
    axes[1].set_ylabel("$||v_{safe}-v_{nom}||$ (m/s)"); axes[1].set_xlabel("Time (s)")
    for ax in axes: ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=8)
    fig.suptitle("FOV-CBF guard margins and velocity corrections")
    fig.tight_layout()
    p = OUTPUT_DIR / "heading_forward_fov_cbf_diagnostics.png"
    fig.savefig(p, dpi=190, bbox_inches="tight"); paths.append(p)

    print("Generated plots:")
    for p in paths: print(" ", p)
    if show: plt.show()
    else: plt.close("all")
    return paths


def save_metrics(results: list[RunResult]) -> Path:
    path = OUTPUT_DIR / "heading_forward_metrics.csv"
    fields = ["case", "tail_rms_m", "final_error_m", "peak_error_m",
              "physical_fov_violation_samples", "margined_fov_violation_samples",
              "first_physical_fov_loss_s", "minimum_horizontal_margin_deg",
              "minimum_vertical_margin_deg", "cbf_active_samples", "cbf_infeasible_samples",
              "mean_cbf_correction_mps", "max_cbf_correction_mps",
              "mean_abs_course_error_deg", "max_abs_course_error_deg",
              "rms_sway_speed_mps", "max_abs_sway_speed_mps",
              "mean_sway_fraction_pct", "max_yaw_rate_deg_s",
              "mean_camera_motion_alignment_error_deg",
              "max_camera_motion_alignment_error_deg", "tracking_converged"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in results:
            row = {"case": r.label}; row.update({k: r.metrics[k] for k in fields[1:]}); writer.writerow(row)
    print("Saved metrics:", path)
    return path


def print_summary(results: list[RunResult]) -> bool:
    print("\nHeading-forward camera-FOV closed-loop results")
    print("=" * 150)
    print(f"{'Case':<22} {'Tail RMS':>9} {'Final':>8} {'FOV loss':>9} {'Guard loss':>10} "
          f"{'Min H':>8} {'Min V':>8} {'RMS sway':>10} {'Mean camera/motion':>18} {'QP infeas':>10}")
    print("-" * 150)
    for r in results:
        m = r.metrics
        print(f"{r.label:<22} {m['tail_rms_m']:>9.3f} {m['final_error_m']:>8.3f} "
              f"{m['physical_fov_violation_samples']:>9d} {m['margined_fov_violation_samples']:>10d} "
              f"{m['minimum_horizontal_margin_deg']:>8.2f} {m['minimum_vertical_margin_deg']:>8.2f} "
              f"{m['rms_sway_speed_mps']:>10.4f} {m['mean_camera_motion_alignment_error_deg']:>14.3f} "
              f"{m['cbf_infeasible_samples']:>10d}")
    protected = [r for r in results if r.use_cbf]
    baseline = [r for r in results if not r.use_cbf]
    safety_ok = all(r.metrics["physical_fov_violation_samples"] == 0
                    and r.metrics["margined_fov_violation_samples"] == 0
                    and r.metrics["cbf_infeasible_samples"] == 0 for r in protected)
    baseline_challenged = all(r.metrics["physical_fov_violation_samples"] > 0 for r in baseline)
    sway_ok = all(r.metrics["rms_sway_speed_mps"] < 0.03 for r in results)
    alignment_ok = all(r.metrics["mean_camera_motion_alignment_error_deg"] < 2.0 for r in results)
    tracking_ok = all(r.metrics["tracking_converged"] for r in protected)
    print("Protected FOV invariance (finite run):", "PASS" if safety_ok else "CHECK / FAIL")
    print("No-CBF baseline loses target visibility:", "PASS" if baseline_challenged else "CHECK / FAIL")
    print("Sway suppressed (RMS < 0.03 m/s):", "PASS" if sway_ok else "CHECK / FAIL")
    print("Camera aligned with actual motion (< 2 deg mean):", "PASS" if alignment_ok else "CHECK / FAIL")
    print("Tracking bounded/recovered:", "PASS" if tracking_ok else "CHECK / FAIL")
    print("This is a finite-step kinematic test, not a formal stability or safety proof.")
    return safety_ok and baseline_challenged and sway_ok and alignment_ok and tracking_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    results = [simulate("pid", False), simulate("pid", True),
               simulate("alos", False), simulate("alos", True)]
    plot_results(results, show=not args.no_show)
    save_metrics(results)
    if not print_summary(results):
        raise SystemExit("Heading-forward FOV test did not meet all configured checks")


if __name__ == "__main__":
    main()
