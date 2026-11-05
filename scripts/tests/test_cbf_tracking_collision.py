"""Closed-loop NED tracking simulation with a body-FRD CBF safety filter.

Runs four cases using the same target and obstacle trajectories:
  1) PID without CBF (comparison / baseline)
  2) PID + CBF
  3) 3D ALOS without CBF (comparison / baseline)
  4) 3D ALOS + CBF

The tracking controller receives only target relative position measurements.
Target ground velocity is estimated from successive relative positions and
controlled-vehicle ground velocity. The CBF receives obstacle/fleet relative
positions in body FRD and estimates their radial relative motion from history.

Kinematics are integrated in NED. Vehicle attitude is fixed in this example,
so body-to-NED rotation is constant. The vehicle has a first-order velocity
response and an acceleration limit; therefore this is a simulation of a
velocity-level CBF with a conservative response-distance buffer, not a formal
higher-order CBF for acceleration-controlled dynamics.

Run:
    pip install numpy scipy matplotlib
    python test_closed_loop_with_cbf.py
    python test_closed_loop_with_cbf.py --no-show
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.target_tracking import TargetTrackingController, TargetVelocityEstimator
from scripts.cbf_collision import CBFCollisionAvoidance


# ----------------------------- Simulation settings ---------------------- #
DT = 0.1
SIM_TIME = 100.0
DESIRED_REL_FRD = np.array([10.0, 0.0, 0.0])
INITIAL_VEHICLE_NED = np.array([0.0, 0.0, 20.0])
INITIAL_TARGET_NED = np.array([35.0, 10.0, 18.0])

# Fixed attitude for this translational-kinematics demonstration.
VEHICLE_YAW_RAD = math.radians(14.3239449)
VEHICLE_PITCH_RAD = math.radians(-2.2918312)

# Low-level velocity response approximation.
VEHICLE_VELOCITY_TAU = 0.50
VEHICLE_MAX_ACCEL = 0.65
MAX_SPEED = 2.0

# Target relative-position estimator.
ESTIMATOR_TAU = 0.30
POSITION_NOISE_STD_M = 0.0
RANDOM_SEED = 7

# Vehicle and obstacle geometry [m].
VEHICLE_RADIUS = 0.75
BASE_SAFETY_MARGIN = 0.50
STATIC_OBSTACLE_RADIUS = 1.80
FLEET_VEHICLE_RADIUS = 0.90

# Extra conservative distance for velocity-response lag. This is an
# engineering buffer, not a mathematically derived HOCBF guarantee.
VELOCITY_RESPONSE_BUFFER = 3.0
CBF_ALPHA = 0.50
UNKNOWN_CLOSING_SPEED = 0.60
OBSTACLE_DETECTION_RANGE = 65.0

# Scenario-specific practical thresholds.
TAIL_WINDOW_FRACTION = 0.20
TAIL_RMS_LIMIT_M = 2.0
FINAL_ERROR_LIMIT_M = 3.0
MAX_ERROR_LIMIT_M = 45.0

OUTPUT_DIR = Path(__file__).resolve().parent / "closed_loop_with_cbf_results"


# ----------------------------- Coordinate transforms -------------------- #
def rotation_body_to_ned(yaw: float, pitch: float) -> np.ndarray:
    """FRD-to-NED rotation (roll=0, 3-2-1 yaw-pitch-roll convention)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array(
        [
            [cp * cy, -sy, sp * cy],
            [cp * sy,  cy, sp * sy],
            [-sp,      0.0, cp],
        ], dtype=float,
    )


# ----------------------------- Vehicle models --------------------------- #
@dataclass
class TargetVehicleNED:
    position_ned: np.ndarray

    def velocity_ned(self, time_s: float) -> np.ndarray:
        return np.array(
            [
                0.5,
                0.22 * math.sin(0.06 * time_s),
                0.04 * math.cos(0.04 * time_s),
            ], dtype=float,
        )

    def step(self, time_s: float, dt: float) -> None:
        self.position_ned = self.position_ned + self.velocity_ned(time_s) * dt


@dataclass
class ControlledVehicleNED:
    position_ned: np.ndarray
    rotation_body_to_ned: np.ndarray
    velocity_tau: float = VEHICLE_VELOCITY_TAU
    max_acceleration: float = VEHICLE_MAX_ACCEL

    def __post_init__(self) -> None:
        self.velocity_body = np.zeros(3, dtype=float)
        self.rotation_ned_to_body = self.rotation_body_to_ned.T

    def step(self, velocity_command_body: np.ndarray, dt: float) -> None:
        command = np.asarray(velocity_command_body, dtype=float)
        accel = (command - self.velocity_body) / self.velocity_tau
        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > self.max_acceleration:
            accel *= self.max_acceleration / accel_norm
        self.velocity_body += accel * dt
        self.position_ned += (
            self.rotation_body_to_ned @ self.velocity_body
        ) * dt

    def relative_position_body(self, object_position_ned: np.ndarray) -> np.ndarray:
        return self.rotation_ned_to_body @ (
            object_position_ned - self.position_ned
        )


@dataclass(frozen=True)
class ObstacleSpec:
    obstacle_id: str
    label: str
    radius: float
    kind: str  # "static" or "fleet"
    position_function: Callable[[float], np.ndarray]

    def position_ned(self, t: float) -> np.ndarray:
        return np.asarray(self.position_function(t), dtype=float)


@dataclass
class SimulationResult:
    label: str
    mode: str
    use_cbf: bool
    time: np.ndarray
    target_ned: np.ndarray
    vehicle_ned: np.ndarray
    desired_vehicle_ned: np.ndarray
    relative_position_frd: np.ndarray
    error_frd: np.ndarray
    nominal_command_frd: np.ndarray
    safe_command_frd: np.ndarray
    estimated_target_velocity_frd: np.ndarray
    obstacle_ned: dict[str, np.ndarray]
    obstacle_distances: dict[str, np.ndarray]
    cbf_feasible: np.ndarray
    cbf_active: np.ndarray
    cbf_correction: np.ndarray
    metrics: dict


def target_reference_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Generate target and desired-following-locus arrays for obstacle setup."""
    n = int(round(SIM_TIME / DT))
    target = TargetVehicleNED(INITIAL_TARGET_NED.copy())
    R_b_n = rotation_body_to_ned(VEHICLE_YAW_RAD, VEHICLE_PITCH_RAD)
    target_positions = np.empty((n, 3))
    desired_locus = np.empty((n, 3))
    for k in range(n):
        t = k * DT
        target_positions[k] = target.position_ned
        desired_locus[k] = target.position_ned - R_b_n @ DESIRED_REL_FRD
        target.step(t, DT)
    return target_positions, desired_locus


def make_obstacles() -> list[ObstacleSpec]:
    """Create one static obstacle and one moving fleet vehicle crossing path."""
    _, desired_locus = target_reference_trajectory()
    R_b_n = rotation_body_to_ned(VEHICLE_YAW_RAD, VEHICLE_PITCH_RAD)

    # Static spherical obstacle: placed to the right of the desired locus,
    # so a CBF correction has a natural lateral component.
    static_idx = int(round(30.0 / DT))
    static_center = (
        desired_locus[static_idx] + R_b_n @ np.array([0.0, 2.5, 0.0])
    )

    def static_position(_t: float) -> np.ndarray:
        return static_center.copy()

    # Another fleet vehicle crosses the desired path near t=60 s.
    fleet_initial = np.array([85.0, -20.0, 18.4])
    fleet_velocity = np.array([0.0, 0.55, 0.0])

    def fleet_position(t: float) -> np.ndarray:
        return fleet_initial + fleet_velocity * t

    return [
        ObstacleSpec(
            obstacle_id="static_obstacle_1",
            label="Static obstacle",
            radius=STATIC_OBSTACLE_RADIUS,
            kind="static",
            position_function=static_position,
        ),
        ObstacleSpec(
            obstacle_id="fleet_vehicle_1",
            label="Moving fleet vehicle",
            radius=FLEET_VEHICLE_RADIUS,
            kind="fleet",
            position_function=fleet_position,
        ),
    ]


def make_tracking_controller(mode: str) -> TargetTrackingController:
    common = dict(max_speed=MAX_SPEED)
    if mode == "pid":
        return TargetTrackingController(
            controller="pid",
            kp=(0.8, 0.8, 0.8),
            ki=(0.05, 0.05, 0.05),
            kd=(0.2, 0.2, 0.2),
            integral_limit=5.0,
            derivative_filter=0.25,
            **common,
        )
    if mode == "alos":
        return TargetTrackingController(
            controller="alos",
            delta_h=5.0,
            delta_v=3.0,
            gamma_h=0.05,
            gamma_v=0.05,
            k_along=0.30,
            beta_inner=0.4,
            beta_outer=0.7,
            alpha_inner=0.4,
            alpha_outer=0.7,
            **common,
        )
    raise ValueError("mode must be 'pid' or 'alos'")


def simulate(mode: str, use_cbf: bool, obstacles: list[ObstacleSpec]) -> SimulationResult:
    rng = np.random.default_rng(RANDOM_SEED)
    R_b_n = rotation_body_to_ned(VEHICLE_YAW_RAD, VEHICLE_PITCH_RAD)
    vehicle = ControlledVehicleNED(INITIAL_VEHICLE_NED.copy(), R_b_n)
    target = TargetVehicleNED(INITIAL_TARGET_NED.copy())
    estimator = TargetVelocityEstimator(tau=ESTIMATOR_TAU)
    tracker = make_tracking_controller(mode)

    cbf = None
    if use_cbf:
        cbf = CBFCollisionAvoidance(
            vehicle_radius=VEHICLE_RADIUS,
            safety_margin=BASE_SAFETY_MARGIN + VELOCITY_RESPONSE_BUFFER,
            alpha=CBF_ALPHA,
            max_speed=MAX_SPEED,
            unknown_closing_speed=UNKNOWN_CLOSING_SPEED,
        )

    body_rates = np.zeros(3)  # fixed attitude in this simulation
    n_steps = int(round(SIM_TIME / DT))
    times = np.arange(n_steps, dtype=float) * DT

    target_log = np.empty((n_steps, 3))
    vehicle_log = np.empty((n_steps, 3))
    desired_vehicle_log = np.empty((n_steps, 3))
    relative_log = np.empty((n_steps, 3))
    error_log = np.empty((n_steps, 3))
    nominal_log = np.zeros((n_steps, 3))
    safe_log = np.zeros((n_steps, 3))
    target_velocity_est_log = np.full((n_steps, 3), np.nan)
    cbf_feasible_log = np.ones(n_steps, dtype=bool)
    cbf_active_log = np.zeros(n_steps, dtype=bool)
    cbf_correction_log = np.zeros(n_steps)

    obstacle_ned_log = {
        obs.obstacle_id: np.empty((n_steps, 3)) for obs in obstacles
    }
    obstacle_distance_log = {
        obs.obstacle_id: np.empty(n_steps) for obs in obstacles
    }

    for k, t in enumerate(times):
        # Target relative-position sensor (the controller is not given target truth).
        relative_true = vehicle.relative_position_body(target.position_ned)
        measurement = relative_true.copy()
        if POSITION_NOISE_STD_M > 0.0:
            measurement += rng.normal(0.0, POSITION_NOISE_STD_M, size=3)

        target_velocity_est = estimator.update(
            target_rel_body=measurement,
            vehicle_velocity_body=vehicle.velocity_body,
            body_rates_frd=body_rates,
            dt=DT,
        )

        if target_velocity_est is None:
            v_nom = np.zeros(3)
            adaptive_state_before = None
        else:
            target_velocity_est_log[k] = target_velocity_est
            # Keep a copy of ALOS adaptive estimates. If the CBF overrides
            # the nominal command, we roll back this update below so that
            # obstacle-avoidance displacement is not learned as persistent
            # crab/attack-angle disturbance by the path-following law.
            adaptive_state_before = (tracker.beta_hat, tracker.alpha_hat)
            v_nom = np.asarray(
                tracker.compute(
                    target_rel_body=measurement,
                    desired_rel_body=DESIRED_REL_FRD,
                    target_velocity_body=target_velocity_est,
                    body_rates_frd=body_rates,
                    dt=DT,
                ), dtype=float,
            )

        # Simulated perception: only obstacle relative positions in body FRD
        # are passed to the CBF. Stable IDs provide multi-frame association.
        detected_positions = []
        detected_radii = []
        detected_ids = []
        detected_specs = []
        for obs in obstacles:
            obs_pos_ned = obs.position_ned(t)
            rel_obs_body = vehicle.relative_position_body(obs_pos_ned)
            distance = float(np.linalg.norm(rel_obs_body))
            obstacle_ned_log[obs.obstacle_id][k] = obs_pos_ned
            obstacle_distance_log[obs.obstacle_id][k] = distance
            if distance <= OBSTACLE_DETECTION_RANGE:
                detected_positions.append(rel_obs_body)
                detected_radii.append(obs.radius)
                detected_ids.append(obs.obstacle_id)
                detected_specs.append(obs)

        if cbf is not None:
            v_safe_tuple, cbf_info = cbf.filter_velocity(
                v_nom=v_nom,
                obstacle_positions_body=detected_positions,
                vehicle_velocity_body=vehicle.velocity_body,
                dt=DT,
                obstacle_radii=detected_radii,
                obstacle_ids=detected_ids,
                return_info=True,
            )
            v_safe = np.asarray(v_safe_tuple, dtype=float)
            cbf_feasible_log[k] = bool(cbf_info["feasible"])
            cbf_correction_log[k] = float(np.linalg.norm(v_safe - v_nom))
            cbf_active_log[k] = cbf_correction_log[k] > 1e-3
            if mode == "alos" and adaptive_state_before is not None and cbf_active_log[k]:
                # CBF has priority over tracking: freeze ALOS adaptation on
                # samples where the safety filter modifies the command.
                tracker.beta_hat, tracker.alpha_hat = adaptive_state_before
        else:
            v_safe = v_nom.copy()

        # Store instantaneous state and commands before propagating.
        target_log[k] = target.position_ned
        vehicle_log[k] = vehicle.position_ned
        desired_vehicle_log[k] = (
            target.position_ned - R_b_n @ DESIRED_REL_FRD
        )
        relative_log[k] = relative_true
        error_log[k] = relative_true - DESIRED_REL_FRD
        nominal_log[k] = v_nom
        safe_log[k] = v_safe

        # NED kinematic propagation of controlled and target vehicles.
        vehicle.step(v_safe, DT)
        target.step(t, DT)

    # Tracking metrics.
    error_norm = np.linalg.norm(error_log, axis=1)
    tail_start = int((1.0 - TAIL_WINDOW_FRACTION) * n_steps)
    tail_rms = float(np.sqrt(np.mean(np.sum(error_log[tail_start:] ** 2, axis=1))))
    final_error = float(error_norm[-1])
    peak_error = float(np.max(error_norm))
    all_finite = bool(
        np.isfinite(error_log).all()
        and np.isfinite(vehicle_log).all()
        and np.isfinite(safe_log).all()
    )

    # Geometric clearances: sphere-to-sphere center distance minus radii.
    min_surface_clearance = math.inf
    min_margin_clearance = math.inf
    min_cbf_envelope_clearance = math.inf
    per_obstacle_clearance = {}
    collision_count = 0
    margin_violation_count = 0
    for obs in obstacles:
        center_dist = np.linalg.norm(
            obstacle_ned_log[obs.obstacle_id] - vehicle_log, axis=1
        )
        surface_clearance = center_dist - (VEHICLE_RADIUS + obs.radius)
        margin_clearance = surface_clearance - BASE_SAFETY_MARGIN
        cbf_clearance = margin_clearance - VELOCITY_RESPONSE_BUFFER
        per_obstacle_clearance[obs.obstacle_id] = {
            "min_surface_clearance_m": float(np.min(surface_clearance)),
            "min_margin_clearance_m": float(np.min(margin_clearance)),
            "min_cbf_buffered_clearance_m": float(np.min(cbf_clearance)),
        }
        min_surface_clearance = min(min_surface_clearance, float(np.min(surface_clearance)))
        min_margin_clearance = min(min_margin_clearance, float(np.min(margin_clearance)))
        min_cbf_envelope_clearance = min(
            min_cbf_envelope_clearance, float(np.min(cbf_clearance))
        )
        collision_count += int(np.count_nonzero(surface_clearance < 0.0))
        margin_violation_count += int(np.count_nonzero(margin_clearance < 0.0))

    bounded = bool(all_finite and peak_error < MAX_ERROR_LIMIT_M)
    converged = bool(
        bounded and tail_rms < TAIL_RMS_LIMIT_M and final_error < FINAL_ERROR_LIMIT_M
    )
    metrics = {
        "tail_rms_m": tail_rms,
        "final_error_m": final_error,
        "peak_error_m": peak_error,
        "bounded": bounded,
        "converged": converged,
        "min_surface_clearance_m": min_surface_clearance,
        "min_margin_clearance_m": min_margin_clearance,
        "min_cbf_buffered_clearance_m": min_cbf_envelope_clearance,
        "collision_samples": collision_count,
        "safety_margin_violation_samples": margin_violation_count,
        "cbf_active_samples": int(np.count_nonzero(cbf_active_log)),
        "cbf_infeasible_samples": int(np.count_nonzero(~cbf_feasible_log)),
        "mean_cbf_correction_mps": float(np.mean(cbf_correction_log)),
        "max_cbf_correction_mps": float(np.max(cbf_correction_log)),
        "per_obstacle_clearance": per_obstacle_clearance,
    }

    label = f"{mode.upper()} + CBF" if use_cbf else f"{mode.upper()} (no CBF)"
    return SimulationResult(
        label=label,
        mode=mode,
        use_cbf=use_cbf,
        time=times,
        target_ned=target_log,
        vehicle_ned=vehicle_log,
        desired_vehicle_ned=desired_vehicle_log,
        relative_position_frd=relative_log,
        error_frd=error_log,
        nominal_command_frd=nominal_log,
        safe_command_frd=safe_log,
        estimated_target_velocity_frd=target_velocity_est_log,
        obstacle_ned=obstacle_ned_log,
        obstacle_distances=obstacle_distance_log,
        cbf_feasible=cbf_feasible_log,
        cbf_active=cbf_active_log,
        cbf_correction=cbf_correction_log,
        metrics=metrics,
    )


# ----------------------------- Plot helpers ----------------------------- #
def draw_sphere(ax, center, radius, color, alpha=0.12, resolution=14):
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def plot_results(results: list[SimulationResult], obstacles: list[ObstacleSpec], show: bool):
    # 3D NED trajectories with obstacles and clearance envelopes.
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    first = results[0]

    ax.plot(
        first.target_ned[:, 0], first.target_ned[:, 1], first.target_ned[:, 2],
        color="black", linewidth=2.2, label="Autonomous target trajectory",
    )
    ax.plot(
        first.desired_vehicle_ned[:, 0], first.desired_vehicle_ned[:, 1],
        first.desired_vehicle_ned[:, 2], color="gray", linestyle="--",
        linewidth=1.6, label="Desired vehicle locus (10 m FRD stand-off)",
    )

    style = {
        ("pid", False): ("#9ecae1", ":", 1.3),
        ("pid", True): ("#08519c", "-", 2.4),
        ("alos", False): ("#fdae6b", ":", 1.3),
        ("alos", True): ("#a63603", "-", 2.4),
    }
    for result in results:
        color, linestyle, width = style[(result.mode, result.use_cbf)]
        ax.plot(
            result.vehicle_ned[:, 0], result.vehicle_ned[:, 1],
            result.vehicle_ned[:, 2], color=color, linestyle=linestyle,
            linewidth=width, label=result.label,
        )
        ax.scatter(*result.vehicle_ned[0], color=color, marker="o", s=22)

    # Plot obstacle physical geometry and the inflated CBF exclusion envelope.
    for obs in obstacles:
        positions = first.obstacle_ned[obs.obstacle_id]
        if obs.kind == "static":
            center = positions[0]
            envelope_radius = (
                VEHICLE_RADIUS + obs.radius + BASE_SAFETY_MARGIN
                + VELOCITY_RESPONSE_BUFFER
            )
            draw_sphere(ax, center, envelope_radius, color="#fdae6b", alpha=0.10)
            draw_sphere(ax, center, obs.radius, color="#636363", alpha=0.42)
            ax.scatter(*center, color="#252525", marker="X", s=65,
                       label=f"{obs.label} center")
            ax.text(center[0], center[1], center[2], "  static", fontsize=8)
        else:
            ax.plot(
                positions[:, 0], positions[:, 1], positions[:, 2],
                color="#31a354", linestyle="-.", linewidth=2.0,
                label=f"{obs.label} trajectory",
            )
            # Show selected snapshots with physical sphere and CBF envelope.
            for j, fraction in enumerate((0.20, 0.60, 0.90)):
                idx = min(len(positions)-1, int(fraction * (len(positions)-1)))
                center = positions[idx]
                envelope_radius = (
                    VEHICLE_RADIUS + obs.radius + BASE_SAFETY_MARGIN
                    + VELOCITY_RESPONSE_BUFFER
                )
                draw_sphere(ax, center, envelope_radius, color="#74c476", alpha=0.055)
                draw_sphere(ax, center, obs.radius, color="#238b45", alpha=0.42)
                if j == 0:
                    ax.scatter(*center, color="#006d2c", marker="^", s=55,
                               label="Fleet vehicle snapshots")
                else:
                    ax.scatter(*center, color="#006d2c", marker="^", s=55)

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Down (m)")
    # Equal data-unit scaling makes the spherical obstacle geometry look
    # spherical rather than stretched by Matplotlib's default 3D aspect.
    x_all = np.concatenate([first.target_ned[:, 0], first.vehicle_ned[:, 0]])
    y_all = np.concatenate([first.target_ned[:, 1], first.vehicle_ned[:, 1]])
    z_all = np.concatenate([first.target_ned[:, 2], first.vehicle_ned[:, 2]])
    for obs in obstacles:
        obs_pos = first.obstacle_ned[obs.obstacle_id]
        envelope_radius = (
            VEHICLE_RADIUS + obs.radius + BASE_SAFETY_MARGIN
            + VELOCITY_RESPONSE_BUFFER
        )
        x_all = np.concatenate([
            x_all, obs_pos[:, 0] - envelope_radius,
            obs_pos[:, 0] + envelope_radius,
        ])
        y_all = np.concatenate([
            y_all, obs_pos[:, 1] - envelope_radius,
            obs_pos[:, 1] + envelope_radius,
        ])
        z_all = np.concatenate([
            z_all, obs_pos[:, 2] - envelope_radius,
            obs_pos[:, 2] + envelope_radius,
        ])
    ranges = np.maximum(
        [np.ptp(x_all), np.ptp(y_all), np.ptp(z_all)], 1.0
    )
    # Vertical exaggeration makes the modest Down variation visible; NED data
    # and all calculations remain in metres. The exact factor is stated on plot.
    display_aspect = ranges.copy()
    display_aspect[2] *= 2.0
    ax.set_box_aspect(display_aspect)
    ax.text2D(
        0.02, 0.02, "Down axis displayed with 2x vertical exaggeration",
        transform=ax.transAxes, fontsize=8, color="dimgray",
    )
    ax.set_title("NED closed-loop target tracking with CBF collision avoidance\n"
                 "Solid: CBF-filtered; dotted: nominal tracking without CBF")
    ax.invert_zaxis()
    ax.grid(True, alpha=0.3)
    # Explicit legend entries for the physical bodies and inflated safety shells.
    shell_handle = Line2D(
        [0], [0], marker="o", linestyle="None", markersize=9,
        markerfacecolor="#fdae6b", alpha=0.35,
        label="Inflated CBF safety envelope",
    )
    physical_handle = Line2D(
        [0], [0], marker="o", linestyle="None", markersize=7,
        markerfacecolor="#636363", markeredgecolor="#252525",
        label="Obstacle physical body",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([physical_handle, shell_handle])
    labels.extend(["Obstacle physical body", "Inflated CBF safety envelope"])
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    trajectory_path = OUTPUT_DIR / "cbf_closed_loop_ned_trajectories.png"
    fig.savefig(trajectory_path, dpi=190, bbox_inches="tight")

    # FRD tracking-error histories (same plot style as original test).
    fig_err, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    labels = ["Forward error $e_x$ (m)", "Right error $e_y$ (m)", "Down error $e_z$ (m)"]
    for axis_idx, ax in enumerate(axes):
        for result in results:
            color, linestyle, width = style[(result.mode, result.use_cbf)]
            ax.plot(
                result.time, result.error_frd[:, axis_idx],
                color=color, linestyle=linestyle, linewidth=width,
                label=result.label,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(labels[axis_idx])
        ax.grid(True, alpha=0.3)
        if axis_idx == 0:
            ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig_err.suptitle("Target-relative tracking errors in body FRD\n"
                     "Obstacle avoidance causes temporary tracking deviations")
    fig_err.tight_layout()
    error_path = OUTPUT_DIR / "cbf_closed_loop_frd_tracking_errors.png"
    fig_err.savefig(error_path, dpi=190, bbox_inches="tight")

    # Extra diagnostic: minimum physical surface clearance and nominal-to-safe
    # command correction for CBF cases.
    fig_diag, axes_diag = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    cbf_results = [r for r in results if r.use_cbf]
    for result in cbf_results:
        color = style[(result.mode, True)][0]
        min_clearance = np.full_like(result.time, np.inf, dtype=float)
        for obs in obstacles:
            center_dist = np.linalg.norm(
                result.obstacle_ned[obs.obstacle_id] - result.vehicle_ned, axis=1
            )
            clearance = center_dist - (VEHICLE_RADIUS + obs.radius)
            min_clearance = np.minimum(min_clearance, clearance)
        axes_diag[0].plot(result.time, min_clearance, color=color,
                          linewidth=1.6, label=result.label)
        axes_diag[1].plot(result.time, result.cbf_correction, color=color,
                          linewidth=1.4, label=result.label)
    axes_diag[0].axhline(BASE_SAFETY_MARGIN, color="black", linestyle="--",
                         linewidth=1.0, label="Required extra safety margin")
    axes_diag[0].axhline(0.0, color="red", linewidth=0.9, alpha=0.7,
                         label="Physical contact boundary")
    axes_diag[0].set_ylabel("Minimum surface clearance (m)")
    axes_diag[0].legend(fontsize=8, ncol=2)
    axes_diag[0].grid(True, alpha=0.3)
    axes_diag[1].set_ylabel(r"$\|v_{safe}-v_{nom}\|$ (m/s)")
    axes_diag[1].set_xlabel("Time (s)")
    axes_diag[1].grid(True, alpha=0.3)
    axes_diag[1].legend(fontsize=8)
    fig_diag.suptitle("CBF safety-filter diagnostics")
    fig_diag.tight_layout()
    diagnostic_path = OUTPUT_DIR / "cbf_safety_diagnostics.png"
    fig_diag.savefig(diagnostic_path, dpi=190, bbox_inches="tight")

    print(f"Saved NED trajectory plot: {trajectory_path}")
    print(f"Saved FRD error plot:      {error_path}")
    print(f"Saved CBF diagnostics:     {diagnostic_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def print_summary(results: list[SimulationResult]) -> bool:
    print("\nClosed-loop tracking + obstacle avoidance summary")
    print("=" * 132)
    header = (
        f"{'Case':<17} {'Tail RMS':>9} {'Final |e|':>10} {'Peak |e|':>10} "
        f"{'Min surface':>12} {'Min margin':>11} {'Contacts':>9} "
        f"{'CBF active':>11} {'Infeasible':>11}"
    )
    print(header)
    print("-" * 132)
    for result in results:
        m = result.metrics
        print(
            f"{result.label:<17} {m['tail_rms_m']:>9.3f} "
            f"{m['final_error_m']:>10.3f} {m['peak_error_m']:>10.3f} "
            f"{m['min_surface_clearance_m']:>12.3f} "
            f"{m['min_margin_clearance_m']:>11.3f} "
            f"{m['collision_samples']:>9d} {m['cbf_active_samples']:>11d} "
            f"{m['cbf_infeasible_samples']:>11d}"
        )
    print("-" * 132)
    print("Surface clearance = center distance - (vehicle radius + obstacle radius).")
    print("Margin clearance = surface clearance - base safety margin.")
    print("CBF uses the base margin plus an extra velocity-response buffer of "
          f"{VELOCITY_RESPONSE_BUFFER:.1f} m.")

    # This is a numerical scenario check, not a formal stability certificate.
    cbf_cases = [r for r in results if r.use_cbf]
    safe = all(
        r.metrics["collision_samples"] == 0
        and r.metrics["min_margin_clearance_m"] >= -0.05
        and r.metrics["min_cbf_buffered_clearance_m"] >= -0.05
        and r.metrics["cbf_infeasible_samples"] == 0
        for r in cbf_cases
    )
    tracking_ok = all(r.metrics["converged"] for r in cbf_cases)
    overall_pass = safe and tracking_ok
    baseline_cases = [r for r in results if not r.use_cbf]
    baseline_exercised = all(r.metrics["collision_samples"] > 0 for r in baseline_cases)
    overall_pass = safe and tracking_ok and baseline_exercised
    print("\nCBF scenario safety check:", "PASS" if safe else "CHECK / FAIL")
    print("CBF-case tracking recovery:", "PASS" if tracking_ok else "CHECK / FAIL")
    print(
        "Unfiltered baseline collision demonstration:",
        "PASS" if baseline_exercised else "CHECK / FAIL",
    )
    for r in cbf_cases:
        print(
            f"  {r.label}: min surface clearance = "
            f"{r.metrics['min_surface_clearance_m']:.3f} m; "
            f"min base-margin clearance = "
            f"{r.metrics['min_margin_clearance_m']:.3f} m; "
            f"min expanded-envelope clearance = "
            f"{r.metrics['min_cbf_buffered_clearance_m']:.3f} m; "
            f"CBF infeasible samples = {r.metrics['cbf_infeasible_samples']}"
        )
    print("Note: collision counts are time samples inside the physical contact boundary, not distinct collision events.")
    print("Note: finite-step simulation with actuator lag is not a formal CBF proof.")
    return overall_pass


def save_metrics_csv(results: list[SimulationResult]) -> Path:
    path = OUTPUT_DIR / "cbf_simulation_metrics.csv"
    fields = [
        "case", "tail_rms_m", "final_error_m", "peak_error_m",
        "min_surface_clearance_m", "min_margin_clearance_m",
        "min_cbf_buffered_clearance_m", "collision_samples",
        "safety_margin_violation_samples", "cbf_active_samples",
        "cbf_infeasible_samples", "mean_cbf_correction_mps",
        "max_cbf_correction_mps", "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"case": result.label}
            row.update({key: result.metrics[key] for key in fields[1:]})
            writer.writerow(row)
    print(f"Saved simulation metrics: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PID/ALOS tracking with and without CBF obstacle avoidance."
    )
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without opening interactive windows.")
    args = parser.parse_args()

    obstacles = make_obstacles()
    results = [
        simulate("pid", False, obstacles),
        simulate("pid", True, obstacles),
        simulate("alos", False, obstacles),
        simulate("alos", True, obstacles),
    ]

    passed = print_summary(results)
    save_metrics_csv(results)
    plot_results(results, obstacles, show=not args.no_show)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
