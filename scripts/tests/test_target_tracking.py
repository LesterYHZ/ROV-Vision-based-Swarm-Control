"""Closed-loop NED kinematic simulation for PID and 3D ALOS tracking.

Run from this directory:
    python test_closed_loop.py

Dependencies:
    numpy, matplotlib

The script imports TargetTrackingController and TargetVelocityEstimator from
tracker.py. It does NOT pass the target's true velocity to the controller;
target velocity is estimated from relative-position samples and the controlled
vehicle's own ground-referenced velocity and body rates.

Model scope:
    * NED position kinematics only (North, East, Down).
    * Controlled vehicle has fixed attitude in this baseline test.
    * Translational velocity response is first-order with an acceleration cap.
    * No current, hydrodynamic force model, actuator allocation, attitude
      dynamics, or sensor latency is included.

A successful run demonstrates practical convergence for this test scenario,
not a formal proof of closed-loop stability for all initial conditions.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

from scripts.target_tracking import TargetTrackingController, TargetVelocityEstimator


# ---------------------------- User settings ---------------------------- #
DT = 0.05                         # simulation sample time [s]
SIM_TIME = 100.0                  # total simulation time [s]
DESIRED_REL_FRD = np.array([10.0, 0.0, 0.0])  # target 10 m ahead [m]

INITIAL_VEHICLE_NED = np.array([0.0, 0.0, 20.0])
INITIAL_TARGET_NED = np.array([35.0, 10.0, 18.0])

# Fixed controlled-vehicle attitude for this kinematic test.
# Yaw is clockwise from North; pitch is positive nose-up.
VEHICLE_YAW_RAD = math.radians(14.3239449)  # about 0.25 rad
VEHICLE_PITCH_RAD = math.radians(-2.2918312)  # about -0.04 rad

# Controlled-vehicle translational response model.
VEHICLE_VELOCITY_TAU = 0.50       # first-order velocity response [s]
VEHICLE_MAX_ACCEL = 0.65          # acceleration magnitude limit [m/s^2]
MAX_SPEED = 2.0                   # controller speed limit [m/s]

# Target motion estimator. Set > 0 to add relative-position measurement noise.
ESTIMATOR_TAU = 0.30              # low-pass time constant [s]
POSITION_NOISE_STD_M = 0.0        # baseline test is noise-free
RANDOM_SEED = 7

# Practical convergence thresholds for this chosen scenario.
TAIL_WINDOW_FRACTION = 0.20
TAIL_RMS_LIMIT_M = 0.50
FINAL_ERROR_LIMIT_M = 0.50
MAX_ERROR_LIMIT_M = 40.0

OUTPUT_DIR = Path(__file__).resolve().parent


# -------------------------- Coordinate transforms ---------------------- #
def rotation_body_to_ned(yaw: float, pitch: float) -> np.ndarray:
    """FRD-to-NED rotation for roll=0, 3-2-1 yaw-pitch-roll convention.

    Body FRD: x forward, y starboard/right, z down.
    NED:      x north,   y east,          z down.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array(
        [
            [cp * cy, -sy, sp * cy],
            [cp * sy,  cy, sp * sy],
            [-sp,      0.0, cp],
        ],
        dtype=float,
    )


# --------------------------- Vehicle models ---------------------------- #
@dataclass
class TargetVehicleNED:
    """Autonomous target with a prescribed smooth NED velocity profile."""
    position_ned: np.ndarray

    def velocity_ned(self, time_s: float) -> np.ndarray:
        """Ground velocity [m/s]; smooth, gently curving 3D trajectory."""
        return np.array(
            [
                1.00,
                0.22 * math.sin(0.06 * time_s),
                0.04 * math.cos(0.04 * time_s),
            ],
            dtype=float,
        )

    def step(self, time_s: float, dt: float) -> None:
        self.position_ned = self.position_ned + self.velocity_ned(time_s) * dt


@dataclass
class ControlledVehicleNED:
    """Fully actuated 3-axis translational kinematic vehicle.

    Attitude is fixed in this baseline case. The velocity state is stored in
    body FRD but represents ground-referenced velocity (no ocean current).
    A first-order response and acceleration-magnitude limit approximate the
    low-level velocity controller without modeling forces/thrusters.
    """
    position_ned: np.ndarray
    rotation_body_to_ned: np.ndarray
    velocity_tau: float = VEHICLE_VELOCITY_TAU
    max_acceleration: float = VEHICLE_MAX_ACCEL

    def __post_init__(self) -> None:
        self.velocity_body = np.zeros(3, dtype=float)
        self.rotation_ned_to_body = self.rotation_body_to_ned.T

    def step(self, velocity_command_body: np.ndarray, dt: float) -> None:
        command = np.asarray(velocity_command_body, dtype=float)
        acceleration_body = (command - self.velocity_body) / self.velocity_tau
        acceleration_norm = float(np.linalg.norm(acceleration_body))
        if acceleration_norm > self.max_acceleration:
            acceleration_body *= self.max_acceleration / acceleration_norm

        self.velocity_body += acceleration_body * dt
        velocity_ned = self.rotation_body_to_ned @ self.velocity_body
        self.position_ned += velocity_ned * dt

    def target_relative_position_body(self, target_position_ned: np.ndarray) -> np.ndarray:
        """Target relative position, expressed in controlled-vehicle FRD."""
        delta_ned = target_position_ned - self.position_ned
        return self.rotation_ned_to_body @ delta_ned


@dataclass
class SimulationResult:
    mode: str
    time: np.ndarray
    target_ned: np.ndarray
    vehicle_ned: np.ndarray
    desired_vehicle_ned: np.ndarray
    relative_position_frd: np.ndarray
    error_frd: np.ndarray
    command_frd: np.ndarray
    estimated_target_velocity_frd: np.ndarray
    true_target_velocity_frd: np.ndarray
    beta_hat: np.ndarray
    alpha_hat: np.ndarray
    metrics: dict


def make_controller(mode: str) -> TargetTrackingController:
    """Controller gains for the example comparison; tune for your vehicle."""
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
            gamma_h=0.005,
            gamma_v=0.005,
            k_along=0.30,
            beta_inner=0.4,
            beta_outer=0.7,
            alpha_inner=0.4,
            alpha_outer=0.7,
            **common,
        )

    raise ValueError("mode must be 'pid' or 'alos'")


def simulate(mode: str) -> SimulationResult:
    """Run one closed-loop case. The controller never receives true target v."""
    rng = np.random.default_rng(RANDOM_SEED)
    rotation_b_to_n = rotation_body_to_ned(
        VEHICLE_YAW_RAD, VEHICLE_PITCH_RAD
    )

    vehicle = ControlledVehicleNED(
        position_ned=INITIAL_VEHICLE_NED.copy(),
        rotation_body_to_ned=rotation_b_to_n,
    )
    target = TargetVehicleNED(position_ned=INITIAL_TARGET_NED.copy())

    estimator = TargetVelocityEstimator(tau=ESTIMATOR_TAU)
    controller = make_controller(mode)

    body_rates_frd = np.zeros(3)  # fixed attitude in this test
    n_steps = int(round(SIM_TIME / DT))

    times = np.empty(n_steps)
    target_log = np.empty((n_steps, 3))
    vehicle_log = np.empty((n_steps, 3))
    desired_vehicle_log = np.empty((n_steps, 3))
    relative_log = np.empty((n_steps, 3))
    error_log = np.empty((n_steps, 3))
    command_log = np.empty((n_steps, 3))
    estimated_target_velocity_log = np.full((n_steps, 3), np.nan)
    true_target_velocity_log = np.empty((n_steps, 3))
    beta_log = np.zeros(n_steps)
    alpha_log = np.zeros(n_steps)

    for k in range(n_steps):
        time_s = k * DT

        # Relative target position is the tracking sensor measurement.
        relative_true_frd = vehicle.target_relative_position_body(
            target.position_ned
        )
        measurement = relative_true_frd.copy()
        if POSITION_NOISE_STD_M > 0.0:
            measurement += rng.normal(
                0.0, POSITION_NOISE_STD_M, size=3
            )

        # Crucially, estimate target velocity from relative position history.
        # Only the controlled vehicle's own actual velocity and IMU rates are
        # supplied; the target's true velocity is used only for simulation.
        target_velocity_est_frd = estimator.update(
            target_rel_body=measurement,
            vehicle_velocity_body=vehicle.velocity_body,
            body_rates_frd=body_rates_frd,
            dt=DT,
        )

        if target_velocity_est_frd is None:
            command_frd = np.zeros(3)
        else:
            command_frd = np.asarray(
                controller.compute(
                    target_rel_body=measurement,
                    desired_rel_body=DESIRED_REL_FRD,
                    target_velocity_body=target_velocity_est_frd,
                    body_rates_frd=body_rates_frd,
                    dt=DT,
                ),
                dtype=float,
            )
            estimated_target_velocity_log[k] = target_velocity_est_frd

        # Ground-truth velocity is logged only for estimator diagnostics.
        true_v_ned = target.velocity_ned(time_s)
        true_v_frd = vehicle.rotation_ned_to_body @ true_v_ned

        # Desired vehicle location in NED for the specified FRD stand-off.
        desired_vehicle_ned = (
            target.position_ned
            - rotation_b_to_n @ DESIRED_REL_FRD
        )

        times[k] = time_s
        target_log[k] = target.position_ned
        vehicle_log[k] = vehicle.position_ned
        desired_vehicle_log[k] = desired_vehicle_ned
        relative_log[k] = relative_true_frd
        error_log[k] = relative_true_frd - DESIRED_REL_FRD
        command_log[k] = command_frd
        true_target_velocity_log[k] = true_v_frd
        beta_log[k] = controller.beta_hat
        alpha_log[k] = controller.alpha_hat

        # Integrate both vehicle kinematic models in NED.
        vehicle.step(command_frd, DT)
        target.step(time_s, DT)

    error_norm = np.linalg.norm(error_log, axis=1)
    tail_start = int((1.0 - TAIL_WINDOW_FRACTION) * n_steps)
    tail_error = error_log[tail_start:]
    tail_rms = float(np.sqrt(np.mean(np.sum(tail_error**2, axis=1))))
    final_error = float(np.linalg.norm(error_log[-1]))
    peak_error = float(np.max(error_norm))
    all_finite = bool(
        np.isfinite(error_log).all()
        and np.isfinite(vehicle_log).all()
        and np.isfinite(command_log).all()
    )
    bounded = bool(all_finite and peak_error < MAX_ERROR_LIMIT_M)
    converged = bool(
        bounded
        and tail_rms < TAIL_RMS_LIMIT_M
        and final_error < FINAL_ERROR_LIMIT_M
    )

    metrics = {
        "tail_rms_m": tail_rms,
        "final_error_m": final_error,
        "peak_error_m": peak_error,
        "bounded": bounded,
        "converged": converged,
        "tail_start_s": float(times[tail_start]),
    }

    return SimulationResult(
        mode=mode,
        time=times,
        target_ned=target_log,
        vehicle_ned=vehicle_log,
        desired_vehicle_ned=desired_vehicle_log,
        relative_position_frd=relative_log,
        error_frd=error_log,
        command_frd=command_log,
        estimated_target_velocity_frd=estimated_target_velocity_log,
        true_target_velocity_frd=true_target_velocity_log,
        beta_hat=beta_log,
        alpha_hat=alpha_log,
        metrics=metrics,
    )


def plot_results(results: list[SimulationResult], show: bool = True) -> None:
    """Save NED 3D trajectories and FRD tracking-error time histories."""
    # 3D NED trajectories.
    fig = plt.figure(figsize=(11.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    first = results[0]
    ax.plot(
        first.target_ned[:, 0],
        first.target_ned[:, 1],
        first.target_ned[:, 2],
        linewidth=2.6,
        label="Autonomous target",
    )

    # Desired vehicle locus is common because the stand-off vector and
    # controlled-vehicle attitude are the same in both runs.
    ax.plot(
        first.desired_vehicle_ned[:, 0],
        first.desired_vehicle_ned[:, 1],
        first.desired_vehicle_ned[:, 2],
        linestyle="--",
        linewidth=1.8,
        label="Desired vehicle locus (10 m FRD stand-off)",
    )

    for result in results:
        ax.plot(
            result.vehicle_ned[:, 0],
            result.vehicle_ned[:, 1],
            result.vehicle_ned[:, 2],
            linewidth=1.8,
            label=f"Controlled vehicle ({result.mode.upper()})",
        )
        ax.scatter(
            result.vehicle_ned[0, 0],
            result.vehicle_ned[0, 1],
            result.vehicle_ned[0, 2],
            marker="o",
            s=26,
        )

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Down (m)")
    ax.set_title("Closed-loop underwater vehicle tracking in NED")
    ax.invert_zaxis()  # NED down-positive axis is displayed downward.
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    trajectory_path = OUTPUT_DIR / "closed_loop_ned_trajectories.png"
    fig.savefig(trajectory_path, dpi=180, bbox_inches="tight")

    # FRD tracking errors; positive error means target is farther +axis than
    # desired, e.g. positive x error means the vehicle is behind the target.
    fig_err, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axis_labels = [
        ("Forward error, $e_x$ (m)", "Forward (x, FRD)"),
        ("Right error, $e_y$ (m)", "Right (y, FRD)"),
        ("Down error, $e_z$ (m)", "Down (z, FRD)"),
    ]

    for i, ax_err in enumerate(axes):
        for result in results:
            ax_err.plot(
                result.time,
                result.error_frd[:, i],
                linewidth=1.5,
                label=result.mode.upper(),
            )
        ax_err.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax_err.set_ylabel(axis_labels[i][0])
        ax_err.grid(True, alpha=0.3)
        if i == 0:
            ax_err.legend()

    axes[-1].set_xlabel("Time (s)")
    fig_err.suptitle("Target-relative tracking error in body FRD")
    fig_err.tight_layout()
    error_path = OUTPUT_DIR / "closed_loop_frd_tracking_errors.png"
    fig_err.savefig(error_path, dpi=180, bbox_inches="tight")

    print(f"Saved trajectory plot: {trajectory_path}")
    print(f"Saved FRD error plot:  {error_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def print_summary(results: list[SimulationResult]) -> bool:
    print("\nClosed-loop tracking summary")
    print("=" * 76)
    print(
        f"{'Controller':<12} {'Tail RMS (m)':>14} "
        f"{'Final |e| (m)':>15} {'Peak |e| (m)':>14} {'Result':>10}"
    )
    print("-" * 76)
    all_passed = True
    for result in results:
        m = result.metrics
        passed = m["bounded"] and m["converged"]
        all_passed = all_passed and passed
        print(
            f"{result.mode.upper():<12} "
            f"{m['tail_rms_m']:>14.4f} "
            f"{m['final_error_m']:>15.4f} "
            f"{m['peak_error_m']:>14.4f} "
            f"{'PASS' if passed else 'FAIL':>10}"
        )
    print("-" * 76)
    print(
        "Pass criteria: finite/bounded error, final error < "
        f"{FINAL_ERROR_LIMIT_M:.2f} m, and final-window RMS < "
        f"{TAIL_RMS_LIMIT_M:.2f} m."
    )
    print(
        "These are scenario-specific numerical checks, not a formal "
        "Lyapunov stability proof."
    )
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PID and ALOS NED closed-loop tracking simulations."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive plot windows.",
    )
    args = parser.parse_args()

    results = [simulate("pid"), simulate("alos")]
    passed = print_summary(results)
    plot_results(results, show=not args.no_show)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
