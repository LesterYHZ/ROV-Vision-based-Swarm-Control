"""Velocity-level control-barrier-function collision avoidance.

The filter modifies a nominal body-FRD velocity command so that spherical
obstacle/vehicle separation satisfies a continuous-time CBF condition, when
that condition is feasible under the configured speed limit.

Dependencies:
    numpy
    scipy

Important assumptions:
- Obstacle positions are relative to the controlled vehicle, in body FRD.
- Obstacle IDs remain consistent between successive calls. If IDs are omitted,
  input-list order is used as the identity.
- vehicle_velocity_body is the controlled vehicle's ACTUAL ground-relative
  velocity in body FRD, not merely the new requested velocity.
- Relative obstacle motion is estimated from successive position detections.
- The vehicle is treated as a velocity-controlled kinematic point/sphere.
  Acceleration limits, actuator dynamics, sensing latency, and inter-sample
  collision checking require additional treatment in a real vehicle.

For each obstacle, define r = p_obstacle - p_vehicle in body FRD and
h(r) = r.T @ r - R_safe**2. With measured relative-position rate r_dot and
actual vehicle velocity v_actual, the instantaneous radial CBF condition for
candidate command v is

    h_dot + alpha*h >= 0

which is equivalent to

    2*r.T @ v <= 2*r.T @ (r_dot + v_actual) + alpha*h.

For a spherical distance barrier, the rotating-frame term -omega x r is
orthogonal to r and therefore vanishes from r.T @ r_dot. This is why body
angular rates are not required in this particular radial CBF inequality.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Optional, Sequence
import math

import numpy as np
from scipy.optimize import minimize


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 elements")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _positions(value: Iterable[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(list(value), dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("obstacle_positions_body must have shape (N, 3)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Obstacle positions must contain only finite values")
    return arr


def _expand_radii(radii, n: int) -> np.ndarray:
    if np.isscalar(radii):
        arr = np.full(n, float(radii), dtype=float)
    else:
        arr = np.asarray(radii, dtype=float)
        if arr.shape != (n,):
            raise ValueError("obstacle_radii must be a scalar or have N values")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Obstacle radii must be finite and nonnegative")
    return arr


class CBFCollisionAvoidance:
    """Stateful multi-obstacle CBF filter for 3D body-FRD velocity commands."""

    def __init__(
        self,
        vehicle_radius: float = 0.75,
        safety_margin: float = 0.50,
        alpha: float = 1.0,
        max_speed: float = 2.0,
        unknown_closing_speed: float = 0.5,
        solver_tolerance: float = 1e-8,
    ) -> None:
        values = (vehicle_radius, safety_margin, alpha,
                  max_speed, unknown_closing_speed, solver_tolerance)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("All constructor parameters must be finite")
        if vehicle_radius < 0 or safety_margin < 0:
            raise ValueError("Radii and safety_margin must be nonnegative")
        if alpha <= 0 or max_speed <= 0:
            raise ValueError("alpha and max_speed must be positive")
        if unknown_closing_speed < 0 or solver_tolerance <= 0:
            raise ValueError("Invalid unknown_closing_speed or solver_tolerance")

        self.vehicle_radius = float(vehicle_radius)
        self.safety_margin = float(safety_margin)
        self.alpha = float(alpha)
        self.max_speed = float(max_speed)
        self.unknown_closing_speed = float(unknown_closing_speed)
        self.solver_tolerance = float(solver_tolerance)
        self._previous_positions: dict[Hashable, np.ndarray] = {}

    def reset(self) -> None:
        """Clear obstacle history; next detections use the unknown-rate bound."""
        self._previous_positions.clear()

    def filter_velocity(
        self,
        v_nom: Sequence[float],
        obstacle_positions_body: Iterable[Sequence[float]],
        vehicle_velocity_body: Sequence[float],
        dt: float,
        obstacle_radii: float | Sequence[float] = 0.0,
        obstacle_ids: Optional[Sequence[Hashable]] = None,
        return_info: bool = False,
    ):
        """Filter a nominal velocity command using one CBF constraint/obstacle.

        Args:
            v_nom: Nominal tracking velocity [vx, vy, vz] in body FRD (m/s).
            obstacle_positions_body: N obstacle/vehicle positions relative to
                the controlled vehicle, each [x, y, z] in body FRD (m).
            vehicle_velocity_body: Actual current vehicle ground velocity in
                body FRD (m/s), preferably from INS/DVL. Do not pass the newly
                requested command unless ideal instantaneous tracking is an
                intentional simulation assumption.
            dt: Time since the previous obstacle-position update (seconds).
            obstacle_radii: Obstacle radii (scalar or N-vector, metres).
            obstacle_ids: Stable IDs in the same order as obstacle positions.
                If omitted, list indices are used; keep ordering consistent.
            return_info: If True, return (v_safe, diagnostic_dictionary).

        Returns:
            v_safe: Collision-filtered body-FRD velocity command (m/s).
            With return_info=True, also returns solver/safety diagnostics.

        New detections have no finite-difference history. For those, a
        conservative radial closing-rate assumption of
        -unknown_closing_speed is used. Tune this to the fastest plausible
        closing speed of an undetected/moving obstacle, or initialize history
        before enabling nominal motion.
        """
        v_nom = _vec3(v_nom, "v_nom")
        v_actual = _vec3(vehicle_velocity_body, "vehicle_velocity_body")
        positions = _positions(obstacle_positions_body)

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        n_obs = len(positions)
        radii = _expand_radii(obstacle_radii, n_obs)

        if obstacle_ids is None:
            ids = list(range(n_obs))
        else:
            ids = list(obstacle_ids)
            if len(ids) != n_obs:
                raise ValueError("obstacle_ids must have one ID per obstacle")
            if len(set(ids)) != len(ids):
                raise ValueError("obstacle_ids must be unique within a call")

        # No obstacle detected: only enforce the configured speed limit.
        if n_obs == 0:
            self._previous_positions = {}
            v_out = self._limit_speed(v_nom)
            info = {
                "feasible": True,
                "status": "no_obstacles",
                "min_cbf_margin": math.inf,
                "cbf_margins": [],
                "estimated_radial_rates": [],
            }
            return (v_out, info) if return_info else v_out

        # Build radial CBF constraints A @ v <= b.
        A = np.zeros((n_obs, 3), dtype=float)
        b = np.zeros(n_obs, dtype=float)
        margins_at_nominal = []
        radial_rates = []
        safe_distances = []
        distances = []

        current_history: dict[Hashable, np.ndarray] = {}

        for i, (obs_id, r) in enumerate(zip(ids, positions)):
            distance = float(np.linalg.norm(r))
            if distance < 1e-9:
                # Direction is undefined at coincident centers; no radial CBF
                # can recover the geometry. Keep a zero direction and flag it.
                raise ValueError(
                    f"Obstacle {obs_id!r} is at/near the vehicle origin; "
                    "collision is already occurring or the detection is invalid"
                )

            previous = self._previous_positions.get(obs_id)
            if previous is None:
                # On first sighting, assume worst-case radial approach.
                r_dot = -self.unknown_closing_speed * (r / distance)
            else:
                # Body-frame relative displacement. For a spherical barrier,
                # omega x r is perpendicular to r and does not affect h_dot.
                r_dot = (r - previous) / dt

            current_history[obs_id] = r.copy()
            radial_rate = float(np.dot(r / distance, r_dot))

            safe_distance = (
                self.vehicle_radius + float(radii[i]) + self.safety_margin
            )
            h = distance**2 - safe_distance**2

            # h_dot + alpha*h >= 0  <=>  A_i @ v <= b_i
            A[i, :] = 2.0 * r
            b[i] = 2.0 * float(np.dot(r, r_dot + v_actual)) + self.alpha * h

            radial_rates.append(radial_rate)
            safe_distances.append(safe_distance)
            distances.append(distance)

        # Replace the history so disappeared IDs do not retain stale tracks.
        self._previous_positions = current_history

        # Minimize 1/2 ||v - v_nom||^2 subject to CBF halfspaces and speed ball.
        x0 = self._limit_speed(v_nom)

        def objective(v):
            d = v - v_nom
            return 0.5 * float(np.dot(d, d))

        def objective_jac(v):
            return v - v_nom

        constraints = [
            {
                "type": "ineq",
                "fun": lambda v, A=A, b=b: b - A @ v,
                "jac": lambda v, A=A, b=b: -A,
            },
            {
                "type": "ineq",
                "fun": lambda v: self.max_speed**2 - float(np.dot(v, v)),
                "jac": lambda v: -2.0 * v,
            },
        ]

        result = minimize(
            objective,
            x0,
            jac=objective_jac,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": self.solver_tolerance, "maxiter": 200, "disp": False},
        )

        candidate = np.asarray(result.x, dtype=float)
        cbf_margins = b - A @ candidate
        speed_ok = np.linalg.norm(candidate) <= self.max_speed + 1e-6
        constraints_ok = np.all(cbf_margins >= -1e-6)
        feasible = bool(np.all(np.isfinite(candidate)) and speed_ok and constraints_ok)

        if feasible:
            v_safe = candidate
            status = "optimal" if result.success else "feasible_iterate"
        else:
            # Emergency fallback. A stop command is not guaranteed safe when
            # an obstacle is closing or the vehicle cannot stop instantaneously.
            v_safe = np.zeros(3, dtype=float)
            fallback_margins = b - A @ v_safe
            fallback_feasible = bool(np.all(fallback_margins >= -1e-6))
            status = "infeasible_emergency_stop"
            feasible = fallback_feasible
            cbf_margins = fallback_margins

        info = {
            "feasible": feasible,
            "solver_success": bool(result.success),
            "status": status,
            "solver_message": str(result.message),
            "min_cbf_margin": float(np.min(cbf_margins)),
            "cbf_margins": [float(x) for x in cbf_margins],
            "distances": distances,
            "safe_distances": safe_distances,
            "estimated_radial_rates": radial_rates,
            "nominal_speed": float(np.linalg.norm(v_nom)),
            "filtered_speed": float(np.linalg.norm(v_safe)),
        }

        output = tuple(float(x) for x in v_safe)
        return (output, info) if return_info else output

    def _limit_speed(self, v: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(v))
        if speed > self.max_speed:
            return v * (self.max_speed / speed)
        return v.copy()


if __name__ == "__main__":
    # Minimal example: a stationary obstacle ahead of the vehicle.
    cbf = CBFCollisionAvoidance(
        vehicle_radius=0.75,
        safety_margin=0.50,
        alpha=1.0,
        max_speed=2.0,
        unknown_closing_speed=0.5,
    )

    # Prime the obstacle history, then move closer over one sample.
    cbf.filter_velocity(
        v_nom=(1.5, 0.0, 0.0),
        obstacle_positions_body=[(2.1, 0.0, 0.0)],
        vehicle_velocity_body=(1.0, 0.0, 0.0),
        dt=0.1,
        obstacle_ids=["obstacle_1"],
    )

    v_safe, diagnostics = cbf.filter_velocity(
        v_nom=(1.5, 0.0, 0.0),
        obstacle_positions_body=[(2.0, 0.0, 0.0)],
        vehicle_velocity_body=(1.0, 0.0, 0.0),
        dt=0.1,
        obstacle_ids=["obstacle_1"],
        return_info=True,
    )

    print("Filtered velocity [FRD] (m/s):", v_safe)
    print("Diagnostics:", diagnostics)
