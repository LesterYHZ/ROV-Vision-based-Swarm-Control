"""CBF-QP safety filter for keeping a leader inside a body-fixed camera FOV.

Camera convention:
    Camera optical axis: body +x (FRD forward)
    Image horizontal axis: body +y (FRD right)
    Image vertical axis: body +z (FRD down)

The horizontal and vertical FOV are independent full-angle values.
The admissible set is the camera frustum intersected with x >= min_depth.
The QP modifies a nominal body-FRD translational velocity command as little
as possible while enforcing exponential CBF inequalities for all frustum
faces and the positive-depth face.

The default camera alignment is body FRD, but a fixed body-to-camera
extrinsic rotation can be supplied. Body angular rates are measured/exogenous;
only translational velocity is optimized. Relative target velocity is estimated from successive camera-pose
measurements and the controlled vehicle's actual ground velocity.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return arr


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


class FOVCBFVelocityFilter:
    """Perception-safety CBF-QP filter for a leader's camera visibility.

    Parameters
    ----------
    horizontal_fov_deg, vertical_fov_deg:
        Full camera field-of-view angles, in degrees.
    fov_margin_deg:
        Inward angular margin applied symmetrically to each FOV edge.
        A scalar applies to both axes; a pair means (horizontal, vertical).
    alpha:
        Exponential CBF rate, 1/s.
    min_depth:
        Minimum forward distance required for visibility, metres.
    max_speed:
        Euclidean norm limit on the output command. It is implemented as
        an inscribed L1-ball polytope so the optimization remains a QP.
    velocity_weights:
        Positive diagonal weights for modifying body x/y/z commands.
    velocity_filter_tau:
        Low-pass time constant for the finite-difference relative velocity.
    solver_tolerance:
        Numerical feasibility tolerance for the QP check.
    """

    def __init__(
        self,
        horizontal_fov_deg: float = 90.0,
        vertical_fov_deg: float = 60.0,
        fov_margin_deg: float | Sequence[float] = 4.0,
        alpha: float = 1.5,
        min_depth: float = 0.25,
        barrier_buffer_m: float = 0.20,
        max_speed: float = 2.0,
        velocity_weights: Sequence[float] = (1.0, 1.0, 1.0),
        R_body_to_camera: Optional[Sequence[Sequence[float]]] = None,
        velocity_filter_tau: float = 0.15,
        solver_tolerance: float = 2e-6,
        max_iterations: int = 150,
    ) -> None:
        values = [horizontal_fov_deg, vertical_fov_deg, alpha,
                  min_depth, barrier_buffer_m, max_speed, velocity_filter_tau,
                  solver_tolerance]
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("All scalar parameters must be finite")
        if not (0.0 < horizontal_fov_deg < 179.0):
            raise ValueError("horizontal_fov_deg must be in (0, 179)")
        if not (0.0 < vertical_fov_deg < 179.0):
            raise ValueError("vertical_fov_deg must be in (0, 179)")
        if alpha <= 0 or min_depth <= 0 or max_speed <= 0 or barrier_buffer_m < 0:
            raise ValueError("alpha, min_depth and max_speed must be positive; barrier_buffer_m must be nonnegative")
        if velocity_filter_tau <= 0 or solver_tolerance <= 0:
            raise ValueError("Filter time constant and tolerance must be positive")
        if isinstance(fov_margin_deg, (int, float)):
            margin_h = margin_v = float(fov_margin_deg)
        else:
            if len(fov_margin_deg) != 2:
                raise ValueError("fov_margin_deg must be scalar or (horizontal, vertical)")
            margin_h, margin_v = map(float, fov_margin_deg)
        if margin_h < 0 or margin_v < 0:
            raise ValueError("FOV margins must be nonnegative")

        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.half_h = math.radians(horizontal_fov_deg / 2.0 - margin_h)
        self.half_v = math.radians(vertical_fov_deg / 2.0 - margin_v)
        if not (0.0 < self.half_h < math.pi / 2 and 0.0 < self.half_v < math.pi / 2):
            raise ValueError("Margins must leave positive half-FOV angles below 90 deg")

        self.alpha = float(alpha)
        self.min_depth = float(min_depth)
        self.barrier_buffer_m = float(barrier_buffer_m)
        self.max_speed = float(max_speed)
        self.weights = _vec3(velocity_weights, "velocity_weights")
        if R_body_to_camera is None:
            self.R_body_to_camera = np.eye(3)
        else:
            self.R_body_to_camera = np.asarray(R_body_to_camera, dtype=float)
            if self.R_body_to_camera.shape != (3, 3) or not np.all(np.isfinite(self.R_body_to_camera)):
                raise ValueError("R_body_to_camera must be a finite 3x3 matrix")
            if not np.allclose(self.R_body_to_camera @ self.R_body_to_camera.T, np.eye(3), atol=1e-6):
                raise ValueError("R_body_to_camera must be orthonormal")
            if not np.isclose(np.linalg.det(self.R_body_to_camera), 1.0, atol=1e-6):
                raise ValueError("R_body_to_camera must be a proper rotation")
        if np.any(self.weights <= 0):
            raise ValueError("velocity_weights must be positive")
        self.velocity_filter_tau = float(velocity_filter_tau)
        self.solver_tolerance = float(solver_tolerance)
        self.max_iterations = int(max_iterations)

        self._previous_relative_position: Optional[np.ndarray] = None
        self._relative_velocity_estimate: Optional[np.ndarray] = None

        # Eight half-spaces describe |vx| + |vy| + |vz| <= max_speed.
        # Since ||v||2 <= ||v||1, this conservatively guarantees ||v||2 <= max_speed.
        self._speed_A = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
        self._speed_b = np.full(8, self.max_speed, dtype=float)

    def reset(self) -> None:
        """Clear relative-motion history (e.g. after target re-acquisition)."""
        self._previous_relative_position = None
        self._relative_velocity_estimate = None

    @property
    def effective_half_fov_deg(self) -> tuple[float, float]:
        return math.degrees(self.half_h), math.degrees(self.half_v)

    def fov_state(self, leader_rel_body: Sequence[float]) -> dict:
        """Return measured angular coordinates and geometric FOV margins (using camera extrinsics)."""
        r_body = _vec3(leader_rel_body, "leader_rel_body")
        r = self.R_body_to_camera @ r_body
        x, y, z = r
        rho = math.hypot(x, y)
        az = math.degrees(math.atan2(y, x))
        el = math.degrees(math.atan2(z, rho))
        full_half_h = self.horizontal_fov_deg / 2.0
        full_half_v = self.vertical_fov_deg / 2.0
        eff_h, eff_v = self.effective_half_fov_deg
        return {
            "depth_m": float(x),
            "azimuth_deg": az,
            "elevation_deg": el,
            "horizontal_margin_deg": float(eff_h - abs(az)),
            "vertical_margin_deg": float(eff_v - abs(el)),
            "inside_physical_fov": bool(
                x > 0.0 and abs(az) <= full_half_h and abs(el) <= full_half_v
            ),
            "inside_margined_fov": bool(
                x >= self.min_depth
                and abs(az) <= eff_h
                and abs(el) <= eff_v
            ),
        }

    def _frustum_barriers(self, r: np.ndarray):
        """Return (h, grad_h, labels) for five convex frustum half-spaces.

        With optical axis +x, the exact angular frustum conditions are:
          x > 0, |y| <= x tan(horizontal_half_angle),
          |z| <= hypot(x,y) tan(vertical_half_angle).
        """
        x, y, z = r
        rho = math.hypot(x, y)
        if rho < 1e-8:
            raise ValueError("Leader is too close to the camera lateral/depth axis singularity")

        th = math.tan(self.half_h)
        tv = math.tan(self.half_v)

        h = np.array([
            x * th - y,       # horizontal right edge
            x * th + y,       # horizontal left edge
            rho * tv - z,     # vertical down edge (+z is down)
            rho * tv + z,     # vertical up edge
            x - self.min_depth,  # in front of camera / minimum depth
        ], dtype=float)

        G = np.array([
            [th, -1.0, 0.0],
            [th,  1.0, 0.0],
            [tv * x / rho, tv * y / rho, -1.0],
            [tv * x / rho, tv * y / rho,  1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
        labels = ["horizontal_right", "horizontal_left",
                  "vertical_down", "vertical_up", "minimum_depth"]
        return h, G, labels

    def _estimate_relative_velocity(
        self, r: np.ndarray, actual_velocity: np.ndarray, omega: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate r_dot in camera coordinates and implied target velocity.

        r_dot = v_target^B - v_vehicle^B - omega x r.
        Therefore v_target_hat = r_dot_hat + v_vehicle + omega x r.
        For a candidate v, predicted r_dot = r_dot_hat + v_actual - v.
        The angular term cancels because omega is measured and held fixed over
        the instantaneous QP prediction horizon.
        """
        if self._previous_relative_position is None:
            raw = np.zeros(3, dtype=float)
            self._relative_velocity_estimate = raw.copy()
        else:
            raw = (r - self._previous_relative_position) / dt
            a = 1.0 - math.exp(-dt / self.velocity_filter_tau)
            if self._relative_velocity_estimate is None:
                self._relative_velocity_estimate = raw.copy()
            else:
                self._relative_velocity_estimate = (
                    (1.0 - a) * self._relative_velocity_estimate + a * raw
                )

        self._previous_relative_position = r.copy()
        r_dot = self._relative_velocity_estimate.copy()
        target_velocity_hat = r_dot + actual_velocity + _cross(omega, r)
        return r_dot, target_velocity_hat

    def filter_velocity(
        self,
        v_nom: Sequence[float],
        leader_rel_body: Sequence[float],
        vehicle_velocity_body: Sequence[float],
        body_rates_frd: Sequence[float] = (0.0, 0.0, 0.0),
        dt: float = 0.05,
        fallback_velocity: Optional[Sequence[float]] = None,
        return_info: bool = False,
    ):
        """Optimize the nominal velocity to keep the leader inside camera FOV.

        Args:
            v_nom: nominal tracking velocity [m/s] in body FRD.
            leader_rel_body: leader position [m] in camera/body FRD.
            vehicle_velocity_body: actual vehicle ground velocity [m/s] in FRD.
            body_rates_frd: measured angular velocity [rad/s] in FRD.
            dt: sample time [s].
            fallback_velocity: command returned if QP is infeasible; defaults to zero.
            return_info: when True, return (v_safe, diagnostics).

        Returns:
            v_safe (vx, vy, vz), optionally plus diagnostics.

        The CBF guarantee applies only if initially inside the margined frustum,
        the relative-motion estimate is sufficiently accurate, and the actual
        velocity tracks the command quickly enough. It does not model camera
        occlusion, detector dropout, or attitude-rate decision variables.
        """
        v_nom = _vec3(v_nom, "v_nom")
        r_body = _vec3(leader_rel_body, "leader_rel_body")
        v_actual_body = _vec3(vehicle_velocity_body, "vehicle_velocity_body")
        omega_body = _vec3(body_rates_frd, "body_rates_frd")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        R_bc = self.R_body_to_camera
        r = R_bc @ r_body
        v_actual_camera = R_bc @ v_actual_body
        omega_camera = R_bc @ omega_body

        h, G, labels = self._frustum_barriers(r)
        # Tighten each face inward to account for finite sampling, pose noise,
        # and small command-tracking lag. This is a geometric guard band, not
        # a formal actuator-dynamics/HOCBF guarantee.
        h = h - self.barrier_buffer_m
        r_dot, v_target_hat = self._estimate_relative_velocity(
            r, v_actual_camera, omega_camera, dt
        )

        # CBF: grad(h_j) dot (r_dot_hat + R_bc*v_actual - R_bc*v) + alpha*h_j >= 0
        # Rearranged into standard QP half-space: A v_body <= b.
        A_fov = G @ R_bc
        b_fov = np.einsum("ij,j->i", G, r_dot + v_actual_camera) + self.alpha * h

        A = np.vstack((A_fov, self._speed_A))
        b = np.concatenate((b_fov, self._speed_b))

        W = np.diag(self.weights)
        def objective(v):
            d = v - v_nom
            return 0.5 * float(d @ W @ d)

        def objective_jac(v):
            return W @ (v - v_nom)

        # A strictly convex QP with linear inequalities. SLSQP is used as the
        # numerical optimizer; the mathematical optimization remains a QP.
        # Start from a speed-feasible command; SLSQP finds a feasible point if one exists.
        v0 = v_nom.copy()
        l1 = np.sum(np.abs(v0))
        if l1 > self.max_speed:
            v0 *= self.max_speed / l1

        result = minimize(
            objective,
            v0,
            jac=objective_jac,
            method="SLSQP",
            constraints=[{
                "type": "ineq",
                "fun": lambda v: b - A @ v,
                "jac": lambda v: -A,
            }],
            options={"ftol": 1e-10, "maxiter": self.max_iterations, "disp": False},
        )

        candidate = np.asarray(result.x, dtype=float)
        residual = b - A @ candidate
        feasible = bool(
            np.all(np.isfinite(candidate))
            and np.all(residual >= -self.solver_tolerance)
            and np.sum(np.abs(candidate)) <= self.max_speed + self.solver_tolerance
        )

        if feasible:
            v_safe = candidate
            status = "optimal" if result.success else "feasible_iterate"
        else:
            v_safe = (
                np.zeros(3, dtype=float) if fallback_velocity is None
                else _vec3(fallback_velocity, "fallback_velocity")
            )
            status = "infeasible_or_solver_failure"

        fov = self.fov_state(r_body)
        diagnostics = {
            "feasible": feasible,
            "solver_success": bool(result.success),
            "solver_message": str(result.message),
            "status": status,
            "barrier_values": {name: float(value) for name, value in zip(labels, h)},
            "barrier_buffer_m": self.barrier_buffer_m,
            "cbf_margins": {name: float(value) for name, value in zip(labels, residual[:len(h)])},
            "min_cbf_margin": float(np.min(residual[:len(h)])),
            "relative_velocity_est_camera": r_dot.tolist(),
            "target_velocity_est_camera": v_target_hat.tolist(),
            "nominal_speed": float(np.linalg.norm(v_nom)),
            "safe_speed": float(np.linalg.norm(v_safe)),
            "correction_norm": float(np.linalg.norm(v_safe - v_nom)),
            "fov": fov,
        }
        safe_tuple = tuple(float(x) for x in v_safe)
        return (safe_tuple, diagnostics) if return_info else safe_tuple
