"""Unified CBF-QP for BlueROV2 Heavy tracking of a tail-AprilTag leader.

Decision q=[u_ref,v_ref,w_ref,r_ref] in body FRD. One QP stacks:
  * spherical collision-clearance barriers;
  * camera angular FOV and near/far range barriers;
  * leader relative-heading barriers; and
  * AprilTag face-incidence barrier.

The controller assumes roll/pitch stabilization; measured p/q rates are treated
as exogenous. It is a velocity-reference CBF (relative-degree one with respect
to the virtual command), not a formal safety guarantee for arbitrary actuator
lag, perception delay, occlusion, illumination, blur, or detector failure.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Sequence
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def _v3(x, name):
    a = np.asarray(x, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return a


def _R3(x, name):
    a = np.asarray(x, dtype=float)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(a.T @ a, np.eye(3), atol=2e-2) or not np.isclose(np.linalg.det(a), 1.0, atol=2e-2):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return a


def _cross(a, b):
    return np.cross(a, b)


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_log(R):
    """SO(3) logarithm as a rotation vector."""
    R = _R3(R, "R")
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(c)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-7:
        return 0.5 * vee
    s = max(math.sin(angle), 1e-8)
    return angle / (2.0 * s) * vee


def apriltag_pose_to_leader_pose(
    tag_position_camera,
    R_camera_tag,
    p_leader_tag,
    R_leader_tag,
    tag_front_normal_tag=(0.0, 0.0, 1.0),
):
    """Convert an AprilTag pose to the leader ROV pose and tag normal.

    Transform convention:
      R_camera_tag maps tag-frame vectors into camera coordinates.
      R_leader_tag maps tag-frame vectors into leader-body coordinates.
      p_leader_tag is the tag-center position relative to leader origin,
      expressed in leader FRD coordinates.
      tag_front_normal_tag is the outward normal of the printed/front face
      expressed in tag coordinates (must match the chosen detector convention).

    Returns (leader_origin_camera, R_camera_leader, tag_normal_camera).
    The tag position itself remains tag_position_camera and should be used for
    camera-FOV constraints because that is the feature being detected.
    """
    p_ct = _v3(tag_position_camera, "tag_position_camera")
    R_ct = _R3(R_camera_tag, "R_camera_tag")
    p_lt = _v3(p_leader_tag, "p_leader_tag")
    R_lt = _R3(R_leader_tag, "R_leader_tag")
    n_t = _v3(tag_front_normal_tag, "tag_front_normal_tag")
    n_t = n_t / np.linalg.norm(n_t)
    R_cl = R_ct @ R_lt.T
    p_cl = p_ct - R_cl @ p_lt
    n_c = R_ct @ n_t
    n_c = n_c / np.linalg.norm(n_c)
    return p_cl, R_cl, n_c


@dataclass
class CBFResult:
    command: np.ndarray
    feasible: bool
    solver_success: bool
    status: str
    min_cbf_margin: float
    collision_margins: np.ndarray
    fov_margins: np.ndarray
    heading_margins: np.ndarray
    tag_margins: np.ndarray
    active_constraints: int
    nominal_command: np.ndarray
    constraint_labels: tuple[str, ...]


class RelativeObjectVelocityEstimator:
    """Estimate translational object velocity in body FRD from relative position.

    v_obj^B = p_rel_dot^B + v_vehicle^B + omega_vehicle^B x p_rel^B.
    """
    def __init__(self, filter_tau=0.18):
        if filter_tau <= 0:
            raise ValueError("filter_tau must be positive")
        self.tau = float(filter_tau)
        self.previous = {}
        self.filtered = {}

    def reset(self):
        self.previous.clear()
        self.filtered.clear()

    def update(self, object_id: Hashable, p_rel_body, actual_nu, dt, roll_rate=0.0, pitch_rate=0.0):
        p = _v3(p_rel_body, "p_rel_body")
        nu = np.asarray(actual_nu, dtype=float)
        if nu.shape != (4,) or not np.all(np.isfinite(nu)):
            raise ValueError("actual_nu must be finite [u,v,w,r]")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be positive")
        omega = np.array([roll_rate, pitch_rate, nu[3]], dtype=float)
        if object_id not in self.previous:
            self.previous[object_id] = p.copy()
            self.filtered[object_id] = np.zeros(3)
            return np.zeros(3)
        p_dot = (p - self.previous[object_id]) / dt
        raw = p_dot + nu[:3] + _cross(omega, p)
        a = 1.0 - math.exp(-dt / self.tau)
        est = (1.0-a)*self.filtered[object_id] + a*raw
        self.previous[object_id] = p.copy()
        self.filtered[object_id] = est.copy()
        return est


class RelativeOrientationRateEstimator:
    """Estimate leader angular velocity in camera coordinates from pose history.

    R_camera_leader is the leader-body orientation relative to camera/body.
    log(R_k R_(k-1)^T)/dt estimates relative spatial angular velocity. Adding
    the measured follower angular velocity recovers leader angular velocity in
    the current camera frame (small-sample-step approximation).
    """
    def __init__(self, filter_tau=0.25):
        if filter_tau <= 0:
            raise ValueError("filter_tau must be positive")
        self.tau = float(filter_tau)
        self.previous = None
        self.filtered_target_omega = np.zeros(3)

    def reset(self):
        self.previous = None
        self.filtered_target_omega[:] = 0.0

    def update(self, R_camera_leader, actual_omega_camera, dt):
        R = _R3(R_camera_leader, "R_camera_leader")
        omega_vehicle = _v3(actual_omega_camera, "actual_omega_camera")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be positive")
        if self.previous is None:
            self.previous = R.copy()
            self.filtered_target_omega = omega_vehicle.copy() * 0.0
            return self.filtered_target_omega.copy()
        R_delta = R @ self.previous.T
        relative_omega = rotation_log(R_delta) / dt
        target_omega = relative_omega + omega_vehicle
        a = 1.0 - math.exp(-dt / self.tau)
        self.filtered_target_omega = (1.0-a)*self.filtered_target_omega + a*target_omega
        self.previous = R.copy()
        return self.filtered_target_omega.copy()


class UnifiedCollisionFOVCBF:
    """One QP for collision avoidance + camera FOV + AprilTag orientation.

    Decision q=[u,v,w,r] are desired body FRD surge, sway, heave, yaw rate.
    leader_rel_camera is the AprilTag center position in camera FRD.
    leader_rotation_camera maps leader-body FRD axes into camera FRD.
    The printed tag face's outward normal is tag_normal_leader in leader FRD.
    """
    def __init__(
        self,
        horizontal_fov_deg=70.0,
        vertical_fov_deg=48.0,
        fov_margin_deg=(5.0, 4.0),
        camera_offset_body=(0.22, 0.0, 0.0),
        min_camera_depth=0.35,
        max_camera_x=12.0,
        max_range_margin_m=0.10,
        fov_linear_margin=0.05,
        max_relative_heading_deg=42.0,
        heading_margin_deg=3.0,
        max_tag_incidence_deg=52.0,
        tag_normal_leader=(-1.0, 0.0, 0.0),
        vehicle_radius=0.34,
        obstacle_margin=0.28,
        alpha_fov=2.0,
        alpha_collision=1.7,
        alpha_heading=1.8,
        alpha_tag=1.8,
        command_lower=(-0.50, -0.32, -0.27, -0.60),
        command_upper=(0.62, 0.32, 0.27, 0.60),
        command_slew=(1.5, 1.5, 1.2, 6.0),
        weights=(1.0, 3.5, 1.8, 0.25),
        slew_weight=0.12,
        estimator_tau=0.18,
        orientation_estimator_tau=0.25,
    ):
        hhalf = 0.5*math.radians(horizontal_fov_deg)-math.radians(fov_margin_deg[0])
        vhalf = 0.5*math.radians(vertical_fov_deg)-math.radians(fov_margin_deg[1])
        if not (0 < hhalf < math.pi/2 and 0 < vhalf < math.pi/2):
            raise ValueError("Effective half-FOV angles must be between 0 and 90 degrees")
        if not (0 < max_relative_heading_deg-heading_margin_deg < 90):
            raise ValueError("Effective relative-heading limit must be between 0 and 90 degrees")
        if not (0 < max_tag_incidence_deg < 90):
            raise ValueError("Tag incidence limit must be between 0 and 90 degrees")
        if (min_camera_depth <= 0 or max_camera_x <= min_camera_depth
                or max_range_margin_m < 0 or max_camera_x-max_range_margin_m <= min_camera_depth
                or vehicle_radius <= 0 or obstacle_margin < 0):
            raise ValueError("Invalid camera range or obstacle geometry")
        self.tan_h, self.tan_v = math.tan(hhalf), math.tan(vhalf)
        self.effective_half_h_deg = math.degrees(hhalf)
        self.effective_half_v_deg = math.degrees(vhalf)
        self.camera_offset = _v3(camera_offset_body, "camera_offset_body")
        self.min_camera_depth = float(min_camera_depth)
        self.max_camera_x = float(max_camera_x)
        self.max_range_margin_m = float(max_range_margin_m)
        self.effective_max_camera_x = self.max_camera_x-self.max_range_margin_m
        self.fov_linear_margin = float(fov_linear_margin)
        self.max_relative_heading = math.radians(max_relative_heading_deg-heading_margin_deg)
        self.max_relative_heading_physical = math.radians(max_relative_heading_deg)
        self.max_tag_incidence = math.radians(max_tag_incidence_deg)
        self.tag_normal_leader = _v3(tag_normal_leader, "tag_normal_leader")
        self.tag_normal_leader /= np.linalg.norm(self.tag_normal_leader)
        self.vehicle_radius = float(vehicle_radius)
        self.obstacle_margin = float(obstacle_margin)
        self.alpha_fov = float(alpha_fov)
        self.alpha_collision = float(alpha_collision)
        self.alpha_heading = float(alpha_heading)
        self.alpha_tag = float(alpha_tag)
        self.lower = np.asarray(command_lower, dtype=float)
        self.upper = np.asarray(command_upper, dtype=float)
        self.slew = np.asarray(command_slew, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        if any(a.shape != (4,) for a in (self.lower, self.upper, self.slew, self.weights)):
            raise ValueError("Command bounds, slew, and weights must be 4-vectors")
        if np.any(self.lower >= self.upper) or np.any(self.slew <= 0) or np.any(self.weights <= 0):
            raise ValueError("Invalid command bounds/slew/weights")
        self.slew_weight = float(slew_weight)
        self.estimator = RelativeObjectVelocityEstimator(estimator_tau)
        self.orientation_estimator = RelativeOrientationRateEstimator(orientation_estimator_tau)
        self.previous_command = np.zeros(4)

    def reset(self):
        self.estimator.reset()
        self.orientation_estimator.reset()
        self.previous_command[:] = 0.0

    @staticmethod
    def _B(p_body):
        """p_dot = f + B q, q=[u,v,w,r]."""
        x, y, _ = p_body
        return np.array([[-1.0, 0.0, 0.0, y],
                         [0.0, -1.0, 0.0, -x],
                         [0.0, 0.0, -1.0, 0.0]])

    @staticmethod
    def _wrap(a):
        return wrap_angle(a)

    def filter(
        self,
        nominal_command,
        leader_rel_camera,
        obstacle_positions_body: Iterable[Sequence[float]],
        actual_nu,
        dt,
        obstacle_ids: Optional[Sequence[Hashable]] = None,
        obstacle_radii=0.25,
        roll_rate=0.0,
        pitch_rate=0.0,
        leader_rotation_camera=None,
        tag_normal_camera=None,
    ) -> CBFResult:
        q_nom = np.asarray(nominal_command, dtype=float)
        nu = np.asarray(actual_nu, dtype=float)
        if q_nom.shape != (4,) or not np.all(np.isfinite(q_nom)):
            raise ValueError("nominal_command must be finite [u,v,w,r]")
        if nu.shape != (4,) or not np.all(np.isfinite(nu)):
            raise ValueError("actual_nu must be finite [u,v,w,r]")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be positive")
        p_cam = _v3(leader_rel_camera, "leader_rel_camera")
        obs = [_v3(p, "obstacle position") for p in obstacle_positions_body]
        nobs = len(obs)
        if obstacle_ids is None:
            obstacle_ids = list(range(nobs))
        if len(obstacle_ids) != nobs or len(set(obstacle_ids)) != nobs:
            raise ValueError("obstacle_ids must be unique and match obstacles")
        if np.isscalar(obstacle_radii):
            radii = np.full(nobs, float(obstacle_radii))
        else:
            radii = np.asarray(obstacle_radii, dtype=float)
        if radii.shape != (nobs,) or np.any(radii < 0):
            raise ValueError("obstacle_radii must be nonnegative scalar or N-vector")

        omega_vehicle = np.array([float(roll_rate), float(pitch_rate), float(nu[3])])
        omega_pq = np.array([float(roll_rate), float(pitch_rate), 0.0])
        rows, rhs, labels = [], [], []
        fov_h, collision_h, heading_h, tag_h = [], [], [], []

        # Tag center's body-origin-relative position and translational velocity.
        p_tag_body = self.camera_offset + p_cam
        v_tag = self.estimator.update("__leader_tag__", p_tag_body, nu, dt,
                                      roll_rate=roll_rate, pitch_rate=pitch_rate)
        f_tag = v_tag - _cross(omega_pq, p_tag_body)
        B_tag = self._B(p_tag_body)

        # Six camera barriers: four rectangular frustum faces, minimum x,
        # maximum x. These apply to the actual AprilTag center.
        x, y, z = p_cam
        rho_xy = max(math.hypot(x, y), 1e-6)
        th, tv, m = self.tan_h, self.tan_v, self.fov_linear_margin
        fov_barriers = [
            (x*th-y-m, np.array([th,-1.0,0.0]), "fov_right"),
            (x*th+y-m, np.array([th, 1.0,0.0]), "fov_left"),
            (rho_xy*tv-z-m, np.array([tv*x/rho_xy,tv*y/rho_xy,-1.0]), "fov_down"),
            (rho_xy*tv+z-m, np.array([tv*x/rho_xy,tv*y/rho_xy, 1.0]), "fov_up"),
            (x-self.min_camera_depth, np.array([1.0,0.0,0.0]), "camera_min_range"),
            (self.max_camera_x-x-self.max_range_margin_m, np.array([-1.0,0.0,0.0]), "camera_max_range"),
        ]
        for h, grad, label in fov_barriers:
            rows.append(-(grad @ B_tag))
            rhs.append(float(grad @ f_tag + self.alpha_fov*h))
            labels.append(label)
            fov_h.append(float(h))

        # Optional orientation constraints are enabled when the AprilTag pose
        # estimator provides leader rotation in camera coordinates.
        if leader_rotation_camera is not None:
            R_cl = _R3(leader_rotation_camera, "leader_rotation_camera")
            yaw_rel = math.atan2(R_cl[1, 0], R_cl[0, 0])
            omega_target = self.orientation_estimator.update(R_cl, omega_vehicle, dt)
            yaw_target_rate = float(omega_target[2])

            # Relative heading safe set: -psi_max <= psi_rel <= psi_max.
            h_yaw_pos = self.max_relative_heading - yaw_rel
            h_yaw_neg = self.max_relative_heading + yaw_rel
            # hdot_pos = -psi_target_dot + r_cmd
            rows.append(np.array([0.0, 0.0, 0.0, -1.0]))
            rhs.append(float(-yaw_target_rate + self.alpha_heading*h_yaw_pos))
            labels.append("relative_heading_positive")
            heading_h.append(float(h_yaw_pos))
            # hdot_neg = psi_target_dot - r_cmd
            rows.append(np.array([0.0, 0.0, 0.0, 1.0]))
            rhs.append(float(yaw_target_rate + self.alpha_heading*h_yaw_neg))
            labels.append("relative_heading_negative")
            heading_h.append(float(h_yaw_neg))

            # AprilTag face-incidence barrier. The outward normal is recovered
            # from leader attitude and the calibrated tail-tag normal.
            if tag_normal_camera is None:
                n_tag = R_cl @ self.tag_normal_leader
            else:
                n_tag = _v3(tag_normal_camera, "tag_normal_camera")
            n_norm = np.linalg.norm(n_tag)
            if n_norm < 1e-8:
                raise ValueError("tag_normal_camera must be nonzero")
            n_tag = n_tag / n_norm
            p_norm = np.linalg.norm(p_cam)
            if p_norm < 1e-6:
                raise ValueError("Tag position is too close to camera origin")
            p_hat = p_cam / p_norm
            d_camera_to_camera = -p_hat  # tag-to-camera unit viewing direction
            P = np.eye(3) - np.outer(p_hat, p_hat)
            h_inc = float(n_tag @ d_camera_to_camera - math.cos(self.max_tag_incidence))

            # q-affine incidence-rate calculation:
            # p_dot = f_tag + B_tag q
            # n_dot = (omega_target - omega_vehicle_candidate) x n_tag.
            ez = np.array([0.0, 0.0, 1.0])
            omega_target_without_vehicle_yaw = omega_target - np.array([roll_rate, pitch_rate, 0.0])

            def incidence_rate(q):
                q = np.asarray(q, dtype=float)
                p_dot = f_tag + B_tag @ q
                omega_rel = omega_target_without_vehicle_yaw - np.array([0.0, 0.0, q[3]])
                n_dot = _cross(omega_rel, n_tag)
                d_dot = -(P @ p_dot) / p_norm
                return float(n_dot @ d_camera_to_camera + n_tag @ d_dot)

            q_zero = np.zeros(4)
            rate_zero = incidence_rate(q_zero)
            rate_coeff = np.array([incidence_rate(np.eye(4)[j]) - rate_zero for j in range(4)])
            rows.append(-rate_coeff)
            rhs.append(float(rate_zero + self.alpha_tag*h_inc))
            labels.append("apriltag_face_incidence")
            tag_h.append(h_inc)

        # Collision barriers are appended to the same QP.
        for idx, (oid, p) in enumerate(zip(obstacle_ids, obs)):
            v_obj = self.estimator.update(oid, p, nu, dt,
                                          roll_rate=roll_rate, pitch_rate=pitch_rate)
            f_obj = v_obj - _cross(omega_pq, p)
            B_obj = self._B(p)
            safe_radius = self.vehicle_radius + float(radii[idx]) + self.obstacle_margin
            h = float(p@p - safe_radius**2)
            grad = 2.0*p
            rows.append(-(grad @ B_obj))
            rhs.append(float(grad @ f_obj + self.alpha_collision*h))
            labels.append(f"collision_{oid}")
            collision_h.append(h)

        A = np.asarray(rows, dtype=float)
        b = np.asarray(rhs, dtype=float)
        self.last_qp_A = A.copy()
        self.last_qp_b = b.copy()
        self.last_qp_labels = tuple(labels)
        lower = np.maximum(self.lower, self.previous_command-self.slew*dt)
        upper = np.minimum(self.upper, self.previous_command+self.slew*dt)
        if np.any(lower > upper):
            lower, upper = self.lower.copy(), self.upper.copy()
        bounds = list(zip(lower, upper))
        lp = linprog(c=np.zeros(4), A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not lp.success:
            return CBFResult(
                command=np.zeros(4), feasible=False, solver_success=False,
                status=f"CBF-QP infeasible: {lp.message}",
                min_cbf_margin=float(np.min(b-A@np.zeros(4))),
                collision_margins=np.asarray(collision_h), fov_margins=np.asarray(fov_h),
                heading_margins=np.asarray(heading_h), tag_margins=np.asarray(tag_h),
                active_constraints=0, nominal_command=q_nom.copy(), constraint_labels=tuple(labels)
            )

        W = np.diag(self.weights)
        def objective(q):
            d, s = q-q_nom, q-self.previous_command
            return 0.5*d@W@d + 0.5*self.slew_weight*(s@s)
        def jac(q):
            return W@(q-q_nom) + self.slew_weight*(q-self.previous_command)
        result = minimize(objective, lp.x, jac=jac, method="SLSQP",
                          bounds=Bounds(lower, upper),
                          constraints=[LinearConstraint(A, -np.inf, b)],
                          options={"ftol":1e-9, "maxiter":180, "disp":False})
        q = np.asarray(result.x, dtype=float)
        residual = b-A@q
        feasible = bool(np.all(np.isfinite(q)) and np.all(q>=lower-1e-6)
                        and np.all(q<=upper+1e-6) and np.all(residual>=-1e-5))
        if not feasible:
            return CBFResult(
                command=np.zeros(4), feasible=False, solver_success=bool(result.success),
                status="Numerical QP result failed feasibility check",
                min_cbf_margin=float(np.min(residual)),
                collision_margins=np.asarray(collision_h), fov_margins=np.asarray(fov_h),
                heading_margins=np.asarray(heading_h), tag_margins=np.asarray(tag_h),
                active_constraints=int(np.sum(residual<1e-4)), nominal_command=q_nom.copy(),
                constraint_labels=tuple(labels)
            )
        self.previous_command = q.copy()
        return CBFResult(
            command=q, feasible=True, solver_success=bool(result.success),
            status="optimal" if result.success else "feasible_qp_iterate",
            min_cbf_margin=float(np.min(residual)),
            collision_margins=np.asarray(collision_h), fov_margins=np.asarray(fov_h),
            heading_margins=np.asarray(heading_h), tag_margins=np.asarray(tag_h),
            active_constraints=int(np.sum(residual<1e-4)), nominal_command=q_nom.copy(),
            constraint_labels=tuple(labels)
        )
