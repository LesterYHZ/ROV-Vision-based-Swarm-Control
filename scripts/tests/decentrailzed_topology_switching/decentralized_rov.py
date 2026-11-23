"""Decentralized vision-only topology switching and local unified CBF for a small ROV fleet.

The topology controller receives only the observing follower's locally detected AprilTag
measurements. No parent assignments, link-quality matrices, or states are exchanged.

Simulation coordinates: NED world; body/camera coordinates: FRD (x forward, y right, z down).
This is a reduced-order 4-DOF simulation, not a validated vehicle plant model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, List
import math
import numpy as np
from scipy.optimize import minimize

from fleet_model import (
    ROV, SphereObstacle, LinkObservation, AprilTagLinkEvaluator,
    rot_b_to_n, rot_n_to_b, wrap_pi, unit, segment_point_distance,
)


@dataclass
class LocalTagObservation:
    """One follower's camera-derived pose estimate of one other ROV's tail tag."""
    target_id: int
    quality: float
    visible: bool
    range: float
    azimuth: float
    elevation: float
    relative_yaw: float
    incidence: float
    occluded: bool
    tag_position_camera: np.ndarray
    target_origin_body: np.ndarray
    tag_position_body: np.ndarray
    tag_normal_body: np.ndarray


@dataclass
class LocalObjectDetection:
    """Camera-derived relative detection used by this follower's collision CBF."""
    object_id: str
    position_body: np.ndarray
    relative_velocity_body: np.ndarray
    radius: float
    is_static: bool


def make_local_tag_observation(observer: ROV, target: ROV,
                               evaluator: AprilTagLinkEvaluator) -> LocalTagObservation:
    """Generate a synthetic local camera pose observation; returns no target world state."""
    quality: LinkObservation = evaluator.evaluate(observer, target)
    R_i = rot_b_to_n(observer.yaw)
    R_rel = rot_n_to_b(observer.yaw) @ rot_b_to_n(target.yaw)
    cam_n = observer.pos + R_i @ evaluator.camera_offset
    tag_n = target.pos + rot_b_to_n(target.yaw) @ evaluator.tag_tail_offset
    p_tag_camera = rot_n_to_b(observer.yaw) @ (tag_n - cam_n)
    # Recover the target body origin from the measured tag pose and calibrated tail mount.
    target_origin_body = (
        p_tag_camera + evaluator.camera_offset - R_rel @ evaluator.tag_tail_offset
    )
    tag_position_body = p_tag_camera + evaluator.camera_offset
    tag_normal_body = R_rel @ np.array([-1.0, 0.0, 0.0])
    return LocalTagObservation(
        target_id=target.idx,
        quality=quality.quality,
        visible=quality.visible,
        range=quality.range,
        azimuth=quality.azimuth,
        elevation=quality.elevation,
        relative_yaw=quality.rel_heading,
        incidence=quality.incidence,
        occluded=quality.occluded,
        tag_position_camera=p_tag_camera.copy(),
        target_origin_body=target_origin_body.copy(),
        tag_position_body=tag_position_body.copy(),
        tag_normal_body=tag_normal_body.copy(),
    )


class LocalRelativeMotionEstimator:
    """Estimate relative motion from successive local camera detections only."""
    def __init__(self, tau: float = 0.30, first_detection_closing_speed: float = 0.18):
        self.tau = float(tau)
        self.first_detection_closing_speed = float(first_detection_closing_speed)
        self.position_history: Dict[str, np.ndarray] = {}
        self.velocity_history: Dict[str, np.ndarray] = {}
        self.heading_history: Dict[str, float] = {}
        self.yaw_rate_history: Dict[str, float] = {}

    def reset(self, key: str):
        self.position_history.pop(key, None)
        self.velocity_history.pop(key, None)
        self.heading_history.pop(key, None)
        self.yaw_rate_history.pop(key, None)

    def update_position(self, key: str, position_body: np.ndarray,
                        observer: ROV, dt: float, is_static: bool = False) -> np.ndarray:
        p = np.asarray(position_body, dtype=float).reshape(3)
        previous = self.position_history.get(key)
        omega_cross_p = np.cross(np.array([0.0, 0.0, observer.r]), p)
        if previous is None:
            # Static-scene visual odometry gives the expected optical flow of a stationary object.
            raw = -observer.nu - omega_cross_p
            if not is_static:
                raw = raw - self.first_detection_closing_speed * unit(p)
            estimate = raw
        else:
            raw = (p - previous) / dt
            old = self.velocity_history.get(key, raw)
            a = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
            estimate = (1.0 - a) * old + a * raw
        self.position_history[key] = p.copy()
        self.velocity_history[key] = estimate.copy()
        return estimate

    def update_relative_yaw(self, key: str, relative_yaw: float,
                            observer: ROV, dt: float) -> float:
        previous = self.heading_history.get(key)
        if previous is None:
            estimate = observer.r
        else:
            raw = wrap_pi(relative_yaw - previous) / dt + observer.r
            old = self.yaw_rate_history.get(key, raw)
            a = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
            estimate = (1.0 - a) * old + a * raw
        self.heading_history[key] = float(relative_yaw)
        self.yaw_rate_history[key] = float(estimate)
        return float(estimate)


class DecentralizedTopologySelector:
    """Independent local parent choice with hysteresis; arbitrary parent IDs are allowed.

    The selector receives only this follower's local camera detections. It may choose any
    other vehicle ID in that local observation dictionary, including a higher-ID follower.
    No rank restriction, shared topology state, or inter-ROV communication is used.

    Because cycle prevention is intentionally not imposed, independent choices can create
    directed cycles (for example F1 -> F2 and F2 -> F1). This is an explicit limitation of
    the fully decentralized, communication-free mechanism.
    """
    def __init__(self, own_id: int,
                 q_enter: float = 0.055, q_keep: float = 0.035,
                 desired_range: float = 4.5, switch_penalty: float = 0.32,
                 switch_margin: float = 0.10, min_dwell: float = 2.0,
                 hold_cost: float = 2.4, quality_weight: float = 1.8,
                 range_weight: float = 0.22, heading_weight: float = 0.18,
                 incidence_weight: float = 0.12):
        self.own_id = int(own_id)
        self.q_enter = float(q_enter)
        self.q_keep = float(q_keep)
        self.desired_range = float(desired_range)
        self.switch_penalty = float(switch_penalty)
        self.switch_margin = float(switch_margin)
        self.min_dwell = float(min_dwell)
        self.hold_cost = float(hold_cost)
        self.quality_weight = float(quality_weight)
        self.range_weight = float(range_weight)
        self.heading_weight = float(heading_weight)
        self.incidence_weight = float(incidence_weight)
        self.parent_id: Optional[int] = None
        self.last_switch_time = -math.inf
        self.last_decision_cost = self.hold_cost
        self.switch_count = 0

    def _cost(self, obs: LocalTagObservation) -> float:
        q = max(float(obs.quality), 1e-6)
        range_term = ((obs.range - self.desired_range) / 3.0) ** 2
        heading_term = (abs(obs.relative_yaw) / math.radians(125.0)) ** 2
        incidence_term = (obs.incidence / math.radians(86.0)) ** 2
        return (
            self.quality_weight * (-math.log(q))
            + self.range_weight * range_term
            + self.heading_weight * heading_term
            + self.incidence_weight * incidence_term
        )

    def select(self, local_observations: Dict[int, LocalTagObservation], now: float
               ) -> Tuple[Optional[int], float, Dict[str, object]]:
        """Return (parent_id or None, local cost, diagnostics) using local detections only."""
        candidates: Dict[int, LocalTagObservation] = {}
        for target_id, obs in local_observations.items():
            if (int(target_id) != self.own_id and obs.target_id != self.own_id
                    and obs.visible and obs.quality >= self.q_enter):
                candidates[int(target_id)] = obs

        current_obs = (local_observations.get(self.parent_id)
                       if self.parent_id is not None else None)
        current_valid = (
            self.parent_id is not None
            and self.parent_id != self.own_id
            and current_obs is not None
            and current_obs.visible
            and current_obs.quality >= self.q_keep
        )

        if current_valid:
            current_cost = self._cost(current_obs)
            # Hysteresis: don't switch during dwell unless the current link is lost.
            if now - self.last_switch_time < self.min_dwell:
                self.last_decision_cost = current_cost
                return self.parent_id, current_cost, {
                    "mode": "retain_dwell", "candidate_ids": sorted(candidates),
                    "current_quality": current_obs.quality,
                }
            challengers = {j: o for j, o in candidates.items() if j != self.parent_id}
            if challengers:
                best_id, best_obs = min(
                    challengers.items(), key=lambda item: (self._cost(item[1]), item[0])
                )
                challenger_cost = self._cost(best_obs) + self.switch_penalty
                if challenger_cost + self.switch_margin < current_cost:
                    old = self.parent_id
                    self.parent_id = best_id
                    self.last_switch_time = now
                    self.switch_count += 1
                    self.last_decision_cost = challenger_cost
                    return best_id, challenger_cost, {
                        "mode": "switch_better_link", "old_parent": old,
                        "candidate_ids": sorted(candidates),
                        "selected_quality": best_obs.quality,
                    }
            self.last_decision_cost = current_cost
            return self.parent_id, current_cost, {
                "mode": "retain_hysteresis", "candidate_ids": sorted(candidates),
                "current_quality": current_obs.quality,
            }

        # Current link is unavailable: immediately select the best locally visible vehicle,
        # irrespective of its ID or nominal fleet role.
        if candidates:
            best_id, best_obs = min(
                candidates.items(), key=lambda item: (self._cost(item[1]), item[0])
            )
            old = self.parent_id
            self.parent_id = best_id
            self.last_switch_time = now
            if old is not None and old != best_id:
                self.switch_count += 1
            cost = self._cost(best_obs) + (
                self.switch_penalty if old is not None and old != best_id else 0.0
            )
            self.last_decision_cost = cost
            return best_id, cost, {
                "mode": "reacquire_or_switch", "old_parent": old,
                "candidate_ids": sorted(candidates), "selected_quality": best_obs.quality,
            }

        old = self.parent_id
        self.parent_id = None
        self.last_decision_cost = self.hold_cost
        return None, self.hold_cost, {
            "mode": "hold_reacquire", "old_parent": old, "candidate_ids": []
        }


class LocalParentKinematicsEstimator:
    """Estimate a selected parent's translational velocity and yaw rate from visual pose history."""
    def __init__(self, tau=0.35):
        self.motion = LocalRelativeMotionEstimator(tau=tau)

    def reset(self, follower_id: int):
        self.motion.reset(f"parent_origin_{follower_id}")
        self.motion.reset(f"parent_yaw_{follower_id}")

    def update(self, follower: ROV, obs: LocalTagObservation, dt: float):
        key = f"parent_origin_{follower.idx}"
        rdot = self.motion.update_position(key, obs.target_origin_body, follower, dt, is_static=False)
        omega = np.array([0.0, 0.0, follower.r])
        parent_velocity_body = rdot + follower.nu + np.cross(omega, obs.target_origin_body)
        parent_yaw_rate = self.motion.update_relative_yaw(
            f"parent_yaw_{follower.idx}", obs.relative_yaw, follower, dt
        )
        return parent_velocity_body, parent_yaw_rate


class LocalPIDParentTracker:
    """PID tracker using only one selected parent AprilTag's local relative pose."""
    def __init__(self, desired_range=4.5, desired_lateral=0.0, desired_down=0.0,
                 kp=(0.52,0.62,0.55), ki=(0.012,0.018,0.015), kd=(0.08,0.10,0.08),
                 kp_yaw=2.0, kd_yaw=0.30):
        self.desired = np.array([desired_range, desired_lateral, desired_down], float)
        self.kp = np.asarray(kp, float); self.ki = np.asarray(ki, float); self.kd = np.asarray(kd, float)
        self.kp_yaw = kp_yaw; self.kd_yaw = kd_yaw
        self.integral: Dict[int, np.ndarray] = {}
        self.prev_error: Dict[int, np.ndarray] = {}

    def reset(self, follower_id: int):
        self.integral.pop(follower_id, None); self.prev_error.pop(follower_id, None)

    def compute(self, follower: ROV, rel_parent_body: np.ndarray,
                parent_velocity_body: np.ndarray, relative_yaw: float,
                parent_yaw_rate: float, dt: float) -> np.ndarray:
        err = np.asarray(rel_parent_body, float) - self.desired
        integ = np.clip(self.integral.get(follower.idx, np.zeros(3)) + err * dt, -4.0, 4.0)
        prev = self.prev_error.get(follower.idx, err)
        derr = (err - prev) / dt
        self.integral[follower.idx] = integ.copy(); self.prev_error[follower.idx] = err.copy()
        omega = np.array([0.0, 0.0, follower.r])
        trans = (parent_velocity_body - np.cross(omega, rel_parent_body)
                 + self.kp*err + self.ki*integ + self.kd*derr)
        r_cmd = parent_yaw_rate + self.kp_yaw*wrap_pi(relative_yaw) + self.kd_yaw*(parent_yaw_rate-follower.r)
        return np.clip(np.r_[trans, r_cmd], [-0.65,-0.40,-0.32,-1.0], [0.90,0.40,0.32,1.0])


class DecentralizedVisionCBF:
    """Per-follower unified QP using only locally detected camera objects and selected tag pose."""
    def __init__(self, camera_offset=(0.32,0,0), hfov_deg=80, vfov_deg=64,
                 fov_margin_deg=(5,4), min_range=1.2, max_range=18.0,
                 range_margin=0.12, max_rel_heading_deg=125, max_incidence_deg=86,
                 collision_alpha=1.7, perception_alpha=1.8, follower_radius=0.72,
                 command_lower=(-0.65,-0.40,-0.32,-1.0),
                 command_upper=(0.90,0.40,0.32,1.0),
                 max_delta=(0.40,0.35,0.30,0.50),
                 weights=(1.0,2.0,2.0,1.25), smooth_weight=0.12):
        self.camera_offset=np.asarray(camera_offset,float)
        self.hh=math.radians(hfov_deg/2-fov_margin_deg[0]); self.hv=math.radians(vfov_deg/2-fov_margin_deg[1])
        self.min_range=min_range; self.max_range=max_range; self.range_margin=range_margin
        self.max_heading=math.radians(max_rel_heading_deg); self.max_inc=math.radians(max_incidence_deg)
        self.ac=collision_alpha; self.ap=perception_alpha; self.follower_radius=follower_radius
        self.lower=np.asarray(command_lower,float); self.upper=np.asarray(command_upper,float)
        self.max_delta=np.asarray(max_delta,float); self.W=np.diag(np.asarray(weights,float)); self.smooth_weight=smooth_weight

    @staticmethod
    def _barrier(A,b,labels,h,grad,f,G,alpha,label):
        coeff=np.asarray(grad @ G,float); const=float(grad @ f)
        A.append(-coeff); b.append(const+alpha*float(h)); labels.append(label)

    def filter(self, observer: ROV, q_nom: np.ndarray,
               parent_obs: Optional[LocalTagObservation], parent_tag_velocity_body: Optional[np.ndarray],
               parent_yaw_rate: Optional[float], collision_detections: Sequence[LocalObjectDetection],
               dt: float, previous_cmd: np.ndarray):
        A=[]; b=[]; labels=[]
        v_actual=np.asarray(observer.nu,float)
        omega=np.array([0.0,0.0,observer.r]); ez=np.array([0.0,0.0,1.0])
        # Collision CBFs use only locally detected obstacle/vehicle body positions and visual-flow rates.
        for det in collision_detections:
            p=np.asarray(det.position_body,float); pdot=np.asarray(det.relative_velocity_body,float)
            G=np.array([[-1.,0.,0.,p[1]],[0.,-1.,0.,-p[0]], [0.,0.,-1.,0.]])
            f=pdot+v_actual+np.cross(omega,p)
            safe=self.follower_radius+det.radius+0.24
            h=float(p@p-safe*safe); grad=2*p
            self._barrier(A,b,labels,h,grad,f,G,self.ac,f"collision:{det.object_id}")

        if parent_obs is not None and parent_tag_velocity_body is not None and parent_yaw_rate is not None:
            pcam=np.asarray(parent_obs.tag_position_camera,float)
            pbody=pcam+self.camera_offset
            pdot=np.asarray(parent_tag_velocity_body,float)
            G=np.array([[-1.,0.,0.,pbody[1]],[0.,-1.,0.,-pbody[0]], [0.,0.,-1.,0.]])
            f=pdot+v_actual+np.cross(omega,pbody)
            x,y,z=pcam; rho=max(math.hypot(x,y),1e-8)
            specs=[
                (x*math.tan(self.hh)-y-self.range_margin, np.array([math.tan(self.hh),-1.,0.]), 'fov_h_right'),
                (x*math.tan(self.hh)+y-self.range_margin, np.array([math.tan(self.hh),1.,0.]), 'fov_h_left'),
                (rho*math.tan(self.hv)-z-self.range_margin, np.array([math.tan(self.hv)*x/rho,math.tan(self.hv)*y/rho,-1.]), 'fov_v_down'),
                (rho*math.tan(self.hv)+z-self.range_margin, np.array([math.tan(self.hv)*x/rho,math.tan(self.hv)*y/rho,1.]), 'fov_v_up'),
                (x-self.min_range-self.range_margin, np.array([1.,0.,0.]), 'range_min'),
                (self.max_range-self.range_margin-x, np.array([-1.,0.,0.]), 'range_max'),
            ]
            for h,grad,label in specs:
                self._barrier(A,b,labels,h,grad,f,G,self.ap,label)

            delta=wrap_pi(parent_obs.relative_yaw)
            hp=self.max_heading-delta; hm=self.max_heading+delta
            A.append(np.array([0.,0.,0.,-1.])); b.append(-parent_yaw_rate+self.ap*hp); labels.append('relative_heading_plus')
            A.append(np.array([0.,0.,0.,1.])); b.append(parent_yaw_rate+self.ap*hm); labels.append('relative_heading_minus')

            # Tail tag outward normal and incidence barrier, all in observer body FRD.
            c,s=math.cos(delta),math.sin(delta)
            n=np.array([-c,-s,0.]); d=-pcam/max(float(np.linalg.norm(pcam)),1e-8)
            dist=max(float(np.linalg.norm(pcam)),1e-8)
            hinc=float(n@d-math.cos(self.max_inc))
            P=np.eye(3)-np.outer(d,d); D=-P/dist
            ez_cross_n=np.cross(ez,n)
            ndot_const=parent_yaw_rate*float(ez_cross_n@d)
            nr_coeff=-float(ez_cross_n@d)
            inc_const=ndot_const+float(n@D@f)
            inc_coeff=np.asarray(n@D@G,float); inc_coeff[3]+=nr_coeff
            A.append(-inc_coeff); b.append(inc_const+self.ap*hinc); labels.append('tag_incidence')

        A=np.asarray(A,float) if A else np.zeros((0,4)); b=np.asarray(b,float) if b else np.zeros(0)
        q_nom=np.asarray(q_nom,float); prev=np.asarray(previous_cmd,float)
        lo=np.maximum(self.lower,prev-self.max_delta); hi=np.minimum(self.upper,prev+self.max_delta)
        q0=np.clip(q_nom,lo,hi)
        def fun(q):
            e=q-q_nom; ds=q-prev
            return 0.5*float(e@self.W@e)+0.5*self.smooth_weight*float(ds@ds)
        def jac(q): return self.W@(q-q_nom)+self.smooth_weight*(q-prev)
        cons=[]
        if len(A): cons=[{'type':'ineq','fun':lambda q:b-A@q,'jac':lambda q:-A}]
        res=minimize(fun,q0,jac=jac,method='SLSQP',bounds=list(zip(lo,hi)),constraints=cons,
                     options={'maxiter':120,'ftol':1e-8,'disp':False})
        margins=b-A@res.x if len(A) else np.array([np.inf])
        feasible=bool(np.all(np.isfinite(res.x)) and np.all(margins>=-3e-5))
        if feasible: cmd=res.x; status='optimal' if res.success else 'feasible_iterate'
        else: cmd=np.zeros(4); status='infeasible_damped_hold'
        return cmd, {'feasible':feasible,'solver_success':bool(res.success),'status':status,
                     'min_margin':float(np.min(margins)),'labels':labels,
                     'correction_norm':float(np.linalg.norm(cmd-q_nom))}


def detect_local_objects(observer: ROV, fleet: Sequence[ROV], obstacles: Sequence[SphereObstacle],
                         motion_estimator: LocalRelativeMotionEstimator, dt: float,
                         camera_offset=(0.32,0,0), hfov_deg=150, vfov_deg=105,
                         max_range=17.0) -> List[LocalObjectDetection]:
    """Synthetic local visual object detector; returns only local camera-array detections.

    In the example, AprilTag identity/pose uses a forward camera, while collision object
    detection uses a 360-degree panoramic/multi-camera visual envelope. This is still
    vision-only and requires no inter-ROV communication. A single forward camera cannot
    detect objects behind the vehicle; see README limitations.
    """
    Rbn=rot_b_to_n(observer.yaw); Rnb=Rbn.T
    cam_n=observer.pos+Rbn@np.asarray(camera_offset,float)
    half_h=math.radians(hfov_deg/2); half_v=math.radians(vfov_deg/2)
    detections=[]
    objects=[]
    for ob in obstacles: objects.append((ob.name,ob.pos,ob.radius,True))
    for other in fleet:
        if other.idx==observer.idx: continue
        objects.append((f"rov_{other.idx}",other.pos,0.82,False))
    for name,pos_n,radius,is_static in objects:
        pbody=Rnb@(np.asarray(pos_n)-observer.pos)
        pcam=pbody-np.asarray(camera_offset,float)
        rng=float(np.linalg.norm(pcam))
        if rng>max_range or rng<0.5: continue
        az=math.atan2(pcam[1],pcam[0]); el=math.atan2(pcam[2],max(math.hypot(pcam[0],pcam[1]),1e-9))
        if abs(az)>half_h or abs(el)>half_v: continue
        # Local visual occlusion check: another static object may block the line of sight.
        blocked=False
        for other_ob in obstacles:
            if other_ob.name==name: continue
            if segment_point_distance(cam_n,np.asarray(pos_n),other_ob.pos)<other_ob.radius+0.10:
                blocked=True; break
        if blocked: continue
        key=f"collision_{name}"
        rel_vel=motion_estimator.update_position(key,pbody,observer,dt,is_static=is_static)
        detections.append(LocalObjectDetection(name,pbody.copy(),rel_vel.copy(),radius,is_static))
    # Remove stale visual tracks for objects that are outside the camera view.
    active={f"collision_{d.object_id}" for d in detections}
    stale=[key for key in motion_estimator.position_history if key.startswith('collision_') and key not in active]
    for key in stale: motion_estimator.reset(key)
    return detections
