"""Nominal camera-based target tracker producing [u,v,w,r] references."""
import math
import numpy as np


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class BodyVelocityTargetTracker:
    """Track a leader's tail AprilTag in camera FRD and align ROV headings.

    q_nom=[u,v,w,r]. Position feedback regulates tag stand-off; relative-yaw
    feedback encourages the follower to align with the leader's heading.
    """
    def __init__(self, desired_rel_camera=(6.0, 0.0, 0.0),
                 kp=(0.27, 0.40, 0.32), yaw_gain=0.75,
                 bearing_yaw_gain=0.08,
                 max_surge=0.62, max_sway=0.16, max_heave=0.25, max_yaw_rate=0.45):
        self.desired = np.asarray(desired_rel_camera, dtype=float)
        self.kp = np.asarray(kp, dtype=float)
        if self.desired.shape != (3,) or self.kp.shape != (3,):
            raise ValueError("desired_rel_camera and kp must be 3-vectors")
        self.yaw_gain = float(yaw_gain)
        self.bearing_yaw_gain = float(bearing_yaw_gain)
        self.limits = np.array([max_surge, max_sway, max_heave, max_yaw_rate], dtype=float)

    def compute(self, tag_rel_camera, tag_velocity_body,
                camera_offset_body=(0.22, 0.0, 0.0),
                relative_heading=0.0, leader_yaw_rate=0.0):
        p = np.asarray(tag_rel_camera, dtype=float)
        vt = np.asarray(tag_velocity_body, dtype=float)
        l = np.asarray(camera_offset_body, dtype=float)
        if p.shape != (3,) or vt.shape != (3,) or l.shape != (3,):
            raise ValueError("Position, velocity and camera offset must be 3-vectors")
        e = p - self.desired
        bearing = math.atan2(p[1], max(p[0], 1e-6))
        psi_rel = wrap_angle(relative_heading)
        # Align follower heading with leader heading while retaining a small
        # bearing-centering contribution from the image geometry.
        r_cmd = leader_yaw_rate + self.yaw_gain*psi_rel + self.bearing_yaw_gain*bearing
        r_cmd = clip(r_cmd, -self.limits[3], self.limits[3])
        p_body = l + p
        u = vt[0] + self.kp[0]*e[0] + r_cmd*p_body[1]
        v = vt[1] + self.kp[1]*e[1] - r_cmd*p_body[0]
        w = vt[2] + self.kp[2]*e[2]
        cmd = np.array([u, v, w, r_cmd], dtype=float)
        return np.clip(cmd, -self.limits, self.limits)
