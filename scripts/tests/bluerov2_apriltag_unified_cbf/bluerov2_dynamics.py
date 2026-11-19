"""4-DOF BlueROV2-Heavy-inspired closed-loop velocity dynamics.

The lower-level loop tracks [u_ref,v_ref,w_ref,r_ref] and produces a bounded
body wrench. This is a transparent reduced-order Fossen-style simulation model,
not an official Blue Robotics/ArduSub firmware model or thruster-level SITL.
"""
import numpy as np


class BlueROV2Heavy4DOF:
    def __init__(self, position_ned=(0.0, 0.0, 10.0), yaw=0.0, dt=0.05):
        self.position = np.asarray(position_ned, dtype=float)
        self.yaw = float(yaw)
        self.nu = np.zeros(4)  # [u,v,w,r]
        self.dt = float(dt)
        # Representative reduced-order effective inertia and damping values.
        # Effective diagonal masses = rigid-body mass + added mass.
        self.M = np.diag([17.0, 24.2, 26.07, 0.28])
        self.D1 = np.array([4.03, 6.22, 5.18, 0.07])
        self.D2 = np.array([18.18, 21.66, 36.99, 1.55])
        # Inner velocity-loop gains (simulation assumptions, tune to actual stack).
        self.kp = np.array([2.8, 2.5, 2.4, 3.0])
        self.ki = np.array([0.30, 0.35, 0.30, 0.35])
        self.integral = np.zeros(4)
        self.wrench_limit = np.array([65.0, 65.0, 85.0, 12.0])
        self.accel_disturbance = np.zeros(4)

    def velocity_loop_offset(self, disturbance=None):
        """Return b in nu_dot = diag(kp) * (nu_ref - nu) + b."""
        disturbance = np.zeros(4) if disturbance is None else np.asarray(disturbance, dtype=float)
        return self.ki * self.integral + disturbance

    def rotation_body_to_ned(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def step(self, command, dt=None, disturbance=None):
        dt = self.dt if dt is None else float(dt)
        command = np.asarray(command, dtype=float)
        if command.shape != (4,):
            raise ValueError("command must be [u,v,w,r]")
        disturbance = np.zeros(4) if disturbance is None else np.asarray(disturbance, dtype=float)
        err = command - self.nu
        self.integral = np.clip(self.integral + err * dt, -1.2, 1.2)
        drag = self.D1 * self.nu + self.D2 * np.abs(self.nu) * self.nu
        tau = np.diag(self.M) * (self.kp * err + self.ki * self.integral) + drag
        tau = np.clip(tau, -self.wrench_limit, self.wrench_limit)
        nu_dot = (tau - drag) / np.diag(self.M) + disturbance
        self.nu += nu_dot * dt
        # Numerical and operating envelopes used for this simulation.
        self.nu = np.clip(self.nu, [-0.55, -0.55, -0.30, -0.65], [0.65, 0.55, 0.30, 0.65])
        R = self.rotation_body_to_ned()
        self.position += R @ self.nu[:3] * dt
        self.yaw = (self.yaw + self.nu[3] * dt + np.pi) % (2*np.pi) - np.pi
        return self.position.copy(), self.yaw, self.nu.copy()
