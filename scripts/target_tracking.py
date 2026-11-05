import math
from numbers import Real


class TargetVelocityEstimator:
    """
    Estimate target translational velocity from successive
    target-relative position measurements.

    All vectors are expressed in body FRD coordinates.

    vehicle_velocity_body must be the actual vehicle velocity
    over ground, not its commanded velocity.
    """

    def __init__(self, tau=0.3):
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be positive")

        self.tau = tau
        self.prev_position = None
        self.velocity_estimate = None

    def reset(self):
        self.prev_position = None
        self.velocity_estimate = None

    @staticmethod
    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def update(
        self,
        target_rel_body,
        vehicle_velocity_body,
        body_rates_frd,
        dt,
    ):
        r = tuple(float(x) for x in target_rel_body)
        vv = tuple(float(x) for x in vehicle_velocity_body)
        omega = tuple(float(x) for x in body_rates_frd)

        if any(len(v) != 3 for v in (r, vv, omega)):
            raise ValueError("All vectors must have 3 elements")

        if not all(math.isfinite(x) for v in (r, vv, omega) for x in v):
            raise ValueError("Inputs must be finite")

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive")

        # First measurement: initialize history
        if self.prev_position is None:
            self.prev_position = r
            return None

        # 1. Relative position derivative
        r_dot = tuple((r[i] - self.prev_position[i]) / dt for i in range(3))

        # 2. Rotational compensation
        omega_cross_r = self.cross(omega, r)

        # 3. Reconstruct target velocity
        v_raw = tuple(r_dot[i] + vv[i] + omega_cross_r[i] for i in range(3))

        # 4. First-order low-pass filter
        alpha = 1.0 - math.exp(-dt / self.tau)

        if self.velocity_estimate is None:
            self.velocity_estimate = v_raw
        else:
            self.velocity_estimate = tuple(
                (1.0 - alpha) * self.velocity_estimate[i] + alpha * v_raw[i]
                for i in range(3)
            )

        self.prev_position = r

        return self.velocity_estimate


def vec3(value, name):
    if len(value) != 3:
        raise ValueError(f"{name} must contain 3 elements")

    v = tuple(float(x) for x in value)

    if not all(math.isfinite(x) for x in v):
        raise ValueError(f"{name} must contain finite values")

    return v


def gain3(value, name):
    if isinstance(value, Real):
        return (float(value),) * 3
    return vec3(value, name)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def limit_norm(v, max_speed):
    speed = norm(v)

    if speed > max_speed:
        return tuple(x * max_speed / speed for x in v)

    return v


def projected_rate(theta_hat, rate, inner, outer):
    """
    Smooth parameter projection following the Fossen ALOS
    projection principle.
    """
    if abs(theta_hat) > inner and theta_hat * rate > 0:
        c = min(1.0, (theta_hat**2 - inner**2) / (outer**2 - inner**2))
        return (1.0 - c) * rate

    return rate


class TargetTrackingController:
    def __init__(
        self,
        controller="pid",
        # PID parameters
        kp=(0.8, 0.8, 0.8),
        ki=(0.05, 0.05, 0.05),
        kd=(0.2, 0.2, 0.2),
        # ALOS parameters
        delta_h=5.0,
        delta_v=5.0,
        gamma_h=0.05,
        gamma_v=0.05,
        k_along=0.3,

        beta_inner=0.4,
        beta_outer=0.7,
        alpha_inner=0.4,
        alpha_outer=0.7,
        # Common parameters
        max_speed=2.0,
        integral_limit=5.0,
        derivative_filter=0.25,
    ):
        if controller not in ("pid", "alos"):
            raise ValueError("controller must be 'pid' or 'alos'")

        self.controller = controller

        # PID gains
        self.kp = gain3(kp, "kp")
        self.ki = gain3(ki, "ki")
        self.kd = gain3(kd, "kd")

        self.integral_limit = float(integral_limit)
        self.derivative_filter = float(derivative_filter)

        # ALOS gains and look-ahead distances
        self.delta_h = float(delta_h)
        self.delta_v = float(delta_v)
        self.gamma_h = float(gamma_h)
        self.gamma_v = float(gamma_v)
        self.k_along = float(k_along)

        # Adaptive parameter projection limits (rad)
        self.beta_inner = float(beta_inner)
        self.beta_outer = float(beta_outer)
        self.alpha_inner = float(alpha_inner)
        self.alpha_outer = float(alpha_outer)

        self.max_speed = float(max_speed)

        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")

        if self.delta_h <= 0.0 or self.delta_v <= 0.0:
            raise ValueError("Look-ahead distances must be positive")

        if self.gamma_h < 0.0 or self.gamma_v < 0.0:
            raise ValueError("Adaptive gains must be nonnegative")

        if not (0 <= self.beta_inner < self.beta_outer):
            raise ValueError("Invalid beta projection limits")

        if not (0 <= self.alpha_inner < self.alpha_outer):
            raise ValueError("Invalid alpha projection limits")

        self.reset()

    def reset(self):
        """Reset controller and adaptive states."""

        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = None
        self.derivative_filtered = [0.0, 0.0, 0.0]

        # ALOS adaptive states
        self.beta_hat = 0.0
        self.alpha_hat = 0.0

        self.last_mode = None

    @staticmethod
    def body_to_path(v, azimuth, elevation):
        ch = math.cos(azimuth)
        sh = math.sin(azimuth)
        cv = math.cos(elevation)
        sv = math.sin(elevation)

        return (
            cv * (ch * v[0] + sh * v[1]) - sv * v[2],
            -sh * v[0] + ch * v[1],
            sv * (ch * v[0] + sh * v[1]) + cv * v[2],
        )

    @staticmethod
    def path_to_body(v, azimuth, elevation):
        ch = math.cos(azimuth)
        sh = math.sin(azimuth)
        cv = math.cos(elevation)
        sv = math.sin(elevation)

        return (
            ch * cv * v[0] - sh * v[1] + ch * sv * v[2],
            sh * cv * v[0] + ch * v[1] + sh * sv * v[2],
            -sv * v[0] + cv * v[2],
        )

    def compute(
        self,
        target_rel_body,
        desired_rel_body=(5.0, 0.0, 0.0),
        target_velocity_body=(0.0, 0.0, 0.0),
        body_rates_frd=(0.0, 0.0, 0.0),
        dt=0.1,
        controller=None,
    ):
        """
        Returns desired vehicle velocity [vx, vy, vz]
        in body FRD coordinates.

        target_rel_body:
            Target position relative to vehicle [m].

        desired_rel_body:
            Desired target relative position [m].

        target_velocity_body:
            Target absolute/ground velocity expressed in
            vehicle body FRD [m/s].

        body_rates_frd:
            Vehicle angular velocity [p, q, r] [rad/s].

        dt:
            Sampling time [s].

        controller:
            'pid' or 'alos'.
        """

        mode = self.controller if controller is None else controller

        if mode not in ("pid", "alos"):
            raise ValueError("controller must be 'pid' or 'alos'")

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")

        r = vec3(target_rel_body, "target_rel_body")
        rd = vec3(desired_rel_body, "desired_rel_body")
        vt = vec3(target_velocity_body, "target_velocity_body")
        omega = vec3(body_rates_frd, "body_rates_frd")

        # Target-relative position error
        error = tuple(r[i] - rd[i] for i in range(3))

        # Rotating body-frame compensation
        rotation_comp = tuple(-x for x in cross(omega, r))

        # ==================================================
        # CONTROLLER 1: PID
        # ==================================================
        if mode == "pid":

            feedforward = tuple(vt[i] + rotation_comp[i] for i in range(3))

            if self.last_mode != "pid":
                self.prev_error = None
                self.derivative_filtered = [0.0, 0.0, 0.0]
                self.integral = [0.0, 0.0, 0.0]

            # Error derivative
            if self.prev_error is None:
                derivative = [0.0, 0.0, 0.0]
            else:
                derivative = [
                    (error[i] - self.prev_error[i]) / dt
                    for i in range(3)
                ]

            # Derivative filtering
            a = self.derivative_filter

            derivative_filtered = [
                a * derivative[i] + (1.0 - a) * self.derivative_filtered[i]
                for i in range(3)
            ]

            # Integral with component-wise limiting
            integral_candidate = [
                max(
                    -self.integral_limit,
                    min(
                        self.integral_limit,
                        self.integral[i] + error[i] * dt,
                    ),
                )
                for i in range(3)
            ]

            def pid_command(integral_state):
                return tuple(
                    feedforward[i]
                    + self.kp[i] * error[i]
                    + self.ki[i] * integral_state[i]
                    + self.kd[i] * derivative_filtered[i]
                    for i in range(3)
                )

            candidate = pid_command(integral_candidate)

            # Basic anti-windup
            if norm(candidate) <= self.max_speed:
                self.integral = integral_candidate
                command = candidate
            else:
                command = pid_command(self.integral)

            self.prev_error = list(error)
            self.derivative_filtered = derivative_filtered

        # ==================================================
        # CONTROLLER 2: 3D ADAPTIVE LOS (FOSSEN)
        # ==================================================
        else:

            target_speed = norm(vt)

            if target_speed < 1e-4:
                raise ValueError(
                    "ALOS requires a nonzero target velocity. "
                    "Use PID for a stationary target."
                )

            # ----------------------------------------------
            # 1. Construct local path frame from target motion
            # ----------------------------------------------

            path_azimuth = math.atan2(vt[1], vt[0])

            horizontal_speed = math.hypot(vt[0], vt[1])

            path_elevation = math.atan2(-vt[2], horizontal_speed)

            # ----------------------------------------------
            # 2. Transform vehicle path-following error
            #    into path coordinates
            # ----------------------------------------------

            vehicle_error_body = tuple(-x for x in error)

            ex_p, ey_p, ez_p = self.body_to_path(
                vehicle_error_body,
                path_azimuth,
                path_elevation
            )

            # ----------------------------------------------
            # 3. Adaptive parameter update laws
            # ----------------------------------------------

            beta_rate = self.gamma_h * self.delta_h * ey_p / math.sqrt(
                self.delta_h**2 + ey_p**2
            )
            alpha_rate = self.gamma_v * self.delta_v * ez_p / math.sqrt(
                self.delta_v**2 + ez_p**2
            )

            # Parameter projection
            beta_rate = projected_rate(
                self.beta_hat,
                beta_rate,
                self.beta_inner,
                self.beta_outer
            )

            alpha_rate = projected_rate(
                self.alpha_hat,
                alpha_rate,
                self.alpha_inner,
                self.alpha_outer
            )

            # Integrate adaptive estimates
            self.beta_hat = max(
                -self.beta_outer,
                min(self.beta_outer, self.beta_hat + beta_rate * dt),
            )
            self.alpha_hat = max(
                -self.alpha_outer,
                min(self.alpha_outer, self.alpha_hat + alpha_rate * dt),
            )

            # ----------------------------------------------
            # 4. Fossen ALOS heading and pitch commands
            # ----------------------------------------------

            psi_los = (
                -self.beta_hat
                - math.atan2(ey_p, self.delta_h)
            )

            theta_los = (
                self.alpha_hat
                + math.atan2(ez_p, self.delta_v)
            )

            # ----------------------------------------------
            # 5. Along-track synchronization
            # ----------------------------------------------

            # ex_p > 0: vehicle is ahead of desired path
            # ex_p < 0: vehicle is behind desired path

            speed_cmd = max(0.0, target_speed - self.k_along * ex_p)

            # ----------------------------------------------
            # 6. Construct 3D desired velocity in path frame
            # ----------------------------------------------

            direction_path = (
                math.cos(theta_los) * math.cos(psi_los),
                math.cos(theta_los) * math.sin(psi_los),
                -math.sin(theta_los),
            )

            velocity_path = tuple(
                speed_cmd * component for component in direction_path
            )

            # ----------------------------------------------
            # 7. Convert path-frame velocity into body FRD
            # ----------------------------------------------

            velocity_body = self.path_to_body(
                velocity_path,
                path_azimuth,
                path_elevation
            )

            # Compensate for rotation of the body frame
            command = tuple(
                velocity_body[i] + rotation_comp[i]
                for i in range(3)
            )

        self.last_mode = mode

        # Common total speed constraint
        return limit_norm(command, self.max_speed)


if __name__ == "__main__":
    velocity_estimator = TargetVelocityEstimator(tau=0.3)
    tracker = TargetTrackingController(
        controller="alos",
        delta_h=5.0,
        delta_v=3.0,
        gamma_h=0.05,
        gamma_v=0.05,
        k_along=0.3,
        max_speed=2.0,
    )

    def tracking_step(
        target_rel_body,
        vehicle_velocity_body,
        body_rates_frd,
        dt,
    ):
        target_velocity_est = velocity_estimator.update(
            target_rel_body=target_rel_body,
            vehicle_velocity_body=vehicle_velocity_body,
            body_rates_frd=body_rates_frd,
            dt=dt,
        )

        if target_velocity_est is None:
            return (0.0, 0.0, 0.0)

        return tracker.compute(
            target_rel_body=target_rel_body,
            desired_rel_body=(5.0, 0.0, 0.0),
            target_velocity_body=target_velocity_est,
            body_rates_frd=body_rates_frd,
            dt=dt,
        )

    for time_step in range(100):
        velocity_cmd = tracking_step(
            target_rel_body=(0.5 * time_step, 2.0, 1.0),
            vehicle_velocity_body=(0.5, 0.1, 0.0),
            body_rates_frd=(0.0, 0.0, 0.0),
            dt=0.1,
        )
        vx, vy, vz = velocity_cmd
        print(f"Computed velocity: ({vx}, {vy}, {vz})")