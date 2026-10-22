#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64


# The front and rear horizontal thrusters have slightly different yaw moment
# arms in the Heavy SDF (|x| + |y|, with 45-degree thrust axes).
REAR_SWAY_RATIO = (0.14 + 0.092) / (0.15 + 0.092)
WORLD_ENU_TO_NED = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
# FLU (forward-left-up) to FRD (forward-right-down) is a 180-degree
# rotation about the forward axis.
BODY_FLU_TO_FRD = (1.0, 0.0, 0.0, 0.0)


def quaternion_normalize(q):
    norm = math.sqrt(sum(component * component for component in q))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("orientation quaternion must have a finite nonzero norm")
    return tuple(component / norm for component in q)


def quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_inverse(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_rotate(q, vector):
    x, y, z, w = q
    vx, vy, vz = vector
    # Efficiently compute q * [v, 0] * conjugate(q).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def enu_flu_to_ned_frd(position_enu, orientation_enu_flu):
    """Convert a Gazebo ENU/FLU pose into a Fossen NED/FRD pose."""
    x, y, z = position_enu
    position_ned = (y, x, -z)
    orientation_ned_frd = quaternion_normalize(
        quaternion_multiply(
            quaternion_multiply(WORLD_ENU_TO_NED, orientation_enu_flu),
            BODY_FLU_TO_FRD,
        )
    )
    return position_ned, orientation_ned_frd


def quaternion_error_vector(target, current):
    """Return the shortest target-current rotation vector in the world frame."""
    error = quaternion_normalize(
        quaternion_multiply(target, quaternion_inverse(current))
    )
    if error[3] < 0.0:
        error = tuple(-component for component in error)
    vector_norm = math.sqrt(sum(component * component for component in error[:3]))
    if vector_norm < 1e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, error[3]))
    scale = angle / vector_norm
    return (error[0] * scale, error[1] * scale, error[2] * scale)


def mix_thruster_commands(surge, sway, heave, roll, pitch, yaw):
    """Allocate a body FRD wrench across the eight Heavy thrusters."""
    return [
        -surge + sway + yaw,
        -surge - sway - yaw,
        -surge - REAR_SWAY_RATIO * sway + yaw,
        -surge + REAR_SWAY_RATIO * sway - yaw,
        heave + roll - pitch,
        -heave + roll + pitch,
        -heave - roll - pitch,
        heave - roll + pitch,
    ]


class BlueROVMotionController(Node):
    """Track Fossen eta=[x,y,z,phi,theta,psi] pose targets.

    The reference and measured pose are expressed in NED/FRD. The controller
    uses the Fossen kinematic structure: an outer pose loop generates body
    velocity/rate references nu=[u,v,w,p,q,r], followed by six-axis feedback
    and thruster allocation.
    """

    def __init__(self):
        super().__init__("bluerov_motion_controller")

        self.declare_parameter(
            "reference_pose_topic", "/model/bluerov2_heavy_camera/reference_pose"
        )
        self.declare_parameter("odometry_topic", "/model/bluerov2_heavy_camera/odometry")
        self.declare_parameter("max_speed", 3.0)
        self.declare_parameter("max_angular_rate", 1.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("publish_rate", 10.0)
        # Conservative Heavy simulation starting gains. Ordering is
        # [north/east/down] and [surge/sway/heave/roll/pitch/yaw].
        self.declare_parameter("position_gains", [0.5, 0.5, 0.35])
        self.declare_parameter("attitude_gains", [0.5, 0.5, 0.4])
        self.declare_parameter(
            "velocity_gains", [0.2, 0.2, 0.12, 0.1, 0.1, 0.08]
        )
        self.declare_parameter(
            "velocity_integral_gains", [0.02, 0.02, 0.015, 0.01, 0.01, 0.008]
        )
        self.declare_parameter(
            "integral_limits", [0.5, 0.5, 0.4, 0.3, 0.3, 0.25]
        )
        self.declare_parameter("max_thruster_command", 10.0)

        self.reference_pose_topic = self.get_parameter("reference_pose_topic").value
        self.odometry_topic = self.get_parameter("odometry_topic").value
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_angular_rate = float(self.get_parameter("max_angular_rate").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.publish_rate = publish_rate
        self.position_gains = self._get_gain_vector("position_gains", 3)
        self.attitude_gains = self._get_gain_vector("attitude_gains", 3)
        self.velocity_gains = self._get_gain_vector("velocity_gains", 6)
        self.velocity_integral_gains = self._get_gain_vector(
            "velocity_integral_gains", 6
        )
        self.integral_limits = self._get_gain_vector("integral_limits", 6)
        self.max_thruster_command = float(
            self.get_parameter("max_thruster_command").value
        )

        if self.max_speed <= 0.0 or self.max_angular_rate <= 0.0:
            raise ValueError("max_speed and max_angular_rate must be positive")
        if self.command_timeout <= 0.0 or publish_rate <= 0.0:
            raise ValueError("command_timeout and publish_rate must be positive")
        if self.max_thruster_command <= 0.0:
            raise ValueError("max_thruster_command must be positive")
        if any(value < 0.0 for value in self.integral_limits):
            raise ValueError("integral_limits cannot be negative")

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            self.reference_pose_topic,
            self._on_reference_pose,
            10,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            self.odometry_topic,
            self._on_odometry,
            10,
        )
        model_name = self._model_name_from_odometry()
        self.thruster_publishers = [
            self.create_publisher(
                Float64,
                f"/model/{model_name}/joint/thruster{i}_joint/cmd_thrust",
                10,
            )
            for i in range(1, 9)
        ]

        self.current_position = None
        self.current_orientation = None
        self.current_velocity = (0.0,) * 6
        self.reference_position = None
        self.reference_orientation = None
        self.reference_time = None
        self.integrals = [0.0] * 6
        self.last_control_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / publish_rate, self._control_step)
        self.get_logger().info(
            "Six-axis pose controller started | "
            f"reference={self.reference_pose_topic} [PoseStamped, frame_id=ned], "
            f"odometry={self.odometry_topic}"
        )

    def _get_gain_vector(self, parameter_name, length):
        values = [float(value) for value in self.get_parameter(parameter_name).value]
        if len(values) != length or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{parameter_name} must contain {length} finite values")
        return values

    def _model_name_from_odometry(self):
        prefix = "/model/"
        suffix = "/odometry"
        if not self.odometry_topic.startswith(prefix) or not self.odometry_topic.endswith(suffix):
            raise ValueError("odometry_topic must have the form /model/<name>/odometry")
        model_name = self.odometry_topic[len(prefix):-len(suffix)].strip("/")
        if not model_name:
            raise ValueError("odometry_topic does not contain a model name")
        return model_name

    def _on_reference_pose(self, message):
        if message.header.frame_id != "ned":
            self.get_logger().warning(
                "Ignoring pose target: header.frame_id must be exactly 'ned'"
            )
            return

        p = message.pose.position
        position = (float(p.x), float(p.y), float(p.z))
        q = message.pose.orientation
        try:
            orientation = quaternion_normalize(
                (float(q.x), float(q.y), float(q.z), float(q.w))
            )
        except ValueError as error:
            self.get_logger().warning(f"Ignoring pose target: {error}")
            return
        if not all(math.isfinite(value) for value in position):
            self.get_logger().warning("Ignoring pose target with non-finite position")
            return

        self.reference_position = position
        self.reference_orientation = orientation
        self.reference_time = self.get_clock().now()
        self.integrals = [0.0] * 6

    def _on_odometry(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        try:
            self.current_position, self.current_orientation = enu_flu_to_ned_frd(
                (float(p.x), float(p.y), float(p.z)),
                (float(q.x), float(q.y), float(q.z), float(q.w)),
            )
        except ValueError:
            self.get_logger().warning("Ignoring odometry with invalid orientation")
            return

        twist = message.twist.twist
        self.current_velocity = (
            float(twist.linear.x),
            -float(twist.linear.y),
            -float(twist.linear.z),
            float(twist.angular.x),
            -float(twist.angular.y),
            -float(twist.angular.z),
        )

    def _hold_current_pose(self):
        if self.current_position is None or self.reference_position is None:
            return
        self.reference_position = self.current_position
        self.reference_orientation = self.current_orientation
        self.reference_time = None
        self.integrals = [0.0] * 6

    def _control_step(self):
        # Do not actuate during startup. A spawned vehicle can have transient
        # odometry velocity, and no pose target means there is no control task.
        if self.reference_position is None:
            self._publish_thrusters([0.0] * 8)
            self.integrals = [0.0] * 6
            self.last_control_time = self.get_clock().now()
            return

        if self.current_position is None or self.reference_position is None:
            self._publish_thrusters([0.0] * 8)
            return

        now = self.get_clock().now()
        if self.reference_time is not None:
            target_age = (now - self.reference_time).nanoseconds * 1e-9
            if target_age > self.command_timeout:
                self._hold_current_pose()

        dt = (now - self.last_control_time).nanoseconds * 1e-9
        self.last_control_time = now
        if dt <= 0.0 or dt > self.command_timeout:
            dt = 1.0 / self.publish_rate

        position_error_ned = tuple(
            self.reference_position[i] - self.current_position[i]
            for i in range(3)
        )
        velocity_reference_ned = tuple(
            self.position_gains[i] * position_error_ned[i] for i in range(3)
        )
        velocity_reference_ned = self._limit_vector(
            velocity_reference_ned, self.max_speed
        )
        velocity_reference_body = quaternion_rotate(
            quaternion_inverse(self.current_orientation), velocity_reference_ned
        )

        attitude_error_ned = quaternion_error_vector(
            self.reference_orientation, self.current_orientation
        )
        attitude_error_body = quaternion_rotate(
            quaternion_inverse(self.current_orientation), attitude_error_ned
        )
        angular_rate_reference = tuple(
            self.attitude_gains[i] * attitude_error_body[i] for i in range(3)
        )
        angular_rate_reference = self._limit_vector(
            angular_rate_reference, self.max_angular_rate
        )

        reference = velocity_reference_body + angular_rate_reference
        errors = tuple(reference[i] - self.current_velocity[i] for i in range(6))
        old_integrals = self.integrals[:]
        efforts = []
        for i, error in enumerate(errors):
            self.integrals[i] += error * dt
            self.integrals[i] = max(
                -self.integral_limits[i],
                min(self.integral_limits[i], self.integrals[i]),
            )
            effort = (
                self.velocity_gains[i] * error
                + self.velocity_integral_gains[i] * self.integrals[i]
            )
            efforts.append(max(-1.0, min(1.0, effort)))

        commands = mix_thruster_commands(*efforts)
        peak = max(1.0, max(abs(command) for command in commands))
        if peak > 1.0:
            # Allocation scales all thrusters together. Freeze the integrator
            # while saturated, then use the pre-step integral state so it
            # cannot wind up against an unreachable wrench.
            self.integrals = old_integrals
            efforts = [
                max(
                    -1.0,
                    min(
                        1.0,
                        self.velocity_gains[i] * errors[i]
                        + self.velocity_integral_gains[i] * self.integrals[i],
                    ),
                )
                for i in range(6)
            ]
            commands = mix_thruster_commands(*efforts)
            peak = max(1.0, max(abs(command) for command in commands))
        self._publish_thrusters(
            [command / peak * self.max_thruster_command for command in commands]
        )
        # log the final commands for debugging purposes
        self.get_logger().info(f"Published thruster commands: {commands}")

    @staticmethod
    def _limit_vector(vector, maximum):
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= maximum:
            return vector
        scale = maximum / norm
        return tuple(value * scale for value in vector)

    def _publish_thrusters(self, commands):
        for publisher, command in zip(self.thruster_publishers, commands):
            publisher.publish(Float64(data=float(command)))


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVMotionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
