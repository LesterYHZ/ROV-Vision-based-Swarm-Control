#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64


class BlueROVVelocityController(Node):
    """Track body-frame u, v, w velocities and yaw rate r."""

    def __init__(self):
        super().__init__("bluerov_velocity_controller")

        self.declare_parameter("input_topic", "/motion_vector")
        self.declare_parameter("odometry_topic", "/model/bluerov2_heavy_camera/odometry")
        self.declare_parameter("max_speed", 2.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("surge_gain", 10.0)
        self.declare_parameter("sway_gain", 0.0)
        self.declare_parameter("heave_gain", 10.0)
        self.declare_parameter("yaw_gain", 10.0)
        self.declare_parameter("max_yaw_rate", 1.0)
        self.declare_parameter("max_thruster_command", 1.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.odometry_topic = self.get_parameter("odometry_topic").value
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.surge_gain = float(self.get_parameter("surge_gain").value)
        self.sway_gain = float(self.get_parameter("sway_gain").value)
        self.heave_gain = float(self.get_parameter("heave_gain").value)
        self.yaw_gain = float(self.get_parameter("yaw_gain").value)
        self.max_yaw_rate = float(
            self.get_parameter("max_yaw_rate").value
        )
        self.max_thruster_command = float(
            self.get_parameter("max_thruster_command").value
        )

        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be greater than zero")
        if self.command_timeout <= 0.0:
            raise ValueError("command_timeout must be greater than zero")
        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be greater than zero")
        if self.max_thruster_command <= 0.0:
            raise ValueError("max_thruster_command must be greater than zero")
        if self.max_yaw_rate <= 0.0:
            raise ValueError("max_yaw_rate must be greater than zero")

        self.subscription = self.create_subscription(
            Twist,
            self.input_topic,
            self._on_motion_vector,
            10,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            self.odometry_topic,
            self._on_odometry,
            10,
        )
        self.thruster_publishers = [
            self.create_publisher(
                Float64,
                f"/model/{self._model_name_from_odometry()}/joint/"
                f"thruster{i}_joint/cmd_thrust",
                10,
            )
            for i in range(1, 9)
        ]
        self.last_command = (0.0, 0.0, 0.0, 0.0)
        self.last_command_time = self.get_clock().now()
        self.have_command = False
        self.current_body_velocity = (0.0, 0.0, 0.0)
        self.current_yaw_rate = 0.0
        self.current_yaw = 0.0
        self.have_odometry = False
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_setpoint)

        self.get_logger().info(
            "Velocity controller started | "
            f"input={self.input_topic} (body u/v/w/r), odometry={self.odometry_topic}, "
            f"max_speed={self.max_speed:.2f} m/s"
        )

    def _model_name_from_odometry(self):
        prefix = "/model/"
        suffix = "/odometry"
        if not self.odometry_topic.startswith(prefix) or not self.odometry_topic.endswith(suffix):
            raise ValueError(
                "odometry_topic must have the form /model/<name>/odometry"
            )
        model_name = self.odometry_topic[len(prefix):-len(suffix)].strip("/")
        if not model_name:
            raise ValueError("odometry_topic does not contain a model name")
        return model_name

    def _on_motion_vector(self, message):
        u = float(message.linear.x)
        v = float(message.linear.y)
        w = float(message.linear.z)
        yaw_rate = float(message.angular.z)
        speed = math.sqrt(u * u + v * v + w * w)

        if not all(math.isfinite(value) for value in (u, v, w, yaw_rate, speed)):
            self.get_logger().warning("Ignoring non-finite body velocity command")
            return

        if speed > self.max_speed:
            scale = self.max_speed / speed
            u *= scale
            v *= scale
            w *= scale

        self.last_command = (u, v, w, yaw_rate)
        self.last_command_time = self.get_clock().now()
        self.have_command = True

    def _on_odometry(self, message):
        q = message.pose.pose.orientation
        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(sin_yaw, cos_yaw)
        # Gazebo's odometry publisher reports twist in the child/base frame.
        # These are therefore already the body-frame u, v, w velocities.
        body_u = float(message.twist.twist.linear.x)
        body_v = float(message.twist.twist.linear.y)
        body_w = float(message.twist.twist.linear.z)
        self.current_body_velocity = (
            body_u,
            body_v,
            body_w,
        )
        self.current_yaw_rate = float(message.twist.twist.angular.z)
        self.have_odometry = True

    def _publish_setpoint(self):
        age = (
            self.get_clock().now() - self.last_command_time
        ).nanoseconds * 1e-9
        if not self.have_odometry:
            self._publish_thrusters([0.0] * 8)
            return

        if not self.have_command or age > self.command_timeout:
            self._publish_thrusters([0.0] * 8)
            return

        u, v, w, yaw_rate = self.last_command
        surge = - self.surge_gain * (u - self.current_body_velocity[0])
        sway = - self.sway_gain * (v - self.current_body_velocity[1])
        heave = - self.heave_gain * (w - self.current_body_velocity[2])
        yaw = - self.yaw_gain * (yaw_rate - self.current_yaw_rate)

        scale = max(self.max_speed, 1e-6)
        surge = max(-1.0, min(1.0, surge / scale))
        sway = max(-1.0, min(1.0, sway / scale))
        heave = max(-1.0, min(1.0, heave / scale))
        yaw = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw))

        # The SDF reverses the thrust coefficients of thrusters 3, 4, 6, and
        # 7. These signs compensate for that convention and prevent a pure
        # surge/heave command from becoming a yaw or roll moment.
        commands = [
            surge + sway + yaw,
            surge + sway - yaw,
            -surge + sway + yaw,
            -surge + sway - yaw,
            heave,
            heave,
            heave,
            heave,
        ]
        peak = max(1.0, max(abs(command) for command in commands))
        self._publish_thrusters(
            [command / peak * self.max_thruster_command for command in commands]
        )

    def _publish_thrusters(self, commands):
        for publisher, command in zip(self.thruster_publishers, commands):
            publisher.publish(Float64(data=float(command)))


def main(args=None):
    rclpy.init(args=args)
    node = BlueROVVelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
