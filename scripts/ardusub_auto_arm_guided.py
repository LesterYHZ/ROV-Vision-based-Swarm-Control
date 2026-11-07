#!/usr/bin/env python3
"""Wait for a usable SITL GPS fix, then select GUIDED and arm ArduSub."""

import rclpy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus


class ArduSubAutoArmGuided(Node):
    def __init__(self):
        super().__init__("ardusub_auto_arm_guided")
        self.declare_parameter("mavros_namespace", "bluerov_0/mavros")
        namespace = str(self.get_parameter("mavros_namespace").value).strip("/")
        if not namespace:
            raise ValueError("mavros_namespace must not be empty")

        prefix = f"/{namespace}"
        self.state = None
        self.have_gps_fix = False
        self.mode_future = None
        self.arm_future = None
        self.last_request_time = self.get_clock().now()
        self.retry_period = 3.0

        self.create_subscription(State, f"{prefix}/state", self._on_state, 10)
        self.create_subscription(
            NavSatFix,
            f"{prefix}/global_position/raw/fix",
            self._on_gps,
            10,
        )
        self.mode_client = self.create_client(SetMode, f"{prefix}/set_mode")
        self.arm_client = self.create_client(CommandBool, f"{prefix}/cmd/arming")
        self.timer = self.create_timer(0.5, self._step)
        self.get_logger().info(
            f"Waiting for {namespace} connection and valid GPS before GUIDED/arm"
        )

    def _on_state(self, message):
        self.state = message

    def _on_gps(self, message):
        self.have_gps_fix = message.status.status >= NavSatStatus.STATUS_FIX

    def _step(self):
        if self.state is None or not self.state.connected or not self.have_gps_fix:
            return

        if self.state.mode == "GUIDED" and self.state.armed:
            self.get_logger().info("ArduSub is armed and in GUIDED mode")
            self.timer.cancel()
            return

        now = self.get_clock().now()
        if (now - self.last_request_time).nanoseconds * 1e-9 < self.retry_period:
            return

        if self.state.mode != "GUIDED":
            if self.mode_future is not None or not self.mode_client.service_is_ready():
                return
            request = SetMode.Request()
            request.base_mode = 0
            request.custom_mode = "GUIDED"
            self.mode_future = self.mode_client.call_async(request)
            self.mode_future.add_done_callback(self._on_mode_response)
            self.last_request_time = now
            self.get_logger().info("Requesting GUIDED mode")
            return

        if not self.state.armed:
            if self.arm_future is not None or not self.arm_client.service_is_ready():
                return
            request = CommandBool.Request()
            request.value = True
            self.arm_future = self.arm_client.call_async(request)
            self.arm_future.add_done_callback(self._on_arm_response)
            self.last_request_time = now
            self.get_logger().info("Requesting arm")

    def _on_mode_response(self, future):
        self.mode_future = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(f"GUIDED request failed: {error}")
            return
        self.get_logger().info(
            f"GUIDED request sent={response.mode_sent}; waiting for FCU state"
        )

    def _on_arm_response(self, future):
        self.arm_future = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(f"Arm request failed: {error}")
            return
        self.get_logger().info(
            f"Arm request accepted={response.success}; waiting for FCU state"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArduSubAutoArmGuided()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
