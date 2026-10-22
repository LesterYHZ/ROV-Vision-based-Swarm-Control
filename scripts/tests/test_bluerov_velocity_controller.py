import math
import unittest

from bluerov_velocity_controller import (
    REAR_SWAY_RATIO,
    enu_flu_to_ned_frd,
    mix_thruster_commands,
    quaternion_error_vector,
    quaternion_rotate,
)


class ThrusterMixerTest(unittest.TestCase):
    # Effective force directions in body FRD after applying the joint axes,
    # SDF poses, and the reversed thrust coefficients on thrusters 3 and 4.
    FORCE_DIRECTIONS = (
        (-1.0, 1.0),
        (-1.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
    )
    POSITIONS = (
        (0.14, 0.092),
        (0.14, -0.092),
        (-0.15, 0.092),
        (-0.15, -0.092),
    )

    def horizontal_wrench(self, commands):
        fx = fy = yaw_moment = 0.0
        for command, (dir_x, dir_y), (x, y) in zip(
            commands, self.FORCE_DIRECTIONS, self.POSITIONS
        ):
            force_x = command * dir_x / math.sqrt(2.0)
            force_y = command * dir_y / math.sqrt(2.0)
            fx += force_x
            fy += force_y
            yaw_moment += x * force_y - y * force_x
        return fx, fy, yaw_moment

    def assert_only_axis(self, commands, active_axis):
        wrench = self.horizontal_wrench(commands)
        for axis, value in enumerate(wrench):
            if axis == active_axis:
                self.assertGreater(value, 0.0)
            else:
                self.assertAlmostEqual(value, 0.0, places=12)

    def test_surge_has_no_lateral_force_or_yaw(self):
        commands = mix_thruster_commands(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(commands[:4], [-1.0] * 4)
        self.assert_only_axis(commands[:4], active_axis=0)

    def test_sway_has_no_surge_or_yaw(self):
        commands = mix_thruster_commands(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(
            commands[:4],
            [1.0, -1.0, -REAR_SWAY_RATIO, REAR_SWAY_RATIO],
        )
        self.assert_only_axis(commands[:4], active_axis=1)

    def test_yaw_has_no_linear_force_and_correct_sign(self):
        commands = mix_thruster_commands(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.assertEqual(commands[:4], [1.0, -1.0, 1.0, -1.0])
        self.assert_only_axis(commands[:4], active_axis=2)

    def test_heave_accounts_for_reversed_vertical_coefficients(self):
        commands = mix_thruster_commands(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        self.assertEqual(commands[4:], [1.0, -1.0, -1.0, 1.0])
        self.assert_only_vertical_axis(commands[4:], active_axis=0)

    def vertical_wrench(self, commands):
        q5, q6, q7, q8 = commands
        down_force = q5 - q6 - q7 + q8
        roll = 0.215 * (q5 + q6 - q7 - q8)
        pitch = 0.118 * (-q5 + q6 - q7 + q8)
        return down_force, roll, pitch

    def assert_only_vertical_axis(self, commands, active_axis):
        wrench = self.vertical_wrench(commands)
        for axis, value in enumerate(wrench):
            if axis == active_axis:
                self.assertGreater(value, 0.0)
            else:
                self.assertAlmostEqual(value, 0.0, places=12)

    def test_roll_has_no_heave_or_pitch(self):
        commands = mix_thruster_commands(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.assertEqual(commands[4:], [1.0, 1.0, -1.0, -1.0])
        self.assert_only_vertical_axis(commands[4:], active_axis=1)

    def test_pitch_has_no_heave_or_roll(self):
        commands = mix_thruster_commands(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        self.assertEqual(commands[4:], [-1.0, 1.0, -1.0, 1.0])
        self.assert_only_vertical_axis(commands[4:], active_axis=2)

    def test_odometry_pose_converts_from_enu_flu_to_ned_frd(self):
        position, orientation = enu_flu_to_ned_frd(
            (2.0, 3.0, 4.0), (0.0, 0.0, 0.0, 1.0)
        )
        self.assertEqual(position, (3.0, 2.0, -4.0))
        forward_ned = quaternion_rotate(orientation, (1.0, 0.0, 0.0))
        down_ned = quaternion_rotate(orientation, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(forward_ned[0], 0.0, places=12)
        self.assertAlmostEqual(forward_ned[1], 1.0, places=12)
        self.assertAlmostEqual(down_ned[2], 1.0, places=12)

    def test_attitude_error_uses_shortest_quaternion_rotation(self):
        half_angle = 0.2
        error = quaternion_error_vector(
            (0.0, 0.0, math.sin(half_angle), math.cos(half_angle)),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(error[0], 0.0, places=12)
        self.assertAlmostEqual(error[1], 0.0, places=12)
        self.assertAlmostEqual(error[2], 2.0 * half_angle, places=12)


if __name__ == "__main__":
    unittest.main()
