"""Unit tests for surge-dominant heading-forward vehicle kinematics."""
import math
import unittest

import numpy as np

from scripts.tests.test_heading_forward_fov import HeadingForwardVehicleNED


class TestHeadingForwardVehicle(unittest.TestCase):
    def test_vehicle_turns_to_course_and_moves_forward(self):
        vehicle = HeadingForwardVehicleNED(
            position_ned=np.zeros(3),
            yaw=0.0,
            pitch=0.0,
        )
        dt = 0.02
        desired_course_ned = np.array([0.0, 1.0, 0.0])  # due East

        for k in range(int(12.0 / dt)):
            # Re-express a fixed NED course in the current FRD frame.
            command_body = vehicle.rotation().T @ desired_course_ned
            vehicle.step(command_body, dt, t=k * dt)

        yaw_error = abs((vehicle.yaw - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
        self.assertLess(yaw_error, math.radians(3.0))
        self.assertGreater(vehicle.surge, 0.5)
        self.assertLess(abs(vehicle.sway), 0.02)

    def test_sway_is_small_and_camera_tracks_motion_direction(self):
        vehicle = HeadingForwardVehicleNED(position_ned=np.zeros(3))
        dt = 0.05
        sway_samples = []
        alignment_samples = []

        desired_ned = np.array([0.8, 0.3, 0.05])
        for k in range(int(30.0 / dt)):
            command_body = vehicle.rotation().T @ desired_ned
            vehicle.step(command_body, dt, t=k * dt)
            v_ned = vehicle.velocity_ned()
            speed = np.linalg.norm(v_ned)
            if speed > 1e-5:
                forward_ned = vehicle.rotation()[:, 0]
                cosine = np.clip(np.dot(forward_ned, v_ned / speed), -1.0, 1.0)
                alignment_samples.append(math.degrees(math.acos(cosine)))
            sway_samples.append(vehicle.sway)

        self.assertLess(float(np.sqrt(np.mean(np.square(sway_samples)))), 0.03)
        self.assertLess(float(np.mean(alignment_samples)), 2.0)
        self.assertLessEqual(abs(vehicle.yaw_rate), vehicle.max_yaw_rate + 1e-9)
        self.assertLessEqual(abs(vehicle.pitch_rate), vehicle.max_pitch_rate + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
