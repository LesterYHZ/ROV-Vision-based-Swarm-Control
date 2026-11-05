"""Unit tests for the camera field-of-view CBF-QP filter."""
import unittest
import numpy as np

from scripts.cbf_perception import FOVCBFVelocityFilter


class TestFOVCBF(unittest.TestCase):
    def setUp(self):
        self.cbf = FOVCBFVelocityFilter(
            horizontal_fov_deg=80.0,
            vertical_fov_deg=64.0,
            fov_margin_deg=(10.0, 10.0),
            alpha=1.5,
            max_speed=2.0,
        )

    def test_inside_fov_nominal_command_unchanged(self):
        r = (10.0, 0.5, 0.2)
        v_nom = (0.5, 0.0, 0.0)
        out, info = self.cbf.filter_velocity(
            v_nom, r, (0.5, 0.0, 0.0), dt=0.05, return_info=True
        )
        self.assertTrue(info["feasible"])
        self.assertTrue(np.allclose(out, v_nom, atol=1e-5))
        self.assertTrue(info["fov"]["inside_margined_fov"])

    def test_nominal_lateral_command_is_filtered_at_fov_edge(self):
        self.cbf.reset()
        # Target lies near the right horizontal edge; command drives it farther right.
        r = (8.0, 4.8, 0.0)
        self.cbf.filter_velocity((0.0, 0.0, 0.0), r, (0.0, 0.0, 0.0), dt=0.05)
        out, info = self.cbf.filter_velocity(
            (0.0, -1.0, 0.0), r, (0.0, 0.0, 0.0), dt=0.05,
            return_info=True,
        )
        self.assertTrue(info["feasible"])
        self.assertLess(out[1], 0.0)
        self.assertGreaterEqual(info["min_cbf_margin"], -2e-5)

    def test_vertical_fov_constraint_is_enforced(self):
        self.cbf.reset()
        r = (8.0, 0.0, 3.0)
        self.cbf.filter_velocity((0.0, 0.0, 0.0), r, (0.0, 0.0, 0.0), dt=0.05)
        out, info = self.cbf.filter_velocity(
            (0.0, 0.0, -1.0), r, (0.0, 0.0, 0.0), dt=0.05,
            return_info=True,
        )
        self.assertTrue(info["feasible"])
        # Positive body-down vehicle velocity moves the leader upward in FRD;
        # at minimum, the CBF must reject most of the nominal upward command.
        self.assertGreater(out[2], -0.95)

    def test_velocity_limit(self):
        self.cbf.reset()
        out, info = self.cbf.filter_velocity(
            (2.0, 2.0, 2.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            dt=0.05, return_info=True,
        )
        self.assertTrue(info["feasible"])
        self.assertLessEqual(np.sum(np.abs(out)), 2.0 + 1e-5)
        self.assertLessEqual(np.linalg.norm(out), 2.0 + 1e-5)

    def test_camera_extrinsic_rotation(self):
        # Camera x-axis is mounted along body +y.
        R_body_to_camera = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        camera_cbf = FOVCBFVelocityFilter(
            horizontal_fov_deg=80.0,
            vertical_fov_deg=50.0,
            R_body_to_camera=R_body_to_camera,
        )
        state = camera_cbf.fov_state((0.0, 10.0, 0.0))
        self.assertAlmostEqual(state["depth_m"], 10.0)
        self.assertTrue(state["inside_physical_fov"])

    def test_reset_clears_history(self):
        self.cbf.filter_velocity((0, 0, 0), (10, 0, 0), (0, 0, 0), dt=0.05)
        self.assertIsNotNone(self.cbf._previous_relative_position)
        self.cbf.reset()
        self.assertIsNone(self.cbf._previous_relative_position)

    def test_invalid_fov_rejected(self):
        with self.assertRaises(ValueError):
            FOVCBFVelocityFilter(horizontal_fov_deg=10, vertical_fov_deg=10,
                                 fov_margin_deg=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
