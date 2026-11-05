"""Unit tests for cbf_filter.py. Run: python -m unittest -v"""

import unittest
import numpy as np

from scripts.cbf_collision import CBFCollisionAvoidance


class TestCBFCollisionAvoidance(unittest.TestCase):

    def make_filter(self, **kwargs):
        params = dict(
            vehicle_radius=0.5,
            safety_margin=1.0,
            alpha=1.0,
            max_speed=2.0,
            unknown_closing_speed=0.0,
        )
        params.update(kwargs)
        return CBFCollisionAvoidance(**params)

    def test_no_obstacles_returns_nominal(self):
        cbf = self.make_filter()
        v, info = cbf.filter_velocity(
            v_nom=(0.4, -0.2, 0.1),
            obstacle_positions_body=[],
            vehicle_velocity_body=(0.0, 0.0, 0.0),
            dt=0.1,
            return_info=True,
        )
        np.testing.assert_allclose(v, (0.4, -0.2, 0.1), atol=1e-7)
        self.assertTrue(info["feasible"])

    def test_far_obstacle_does_not_change_command(self):
        cbf = self.make_filter()
        # Prime history while the vehicle is not moving.
        cbf.filter_velocity(
            (0.5, 0.0, 0.0), [(20.0, 0.0, 0.0)],
            (0.0, 0.0, 0.0), 0.1, obstacle_ids=["A"]
        )
        v = cbf.filter_velocity(
            (0.5, 0.0, 0.0), [(20.0, 0.0, 0.0)],
            (0.0, 0.0, 0.0), 0.1, obstacle_ids=["A"]
        )
        np.testing.assert_allclose(v, (0.5, 0.0, 0.0), atol=1e-5)

    def test_static_obstacle_limits_forward_motion(self):
        cbf = self.make_filter(alpha=0.5)
        # Static obstacle: r_dot = -v_actual along the radial direction.
        cbf.filter_velocity(
            (1.5, 0.0, 0.0), [(2.6, 0.0, 0.0)],
            (1.0, 0.0, 0.0), 0.1, obstacle_ids=["A"]
        )
        v, info = cbf.filter_velocity(
            (1.5, 0.0, 0.0), [(2.5, 0.0, 0.0)],
            (1.0, 0.0, 0.0), 0.1,
            obstacle_ids=["A"], return_info=True
        )
        self.assertTrue(info["feasible"], info)
        self.assertLess(v[0], 1.5)
        self.assertGreaterEqual(info["min_cbf_margin"], -1e-5)

    def test_moving_obstacle_closing_is_accounted_for(self):
        cbf = self.make_filter(alpha=0.5)
        # Vehicle stationary, obstacle approaches from ahead at 1 m/s.
        cbf.filter_velocity(
            (0.8, 0.0, 0.0), [(4.1, 0.0, 0.0)],
            (0.0, 0.0, 0.0), 0.1, obstacle_ids=["fleet_vehicle"]
        )
        v, info = cbf.filter_velocity(
            (0.8, 0.0, 0.0), [(4.0, 0.0, 0.0)],
            (0.0, 0.0, 0.0), 0.1,
            obstacle_ids=["fleet_vehicle"], return_info=True
        )
        self.assertTrue(info["feasible"], info)
        self.assertLess(v[0], 0.8)
        self.assertGreaterEqual(info["min_cbf_margin"], -1e-5)

    def test_lateral_obstacle_changes_lateral_command(self):
        cbf = self.make_filter(alpha=0.5)
        # Obstacle is on the right (positive y). Nominally moving right
        # should be filtered toward a less-rightward / leftward command.
        cbf.filter_velocity(
            (0.0, 1.0, 0.0), [(0.0, 2.6, 0.0)],
            (0.0, 1.0, 0.0), 0.1, obstacle_ids=["A"]
        )
        v, info = cbf.filter_velocity(
            (0.0, 1.0, 0.0), [(0.0, 2.5, 0.0)],
            (0.0, 1.0, 0.0), 0.1,
            obstacle_ids=["A"], return_info=True
        )
        self.assertTrue(info["feasible"], info)
        self.assertLess(v[1], 1.0)

    def test_multiple_obstacles(self):
        cbf = self.make_filter(alpha=0.5)
        positions0 = [(3.1, 0.0, 0.0), (0.0, 3.1, 0.0)]
        positions1 = [(3.0, 0.0, 0.0), (0.0, 3.0, 0.0)]
        cbf.filter_velocity(
            (1.0, 1.0, 0.0), positions0,
            (0.5, 0.5, 0.0), 0.1, obstacle_ids=["A", "B"]
        )
        v, info = cbf.filter_velocity(
            (1.0, 1.0, 0.0), positions1,
            (0.5, 0.5, 0.0), 0.1,
            obstacle_ids=["A", "B"], return_info=True
        )
        self.assertTrue(info["feasible"], info)
        self.assertEqual(len(info["cbf_margins"]), 2)
        self.assertGreaterEqual(info["min_cbf_margin"], -1e-5)

    def test_speed_limit(self):
        cbf = self.make_filter(max_speed=1.0)
        v = cbf.filter_velocity(
            (3.0, 0.0, 0.0), [], (0.0, 0.0, 0.0), 0.1
        )
        self.assertLessEqual(np.linalg.norm(v), 1.0 + 1e-8)

    def test_invalid_dt_rejected(self):
        cbf = self.make_filter()
        with self.assertRaises(ValueError):
            cbf.filter_velocity(
                (0.0, 0.0, 0.0), [], (0.0, 0.0, 0.0), 0.0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
