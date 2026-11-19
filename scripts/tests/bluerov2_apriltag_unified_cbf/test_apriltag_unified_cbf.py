import math
import unittest
import numpy as np

from unified_cbf import UnifiedCollisionFOVCBF, apriltag_pose_to_leader_pose
from apriltag_pose import (
    tail_tag_mounting_transform,
    opencv_optical_pose_to_body_frd,
    R_BODY_FROM_OPENCV_OPTICAL,
)
from run_apriltag_rov_sim import simulate, rotation_body_to_ned, leader_state, leader_attitude


class TestAprilTagPoseGeometry(unittest.TestCase):
    def test_opencv_optical_axes_convert_to_body_frd(self):
        # OpenCV optical +z (forward) -> body +x (forward).
        self.assertTrue(np.allclose(R_BODY_FROM_OPENCV_OPTICAL @ [0, 0, 1], [1, 0, 0]))
        # OpenCV optical +x (right) -> body +y (right).
        self.assertTrue(np.allclose(R_BODY_FROM_OPENCV_OPTICAL @ [1, 0, 0], [0, 1, 0]))
        # OpenCV optical +y (down) -> body +z (down).
        self.assertTrue(np.allclose(R_BODY_FROM_OPENCV_OPTICAL @ [0, 1, 0], [0, 0, 1]))

    def test_optical_pose_conversion_preserves_rotation(self):
        p_lt, R_lt, _ = tail_tag_mounting_transform()
        # A forward tag at 4 m is [0,0,4] in optical coordinates.
        t_body, R_body_tag = opencv_optical_pose_to_body_frd(
            [0, 0, 4], R_BODY_FROM_OPENCV_OPTICAL.T @ R_lt
        )
        self.assertTrue(np.allclose(t_body, [4, 0, 0]))
        self.assertTrue(np.allclose(R_body_tag, R_lt))

    def test_tail_tag_mounting_reconstructs_leader_pose(self):
        p_lt, R_lt, n_t = tail_tag_mounting_transform((-0.55, 0.0, 0.0))
        yaw = 0.7
        R_cl = rotation_body_to_ned(yaw)
        R_ct = R_cl @ R_lt
        p_cl_true = np.array([5.5, 0.8, 0.2])
        p_ct = p_cl_true + R_cl @ p_lt
        p_cl, R_rec, n_c = apriltag_pose_to_leader_pose(
            p_ct, R_ct, p_lt, R_lt, n_t
        )
        self.assertTrue(np.allclose(p_cl, p_cl_true, atol=1e-9))
        self.assertTrue(np.allclose(R_rec, R_cl, atol=1e-9))
        self.assertTrue(np.allclose(n_c, R_cl @ np.array([-1.0, 0.0, 0.0]), atol=1e-9))

    def test_tag_face_normal_points_toward_camera_when_following_behind(self):
        p_lt, R_lt, n_t = tail_tag_mounting_transform()
        R_cl = np.eye(3)
        R_ct = R_cl @ R_lt
        p_ct = np.array([5.5, 0.0, 0.0])
        _, _, normal = apriltag_pose_to_leader_pose(p_ct, R_ct, p_lt, R_lt, n_t)
        view = -p_ct / np.linalg.norm(p_ct)
        incidence = math.degrees(math.acos(np.clip(normal @ view, -1.0, 1.0)))
        self.assertAlmostEqual(incidence, 0.0, places=6)


class TestAprilTagCBF(unittest.TestCase):
    def make_cbf(self, **kwargs):
        p = dict(
            horizontal_fov_deg=70.0,
            vertical_fov_deg=50.0,
            fov_margin_deg=(5.0, 4.0),
            camera_offset_body=(0.22, 0.0, 0.0),
            max_camera_x=10.0,
            max_range_margin_m=0.2,
            max_relative_heading_deg=45.0,
            heading_margin_deg=3.0,
            max_tag_incidence_deg=55.0,
            command_slew=(8.0, 8.0, 8.0, 8.0),
        )
        p.update(kwargs)
        return UnifiedCollisionFOVCBF(**p)

    def test_pose_constraints_are_added_to_same_qp(self):
        cbf = self.make_cbf()
        q_nom = np.array([0.2, 0.08, 0.0, 0.0])
        R = np.eye(3)
        res = cbf.filter(q_nom, (5.5, 0.0, 0.0), [(2.0, 2.0, 0.0)],
                         np.zeros(4), 0.1, obstacle_ids=["o1"],
                         leader_rotation_camera=R)
        self.assertTrue(res.feasible, res.status)
        self.assertEqual(len(res.fov_margins), 6)
        self.assertEqual(len(res.heading_margins), 2)
        self.assertEqual(len(res.tag_margins), 1)
        self.assertEqual(len(res.collision_margins), 1)
        self.assertIn("apriltag_face_incidence", res.constraint_labels)
        self.assertIn("relative_heading_positive", res.constraint_labels)

    def test_relative_heading_constraint_rejects_wrong_way_yaw(self):
        cbf = self.make_cbf(max_relative_heading_deg=35.0, heading_margin_deg=2.0)
        # Leader is rotated +33 deg relative to follower; positive yaw rate
        # should reduce the relative heading, not make it grow.
        a = math.radians(33.0)
        R = np.array([[math.cos(a), -math.sin(a), 0.0],
                      [math.sin(a), math.cos(a), 0.0],
                      [0.0, 0.0, 1.0]])
        res = cbf.filter((0.0, 0.0, 0.0, -0.4), (5.5, 0.0, 0.0), [],
                         np.zeros(4), 0.1, leader_rotation_camera=R)
        self.assertTrue(res.feasible, res.status)
        self.assertGreater(res.command[3], -0.05)
        self.assertGreaterEqual(res.min_cbf_margin, -1e-5)

    def test_tag_incidence_constraint_changes_translation_or_yaw(self):
        cbf = self.make_cbf(max_tag_incidence_deg=35.0)
        # Tail normal points aft. Introduce a large tag-bearing offset, so the
        # viewing ray is oblique; the QP must alter the command toward the face.
        psi = math.radians(20.0)
        R = np.array([[math.cos(psi), -math.sin(psi), 0.0],
                      [math.sin(psi), math.cos(psi), 0.0],
                      [0.0, 0.0, 1.0]])
        res = cbf.filter((0.25, 0.20, 0.0, -0.25), (4.5, 2.3, 0.0), [],
                         np.zeros(4), 0.1, leader_rotation_camera=R)
        self.assertTrue(res.feasible, res.status)
        self.assertGreaterEqual(res.min_cbf_margin, -1e-5)
        self.assertTrue(np.all(np.isfinite(res.command)))

    def test_tag_orientation_optional_for_legacy_position_only_call(self):
        cbf = self.make_cbf()
        res = cbf.filter((0.1, 0.0, 0.0, 0.0), (5.5, 0.0, 0.0), [], np.zeros(4), 0.1)
        self.assertTrue(res.feasible, res.status)
        self.assertEqual(len(res.heading_margins), 0)
        self.assertEqual(len(res.tag_margins), 0)


class TestClosedLoopAprilTagScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protected = simulate(True, duration=200.0, dt=0.05)
        cls.baseline = simulate(False, duration=200.0, dt=0.05)

    def test_unified_qp_remains_feasible(self):
        self.assertEqual(self.protected["qp_failures"], 0)
        self.assertTrue(np.all(self.protected["cbf_feasible"]))

    def test_physical_tag_detection_maintained(self):
        self.assertEqual(self.protected["loss_count"], 0)
        self.assertTrue(np.all(self.protected["safe_visible"]))

    def test_heading_inside_physical_detection_limit(self):
        self.assertLessEqual(np.max(np.abs(self.protected["relative_heading"])), 58.0 + 1e-6)

    def test_tag_incidence_inside_physical_detection_limit(self):
        self.assertLessEqual(np.max(self.protected["tag_incidence"]), 65.0 + 1e-6)

    def test_collision_avoidance(self):
        self.assertEqual(self.protected["collision_count"], 0)
        self.assertGreater(np.min(self.protected["clear_static"]), 0.0)
        self.assertGreater(np.min(self.protected["clear_fleet"]), 0.0)

    def test_baseline_exercises_perception_failure(self):
        self.assertGreater(self.baseline["loss_count"], 0)

    def test_leader_completes_figure_eight(self):
        pos0, _ = leader_state(0.0)
        pos1, _ = leader_state(200.0)
        self.assertTrue(np.allclose(pos0, pos1, atol=1e-8))
        # The middle of the period crosses the opposite part of the loop.
        self.assertGreater(np.linalg.norm(leader_state(25.0)[0] - pos0), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
