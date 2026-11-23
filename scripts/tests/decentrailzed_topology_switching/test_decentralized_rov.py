import math
import unittest
import numpy as np

from fleet_model import ROV, SphereObstacle, create_default_scenario, rot_n_to_b
from decentralized_rov import (
    LocalTagObservation, LocalObjectDetection, DecentralizedTopologySelector,
    LocalRelativeMotionEstimator, LocalParentKinematicsEstimator,
    LocalPIDParentTracker, DecentralizedVisionCBF, make_local_tag_observation,
    detect_local_objects,
)


def obs(target_id, quality=0.8, visible=True, distance=4.8, rel_yaw=0.0, incidence=0.2):
    return LocalTagObservation(
        target_id=target_id, quality=quality, visible=visible,
        range=distance, azimuth=0.0, elevation=0.0,
        relative_yaw=rel_yaw, incidence=incidence, occluded=not visible,
        tag_position_camera=np.array([distance,0.0,0.0]),
        target_origin_body=np.array([distance+1.32,0.0,0.0]),
        tag_position_body=np.array([distance+0.32,0.0,0.0]),
        tag_normal_body=np.array([-1.0,0.0,0.0]),
    )


class TestDecentralizedTopology(unittest.TestCase):
    def test_follower_can_choose_higher_id_vehicle(self):
        selector=DecentralizedTopologySelector(1,q_enter=0.02,min_dwell=0.0)
        # F1 sees F2 and the leader. F2 has better local visual quality.
        local={0:obs(0,0.35),2:obs(2,0.92)}
        parent,_,diag=selector.select(local,0.0)
        self.assertEqual(parent,2)
        self.assertIn(2,diag['candidate_ids'])

    def test_f1_switches_to_f2_while_f2_tracks_leader(self):
        # These are two independent local observations. F1 has lost the leader tag but
        # sees F2; F2 sees only the leader. Neither selector receives the other's choice.
        f1=DecentralizedTopologySelector(1,q_enter=0.05,min_dwell=0.0)
        f2=DecentralizedTopologySelector(2,q_enter=0.05,min_dwell=0.0)
        # Initially F1 tracks the leader and F2 independently tracks the leader.
        f1_initial,_,_=f1.select({0:obs(0,0.86)},0.0)
        f2_parent,_,_=f2.select({0:obs(0,0.84)},0.0)
        self.assertEqual(f1_initial,0)
        self.assertEqual(f2_parent,0)
        # Leader tag disappears only in F1's local image; F1 sees F2's tag and switches.
        f1_parent,_,diag=f1.select({2:obs(2,0.88)},10.0)
        # F2 receives no notification and simply continues its own leader selection.
        f2_parent,_,_=f2.select({0:obs(0,0.82)},10.0)
        self.assertEqual(f1_parent,2)
        self.assertEqual(f2_parent,0)
        self.assertEqual(diag['mode'],'reacquire_or_switch')

    def test_cycles_are_possible_and_intentionally_not_prevented(self):
        # With arbitrary local parent choice and no communication, mutually visible tags
        # can cause a cycle. This is documented as a limitation rather than hidden by ranks.
        f1=DecentralizedTopologySelector(1,q_enter=0.05,min_dwell=0.0)
        f2=DecentralizedTopologySelector(2,q_enter=0.05,min_dwell=0.0)
        p1,_,_=f1.select({2:obs(2,0.9)},0.0)
        p2,_,_=f2.select({1:obs(1,0.9)},0.0)
        self.assertEqual(p1,2)
        self.assertEqual(p2,1)

    def test_offline_cycle_monitor_detects_cycle_but_is_not_in_control_path(self):
        from run_decentralized_multi_rov_showcase import find_parent_cycles
        parents={1:2, 2:1, 3:0}
        self.assertEqual(find_parent_cycles(parents), [(1,2)])
        self.assertEqual(find_parent_cycles({1:0, 2:0, 3:1}), [])

    def test_link_loss_immediately_reacquires_best_local_candidate(self):
        selector=DecentralizedTopologySelector(3,q_enter=0.05,q_keep=0.03,min_dwell=10.0)
        p,_,_=selector.select({2:obs(2,0.9),1:obs(1,0.5),0:obs(0,0.3)},0.0)
        self.assertEqual(p,2)
        # Losing F2 allows immediate switch to F1 even during minimum dwell.
        p,_,diag=selector.select({1:obs(1,0.8),0:obs(0,0.4)},0.1)
        self.assertEqual(p,1)
        self.assertEqual(diag['mode'],'reacquire_or_switch')

    def test_hysteresis_prevents_chattering(self):
        selector=DecentralizedTopologySelector(2,q_enter=0.03,q_keep=0.02,
                    min_dwell=2.0,switch_margin=0.2)
        p,_,_=selector.select({0:obs(0,0.96),1:obs(1,0.95)},0.0)
        self.assertEqual(p,0)
        p,_,diag=selector.select({0:obs(0,0.96),1:obs(1,0.98)},0.5)
        self.assertEqual(p,0)
        self.assertEqual(diag['mode'],'retain_dwell')

    def test_hold_when_no_locally_visible_other_tag(self):
        selector=DecentralizedTopologySelector(1)
        p,_,diag=selector.select({},0.0)
        self.assertIsNone(p)
        self.assertEqual(diag['mode'],'hold_reacquire')

    def test_followers_decide_independently_without_global_child_capacity(self):
        # Both followers may independently choose the leader; there is no exchanged
        # child-count or centralized capacity constraint in the no-communication design.
        f1=DecentralizedTopologySelector(1,q_enter=0.02)
        f2=DecentralizedTopologySelector(2,q_enter=0.02)
        p1,_,_=f1.select({0:obs(0,0.85)},0.0)
        p2,_,_=f2.select({0:obs(0,0.80),1:obs(1,0.25)},0.0)
        self.assertEqual(p1,0)
        self.assertEqual(p2,0)


class TestLocalVisionAndTracking(unittest.TestCase):
    def test_tag_pose_is_reconstructed_from_tail_mount(self):
        leader,followers,obstacles=create_default_scenario()
        from fleet_model import AprilTagLinkEvaluator
        evaluator=AprilTagLinkEvaluator(obstacles=[])
        follower=followers[0]
        measurement=make_local_tag_observation(follower,leader,evaluator)
        expected=rot_n_to_b(follower.yaw)@(leader.pos-follower.pos)
        np.testing.assert_allclose(measurement.target_origin_body,expected,atol=1e-8)

    def test_visual_motion_estimator_uses_only_relative_position_sequence(self):
        observer=ROV(1,'F1',np.zeros(3),0.0,nu=np.array([0.2,0,0]))
        estimator=LocalRelativeMotionEstimator(tau=1e-6)
        p0=np.array([5.0,0.0,0.0])
        estimator.update_position('ob',p0,observer,0.1,is_static=False)
        p1=np.array([4.9,0.0,0.0])
        vel=estimator.update_position('ob',p1,observer,0.1,is_static=False)
        np.testing.assert_allclose(vel,[-1.0,0.0,0.0],atol=1e-5)

    def test_parent_estimator_outputs_local_velocity_and_yaw_rate(self):
        follower=ROV(1,'F1',np.zeros(3),0.0,nu=np.array([0.1,0,0]),r=0.05)
        estimator=LocalParentKinematicsEstimator(tau=0.1)
        measurement=obs(0,0.8)
        v,r=estimator.update(follower,measurement,0.1)
        self.assertEqual(v.shape,(3,))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.isfinite(r))

    def test_pid_tracker_uses_only_relative_parent_measurement(self):
        follower=ROV(2,'F2',np.zeros(3),0.0)
        tracker=LocalPIDParentTracker()
        q=tracker.compute(follower,np.array([6.,0.2,0.]),np.array([0.3,0,0]),
                          0.1,0.05,0.1)
        self.assertEqual(q.shape,(4,))
        self.assertTrue(np.all(np.isfinite(q)))

    def test_local_collision_detector_returns_only_camera_visible_objects(self):
        observer=ROV(1,'F1',np.zeros(3),0.0)
        obstacle=SphereObstacle('front',np.array([5.,0.,0.]),0.5)
        behind=SphereObstacle('behind',np.array([-2.,0.,0.]),0.5)
        estimator=LocalRelativeMotionEstimator()
        detections=detect_local_objects(observer,[observer],[obstacle,behind],estimator,0.1)
        ids={d.object_id for d in detections}
        self.assertIn('front',ids)
        self.assertNotIn('behind',ids)


class TestDecentralizedUnifiedCBF(unittest.TestCase):
    def test_local_cbf_returns_bounded_four_axis_command(self):
        follower=ROV(1,'F1',np.zeros(3),0.0)
        cbf=DecentralizedVisionCBF()
        cmd,info=cbf.filter(follower,np.array([0.3,0.02,0.0,0.1]),None,None,None,[],0.1,np.zeros(4))
        self.assertEqual(cmd.shape,(4,))
        self.assertTrue(np.all(np.isfinite(cmd)))
        self.assertTrue(info['feasible'])
        self.assertTrue(np.all(cmd>=cbf.lower-1e-8))
        self.assertTrue(np.all(cmd<=cbf.upper+1e-8))

    def test_local_collision_barrier_is_added(self):
        follower=ROV(1,'F1',np.zeros(3),0.0)
        det=LocalObjectDetection('reef',np.array([2.0,0,0]),np.zeros(3),0.6,True)
        cbf=DecentralizedVisionCBF(command_lower=(-0.8,-0.5,-0.4,-0.8),
                    command_upper=(0.9,0.5,0.4,0.8),max_delta=(0.9,0.5,0.4,0.8))
        cmd,info=cbf.filter(follower,np.array([0.8,0,0,0]),None,None,None,[det],0.1,np.zeros(4))
        self.assertTrue(info['feasible'])
        self.assertIn('collision:reef',info['labels'])
        self.assertLess(cmd[0],0.8)

    def test_parent_fov_heading_incidence_and_range_are_in_same_qp(self):
        follower=ROV(1,'F1',np.zeros(3),0.0)
        parent=obs(0,0.8,distance=7.0,rel_yaw=0.2,incidence=0.25)
        cbf=DecentralizedVisionCBF()
        cmd,info=cbf.filter(follower,np.array([0.35,0.05,0.0,0.1]),parent,
                            np.zeros(3),0.0,[],0.1,np.zeros(4))
        for label in ('fov_h_right','fov_h_left','fov_v_down','fov_v_up',
                      'range_min','range_max','relative_heading_plus',
                      'relative_heading_minus','tag_incidence'):
            self.assertIn(label,info['labels'])
        self.assertEqual(cmd.shape,(4,))


if __name__=='__main__':
    unittest.main(verbosity=2)
