"""Example integration of AprilTag pose detections with the unified CBF.

Replace `get_apriltag_pose()` and the navigation/obstacle inputs with your
camera detector, DVL/INS and obstacle-tracker outputs.
"""
from __future__ import annotations

import math
import numpy as np

from apriltag_pose import (
    tail_tag_mounting_transform,
    estimate_leader_from_apriltag,
    opencv_optical_pose_to_body_frd,
    R_BODY_FROM_OPENCV_OPTICAL,
)
from tracking_controller import BodyVelocityTargetTracker
from unified_cbf import UnifiedCollisionFOVCBF, RelativeObjectVelocityEstimator


CAMERA_OFFSET_BODY = np.array([0.22, 0.0, 0.0])
TAG_OFFSET_LEADER, R_LEADER_TAG, TAG_FRONT_NORMAL_TAG = tail_tag_mounting_transform(
    tag_offset_leader=(-0.55, 0.0, 0.0)
)

tracker = BodyVelocityTargetTracker(
    desired_rel_camera=(3.8, 0.0, 0.0),
    max_surge=0.62,
    max_sway=0.50,
    max_heave=0.24,
    max_yaw_rate=0.55,
)

safety_filter = UnifiedCollisionFOVCBF(
    horizontal_fov_deg=34.0,
    vertical_fov_deg=40.0,
    fov_margin_deg=(2.0, 2.5),
    camera_offset_body=CAMERA_OFFSET_BODY,
    min_camera_depth=0.45,
    max_camera_x=5.9,
    max_range_margin_m=0.20,
    max_relative_heading_deg=45.0,
    heading_margin_deg=4.0,
    max_tag_incidence_deg=55.0,
    tag_normal_leader=(-1.0, 0.0, 0.0),
)

leader_velocity_estimator = RelativeObjectVelocityEstimator(filter_tau=0.18)


def control_step(
    t_camera_tag_optical,
    R_optical_tag,
    actual_nu,
    dt,
    obstacle_positions_body,
    obstacle_ids,
    obstacle_radii,
    roll_rate=0.0,
    pitch_rate=0.0,
    R_body_from_optical=None,
):
    """Return a safe [u,v,w,r] command from one valid AprilTag detection.

    Detector outputs must be timestamp-aligned with actual_nu and obstacle
    positions. `R_optical_tag` must map tag axes into camera optical axes.
    If your detector uses a different axis convention, calibrate the relevant
    conversion before using this example.
    """
    if t_camera_tag_optical is None or R_optical_tag is None:
        # Do not feed stale pose into the CBF as if it were a current detection.
        # Trigger a dedicated target-loss/reacquisition supervisor instead.
        raise RuntimeError(
            "AprilTag pose unavailable: switch to target-loss/reacquisition "
            "supervisor; stale pose cannot certify FOV or tag visibility."
        )

    # 1. Convert standard OpenCV optical pose to body-aligned camera FRD.
    t_camera_tag_frd, R_camera_tag_frd = opencv_optical_pose_to_body_frd(
        t_camera_tag_optical,
        R_optical_tag,
        R_body_from_optical=R_body_from_optical,
    )

    # 2. Recover leader ROV pose from the tail-mounted tag.
    pose = estimate_leader_from_apriltag(
        t_camera_tag_frd,
        R_camera_tag_frd,
        TAG_OFFSET_LEADER,
        R_LEADER_TAG,
        TAG_FRONT_NORMAL_TAG,
    )
    tag_position_camera = pose["tag_position_camera"]
    leader_rotation_camera = pose["leader_rotation_camera"]
    tag_normal_camera = pose["tag_normal_camera"]

    # Relative leader yaw in camera/body FRD.
    relative_heading = math.atan2(
        leader_rotation_camera[1, 0], leader_rotation_camera[0, 0]
    )

    # 3. Estimate the tag's translational velocity in the follower body frame.
    actual_nu = np.asarray(actual_nu, dtype=float)
    tag_position_body_origin = CAMERA_OFFSET_BODY + tag_position_camera
    leader_tag_velocity_body = leader_velocity_estimator.update(
        "leader_tail_tag",
        tag_position_body_origin,
        actual_nu,
        dt,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
    )

    # 4. Nominal target tracking command [surge, sway, heave, yaw rate].
    q_nom = tracker.compute(
        tag_position_camera,
        leader_tag_velocity_body,
        camera_offset_body=CAMERA_OFFSET_BODY,
        relative_heading=relative_heading,
    )

    # 5. One QP enforces obstacle clearance, camera FOV/range, relative
    # heading, and AprilTag face-incidence constraints simultaneously.
    result = safety_filter.filter(
        nominal_command=q_nom,
        leader_rel_camera=tag_position_camera,
        obstacle_positions_body=obstacle_positions_body,
        actual_nu=actual_nu,
        dt=dt,
        obstacle_ids=obstacle_ids,
        obstacle_radii=obstacle_radii,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        leader_rotation_camera=leader_rotation_camera,
        tag_normal_camera=tag_normal_camera,
    )

    if not result.feasible:
        # Do not silently treat an infeasible safety optimization as safe.
        raise RuntimeError(f"Unified CBF infeasible: {result.status}")

    return result.command.copy(), {
        "nominal_command": q_nom,
        "cbf_result": result,
        "tag_position_camera_frd": tag_position_camera,
        "leader_rotation_camera_frd": leader_rotation_camera,
        "tag_normal_camera_frd": tag_normal_camera,
        "relative_heading_rad": relative_heading,
    }


if __name__ == "__main__":
    # Example synthetic AprilTag observation. Here the tag lies 3.8 m ahead
    # of the camera, directly facing it. Actual detector output replaces this.
    t_optical = np.array([0.0, 0.0, 3.8])
    R_optical_tag = R_BODY_FROM_OPENCV_OPTICAL.T @ R_LEADER_TAG

    # For a real optical camera, replace these synthetic values with the
    # detector output and verify the detector-specific tag-axis convention.
    try:
        command, diagnostics = control_step(
            t_camera_tag_optical=t_optical,
            R_optical_tag=R_optical_tag,
            actual_nu=np.zeros(4),
            dt=0.05,
            obstacle_positions_body=[],
            obstacle_ids=[],
            obstacle_radii=[],
        )
        print("Safe [u, v, w, r] command:", command)
        print("CBF status:", diagnostics["cbf_result"].status)
    except RuntimeError as exc:
        print("Example input failed the geometry/safety check:", exc)
