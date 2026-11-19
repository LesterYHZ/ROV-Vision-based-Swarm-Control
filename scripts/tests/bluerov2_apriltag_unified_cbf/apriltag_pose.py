"""AprilTag-to-leader pose conversion helpers.

Detector outputs differ in axis convention. Calibrate R_leader_tag and the
front-face normal axis for the specific AprilTag library and camera pipeline.
"""
import numpy as np
from unified_cbf import apriltag_pose_to_leader_pose


# OpenCV/ROS optical camera axes: x right, y down, z forward.
# Vehicle body FRD axes: x forward, y right, z down.
R_BODY_FROM_OPENCV_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def opencv_optical_pose_to_body_frd(t_camera_optical, R_optical_tag,
                                   R_body_from_optical=None):
    """Convert a pose from OpenCV optical coordinates to body FRD.

    OpenCV optical: +x right, +y down, +z forward.
    Body FRD:       +x forward, +y right, +z down.

    R_body_from_optical may be replaced by a calibrated camera-to-body
    rotation if the camera is not mounted with its optical axis along body +x.
    R_optical_tag maps tag-frame vectors into OpenCV optical coordinates.
    """
    t = np.asarray(t_camera_optical, dtype=float).reshape(3)
    R = np.asarray(R_optical_tag, dtype=float).reshape(3, 3)
    R_bo = (R_BODY_FROM_OPENCV_OPTICAL if R_body_from_optical is None
            else np.asarray(R_body_from_optical, dtype=float).reshape(3, 3))
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(R)):
        raise ValueError("Camera pose must contain finite values")
    if not np.allclose(R_bo.T @ R_bo, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R_bo), 1.0, atol=1e-3):
        raise ValueError("R_body_from_optical must be a proper rotation matrix")
    return R_bo @ t, R_bo @ R


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tail_tag_mounting_transform(tag_offset_leader=(-0.55, 0.0, 0.0)):
    """Example tag mounting: its outward +z normal points aft (-x_leader).

    The returned rotation maps tag-frame vectors into leader FRD. This is a
    calibration example, not a universal AprilTag coordinate convention.
    """
    R_leader_tag = rotation_y(-0.5*np.pi)
    p_leader_tag = np.asarray(tag_offset_leader, dtype=float)
    tag_front_normal_tag = np.array([0.0, 0.0, 1.0])
    return p_leader_tag, R_leader_tag, tag_front_normal_tag


def estimate_leader_from_apriltag(t_camera_tag, R_camera_tag,
                                  p_leader_tag, R_leader_tag,
                                  tag_front_normal_tag=(0.0, 0.0, 1.0)):
    """Return tag position, leader rotation, and outward tag normal in camera FRD."""
    tag_pos = np.asarray(t_camera_tag, dtype=float).reshape(3)
    leader_origin, R_camera_leader, normal_camera = apriltag_pose_to_leader_pose(
        tag_pos, R_camera_tag, p_leader_tag, R_leader_tag, tag_front_normal_tag
    )
    return {
        "tag_position_camera": tag_pos,
        "leader_position_camera": leader_origin,
        "leader_rotation_camera": R_camera_leader,
        "tag_normal_camera": normal_camera,
    }
