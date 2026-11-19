#!/usr/bin/env python3
"""BlueROV2-style 4-DOF tracking with a tail-mounted AprilTag and unified CBF.

Leader follows a 3-D figure-eight. The follower's nominal controller tracks the
leader's tail tag and aligns heading. A single QP stacks obstacle, camera FOV,
range, relative-heading, and AprilTag face-incidence constraints.

The follower dynamics are the same transparent reduced-order 4-DOF model from
this project, not ArduSub SITL or a calibrated BlueROV2 Heavy digital twin.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bluerov2_dynamics import BlueROV2Heavy4DOF
from tracking_controller import BodyVelocityTargetTracker
from unified_cbf import (
    UnifiedCollisionFOVCBF,
    RelativeObjectVelocityEstimator,
    wrap_angle,
)
from apriltag_pose import tail_tag_mounting_transform, estimate_leader_from_apriltag


def rotation_body_to_ned(yaw, pitch=0.0, roll=0.0):
    """FRD body to NED rotation using Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def leader_state(t):
    """Leader 3-D figure-eight trajectory and translational NED velocity."""
    period = 200.0
    om = 2.0 * np.pi / period
    ph = om * t
    N = 20.0 + 5.0 * np.sin(ph)
    E = 4.0 * np.sin(2.0 * ph)
    D = 10.0 + 0.55 * np.sin(ph + 0.4)
    vN = 5.0 * om * np.cos(ph)
    vE = 8.0 * om * np.cos(2.0 * ph)
    vD = 0.55 * om * np.cos(ph + 0.4)
    return np.array([N, E, D]), np.array([vN, vE, vD])


def leader_attitude(t):
    """Leader ROV attitude follows path tangent, with mild roll/pitch motion."""
    period = 200.0
    om = 2.0 * np.pi / period
    ph = om * t
    pos, vel = leader_state(t)
    acc = np.array([
        -5.0 * om**2 * np.sin(ph),
        -16.0 * om**2 * np.sin(2.0 * ph),
        -0.55 * om**2 * np.sin(ph + 0.4),
    ])
    horizontal = max(math.hypot(vel[0], vel[1]), 1e-8)
    yaw = math.atan2(vel[1], vel[0])
    yaw_rate = (vel[0] * acc[1] - vel[1] * acc[0]) / max(horizontal**2, 1e-8)
    pitch = math.atan2(-vel[2], horizontal)
    roll = math.radians(4.0) * math.sin(2.0 * ph + 0.3)
    return yaw, pitch, roll, yaw_rate


def obstacle_states(t):
    fixed = np.array([24.0, 6.0, 10.35])
    fleet = np.array([16.8 + 0.075 * t, -9.0 + 0.145 * t, 10.3])
    return [fixed, fleet], [np.zeros(3), np.array([0.075, 0.145, 0.0])]


def tag_incidence_angle(p_tag_camera, tag_normal_camera):
    p = np.asarray(p_tag_camera, dtype=float)
    n = np.asarray(tag_normal_camera, dtype=float)
    if np.linalg.norm(p) < 1e-9:
        return math.pi
    n = n / max(np.linalg.norm(n), 1e-9)
    view = -p / np.linalg.norm(p)  # from tag toward camera
    return math.acos(float(np.clip(np.dot(n, view), -1.0, 1.0)))


def physical_detection_ok(p_cam, rel_heading, incidence,
                          h_fov=34.0, v_fov=40.0, max_x=7.5,
                          max_heading=65.0, max_incidence=68.0):
    x, y, z = p_cam
    if x <= 0.45 or x > max_x:
        return False
    az = abs(math.degrees(math.atan2(y, x)))
    el = abs(math.degrees(math.atan2(z, max(math.hypot(x, y), 1e-8))))
    return (az <= h_fov/2 and el <= v_fov/2
            and abs(math.degrees(wrap_angle(rel_heading))) <= max_heading
            and math.degrees(incidence) <= max_incidence)


def simulate(use_cbf=True, duration=200.0, dt=0.05):
    steps = int(duration / dt)
    camera_offset = np.array([0.22, 0.0, 0.0])
    p_leader_tag, R_leader_tag, normal_tag = tail_tag_mounting_transform((-0.55, 0.0, 0.0))
    leader0, leader_v0 = leader_state(0.0)
    lyaw0, lpitch0, lroll0, _ = leader_attitude(0.0)
    Rl0 = rotation_body_to_ned(lyaw0, lpitch0, lroll0)
    tag0 = leader0 + Rl0 @ p_leader_tag
    course0 = np.array([math.cos(lyaw0), math.sin(lyaw0), 0.0])
    follower_yaw0 = lyaw0
    Rv0 = rotation_body_to_ned(follower_yaw0)
    camera0 = tag0 - 3.8 * course0
    follower_pos0 = camera0 - Rv0 @ camera_offset
    vehicle = BlueROV2Heavy4DOF(position_ned=follower_pos0, yaw=follower_yaw0, dt=dt)

    tracker = BodyVelocityTargetTracker(
        desired_rel_camera=(3.8, 0.2, 1.0),
        kp=(0.24, 0.34, 0.27),
        yaw_gain=0.90,
        bearing_yaw_gain=0.08,
        max_surge=0.62, max_sway=0.50, max_heave=0.24, max_yaw_rate=0.55,
    )
    cbf = UnifiedCollisionFOVCBF(
        horizontal_fov_deg=80.0,
        vertical_fov_deg=64.0,
        fov_margin_deg=(2.0, 2.5),
        camera_offset_body=camera_offset,
        min_camera_depth=0.45,
        max_camera_x=5.9,
        max_range_margin_m=0.20,
        fov_linear_margin=0.045,
        max_relative_heading_deg=45.0,
        heading_margin_deg=4.0,
        max_tag_incidence_deg=55.0,
        tag_normal_leader=(-1.0, 0.0, 0.0),
        vehicle_radius=0.34,
        obstacle_margin=0.28,
        alpha_fov=4.2,
        alpha_collision=1.7,
        alpha_heading=2.4,
        alpha_tag=4.0,
        command_lower=(-0.50, -0.55, -0.27, -0.65),
        command_upper=(0.62, 0.55, 0.27, 0.65),
        command_slew=(3.0, 6.0, 1.5, 6.0),
        weights=(1.0, 2.6, 1.8, 0.75),
        slew_weight=0.12,
        estimator_tau=0.18,
        orientation_estimator_tau=0.22,
    )
    tracker_velocity_estimator = RelativeObjectVelocityEstimator(filter_tau=0.18)

    keys = [
        "t", "vehicle", "leader", "tag", "yaw_vehicle", "yaw_leader",
        "pitch_leader", "roll_leader", "R_vehicle", "R_leader", "R_camera_leader",
        "nu", "q_nom", "q_cmd", "tag_cam", "az", "el", "relative_heading",
        "tag_incidence", "visible", "safe_visible", "static_obs", "fleet_obs",
        "clear_static", "clear_fleet", "cbf_active", "cbf_feasible", "min_margin",
        "fov_hmin", "fov_vmin", "fov_xmax", "heading_margin", "tag_margin",
        "nominal_to_safe", "tag_velocity_est", "status",
    ]
    hist = {k: [] for k in keys}
    last_tag_cam = None
    last_R_camera_leader = None
    last_normal_camera = None
    qp_failures = 0
    loss_count = 0
    collision_count = 0

    for k in range(steps):
        t = k * dt
        leader_ned, leader_v_ned = leader_state(t)
        yaw_l, pitch_l, roll_l, _ = leader_attitude(t)
        R_leader = rotation_body_to_ned(yaw_l, pitch_l, roll_l)
        tag_ned = leader_ned + R_leader @ p_leader_tag

        obs_ned, obs_v_ned = obstacle_states(t)
        vehicle_pos = vehicle.position.copy()
        yaw_v = vehicle.yaw
        nu = vehicle.nu.copy()
        R_vehicle = rotation_body_to_ned(yaw_v)
        camera_ned = vehicle_pos + R_vehicle @ camera_offset

        # Ground-truth tag pose in follower camera coordinates.
        tag_pos_cam_true = R_vehicle.T @ (tag_ned - camera_ned)
        R_camera_leader_true = R_vehicle.T @ R_leader
        R_camera_tag_true = R_camera_leader_true @ R_leader_tag

        # In real operation this transform comes from AprilTag PnP plus a
        # measured/calibrated rigid tag-to-leader mounting transform.
        pose = estimate_leader_from_apriltag(
            tag_pos_cam_true, R_camera_tag_true,
            p_leader_tag, R_leader_tag, normal_tag
        )
        tag_cam_true = pose["tag_position_camera"]
        R_camera_leader_true = pose["leader_rotation_camera"]
        normal_cam_true = pose["tag_normal_camera"]
        relative_heading_true = math.atan2(R_camera_leader_true[1, 0], R_camera_leader_true[0, 0])
        incidence_true = tag_incidence_angle(tag_cam_true, normal_cam_true)

        visible = physical_detection_ok(tag_cam_true, relative_heading_true, incidence_true)
        if visible:
            # Perfect pose measurements in the baseline experiment. Noise and
            # latency are deliberately separated as future test dimensions.
            tag_cam_meas = tag_cam_true.copy()
            R_camera_leader_meas = R_camera_leader_true.copy()
            normal_cam_meas = normal_cam_true.copy()
            last_tag_cam = tag_cam_meas.copy()
            last_R_camera_leader = R_camera_leader_meas.copy()
            last_normal_camera = normal_cam_meas.copy()
        elif last_tag_cam is not None:
            # Pose hold/predictor placeholder after detector dropout. A real
            # implementation should propagate a timestamped target-motion filter.
            tag_cam_meas = last_tag_cam.copy()
            R_camera_leader_meas = last_R_camera_leader.copy()
            normal_cam_meas = last_normal_camera.copy()
        else:
            tag_cam_meas = tag_cam_true.copy()
            R_camera_leader_meas = R_camera_leader_true.copy()
            normal_cam_meas = normal_cam_true.copy()
        if not visible:
            loss_count += 1

        relative_heading_meas = math.atan2(R_camera_leader_meas[1, 0], R_camera_leader_meas[0, 0])
        tag_body_rel = camera_offset + tag_cam_meas
        tag_velocity_est = tracker_velocity_estimator.update(
            "leader_tag", tag_body_rel, nu, dt
        )
        q_nom = tracker.compute(
            tag_cam_meas, tag_velocity_est, camera_offset,
            relative_heading=relative_heading_meas
        )

        # Time-varying low-level disturbances.
        disturbance = np.array([
            0.008 * np.sin(0.31*t),
            0.012 * np.sin(0.23*t + 0.7),
            0.006 * np.sin(0.19*t),
            0.004 * np.sin(0.17*t),
        ])
        obs_body = [R_vehicle.T @ (p - vehicle_pos) for p in obs_ned]
        obstacle_radii = [0.55, 0.48]
        obstacle_ids = ["fixed_obstacle", "fleet_vehicle"]

        if use_cbf:
            result = cbf.filter(
                nominal_command=q_nom,
                leader_rel_camera=tag_cam_meas,
                obstacle_positions_body=obs_body,
                actual_nu=nu,
                dt=dt,
                obstacle_ids=obstacle_ids,
                obstacle_radii=obstacle_radii,
                leader_rotation_camera=R_camera_leader_meas,
                tag_normal_camera=normal_cam_meas,
            )
            q_cmd = result.command.copy()
            feasible = result.feasible
            if not feasible:
                qp_failures += 1
            active = result.active_constraints
            min_margin = result.min_cbf_margin
            status = result.status
            fov_hm = float(np.min(result.fov_margins[:2]))
            fov_vm = float(np.min(result.fov_margins[2:4]))
            fov_xmax = float(result.fov_margins[5])
            heading_margin = float(np.min(result.heading_margins)) if len(result.heading_margins) else np.nan
            tag_margin = float(result.tag_margins[0]) if len(result.tag_margins) else np.nan
        else:
            q_cmd = np.clip(q_nom, [-0.50, -0.32, -0.27, -0.60], [0.62, 0.32, 0.27, 0.60])
            feasible, active, min_margin, status = True, 0, np.nan, "baseline"
            x, y, z = tag_cam_true
            fov_hm = x*np.tan(np.deg2rad(15.0)) - abs(y)
            rho = max(np.hypot(x, y), 1e-8)
            fov_vm = rho*np.tan(np.deg2rad(17.5)) - abs(z)
            fov_xmax = 5.9 - x - 0.20
            heading_margin = math.radians(54.0) - abs(relative_heading_true)
            tag_margin = float(np.dot(normal_cam_true, -tag_cam_true/max(np.linalg.norm(tag_cam_true), 1e-9)) - np.cos(np.deg2rad(55.0)))

        vehicle.step(q_cmd, dt, disturbance)
        clear = [np.linalg.norm(p - vehicle.position) - (0.34 + r)
                 for p, r in zip(obs_ned, obstacle_radii)]
        if min(clear) < 0:
            collision_count += 1

        az = math.degrees(math.atan2(tag_cam_true[1], max(tag_cam_true[0], 1e-9)))
        el = math.degrees(math.atan2(tag_cam_true[2], max(math.hypot(tag_cam_true[0], tag_cam_true[1]), 1e-9)))
        heading_deg = math.degrees(wrap_angle(relative_heading_true))
        incidence_deg = math.degrees(incidence_true)
        effective_visible = (
            visible and abs(az) <= 15.0 and abs(el) <= 17.5
            and tag_cam_true[0] >= 0.45 and tag_cam_true[0] <= 5.70
            and abs(heading_deg) <= 41.0 and incidence_deg <= 55.0
        )

        vals = {
            "t": t, "vehicle": vehicle_pos, "leader": leader_ned, "tag": tag_ned,
            "yaw_vehicle": yaw_v, "yaw_leader": yaw_l, "pitch_leader": pitch_l,
            "roll_leader": roll_l, "R_vehicle": R_vehicle, "R_leader": R_leader,
            "R_camera_leader": R_camera_leader_true, "nu": nu, "q_nom": q_nom,
            "q_cmd": q_cmd, "tag_cam": tag_cam_true, "az": az, "el": el,
            "relative_heading": heading_deg, "tag_incidence": incidence_deg,
            "visible": visible, "safe_visible": effective_visible,
            "static_obs": obs_ned[0], "fleet_obs": obs_ned[1],
            "clear_static": clear[0], "clear_fleet": clear[1],
            "cbf_active": active, "cbf_feasible": feasible,
            "min_margin": min_margin, "fov_hmin": fov_hm,
            "fov_vmin": fov_vm, "fov_xmax": fov_xmax,
            "heading_margin": heading_margin, "tag_margin": tag_margin,
            "nominal_to_safe": np.linalg.norm(q_cmd-q_nom),
            "tag_velocity_est": tag_velocity_est, "status": status,
        }
        for key, value in vals.items():
            hist[key].append(value)

    for key in hist:
        hist[key] = np.asarray(hist[key])
    hist["qp_failures"] = qp_failures
    hist["loss_count"] = loss_count
    hist["collision_count"] = collision_count
    hist["use_cbf"] = use_cbf
    return hist


def _add_rov(ax, center, R, color, label, length=1.15, width=0.48, height=0.30):
    """Draw a simple oriented ROV hull and its forward heading arrow."""
    c = np.asarray(center, dtype=float)
    local = np.array([
        [-length/2, -width/2, -height/2], [length/2, -width/2, -height/2],
        [length/2, width/2, -height/2], [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2], [length/2, -width/2, height/2],
        [length/2, width/2, height/2], [-length/2, width/2, height/2],
    ])
    pts = c + (R @ local.T).T
    faces_idx = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
    faces = [[pts[j] for j in face] for face in faces_idx]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors="k", linewidths=0.45, alpha=0.55)
    ax.add_collection3d(poly)
    fwd = R[:, 0]
    ax.quiver(c[0], c[1], c[2], fwd[0]*1.25, fwd[1]*1.25, fwd[2]*1.25,
              color=color, linewidth=2.0, arrow_length_ratio=0.22, label=label)


def _add_camera_frustum(ax, camera_pos, R_vehicle, length=5.7,
                         half_h_deg=15.0, half_v_deg=17.5, color="tab:blue", alpha=0.14):
    origin = np.asarray(camera_pos, dtype=float)
    th, tv = math.tan(math.radians(half_h_deg)), math.tan(math.radians(half_v_deg))
    local = np.array([[length, -length*th, -length*tv],
                      [length, length*th, -length*tv],
                      [length, length*th, length*tv],
                      [length, -length*th, length*tv]])
    corners = origin + (R_vehicle @ local.T).T
    for p in corners:
        ax.plot([origin[0], p[0]], [origin[1], p[1]], [origin[2], p[2]], color=color, alpha=alpha+0.25, linewidth=0.9)
    for j in range(4):
        p, q = corners[j], corners[(j+1) % 4]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, alpha=alpha+0.35, linewidth=0.9)
    face = Poly3DCollection([corners], facecolors=color, edgecolors=color, alpha=alpha)
    ax.add_collection3d(face)


def _add_tag_panel(ax, tag_center, R_leader, size=0.42):
    c = np.asarray(tag_center, dtype=float)
    half = size/2
    local = np.array([[0, -half, -half], [0, half, -half], [0, half, half], [0, -half, half]])
    pts = c + (R_leader @ local.T).T
    panel = Poly3DCollection([pts], facecolors="black", edgecolors="yellow", linewidths=1.2, alpha=0.95)
    ax.add_collection3d(panel)
    n = R_leader @ np.array([-1.0, 0.0, 0.0])
    ax.quiver(c[0], c[1], c[2], n[0]*0.55, n[1]*0.55, n[2]*0.55,
              color="black", linewidth=1.8, arrow_length_ratio=0.25)


def create_plots(c, b, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    # 3-D NED path with oriented follower/leader hulls, camera FOV frustums,
    # leader heading arrows, and the tail-mounted AprilTag panel.
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(c["leader"][:, 0], c["leader"][:, 1], c["leader"][:, 2],
            color="tab:orange", linewidth=2.1, label="Leader ROV trajectory")
    ax.plot(c["vehicle"][:, 0], c["vehicle"][:, 1], c["vehicle"][:, 2],
            color="tab:blue", linewidth=1.8, label="Follower BlueROV2 + CBF")
    ax.plot(b["vehicle"][:, 0], b["vehicle"][:, 1], b["vehicle"][:, 2],
            "--", color="0.55", linewidth=1.1, alpha=0.65, label="Follower without CBF")
    ax.plot(c["fleet_obs"][:, 0], c["fleet_obs"][:, 1], c["fleet_obs"][:, 2],
            ":", color="tab:green", linewidth=1.4, label="Fleet obstacle path")
    ax.scatter(*c["static_obs"][0], marker="X", s=100, color="tab:red", label="Static obstacle")

    # Place a small number of oriented ROV and camera-frustum snapshots.
    snapshot_times = [0.0, 50.0, 100.0, 150.0]
    drawn_labels = set()
    for ts in snapshot_times:
        idx = int(np.clip(round(ts / (c["t"][1]-c["t"][0])), 0, len(c["t"])-1))
        _add_rov(ax, c["vehicle"][idx], c["R_vehicle"][idx], "tab:blue",
                 "Follower heading" if "follower" not in drawn_labels else "")
        drawn_labels.add("follower")
        _add_rov(ax, c["leader"][idx], c["R_leader"][idx], "tab:orange",
                 "Leader heading" if "leader" not in drawn_labels else "")
        drawn_labels.add("leader")
        _add_tag_panel(ax, c["tag"][idx], c["R_leader"][idx])
        cam_pos = c["vehicle"][idx] + c["R_vehicle"][idx] @ np.array([0.22, 0.0, 0.0])
        if ts in (0.0, 100.0):
            _add_camera_frustum(ax, cam_pos, c["R_vehicle"][idx], color="tab:cyan", alpha=0.10)

    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_zlabel("Down [m]")
    ax.set_title("AprilTag target tracking: ROV headings, tail tag and follower camera FOV")
    ax.invert_zaxis(); ax.grid(True); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "apriltag_rov_ned_3d_headings_fov.png", dpi=180)
    plt.close(fig)

    # North-East figure eight.
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(c["leader"][:, 0], c["leader"][:, 1], label="Leader figure-eight", linewidth=2)
    ax.plot(c["vehicle"][:, 0], c["vehicle"][:, 1], label="Follower + CBF", linewidth=1.8)
    ax.plot(b["vehicle"][:, 0], b["vehicle"][:, 1], "--", label="Follower no CBF", alpha=0.7)
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True); ax.legend(); ax.set_title("North-East projection of the leader figure-eight")
    fig.tight_layout(); fig.savefig(outdir / "leader_figure8_ne.png", dpi=170); plt.close(fig)

    # Tag-center position tracking errors in camera FRD.
    desired = np.array([3.8, 0.0, 0.0])
    ec, eb = c["tag_cam"] - desired, b["tag_cam"] - desired
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for j, name in enumerate(["Forward error [m]", "Right error [m]", "Down error [m]"]):
        axs[j].plot(c["t"], ec[:, j], label="Unified AprilTag CBF")
        axs[j].plot(b["t"], eb[:, j], "--", label="No CBF", alpha=0.75)
        axs[j].axhline(0, color="k", linewidth=0.7); axs[j].set_ylabel(name); axs[j].grid(True)
    axs[0].legend(); axs[-1].set_xlabel("Time [s]")
    fig.suptitle("AprilTag-center tracking errors in camera FRD")
    fig.tight_layout(); fig.savefig(outdir / "apriltag_rov_frd_tracking_errors.png", dpi=170); plt.close(fig)

    # Image bearing, leader-relative heading, tag incidence, forward range.
    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    axs[0].plot(c["t"], c["az"], label="CBF")
    axs[0].plot(b["t"], b["az"], "--", label="No CBF", alpha=0.75)
    axs[0].axhline(40, color="r", linestyle=":"); axs[0].axhline(-40, color="r", linestyle=":")
    axs[0].set_ylabel("Tag azimuth [deg]"); axs[0].grid(True); axs[0].legend()
    axs[1].plot(c["t"], c["el"], label="CBF")
    axs[1].plot(b["t"], b["el"], "--", label="No CBF", alpha=0.75)
    axs[1].axhline(32, color="r", linestyle=":"); axs[1].axhline(-32, color="r", linestyle=":")
    axs[1].set_ylabel("Tag elevation [deg]"); axs[1].grid(True)
    axs[2].plot(c["t"], c["relative_heading"], label="Relative heading")
    axs[2].plot(c["t"], c["tag_incidence"], label="Tag face incidence")
    axs[2].axhline(41, color="tab:blue", linestyle="--", label="CBF heading limit")
    axs[2].axhline(-41, color="tab:blue", linestyle="--")
    axs[2].axhline(55, color="tab:orange", linestyle=":", label="CBF incidence limit")
    axs[2].set_ylabel("Orientation angle [deg]"); axs[2].grid(True); axs[2].legend(ncol=2)
    axs[3].plot(c["t"], c["tag_cam"][:, 0], label="Tag forward x")
    axs[3].axhline(6.6, color="tab:red", linestyle="--", label="Effective max range")
    axs[3].axhline(7.5, color="0.4", linestyle=":", label="Physical max range")
    axs[3].axhline(0.45, color="k", linestyle=":", label="Minimum range")
    axs[3].set_ylabel("Camera x [m]"); axs[3].set_xlabel("Time [s]"); axs[3].grid(True); axs[3].legend(ncol=2)
    fig.suptitle("AprilTag camera visibility: image FOV, relative heading, tag tilt and range")
    fig.tight_layout(); fig.savefig(outdir / "apriltag_rov_visibility_orientation.png", dpi=170); plt.close(fig)

    # Unified CBF diagnostics.
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axs[0].plot(c["t"], c["clear_static"], label="Static obstacle clearance")
    axs[0].plot(c["t"], c["clear_fleet"], label="Fleet obstacle clearance")
    axs[0].axhline(0, color="r", linestyle="--", label="Collision boundary")
    axs[0].set_ylabel("Surface clearance [m]"); axs[0].grid(True); axs[0].legend()
    axs[1].plot(c["t"], c["heading_margin"], label="Relative-heading barrier")
    axs[1].plot(c["t"], c["tag_margin"], label="Tag-incidence barrier")
    axs[1].plot(c["t"], c["fov_xmax"], label="Maximum camera range barrier")
    axs[1].axhline(0, color="r", linestyle="--")
    axs[1].set_ylabel("CBF margin [rad / dimensionless / m]"); axs[1].grid(True); axs[1].legend()
    axs[2].plot(c["t"], c["nominal_to_safe"], label="||q_safe - q_nom||")
    axs[2].plot(c["t"], c["cbf_active"], label="Active CBF constraints")
    axs[2].set_ylabel("QP intervention"); axs[2].set_xlabel("Time [s]"); axs[2].grid(True); axs[2].legend()
    fig.suptitle("Unified collision, FOV, heading and tag-incidence CBF diagnostics")
    fig.tight_layout(); fig.savefig(outdir / "apriltag_rov_unified_cbf_diagnostics.png", dpi=170); plt.close(fig)

    # Follower and leader heading comparison.
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(c["t"], np.rad2deg(np.unwrap(c["yaw_vehicle"])), label="Follower yaw")
    axs[0].plot(c["t"], np.rad2deg(np.unwrap(c["yaw_leader"])), label="Leader yaw")
    axs[0].set_ylabel("Heading [deg]"); axs[0].grid(True); axs[0].legend()
    axs[1].plot(c["t"], c["relative_heading"], label="Relative heading")
    axs[1].axhline(41, color="r", linestyle="--"); axs[1].axhline(-41, color="r", linestyle="--")
    axs[1].set_ylabel("Relative heading [deg]"); axs[1].set_xlabel("Time [s]"); axs[1].grid(True)
    fig.suptitle("Leader/follower heading alignment")
    fig.tight_layout(); fig.savefig(outdir / "apriltag_rov_heading_alignment.png", dpi=170); plt.close(fig)


def metrics(h):
    err = h["tag_cam"] - np.array([3.8, 0.0, 0.0])
    tail = err[int(0.8*len(err)):]
    return {
        "tail_rms_tag_tracking_error_m": float(np.sqrt(np.mean(np.sum(tail**2, axis=1)))),
        "final_tag_tracking_error_m": float(np.linalg.norm(err[-1])),
        "physical_apriltag_loss_samples": int(np.sum(~h["visible"])),
        "physical_apriltag_loss_seconds": float(np.sum(~h["visible"]) * (h["t"][1]-h["t"][0])),
        "effective_guard_violation_samples": int(np.sum(~h["safe_visible"])),
        "max_abs_relative_heading_deg": float(np.max(np.abs(h["relative_heading"]))),
        "max_tag_incidence_deg": float(np.max(h["tag_incidence"])),
        "max_tag_forward_range_m": float(np.max(h["tag_cam"][:, 0])),
        "collision_samples": int(h["collision_count"]),
        "qp_infeasible_samples": int(h["qp_failures"]),
        "min_static_clearance_m": float(np.min(h["clear_static"])),
        "min_fleet_clearance_m": float(np.min(h["clear_fleet"])),
        "mean_qp_intervention": float(np.mean(h["nominal_to_safe"])),
        "mean_active_constraints": float(np.mean(h["cbf_active"])),
    }


def write_metrics(c, b, outdir):
    mc, mb = metrics(c), metrics(b)
    with (outdir / "simulation_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "with_unified_apriltag_cbf", "without_cbf"])
        for k in mc:
            w.writerow([k, mc[k], mb[k]])
    return mc, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--outdir", type=Path, default=Path("results_apriltag"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print("Running unified AprilTag CBF case...")
    protected = simulate(True, args.duration, args.dt)
    print("Running no-CBF baseline...")
    baseline = simulate(False, args.duration, args.dt)
    create_plots(protected, baseline, args.outdir)
    mc, mb = write_metrics(protected, baseline, args.outdir)
    print("\nMETRICS (unified CBF / baseline)")
    for key in mc:
        print(f"{key:38s}: {mc[key]:10.4f} / {mb[key]:10.4f}")
    print(f"\nPlots and metrics saved to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
