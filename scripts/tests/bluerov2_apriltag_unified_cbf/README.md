# BlueROV2 Heavy AprilTag target tracking with a unified CBF-QP

This project extends the BlueROV2-style 4-DOF target tracker with AprilTag-based relative pose estimation and a single quadratic-programming safety filter. The target is another ROV with a rigid AprilTag mounted on its tail. The follower's camera is fixed at its bow and looks forward.

The command vector is

\[
q=[u,v,w,r]^T
\]

where `u`, `v`, `w` are desired body-FRD surge, sway, and heave velocities (m/s), and `r` is desired yaw rate (rad/s). A lower-level velocity controller and thruster allocator must track these references.

## What is in the package

- `apriltag_pose.py`: converts AprilTag PnP output into leader-ROV pose using a calibrated rigid tag-to-leader transform; includes OpenCV optical-frame to body-FRD conversion.
- `tracking_controller.py`: nominal tail-tag relative-position and relative-heading tracker.
- `unified_cbf.py`: one QP combining obstacle collision avoidance, horizontal/vertical camera FOV, near/far range, relative heading, and tag-face incidence constraints.
- `bluerov2_dynamics.py`: transparent reduced-order 4-DOF BlueROV2-Heavy-style simulation model with a finite-bandwidth velocity loop.
- `run_apriltag_rov_sim.py`: figure-eight closed-loop simulation and 3D visualization of both ROV headings, the tail tag, and follower camera FOV.
- `example_apriltag_control_loop.py`: illustrative interface from detector pose to `[u,v,w,r]`.
- `test_apriltag_unified_cbf.py`: AprilTag geometry, QP constraints, and full closed-loop regression tests.
- `results_apriltag/`: plots and metrics from the tested 200-second figure-eight run.

## AprilTag coordinate frames and mounting calibration

The project uses vehicle body FRD coordinates:

- `+x`: forward
- `+y`: right
- `+z`: down

The sample camera is aligned with the vehicle's forward axis. OpenCV/ROS optical coordinates are commonly `+x` right, `+y` down, `+z` optical-forward. For that convention, `apriltag_pose.py` uses

\[
R_{B\leftarrow O}=\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix}.
\]

Thus optical `+z` maps to body `+x`. If the camera is mounted with a different rotation, supply a calibrated `R_body_from_optical` to `opencv_optical_pose_to_body_frd()`.

The sample tail tag is placed at `p_leader_tag=(-0.55, 0, 0)` m. Its front normal is configured to point aft along leader body `-x`, toward a follower travelling behind the leader. This is only an example mounting calibration. Measure the actual tag position, tag rotation, and visible-face normal on the target ROV. AprilTag libraries differ in tag-axis conventions; choose `tag_front_normal_tag` to match the detector's actual pose convention. Do not assume the example rotation is correct for an arbitrary tag/detector.

Given detector translation `t_camera_tag` and rotation `R_camera_tag` (tag axes mapped into camera coordinates), the leader pose is recovered using the calibrated tag transform:

\[
R_{C,L}=R_{C,T}R_{L,T}^{T},\qquad
p_{C,L}=p_{C,T}-R_{C,L}p_{L,T}.
\]

The tag center `p_camera_tag` is retained for FOV and range checks because that is the feature the detector must see. The leader rotation and tag outward normal are used for relative-heading and incidence constraints.

## CBF safety conditions

Let `p=[x,y,z]` be the AprilTag center in camera FRD. The effective camera half angles are the physical half-FOV less the configured guard margins. The rectangular frustum constraints are

\[
\begin{aligned}
h_1&=x\tan\theta_h-y-m_F, & h_2&=x\tan\theta_h+y-m_F,\\
h_3&=\sqrt{x^2+y^2}\tan\theta_v-z-m_F, &
h_4&=\sqrt{x^2+y^2}\tan\theta_v+z-m_F.
\end{aligned}
\]

The near and far range barriers are

\[
h_5=x-x_{min},\qquad h_6=x_{max}-x-m_x.
\]

For relative leader yaw `psi_rel`, the controller adds

\[
h_{\psi+}=\psi_{max}-\psi_{rel},\qquad
h_{\psi-}=\psi_{max}+\psi_{rel}.
\]

For tag outward normal `n_T^C` and direction from tag toward the camera `d=-p/||p||`, the face-incidence barrier is

\[
h_{tag}=(n_T^C)^T d-\cos(\theta_{inc,max}).
\]

Each detected obstacle `i` gets a spherical separation barrier

\[
h_{c,i}=p_i^T p_i-(R_{vehicle}+R_i+d_{safe})^2.
\]

The same QP imposes the first-order exponential CBF constraints

\[
\dot h_j+\alpha_jh_j\geq0
\]

for every camera, relative-heading, tag-incidence, and collision barrier. It minimizes weighted deviation from the nominal tracking command, with command and command-slew bounds. For two detected obstacles, the QP has six geometric camera/range rows, two relative-heading rows, one tag-incidence row, and two collision rows: 11 safety inequalities in total.

The relative target translational velocity and leader angular velocity are estimated from successive timestamped pose measurements, together with the follower's measured velocity and angular rates. The camera lever arm is included in the relative-position kinematics.

## Run the simulation and tests

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_apriltag_unified_cbf.py
python run_apriltag_rov_sim.py --duration 200 --dt 0.05 --outdir results_apriltag
```

The tested regression suite contains 15 tests. It checks optical-axis conversion, tag-to-leader pose reconstruction, the tail-tag front normal, the stacked QP constraints, relative-heading and face-incidence responses, feasibility, physical AprilTag visibility, obstacle clearance, and completion of a full figure-eight period.

## Example with a real camera detector

See `example_apriltag_control_loop.py`. Replace its synthetic example inputs with the detector's `t_camera_tag` and `R_camera_tag`, and provide synchronized `actual_nu`, IMU rates, obstacle detections, stable obstacle IDs, and obstacle radii.

The example assumes the detector pose has first been transformed into body-aligned camera FRD. It is important not to pass raw optical-frame `[x_right, y_down, z_forward]` coordinates into an FRD controller without applying the calibrated frame transform.

If the tag detector reports no valid pose, do not treat a stale pose as a current observation. Enter a separate target-loss/reacquisition supervisor. A pose-based CBF cannot certify tag visibility when the tag is no longer detected and its relative pose is unavailable.

## Simulation results

The default simulation uses a 200-second 3D figure-eight leader trajectory. The controlled follower tracks the leader's tail tag. The 3D NED plot shows the leader and follower ROV hull orientations, heading arrows, the tail-mounted tag panel, camera FOV frustum snapshots, and obstacle paths.

The final tested run (`results_apriltag/simulation_metrics.csv`) produced:

| Metric | Unified AprilTag CBF | No-CBF baseline |
|---|---:|---:|
| Tail RMS tag tracking error | 0.218 m | 35.516 m |
| Final tag tracking error | 0.014 m | 39.371 m |
| Physical AprilTag loss samples | 0 | 3453 |
| Maximum absolute relative heading | 9.55 deg | 179.98 deg |
| Maximum tag incidence | 19.28 deg | 121.77 deg |
| Maximum tag forward range | 5.37 m | 13.83 m |
| Collision samples | 0 | 0 |
| QP infeasible samples | 0 | 0 |
| Minimum static-obstacle surface clearance | 0.273 m | 2.231 m |
| Minimum fleet-obstacle surface clearance | 3.764 m | 4.336 m |

The leader's figure-eight period is deliberately 200 s. This gives the follower enough time to track the tail-facing tag with the configured yaw/sway authority. A faster leader may turn its tag away faster than the follower can follow; in that case the QP can become infeasible or the tag can leave the physical image FOV. This is a real control-authority limitation, not something an optimizer can remove.

Plots:

- `results_apriltag/apriltag_rov_ned_3d_headings_fov.png`: 3D NED trajectories, both ROV headings, tail tag and camera FOV frustums.
- `results_apriltag/apriltag_rov_frd_tracking_errors.png`: tag-center tracking errors in camera FRD.
- `results_apriltag/apriltag_rov_visibility_orientation.png`: image bearing, relative heading, tag incidence and range.
- `results_apriltag/apriltag_rov_unified_cbf_diagnostics.png`: collision clearance, CBF margins and QP intervention.
- `results_apriltag/apriltag_rov_heading_alignment.png`: leader/follower heading comparison.
- `results_apriltag/leader_figure8_ne.png`: leader's North-East figure-eight path.

## Limitations and deployment notes

1. **This is geometric visibility safety, not an AprilTag detection guarantee.** Actual detection also depends on tag pixel size, camera calibration, image resolution, blur, lighting, turbidity, occlusion, and detector quality. The simulation does not model these image-level effects or pose-estimation noise.
2. **The CBF is first-order in commanded body-velocity references.** The reduced-order follower model has finite bandwidth, but the QP does not constitute a formal high-order safety guarantee under arbitrary actuator lag, saturation, or communication delay. A dynamic/HOCBF or predictive safety filter should use an identified vehicle velocity-loop model before real deployment.
3. **The tag must be calibrated on the target.** An incorrect tag-to-body transform or wrong face-normal sign invalidates the heading/incidence calculation.
4. **Pose timestamps and IDs matter.** Relative-position finite differences require synchronized pose, DVL/INS, and IMU data and reliable target/obstacle association.
5. **Target loss needs separate logic.** Once detection is lost, a stale pose cannot preserve a valid FOV or tag-incidence guarantee. Use a predictor with uncertainty bounds and a dedicated reacquisition policy.
6. **Vehicle model scope.** `bluerov2_dynamics.py` is a transparent reduced-order 4-DOF simulation model, not the official Blue Robotics/ArduSub firmware, a calibrated BlueROV2 Heavy digital twin, or hardware validation.

The implementation is intended for algorithm development and simulation. Validate the camera extrinsics, tag mount, detector convention, vehicle dynamics, actuator limits, and safety behavior in simulation/SITL and controlled test conditions before any real-vehicle operation.
