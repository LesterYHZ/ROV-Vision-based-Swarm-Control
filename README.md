# Vision-based ROV Swarm Control with Dynamic Topology

## AprilTag detector

`rov_tag_detector` detects every `tag36h11` tag visible in each camera frame and
publishes a `dave_robot_models/msg/AprilTagDetectionArray` message on
`/model/<robot_name>/detected_tag_pose`.

The message header identifies the camera optical frame and each entry contains:

- `id`: the AprilTag ID
- `pose`: the tag pose relative to the camera, with translation in metres

## BlueROV2 Heavy six-axis pose controller

The `bluerov_velocity_controller` executable supports two control modes:

- Position mode uses absolute `geometry_msgs/msg/PoseStamped` targets on
  `/model/<robot_name>/reference_pose` with `header.frame_id: ned`. Position is
  expressed as north, east, and down in metres. Orientation is a quaternion
  describing the vehicle FRD body attitude in the NED world frame.
- Velocity mode uses `geometry_msgs/msg/TwistStamped` targets on
  `/model/<robot_name>/reference_velocity` with `header.frame_id: frd`.
  Linear velocity is surge, sway, and heave; angular velocity is roll, pitch,
  and yaw rate. All six values are expressed in the body FRD frame.

The most recently received valid target selects the mode. If velocity targets
become stale, the controller commands zero velocity. If position targets
become stale, it captures and holds the current pose.

In Fossen notation, the controller tracks
`eta = [x, y, z, roll, pitch, yaw]` and uses body velocities
`nu = [u, v, w, p, q, r]`. The outer pose loop generates `nu` references;
the inner loop then allocates all six control efforts to the Heavy thrusters.

The controller converts Gazebo ENU/FLU odometry into NED/FRD, uses an outer
pose loop and inner body velocity/rate loops, and allocates surge, sway,
heave, roll, pitch, and yaw effort across the eight Heavy thrusters.
Conservative simulation starting gains are provided per axis: position gains
are ordered north/east/down, attitude gains roll/pitch/yaw, and velocity and
integral gains surge/sway/heave/roll/pitch/yaw. The default speed and angular
rate limits are 1.0 m/s and 0.5 rad/s. These defaults are starting points for
the Heavy simulation model, not hardware-validated gains.

The controller publishes zero thrust until a target arrives; if targets
become stale, it captures and holds the current pose. Its integrator is frozen
when the combined thruster allocation saturates. With the default 0.5 second
timeout, keep publishing the target at a higher rate while it is being
tracked. Tune in simulation with isolated small steps on all six axes, followed
by coupled translation and attitude steps; check command direction, settling,
overshoot, and thruster saturation before using the gains on hardware. For
example, this requests a pose 1 metre north and 2 metres down, with zero
Fossen roll, pitch, and yaw:

```bash
ros2 topic pub -r 10 /model/bluerov_0/reference_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: ned}, pose: {position: {x: 1.0, y: 0.0, z: 2.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

For body-FRD velocity control, for example 0.2 m/s forward and -0.1 rad/s
yaw:

```bash
ros2 topic pub -r 10 /model/bluerov_0/reference_velocity geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: frd}, twist: {linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.1}}}"
```

Run it with:

```bash
ros2 launch dave_robot_models bluerov_velocity_controller.launch.py
```

The single- and multi-ROV spawn launches start one isolated controller per
vehicle. For example:

```text
/model/bluerov_0/reference_pose (geometry_msgs/msg/PoseStamped)
/model/bluerov_0/reference_velocity (geometry_msgs/msg/TwistStamped)
/model/bluerov_0/odometry
/model/bluerov_1/reference_pose
/model/bluerov_1/reference_velocity
/model/bluerov_1/odometry
```

The controller also supports an ArduSub/MAVROS backend. Set
`control_backend:=mavros` and keep `use_velocity_controller:=true` to stream
the same NED pose and FRD body-velocity references to ArduSub instead of
publishing Gazebo thruster commands:

```bash
ros2 launch dave_robot_models spawn_multiple_rovs.launch.py \
  num_robots:=2 use_ardusub:=true \
  use_velocity_controller:=true control_backend:=mavros
```

In this mode, pose references are published to
`/bluerov_N/mavros/setpoint_position/local` after conversion to MAVROS ENU/FLU
coordinates. Velocity references are published to
`/bluerov_N/mavros/setpoint_velocity/cmd_vel`; the MAVROS configuration uses
`BODY_NED`, and the controller converts FRD inputs to the ROS FLU input
expected by MAVROS. ArduSub must be armed and in a mode that accepts guided
velocity/position setpoints. The direct `gazebo` backend remains the default.

The multi-ROV launch can also create an ArduSub SITL and MAVROS connection for
each vehicle. Direct Gazebo thruster control remains the default; disable it
when handing control to ArduSub:

```bash
ros2 launch dave_robot_models spawn_multiple_rovs.launch.py \
  num_robots:=2 use_ardusub:=true use_velocity_controller:=false
```

`use_ardusub:=true` starts one ArduSub instance and one MAVROS node per ROV.
`use_mavros:=true` starts only the MAVROS nodes, which is useful when the
flight controllers are external. The MAVROS topics are namespaced as
`/bluerov_0/mavros`, `/bluerov_1/mavros`, and so on. The launch places each
MAVROS node under `/bluerov_N`; MAVROS adds its own `mavros` child namespace.
ArduPilot's ten-port
instance spacing is applied to the FDM, MAVLink, and GCS ports; override
`fdm_port_base` or `mavros_port_base` when those port ranges are already in
use.

## CBF collision avoidance example

The standalone `scripts/cbf_collision.py` example uses SciPy's constrained
optimizer. Install its dependencies in the Python environment used to run the
script:

```bash
python -m pip install -r requirements-cbf.txt
python scripts/cbf_collision.py
```

The requirements intentionally keep NumPy below version 2 and use SciPy 1.13,
because newer SciPy releases require the NumPy 2.x API.
