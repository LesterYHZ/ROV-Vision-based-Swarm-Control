# Vision-based ROV Swarm Control with Dynamic Topology

## AprilTag detector

`rov_tag_detector` detects every `tag36h11` tag visible in each camera frame and
publishes a `dave_robot_models/msg/AprilTagDetectionArray` message on
`/model/<robot_name>/detected_tag_pose`.

The message header identifies the camera optical frame and each entry contains:

- `id`: the AprilTag ID
- `pose`: the tag pose relative to the camera, with translation in metres

## BlueROV2 Heavy six-axis pose controller

The `bluerov_velocity_controller` executable tracks absolute `geometry_msgs/msg/PoseStamped`
targets. By default, publish to `/model/<robot_name>/reference_pose` with
`header.frame_id: ned`. Position is expressed as north, east, and down in
metres. Orientation is a quaternion describing the vehicle FRD body attitude
in the NED world frame (equivalent to Fossen roll, pitch, and yaw).

The controller converts Gazebo ENU/FLU odometry into NED/FRD, uses an outer
pose loop and inner body velocity/rate loops, and allocates surge, sway,
heave, roll, pitch, and yaw effort across the eight Heavy thrusters. It
publishes zero thrust until a target arrives; if targets become stale, it
captures and holds the current pose. With the default 0.5 second timeout,
keep publishing the target at a higher rate while it is being tracked. For
example, this requests a pose 1 metre north and 2 metres down, with zero
Fossen roll, pitch, and yaw:

```bash
ros2 topic pub -r 10 /model/bluerov_0/reference_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: ned}, pose: {position: {x: 1.0, y: 0.0, z: 2.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

Run it with:

```bash
ros2 launch dave_robot_models bluerov_velocity_controller.launch.py
```

The single- and multi-ROV spawn launches start one isolated controller per
vehicle. For example:

```text
/model/bluerov_0/reference_pose (geometry_msgs/msg/PoseStamped)
/model/bluerov_0/odometry
/model/bluerov_1/reference_pose
/model/bluerov_1/odometry
```
