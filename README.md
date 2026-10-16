# Vision-based ROV Swarm Control with Dynamic Topology

## AprilTag detector

`rov_tag_detector` detects every `tag36h11` tag visible in each camera frame and
publishes a `dave_robot_models/msg/AprilTagDetectionArray` message on
`/model/<robot_name>/detected_tag_pose`.

The message header identifies the camera optical frame and each entry contains:

- `id`: the AprilTag ID
- `pose`: the tag pose relative to the camera, with translation in metres

## BlueROV independent velocity controller

The `bluerov_velocity_controller` node accepts a `geometry_msgs/msg/Twist`
on `/motion_vector`. The command is interpreted in the ROV body frame:

- `linear.x`: `u`, forward velocity in metres per second
- `linear.y`: `v`, lateral velocity in metres per second
- `linear.z`: `w`, vertical velocity in metres per second
- `angular.z`: absolute heading `psi` in radians

It uses Gazebo odometry feedback to calculate body-frame velocity error and
heading error, then directly publishes normalized
`std_msgs/msg/Float64` commands to the eight thruster topics. The command
magnitude is limited by `max_speed`, and a stale command becomes zero thrust
after `command_timeout` seconds. The controller does not require ArduSub or
MAVROS.

Run it with:

```bash
ros2 launch dave_robot_models bluerov_velocity_controller.launch.py
```

The single- and multi-ROV spawn launches start one isolated controller per
vehicle. For example:

```text
/model/bluerov_0/motion_vector (geometry_msgs/msg/Twist)
/model/bluerov_0/odometry
/model/bluerov_1/motion_vector
/model/bluerov_1/odometry
```
