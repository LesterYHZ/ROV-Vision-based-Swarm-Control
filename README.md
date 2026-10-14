# Vision-based ROV Swarm Control with Dynamic Topology

## AprilTag detector

`rov_tag_detector` detects every `tag36h11` tag visible in each camera frame and
publishes a `dave_robot_models/msg/AprilTagDetectionArray` message on
`/model/<robot_name>/detected_tag_pose`.

The message header identifies the camera optical frame and each entry contains:

- `id`: the AprilTag ID
- `pose`: the tag pose relative to the camera, with translation in metres
