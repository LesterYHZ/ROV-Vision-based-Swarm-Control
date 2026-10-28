from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_name = "bluerov2_heavy_camera"
    thruster_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            f"/model/{model_name}/joint/thruster{i}_joint/cmd_thrust"
            "@std_msgs/msg/Float64]gz.msgs.Double"
            for i in range(1, 9)
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "reference_pose_topic",
                default_value="/model/bluerov2_heavy_camera/reference_pose",
                description="NED PoseStamped target topic",
            ),
            DeclareLaunchArgument(
                "reference_velocity_topic",
                default_value="/model/bluerov2_heavy_camera/reference_velocity",
                description="FRD TwistStamped target topic",
            ),
            DeclareLaunchArgument(
                "odometry_topic",
                default_value="/model/bluerov2_heavy_camera/odometry",
                description="Gazebo odometry topic for the ROV",
            ),
            DeclareLaunchArgument(
                "max_speed",
                default_value="1.0",
                description="Maximum commanded speed in metres per second",
            ),
            DeclareLaunchArgument(
                "command_timeout",
                default_value="0.5",
                description="Seconds before a stale pose target captures the current pose",
            ),
            DeclareLaunchArgument(
                "max_angular_rate",
                default_value="0.5",
                description="Maximum commanded body angular rate in radians per second",
            ),
            Node(
                package="dave_robot_models",
                executable="bluerov_velocity_controller.py",
                name="bluerov_motion_controller",
                output="screen",
                parameters=[
                    {
                        "reference_pose_topic": LaunchConfiguration("reference_pose_topic"),
                        "reference_velocity_topic": LaunchConfiguration(
                            "reference_velocity_topic"
                        ),
                        "odometry_topic": LaunchConfiguration("odometry_topic"),
                        "max_speed": LaunchConfiguration("max_speed"),
                        "max_angular_rate": LaunchConfiguration("max_angular_rate"),
                        "command_timeout": LaunchConfiguration("command_timeout"),
                    }
                ],
            ),
            thruster_bridge,
        ]
    )
