from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_topic",
                default_value="/motion_vector",
                description="Body-frame Twist command topic (u, v, w, psi)",
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
                description="Seconds before a stale command becomes zero velocity",
            ),
            Node(
                package="dave_robot_models",
                executable="bluerov_velocity_controller.py",
                name="bluerov_velocity_controller",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "odometry_topic": LaunchConfiguration("odometry_topic"),
                        "max_speed": LaunchConfiguration("max_speed"),
                        "command_timeout": LaunchConfiguration("command_timeout"),
                    }
                ],
            ),
        ]
    )
