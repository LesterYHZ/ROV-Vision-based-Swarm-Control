import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    # 1. 解析所有的 Launch 传入参数
    tag_id = LaunchConfiguration('apriltag_id').perform(context)
    namespace = LaunchConfiguration('namespace').perform(context)
    x = LaunchConfiguration('x').perform(context)
    y = LaunchConfiguration('y').perform(context)
    z = LaunchConfiguration('z').perform(context)
    roll = LaunchConfiguration('roll').perform(context)
    pitch = LaunchConfiguration('pitch').perform(context)
    yaw = LaunchConfiguration('yaw').perform(context)

    # 2. 定位我们在 description 目录下的原始自定义模型模板 (SDF)
    pkg_share = get_package_share_directory('dave_robot_models')
    template_sdf_path = os.path.join(pkg_share, 'description', 'bluerov2_heavy_camera', 'model.sdf')
    
    if not os.path.exists(template_sdf_path):
        raise FileNotFoundError(f"[ERROR] 找不到原始的 SDF 模板文件: {template_sdf_path}")

    # 3. 顺应系统的 description 路径规则，构建动态 AprilTag 图片路径
    # 这将确保路径能被系统的环境变量原生解析，彻底杜绝黑块现象
    with open(template_sdf_path, 'r') as f:
        sdf_content = f.read()

    target_placeholder_str = (
        "model://description/bluerov2_heavy_camera/materials/"
        "textures/tag36h11_placeholder.png"
    )
    tag_texture_path = os.path.join(
        pkg_share,
        "description",
        "bluerov2_heavy_camera",
        "materials",
        "textures",
        f"tag36h11_{tag_id}.png",
    )
    if not os.path.isfile(tag_texture_path):
        raise FileNotFoundError(
            f"[ERROR] AprilTag texture not found for {namespace}: "
            f"{tag_texture_path}"
        )

    modified_sdf_content = sdf_content.replace(
        target_placeholder_str,
        tag_texture_path,
    )
    modified_sdf_content = modified_sdf_content.replace(
        "<topic>front_camera/image_raw</topic>",
        f"<topic>/model/{namespace}/front_camera/image_raw</topic>",
    )

    # 4. 在系统临时目录生成实例专属的隐藏临时加载 SDF 文件
    tmp_dir = tempfile.gettempdir()
    temp_file_name = f"tmp_{namespace if namespace else 'bluerov2_heavy_camera'}.sdf"
    temp_sdf_path = os.path.join(tmp_dir, temp_file_name)
    
    with open(temp_sdf_path, 'w') as f:
        f.write(modified_sdf_content)

    print(f"[INFO] 已动态生成包含 AprilTag ID={tag_id} 路径的模型加载文件: {temp_sdf_path}")

    # 5. 调用通用加载脚本，将包含 Camera 和 Tag 的模型打入 Gazebo 仿真空间
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("dave_robot_models"),
                        "launch",
                        "upload_robot.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "namespace": namespace,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "use_ned_frame": LaunchConfiguration("use_ned_frame"),
            "use_teleop": LaunchConfiguration("use_teleop"),
            "use_web_joystick": LaunchConfiguration("use_web_joystick"),
            "joystick_ws_host": LaunchConfiguration("joystick_ws_host"),
            "joystick_ws_port": LaunchConfiguration("joystick_ws_port"),
            "zoom_camera_delay": LaunchConfiguration("zoom_camera_delay"),
            "open_qgc": LaunchConfiguration("open_qgc"),
            "open_virtual_joystick": LaunchConfiguration("open_virtual_joystick"),
            "virtual_joystick_url": LaunchConfiguration("virtual_joystick_url"),
            "ui_launch_delay": LaunchConfiguration("ui_launch_delay"),
            "sdf_file": temp_sdf_path, # 强行注入修改过纹理的临时模型文件
        }.items(),
    )

    # 6. 【核心新增】启动 ros_gz_bridge 节点，将 Gazebo 内部的相机视频流无缝引入 ROS 2
    # 规则：<Gazebo内置Topic>@<ROS2消息类型>[<Gazebo消息类型>
    gz_topic = f"/model/{namespace}/front_camera/image_raw"
    ros_topic = f"/model/{namespace}/image_raw"
    
    bridge_argument = f"{gz_topic}@sensor_msgs/msg/Image[gz.msgs.Image"
    
    camera_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[bridge_argument],
        remappings=[
            (gz_topic, ros_topic)
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    

    # 返回加载项，确保生成机器人的同时，相机数据通道也被立刻打通
    return [robot_launch, camera_bridge_node]


def generate_launch_description():
    return LaunchDescription([
        # 自定义可编辑参数
        DeclareLaunchArgument("apriltag_id", default_value="0", description="后置 AprilTag 的数字 ID"),

        # 完美对齐并支持 Project DAVE 的所有基础环境变量
        DeclareLaunchArgument("paused", default_value="true", description="Start the simulation paused"),
        DeclareLaunchArgument("gui", default_value="true", description="Flag to enable the gazebo gui"),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Flag to use simulation time"),
        DeclareLaunchArgument("debug", default_value="false", description="Flag to enable gazebo debug"),
        DeclareLaunchArgument("headless", default_value="false", description="Flag to enable gazebo headless mode"),
        DeclareLaunchArgument("verbose", default_value="0", description="Adjust level of console verbosity"),
        DeclareLaunchArgument("world_name", default_value="camera_tutorial", description="Gazebo world file to launch"),
        DeclareLaunchArgument("namespace", default_value="bluerov2_heavy_camera", description="Robot namespace"),
        DeclareLaunchArgument("x", default_value="0.0", description="Initial x position"),
        DeclareLaunchArgument("y", default_value="0.0", description="Initial y position"),
        DeclareLaunchArgument("z", default_value="-1.0", description="Initial z position"),
        DeclareLaunchArgument("roll", default_value="0.0", description="Initial roll angle"),
        DeclareLaunchArgument("pitch", default_value="0.0", description="Initial pitch angle"),
        DeclareLaunchArgument("yaw", default_value="0.0", description="Initial yaw angle"),
        DeclareLaunchArgument("use_ned_frame", default_value="false", description="Use NED frame"),
        DeclareLaunchArgument("use_teleop", default_value="true", description="Launch teleop bridge"),
        DeclareLaunchArgument("use_web_joystick", default_value="true", description="Launch websocket joystick bridge"),
        DeclareLaunchArgument("joystick_ws_host", default_value="0.0.0.0", description="Bind host for websocket"),
        DeclareLaunchArgument("joystick_ws_port", default_value="8765", description="Bind port for websocket"),
        DeclareLaunchArgument("open_qgc", default_value="false", description="Launch QGroundControl"),
        DeclareLaunchArgument("open_virtual_joystick", default_value="false", description="Open virtual joystick in Firefox"),
        DeclareLaunchArgument("virtual_joystick_url", default_value="https://github.io", description="URL for virtual joystick"),
        DeclareLaunchArgument("ui_launch_delay", default_value="2.0", description="Delay before launching UI"),
        DeclareLaunchArgument("zoom_camera_delay", default_value="2.0", description="Delay for zoom camera"),

        OpaqueFunction(function=launch_setup)
    ])
