import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    # 1. 解析传入的数量和位置参数
    num_robots_str = LaunchConfiguration('num_robots').perform(context)
    start_x_str = LaunchConfiguration('start_x').perform(context)
    start_y_str = LaunchConfiguration('start_y').perform(context)
    base_z_str = LaunchConfiguration('base_z').perform(context)
    max_speed = float(LaunchConfiguration("max_speed").perform(context))
    command_timeout = float(LaunchConfiguration("command_timeout").perform(context))
    
    try:
        num_robots = int(num_robots_str)
        start_x = float(start_x_str)
        start_y = float(start_y_str)
        base_z = float(base_z_str)
    except ValueError:
        print("[ERROR] 传入的数量或坐标参数格式错误，请检查输入！")
        return []

    # 2. 定位我们在 description 目录下的原始自定义模型模板 (SDF)
    pkg_share = get_package_share_directory('dave_robot_models')
    template_sdf_path = os.path.join(pkg_share, 'description', 'bluerov2_heavy_camera', 'model.sdf')
    
    if not os.path.exists(template_sdf_path):
        raise FileNotFoundError(f"[ERROR] 找不到原始的 SDF 模板文件: {template_sdf_path}")

    with open(template_sdf_path, 'r') as f:
        sdf_content = f.read()

    launch_entities = []

    # 3. 核心循环：独立、并行生成每台车辆
    for i in range(num_robots):
        unique_robot_name = f"bluerov_{i}"
        
        # 动态错开摆放位置，避免它们重叠产生剧烈的物理碰撞
        x_pos = str(start_x - (i * 1.0))
        y_pos = str(start_y + (i * 0.5)) 
        z_pos = str(base_z)

        # 动态替换每辆车自己的专属后置 AprilTag 纹理 ID
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
            f"tag36h11_{i}.png",
        )
        if not os.path.isfile(tag_texture_path):
            raise FileNotFoundError(
                f"[ERROR] AprilTag texture not found for {unique_robot_name}: "
                f"{tag_texture_path}"
            )

        car_sdf_content = sdf_content.replace(
            target_placeholder_str,
            tag_texture_path,
        )
        car_sdf_content = car_sdf_content.replace(
            "<topic>front_camera/image_raw</topic>",
            f"<topic>/model/{unique_robot_name}/front_camera/image_raw</topic>",
        )

        # The Heavy SDF contains model-scoped actuator topics. Rewrite both the
        # Gazebo plugin namespaces and ArduPilot command topics per vehicle.
        car_sdf_content = car_sdf_content.replace(
            "<model name=\"bluerov2_heavy_camera\">",
            f"<model name=\"{unique_robot_name}\">",
        )
        car_sdf_content = car_sdf_content.replace(
            "<namespace>bluerov2_heavy_camera</namespace>",
            f"<namespace>{unique_robot_name}</namespace>",
        )
        car_sdf_content = car_sdf_content.replace(
            "/model/bluerov2_heavy_camera/",
            f"/model/{unique_robot_name}/",
        )

        # 存为系统级的临时隐藏专属模型文件
        tmp_dir = tempfile.gettempdir()
        temp_sdf_path = os.path.join(tmp_dir, f"tmp_model_{unique_robot_name}.sdf")
        with open(temp_sdf_path, 'w') as f:
            f.write(car_sdf_content)

        print(f"[INFO] 部署就绪: {unique_robot_name}，分配 AprilTag ID={i}，位置=({x_pos}, {y_pos}, {z_pos})")

        # 4. 召唤底层官方生成节点（Spawn Node）
        spawn_node = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', unique_robot_name, 
                '-file', temp_sdf_path,     
                '-x', x_pos,
                '-y', y_pos,
                '-z', z_pos
            ],
            output='screen'
        )
        launch_entities.append(spawn_node)

        # 5. Bridge the camera and all eight Heavy thruster command topics.
        # The SDF gives each camera an explicit model-scoped Gazebo topic.
        gz_absolute_topic = f"/model/{unique_robot_name}/front_camera/image_raw"
        ros_absolute_topic = f"/model/{unique_robot_name}/image_raw"

        bridge_arguments = [
            f"{gz_absolute_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            (
                f"/model/{unique_robot_name}/odometry@"
                "nav_msgs/msg/Odometry[gz.msgs.Odometry"
            ),
        ]
        bridge_arguments.extend(
            f"/model/{unique_robot_name}/joint/thruster{i}_joint/cmd_thrust"
            "@std_msgs/msg/Float64]gz.msgs.Double"
            for i in range(1, 9)
        )

        camera_bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=bridge_arguments,
            remappings=[
                (gz_absolute_topic, ros_absolute_topic)
            ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        )

        # Start the bridge after the model has been inserted into Gazebo.
        # 这确保了当桥接器去 Gazebo Transport 拓扑网络中握手时，多辆潜水器的相机硬件早已全部在水下初始化并上线完毕！
        delayed_bridge = TimerAction(
            period=3.0,
            actions=[camera_bridge_node]
        )
        
        launch_entities.append(delayed_bridge)

        velocity_controller_node = Node(
            package="dave_robot_models",
            executable="bluerov_velocity_controller.py",
            name=f"{unique_robot_name}_motion_controller",
            output="screen",
            parameters=[{
                "reference_pose_topic": f"/model/{unique_robot_name}/reference_pose",
                "odometry_topic": f"/model/{unique_robot_name}/odometry",
                "max_speed": max_speed,
                "command_timeout": command_timeout,
            }],
        )
        launch_entities.append(velocity_controller_node)

    return launch_entities


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_robots", default_value="2", description="想要生成的 BlueROV2 机器人总数量"),
        DeclareLaunchArgument("start_x", default_value="0.0", description="第一辆车的初始 X 坐标"),
        DeclareLaunchArgument("start_y", default_value="0.0", description="第一辆车的初始 Y 坐标"),
        DeclareLaunchArgument("base_z", default_value="-2.0", description="所有潜水器的统一下潜深度"),
        DeclareLaunchArgument("max_speed", default_value="1.0", description="Maximum velocity command in m/s"),
        DeclareLaunchArgument("command_timeout", default_value="0.5", description="Seconds before a stale pose target captures the current pose"),

        OpaqueFunction(function=launch_setup)
    ])
