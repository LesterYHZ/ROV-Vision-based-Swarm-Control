cmake_minimum_required(VERSION 3.8)
project(dave_robot_models)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AprilTagDetection.msg"
  "msg/AprilTagDetectionArray.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

install(
  PROGRAMS
    scripts/ardusub_manual_control.py
    scripts/keyboard_publisher.py
    scripts/ws_to_joy.py
    scripts/rov_tag_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY description launch meshes config
  DESTINATION share/dave_robot_models
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_export_dependencies(rosidl_default_runtime)
ament_package()
