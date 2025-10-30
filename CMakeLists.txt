cmake_minimum_required(VERSION 3.5)
project(camera_pose)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(camera_pose)

install(PROGRAMS
  camera_pose/detect_node.py
  DESTINATION lib/camera_pose
)

install(DIRECTORY launch
  DESTINATION share/camera_pose
)

ament_package()
