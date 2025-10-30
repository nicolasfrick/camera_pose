################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.0.2)
project(camera_pose VERSION 0.0.1)

add_compile_options(-std=c++11)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS
    std_msgs
    sensor_msgs
    geometry_msgs
    dynamic_reconfigure
    cmake_modules
    cv_bridge
    image_transport
    camera_info_manager
    realsense2_camera
)

find_package(OpenCV)

################################################################################
# Setup for python modules and scripts
################################################################################

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
## Declare ROS dynamic reconfigure parameters
################################################################################
# Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/DetectorConfig.cfg
)

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS  
                  std_msgs
                  sensor_msgs
                  cmake_modules
                  cv_bridge
                  image_transport
                  camera_info_manager
                  realsense2_camera
                  geometry_msgs
                  dynamic_reconfigure
)

################################################################################
# Build
################################################################################
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME} SHARED src/exec.cpp)
# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

################################################################################
# Install
################################################################################
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################
