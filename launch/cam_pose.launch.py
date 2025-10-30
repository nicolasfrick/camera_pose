#!/usr/bin/env python3

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
	start_setup = []

	use_markers_camera = LaunchConfiguration('use_markers_camera', default='true')
	
	start_setup.append( 
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource([
				get_package_share_directory('realsense2_camera'),
				'/launch/rs_launch.py'
			]),
			condition=IfCondition(use_markers_camera),
            launch_arguments={'camera': LaunchConfiguration('camera'),
							  'serial_no': LaunchConfiguration('serial_no'),
							  'publish_tf': LaunchConfiguration('publish_tf'),
							  'enable_color': LaunchConfiguration('enable_color'),
							  'enable_depth': LaunchConfiguration('enable_depth'),
							  'enable_rgbd': LaunchConfiguration('enable_rgbd'),
							  'enable_infra': LaunchConfiguration('enable_infra'),
							  'enable_gyro': LaunchConfiguration('enable_gyro'),
							  'enable_sync': LaunchConfiguration('enable_sync'),
							  'pointcloud.enable': LaunchConfiguration('enable_pointcloud'),
							  'align_depth.enable': LaunchConfiguration('align_depth'),
							  'depth_module.depth_profile': f"{LaunchConfiguration('depth_width')}x{LaunchConfiguration('depth_height')}x{LaunchConfiguration('depth_fps')}",
							  'depth_module.inter_cam_sync_mode': LaunchConfiguration('depth_inter_cam_sync_mode'),
							  'rgb_camera.color_profile': f"{LaunchConfiguration('color_width')}x{LaunchConfiguration('color_height')}x{LaunchConfiguration('color_fps')}",
							  'depth_module.enable_auto_exposure': LaunchConfiguration('enable_depth_auto_exposure'),
							  'rgb_camera.enable_auto_exposure': LaunchConfiguration('enable_color_auto_exposure'),
							  'filters': LaunchConfiguration('filters'),
							  'clip_distance': LaunchConfiguration('clip_distance'),
            				}.items()
		)
	)

	start_setup.append(
		Node(
			package='camera_pose',
			executable='camera_pose_node',
			name='camera_pose',
			namespace='',
			parameters=[{
				'markers_camera_name': LaunchConfiguration('markers_camera_name'),
				'marker_poses_file': LaunchConfiguration('marker_poses_file'),
				'use_reconfigure': LaunchConfiguration('use_reconfigure'),
				'marker_length': LaunchConfiguration('marker_length'),
				'vis': LaunchConfiguration('vis'),
				'filter': LaunchConfiguration('filter'),
				'filter_iters': LaunchConfiguration('filter_iters'),
				'f_ctrl': LaunchConfiguration('f_ctrl'),
				'debug': LaunchConfiguration('debug'),
				'fps': LaunchConfiguration('fps'),
				'err_term': LaunchConfiguration('err_term'),
				'camera_pose': 'true',
			}],
			output='screen',
		),
	)

	return start_setup
		
def generate_launch_description():

	return LaunchDescription([
			DeclareLaunchArgument(
				"use_markers_camera",
				default_value="true",
				description=''
			),
			DeclareLaunchArgument(
				"markers_camera_name",
				default_value="markers_camera",
				description=''
			),
			DeclareLaunchArgument(
				"fps",
				default_value="30",
				description=''
			),
			DeclareLaunchArgument(
				"f_ctrl",
				default_value="5",
				description=''
			) ,
			DeclareLaunchArgument(
				"debug",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"marker_poses_file",
				default_value="marker_holder_poses.yml",
				description=''
			) ,
			DeclareLaunchArgument(
				"err_term",
				default_value="2.0",
				description=''
			),
			# detector params
			DeclareLaunchArgument(
				"marker_length",
				default_value="0.015",
				description=''
			),
			DeclareLaunchArgument(
				"use_reconfigure",
				default_value="false",
				description=''
			),
			DeclareLaunchArgument(
				"filter",
				default_value="none",
				description="Filter marker pose detections.",
				choices=['none', 'mean', 'median', 'kalman_simple', 'kalman'],
			),
			DeclareLaunchArgument(
				"filter_iters",
				default_value="5",
				description=''
			),
			DeclareLaunchArgument(
				"vis",
				default_value="true",
				description=''
			),
			# rs params
			DeclareLaunchArgument(
				"serial_no",
				default_value="",
				description=''
			),
			DeclareLaunchArgument(
				"enable_depth",
				default_value="false",
				description=''
			),
			DeclareLaunchArgument(
				"depth_width",
				default_value="1280",
				description=''
			),
			DeclareLaunchArgument(
				"depth_height",
				default_value="720",
				description=''
			) , 
			DeclareLaunchArgument(
				"depth_fps",
				default_value="30",
				description=''
			),
			DeclareLaunchArgument(
				"enable_color",
				default_value="true",
				description=''
			),
			DeclareLaunchArgument(
				"color_width",
				default_value="1920",
				description=''
			),
			DeclareLaunchArgument(
				"color_height",
				default_value="1080",
				description=''
			),
			DeclareLaunchArgument(
				"color_fps",
				default_value="30",
				description=''
			),
			DeclareLaunchArgument(
				"filters",
				default_value="colorizer",
				description=''
			),
			DeclareLaunchArgument(
				"clip_distance",
				default_value="-2",
				description=''
			),
			DeclareLaunchArgument(
				"align_depth",
				default_value="true",
				description=''
			),
			DeclareLaunchArgument(
				"output",
				default_value="log",
				description=''
			),
			DeclareLaunchArgument(
				'enable_rgbd',
				default_value="false",
				description=''),
			DeclareLaunchArgument(
				'enable_sync',
				default_value="false",
				description=''),
			DeclareLaunchArgument(
				'enable_pointcloud',
				default_value="false",
				description=''),
			DeclareLaunchArgument(
				'enable_infra',
				default_value="false",
				description=''),
			DeclareLaunchArgument(
				'enable_gyro',
				default_value="false",
				description=''),
			DeclareLaunchArgument(
				'publish_tf',
				default_value="true",
				description=''),
			DeclareLaunchArgument(
				'enable_depth_auto_exposure',
				default_value="true",
				description=''),
			DeclareLaunchArgument(
				'enable_color_auto_exposure',
				default_value="true",
				description=''),
			DeclareLaunchArgument(
				'depth_inter_cam_sync_mode',
				default_value="0",
				description=''),
			OpaqueFunction(function=launch_setup),
		])
