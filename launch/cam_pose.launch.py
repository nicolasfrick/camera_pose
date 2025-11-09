#!/usr/bin/env python3

import numpy as np

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
	start_setup = []

	vis = LaunchConfiguration('vis').perform(context).lower() == 'true'
	debug = LaunchConfiguration('debug').perform(context).lower() == 'true'
	cv_window = LaunchConfiguration('cv_window').perform(context).lower() == 'true'
	debug_vis = LaunchConfiguration('debug_vis').perform(context).lower() == 'true'
	
	bagfile = LaunchConfiguration('bagfile', default='').perform(context)
	markers_camera_name = LaunchConfiguration('markers_camera_name', default='').perform(context)
	image_topic = LaunchConfiguration('image_topic').perform(context) if not markers_camera_name and not bagfile else f'/{markers_camera_name if markers_camera_name else "camera"}/camera/color/image_raw'
	camera_info_topic = LaunchConfiguration('camera_info_topic').perform(context) if not markers_camera_name and not bagfile else f'/{markers_camera_name if markers_camera_name else "camera"}/camera/color/camera_info'

	# realsense
	if markers_camera_name != "" and bagfile == "":
		depth_profile = f"{LaunchConfiguration('depth_width').perform(context)}x{LaunchConfiguration('depth_height').perform(context)}x{LaunchConfiguration('depth_fps').perform(context)}"
		color_profile = f"{LaunchConfiguration('color_width').perform(context)}x{LaunchConfiguration('color_height').perform(context)}x{LaunchConfiguration('color_fps').perform(context)}"
		
		start_setup.append( 
			IncludeLaunchDescription(
				PythonLaunchDescriptionSource([
					get_package_share_directory('realsense2_camera'),
					'/launch/rs_launch.py'
				]),
				launch_arguments={'camera': markers_camera_name,
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
								'depth_module.depth_profile': depth_profile,
								'depth_module.inter_cam_sync_mode': LaunchConfiguration('depth_inter_cam_sync_mode'),
								'rgb_camera.color_profile': color_profile,
								'depth_module.enable_auto_exposure': LaunchConfiguration('enable_depth_auto_exposure'),
								'rgb_camera.enable_auto_exposure': LaunchConfiguration('enable_color_auto_exposure'),
								'filters': LaunchConfiguration('filters'),
								'clip_distance': LaunchConfiguration('clip_distance'),
								}.items()
			)
		)
	
	# rosbag
	elif bagfile != "":
		cmd = ['ros2', 'bag', 'play', f'{bagfile}', '--read-ahead-queue-size', '1000', '--rate', '1.0']
		start_setup.append(
			ExecuteProcess(
				cmd=cmd,
				output='screen',
			),
		)

	# viewer
	if vis and not cv_window:
		start_setup.append(
			Node(
				package='rqt_image_view',
				executable='rqt_image_view',
				name='rqt_image_view_camera_pose',
				namespace='',
				arguments=[image_topic],
				output='screen',
			),
		)

	# camera pose node
	start_setup.append(
		Node(
			package='camera_pose',
			executable='camera_pose_node',
			name='camera_pose',
			namespace='',
			parameters=[{
				'camera_ns': LaunchConfiguration('camera_ns'),
				'image_topic': image_topic,
				'camera_info_topic': camera_info_topic,
				'marker_poses_file': LaunchConfiguration('marker_poses_file'),
				'camera_pose_file': LaunchConfiguration('camera_pose_file'),
				'use_reconfigure': LaunchConfiguration('use_reconfigure'),
				'marker_length': LaunchConfiguration('marker_length'),
				'vis': LaunchConfiguration('vis'),
				'test': LaunchConfiguration('test'),
				'cv_window': LaunchConfiguration('cv_window'),
				'refine_pose': LaunchConfiguration('refine_pose'),
				'flip_outliers': LaunchConfiguration('flip_outliers'),
				'filter_type': LaunchConfiguration('filter_type'),
				'filter_iters': LaunchConfiguration('filter_iters'),
				'f_ctrl': LaunchConfiguration('f_ctrl'),
				'debug': debug,
				'fps': LaunchConfiguration('fps'),
				'err_term': LaunchConfiguration('err_term'),
				'cartesian_bounds_low': LaunchConfiguration('cartesian_bounds_low',),
				'rotational_bounds_low': LaunchConfiguration('rotational_bounds_low',),
				'cartesian_bounds_high': LaunchConfiguration('cartesian_bounds_high',),
				'rotational_bounds_high': LaunchConfiguration('rotational_bounds_high',),
			}],
			arguments=['camera_pose' if not debug_vis else ''],
			ros_arguments=['--log-level', 'debug' if debug else 'info'],
			output='screen',
		),
	)

	return start_setup
		

def generate_launch_description():

	return LaunchDescription([
			DeclareLaunchArgument(
				"markers_camera_name",
				default_value="",
				description=''
			),
			DeclareLaunchArgument(
				"image_topic",
				default_value="image_raw",
				description=''
			),
			DeclareLaunchArgument(
				"camera_info_topic",
				default_value="camera_info",
				description=''
			),
			DeclareLaunchArgument(
				"camera_ns",
				default_value="",
				description=''
			),
			DeclareLaunchArgument(
				"fps",
				default_value="30.0",
				description=''
			),
			DeclareLaunchArgument(
				"f_ctrl",
				default_value="10.0",
				description=''
			) ,
			DeclareLaunchArgument(
				"debug",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"debug_vis",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"test",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"cv_window",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"refine_pose",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"flip_outliers",
				default_value="false",
				description=''
			), 
			DeclareLaunchArgument(
				"marker_poses_file",
				default_value="",
				description=''
			) ,
			DeclareLaunchArgument(
				"camera_pose_file",
				default_value="",
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
				'cartesian_bounds_low',
				default_value=f"{3*[-np.pi]}",
				description=''
				),
			DeclareLaunchArgument(
				'rotational_bounds_low',
				default_value=f"{3*[-np.pi]}",
				description=''
				),
			DeclareLaunchArgument(
				'cartesian_bounds_high',
				default_value=f"{3*[np.pi]}",
				description=''
				),
			DeclareLaunchArgument(
				'rotational_bounds_high',
				default_value=f"{3*[np.pi]}",
				description=''
				),
			DeclareLaunchArgument(
				"use_reconfigure",
				default_value="false",
				description=''
			),
			DeclareLaunchArgument(
				"filter_type",
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
			DeclareLaunchArgument(
				"bagfile",
				default_value="",
				description='Path to a rosbag to playback.'
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
				default_value="30.0",
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
				default_value="30.0",
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
