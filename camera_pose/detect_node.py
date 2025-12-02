#!/usr/bin/env python3

import os
import sys
import cv2
import yaml
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.time import Duration
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_share_directory
from rclpy.wait_for_message import wait_for_message
from typing import Optional, Any, Tuple, Union, List

from .util import *
from .pose_filter import *
from .marker_detector import AprilDetector

class DetectBase(Node):
	"""
		@param vis
		@type bool
		@param filter_type
		@type str
		@param filter_iters
		@type int
		@param f_ctrl
		@type float
		@param plt_id
		@type int
		@param refine_pose
		@type bool
		@param flip_outliers
		@type bool
		@param fps
		@type float
		@param cv_window
		@type bool

	"""

	FONT_THCKNS = 2
	FONT_SCALE = 0.7
	FONT_CLR =  (0,0,0)
	TXT_OFFSET = 30
	
	def __init__(self) -> None:
		
		super().__init__('camera_pose')

		# set util logger
		set_logger(self.get_logger().warning)

		self.declare_parameter('camera_ns', '')
		self.declare_parameter('image_topic', 'image_raw')
		self.declare_parameter('camera_info_topic', 'camera_info')
		
		self.declare_parameter('vis', True)
		self.declare_parameter('test', False)
		self.declare_parameter('debug', False)
		self.declare_parameter('cv_window', False)
		self.declare_parameter('refine_pose', False)
		self.declare_parameter('flip_outliers', False)
		self.declare_parameter('use_reconfigure', False)

		self.declare_parameter('fps', 30.0)
		self.declare_parameter('f_ctrl', 30.0)
		self.declare_parameter('filter_iters', 10)
		self.declare_parameter('marker_family', 'tag16h5')
		self.declare_parameter('cam_pose_marker_length', 0.010)
		self.declare_parameter('target_pose_marker_length', -1.0)

		self.declare_parameter('filter_type', 'none')  # 'none' | 'median' | 'mean'

		self.camera_ns = self.get_parameter('camera_ns').get_parameter_value().string_value.lstrip('/')
		self.camera_ns = '/' + self.camera_ns if self.camera_ns else ''
		self.image_topic = self.camera_ns + '/' + self.get_parameter('image_topic').get_parameter_value().string_value.lstrip('/')
		self.camera_info_topic = self.camera_ns + '/' + self.get_parameter('camera_info_topic').get_parameter_value().string_value.lstrip('/')

		self.vis = self.get_parameter('vis').get_parameter_value().bool_value
		self.test = self.get_parameter('test').get_parameter_value().bool_value
		self.cv_window = self.get_parameter('cv_window').get_parameter_value().bool_value and self.vis
		self.refine_pose = self.get_parameter('refine_pose').get_parameter_value().bool_value
		self.flip_outliers = self.get_parameter('flip_outliers').get_parameter_value().bool_value
		self.debug = self.get_parameter('debug').get_parameter_value().bool_value

		self.fps = self.get_parameter('fps').get_parameter_value().double_value
		self.f_loop = self.get_parameter('f_ctrl').get_parameter_value().double_value
		self.marker_family = self.get_parameter('marker_family').get_parameter_value().string_value
		self.cam_pose_marker_length = self.get_parameter('cam_pose_marker_length').get_parameter_value().double_value
		self.target_pose_marker_length = self.get_parameter('target_pose_marker_length').get_parameter_value().double_value
		self.filter_type = self.get_parameter('filter_type').get_parameter_value().string_value
		self.filter_iters = self.get_parameter('filter_iters').get_parameter_value().integer_value
		self.filter_iters = self.filter_iters if (self.filter_type != 'none' and self.filter_iters > 0) else 1
		if self.target_pose_marker_length <= 0.0:
			self.target_pose_marker_length = self.cam_pose_marker_length

		self.det = None
		self.frame_cnt = 0
		self.pwd = get_package_share_directory('camera_pose')

		# dummies
		self.rgb_info = CameraInfo()
		self.rgb_info.k = [1396.5938720703125, 0.0, 944.5514526367188, 0.0, 1395.5264892578125, 547.0949096679688, 0.0, 0.0, 1.0]
		self.rgb_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
		self.img = cv2.imread(os.path.join(self.pwd, 'images/test_img.jpg'), cv2.IMREAD_COLOR)

		# init ros
		if not self.test:
			self.img = None
			self.get_logger().info(f"Waiting for camera_info from {self.camera_info_topic}")
			(success, self.rgb_info) = wait_for_message(msg_type=CameraInfo, node=self, topic=self.camera_info_topic, time_to_wait=600) # wait 10 mins
			if success:
				self.get_logger().info(f"Camera info received. Camera height: {self.rgb_info.height}, width: {self.rgb_info.width}")
			else:
				self.get_logger().error("Failed to receive camera info. Exiting...")
				rclpy.shutdown()

		# init detector
		self.createDetector(self.cam_pose_marker_length)
			
		# init vis	
		if self.vis:
			if self.cv_window:
				cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)
				cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
				cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
			else:
				self.bridge = CvBridge()
				self.raw_image_pub = self.create_publisher(Image, 'camera_pose_raw', 10)
				self.proc_image_pub = self.create_publisher(Image, 'camera_pose_processed', 10)
				self.det_image_pub = self.create_publisher(Image, 'camera_pose_detections', 10)
				self.raw_encoding = 'rgb8' if self.test else 'bgr8'
				self.proc_encoding = 'mono8'

		# executor
		self.timer = self.create_timer(1/self.f_loop, self.run)

	def clean(self) -> None:
		# remove detector
		if self.det is not None:
			del self.det
		# close open cv windows
		if self.cv_window:
			cv2.destroyAllWindows()

	def createDetector(self, marker_length: float) -> None:
		if self.det is not None:
			del self.det

		self.det = AprilDetector(marker_length=marker_length, 
								 K=self.rgb_info.k, 
								 D=self.rgb_info.d,
								 dt=1/self.fps,
								 invert_pose=False,
								 filter_type=self.filter_type,
								 pwd=self.pwd,
								 log_info_fn=self.get_logger().info,
								 marker_family=self.marker_family,
								 )
		
	def flipOutliers(self, marker_detections: dict, tolerance: float=0.5, exclude_ids: list=[6,7,8,9], normal_type: NormalTypes=NormalTypes.XZ) -> bool:
		"""Check if all Z axes are oriented similarly and 
			  flip orientation for outliers. 
		"""
		# TODO: divide between fingers and thumb

		# exclude markers from check
		detections = {id: det for id, det in marker_detections.items() if id not in exclude_ids}
		# get ids
		marker_ids = list(detections.keys())
		# extract filtered rotations
		rotations = [getRotation(marker_det['frot'], RotTypes.EULER, RotTypes.MAT)  for marker_det in detections.values()]

		# get axis idx
		axis_idx = NORMAL_IDX_MAP[normal_type]
		# find outliers
		outliers, axis_avg = findAxisOrientOutliers(rotations, tolerance=tolerance, axis_idx=axis_idx)

		# correct outliers
		fixed = []
		for idx in outliers:
			mid = marker_ids[idx]
			self.get_logger().info(f"Marker {mid} orientation is likely flipped ...")
			# find possible PnP solutions
			num_sols, rvecs, tvecs, repr_err = cv2.solvePnPGeneric(detections[mid]['points'], 
														  			np.array(detections[mid]['corners'], dtype=np.float32), 
																	self.det.cmx, 
																	self.det.dist,
																	getRotation(rotations[idx], RotTypes.MAT, RotTypes.RVEC), 
																	detections[mid]['tvec'], 
																	flags=cv2.SOLVEPNP_IPPE_SQUARE)
			# find solution that matches the average
			for rvec, tvec in zip(rvecs, tvecs):
				# normalize rotation
				mat = getRotation(rvec.flatten(), RotTypes.RVEC, RotTypes.MAT)
				axs = mat[:, axis_idx] / np.linalg.norm(mat[:, axis_idx])
				# check angular distance to average
				if abs( np.dot(axs, axis_avg) ) > tolerance:
					# set other rot
					marker_detections[mid]['rot_mat'] = mat
					marker_detections[mid]['rvec'] = rvec.flatten()
					marker_detections[mid]['frot'] = getRotation(mat, RotTypes.MAT, RotTypes.EULER)
					# set other trans
					marker_detections[mid]['ftrans'] = tvec.flatten()
					self.get_logger().info("fixed")
					fixed.append(idx)
				
		return all([o in fixed for o in outliers])

	def refineDetection(self, detections: dict) -> None:
		"""Minimizes the projection error with respect to the rotation and the translation vectors, 
			 according to a Levenberg-Marquardt iterative minimization process.
		"""
		for id in detections.keys():
			det = detections[id]
			(tvec, rvec) = refinePose(tvec=det['ftrans'], 
						   				rvec=getRotation(det['frot'], RotTypes.EULER, RotTypes.RVEC), 
										corners=det['corners'], 
										obj_points=det['points'], 
										cmx=self.det.cmx, 
										dist=self.det.dist,
										)
			detections[id]['ftrans'] = tvec
			detections[id]['frot'] = getRotation(rvec, RotTypes.RVEC, RotTypes.EULER)
			detections[id]['rot_mat'] = getRotation(rvec, RotTypes.RVEC, RotTypes.MAT)
			detections[id]['rvec'] = rvec
	
	def preProcImage(self, vis: bool=True) -> Tuple[Union[dict, None], Union[cv2.typing.MatLike, None], Union[cv2.typing.MatLike, None], Union[cv2.typing.MatLike, None]]:
		""" Put num filter_iters images into
			fresh detection filter and get last
			detection.
		"""
		# test img
		raw_img = self.img if self.test else None
		marker_det, det_img, proc_img = None, None, None

		self.det.resetFilters()
		for i in range(self.filter_iters):
			self.frame_cnt += 1
			self.get_logger().debug("Capture frame {}".format(self.frame_cnt))

			if not self.test:
				# real image
				self.get_logger().debug("Waiting for image message {}".format(self.frame_cnt))
				(res, rgb) = wait_for_message(msg_type=Image, topic=self.image_topic, node=self, time_to_wait=600) # wait 10 mins
				self.get_logger().debug("Received image {}: {}, encoding: {}".format(self.frame_cnt, res, rgb._encoding if rgb is not None else None))
				if res:
					raw_img = self.bridge.imgmsg_to_cv2(rgb, self.raw_encoding)
				else:
					self.get_logger().error("Waiting time for image message exceeded. Shutdown ...")
					rclpy.shutdown()
		
			if raw_img is not None:
				self.get_logger().debug("Processing frame {}".format(self.frame_cnt))
				(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy(), vis=(vis if (i >= self.filter_iters-1 and self.vis) else False))
		
		self.get_logger().debug(f"Found {len(marker_det) if marker_det else 0} markers in the {'test' if self.test else 'live'} image, in frame {self.frame_cnt}. images present: det_img={det_img is not None}, proc_img={proc_img is not None}, raw_img={raw_img is not None}")

		# align rotations by consens
		if self.flip_outliers and marker_det is not None:
			if not self.flipOutliers(marker_det):
				self.get_logger().debug("No outliers were flipped")

		# improve detection
		if self.refine_pose and marker_det is not None:
			self.refineDetection(marker_det)

		return marker_det, det_img, proc_img, raw_img

	def show_images(self, det_img: Union[None, cv2.typing.MatLike], proc_img: Union[None, cv2.typing.MatLike], raw_img: Union[None, cv2.typing.MatLike], wait_key: int=1) -> None:
		if self.cv_window:
			if raw_img is not None:
				cv2.imshow('Raw', raw_img)
			if proc_img is not None:
				cv2.imshow('Processed', proc_img)
			if det_img is not None:
				cv2.imshow('Detection', det_img)
			if cv2.waitKey(wait_key) == ord("q"):
				rclpy.shutdown()
		else:
			if raw_img is not None:
				self.raw_image_pub.publish(self.bridge.cv2_to_imgmsg(raw_img, self.raw_encoding))
			if proc_img is not None:
				self.proc_image_pub.publish(self.bridge.cv2_to_imgmsg(proc_img, self.proc_encoding))
			if det_img is not None:
				self.det_image_pub.publish(self.bridge.cv2_to_imgmsg(det_img, self.raw_encoding))

	def run(self) -> None:
		try:
			(marker_det, det_img, proc_img, raw_img) = self.preProcImage()
			self.get_logger().info(f"Frame {self.frame_cnt}: Detected {len(marker_det) if marker_det else 0} markers")

			if self.vis and det_img is not None:
				# frame counter
				cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
				self.show_images(det_img, proc_img, raw_img)
			else:
				self.get_logger().debug("Visualization disabled or images not available")

		except Exception as e:
			self.get_logger().error(f"Exception in run: {e}")
			rclpy.shutdown()

	def detectionRoutine(self, arg: Any) -> Union[Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, int], dict]:
		raise NotImplementedError


class CameraPoseDetect(DetectBase):
	"""
		Detect camera world pose from marker 
		poses in static environment.

		@param err_term
		@type float
		@param cart_bound_low
		@type float
		@param cart_bound_high
		@type float
		@param fn
		@type str

	"""

	CAM_LABEL_YPOS = 20

	def __init__(self) -> None:
		
		super().__init__()
		
		self.declare_parameter('err_term', 2.0)
		self.declare_parameter('cartesian_bounds_low', 3*[-np.pi])
		self.declare_parameter('rotational_bounds_low', 3*[-np.pi])
		self.declare_parameter('cartesian_bounds_high', 3*[np.pi])
		self.declare_parameter('rotational_bounds_high', 3*[np.pi])
		self.declare_parameter('camera_marker_poses_file', '')
		self.declare_parameter('results_path', '')
		self.declare_parameter('result_images_path', '')

		# estimation params
		self.err_term = self.get_parameter('err_term').get_parameter_value().double_value
		self.camera_marker_poses_file = self.get_parameter('camera_marker_poses_file').get_parameter_value().string_value
		if not self.camera_marker_poses_file:
			self.camera_marker_poses_file = os.path.join(self.pwd, 'config/camera_marker_poses.yaml')
		# optimization bounds
		cartesian_lower_bounds = self.get_parameter('cartesian_bounds_low').get_parameter_value().double_array_value.tolist()
		cartesian_upper_bounds = self.get_parameter('cartesian_bounds_high').get_parameter_value().double_array_value.tolist()
		rotational_lower_bounds = self.get_parameter('rotational_bounds_low').get_parameter_value().double_array_value.tolist()
		rotational_upper_bounds = self.get_parameter('rotational_bounds_high').get_parameter_value().double_array_value.tolist()
		self.lower_bounds = np.array(cartesian_lower_bounds + rotational_lower_bounds, dtype=np.float32)
		self.upper_bounds = np.array(cartesian_upper_bounds + rotational_upper_bounds, dtype=np.float32)
		# results
		self.results_path = self.get_parameter('results_path').get_parameter_value().string_value
		self.result_file = os.path.join(self.results_path, 'results.yaml') if self.results_path != '' else ''
		self.img_result_path = self.get_parameter('result_images_path').get_parameter_value().string_value

		self.err = np.inf
		self.last_err = None
		self.init = False
		self.success = False
		self.cam_reprojection_errors = {}
		self.camera_pose = None
		self.inv_camera_pose = None
		self.est_camera_pose = np.zeros(6)
		self.target_poses = {}

		self.print_dbg()

		# load config file
		with open(self.camera_marker_poses_file, 'r') as fr:
			self.camera_marker_poses_config = yaml.safe_load(fr)
			# set marker and camera pose config
			self.load_pose_config()

		self.get_logger().info("Running camera_pose node")
	
	@property
	def has_result(self) -> bool:
		return self.success
	@property
	def cam_trans(self) -> Union[None, np.ndarray]:
		return self.camera_pose[:3] if self.camera_pose is not None else None
	@property
	def cam_rot_ext_xyz_euler(self) -> Union[None, np.ndarray]:
		return self.camera_pose[3:] if self.camera_pose is not None else None
	@property
	def cam_rot_ext_xyz_quat(self) -> Union[None, np.ndarray]:
		return getRotation(self.camera_pose[3:], RotTypes.EULER, RotTypes.QUAT) if self.camera_pose is not None else None
	@property
	def cam_rot_ext_xyz_mat(self) -> Union[None, np.ndarray]:
		return getRotation(self.camera_pose[3:], RotTypes.EULER, RotTypes.MAT) if self.camera_pose is not None else None
	@property
	def cam_tf_matrix(self) -> Union[None, np.ndarray]:
		return pose2Matrix(self.camera_pose[:3], self.camera_pose[3:], RotTypes.EULER) if self.camera_pose is not None else None
	@property
	def inv_cam_trans(self) -> Union[None, np.ndarray]:
		return self.inv_camera_pose[:3] if self.inv_camera_pose is not None else None
	@property
	def inv_cam_rot_ext_xyz_euler(self) -> Union[None, np.ndarray]:
		return self.inv_camera_pose[3:] if self.inv_camera_pose is not None else None
	@property
	def inv_cam_rot_ext_xyz_quat(self) -> Union[None, np.ndarray]:
		return getRotation(self.inv_camera_pose[3:], RotTypes.EULER, RotTypes.QUAT) if self.inv_camera_pose is not None else None
	@property
	def inv_cam_rot_ext_xyz_mat(self) -> Union[None, np.ndarray]:
		return getRotation(self.inv_camera_pose[3:], RotTypes.EULER, RotTypes.MAT) if self.inv_camera_pose is not None else None
	@property
	def inv_cam_tf_matrix(self) -> Union[None, np.ndarray]:
		return pose2Matrix(self.inv_camera_pose[:3], self.inv_camera_pose[3:], RotTypes.EULER) if self.inv_camera_pose is not None else None
	
	def print_dbg(self) -> None:
		dbg_msg = f"\ncamera_ns='{self.camera_ns}'" \
				  + f"\nimage_topic='{self.image_topic}'" \
				  + f"\ncamera_info_topic='{self.camera_info_topic}'" \
				  + f"\nvis={self.vis}" \
				  + f"\ntest={self.test}" \
				  + f"\ncv_window={self.cv_window}" \
				  + f"\nrefine_pose={self.refine_pose}" \
				  + f"\nflip_outliers={self.flip_outliers}" \
				  + f"\nfps={self.fps}" \
				  + f"\nf_loop={self.f_loop}" \
				  + f"\ncam_pose_marker_length={self.cam_pose_marker_length}" \
				  + f"\ntarget_pose_marker_length={self.target_pose_marker_length}" \
				  + f"\nfilter_type={self.filter_type}" \
				  + f"\nfilter_iters={self.filter_iters}" \
				  + f"\nK={self.rgb_info.k}" \
				  + f"\nD={self.rgb_info.d}" \
				  + f"\ndebug={self.debug}" \
				  + f"\nerr_term={self.err_term}" \
				  + f"\ncamera_marker_poses_file={self.camera_marker_poses_file}" \
				  + f"\nlower_bounds={self.lower_bounds}" \
				  + f"\nupper_bounds={self.upper_bounds}" \
				  + f"\nresult_file={self.result_file}" \
				  + f"\n\n"
		
		self.get_logger().info(dbg_msg)

	def load_pose_config(self) -> None:
		# check root pose
		self.root_pose = self.camera_marker_poses_config['root']
		if self.root_pose.get('xyz') is None:
			self.root_pose['xyz'] = [0.0, 0.0, 0.0]
			self.get_logger().warning("Setting root translation to zero!")
		if self.root_pose.get('rpy') is None:
			self.root_pose['rpy'] = [0.0, 0.0, 0.0]
			self.get_logger().warning("Setting root rotation to zero!")

		# check camera pose
		self.camera_pose = self.camera_marker_poses_config['camera_pose']
		if self.camera_pose.get('xyz') is None or self.camera_pose.get('rpy') is None:
			self.camera_pose = None
			self.get_logger().info("No camera pose found in configuration.")

		# check marker poses
		self.marker_poses = self.camera_marker_poses_config['camera_pose_marker']
		if self.camera_pose is None:
			assert self.marker_poses and isinstance(self.marker_poses, dict) # at least one marker pose is required for camera pose estimation
		for k, v in self.marker_poses.items():
			assert isinstance(k, int) # marker keys must be integers
			assert v.get('xyz') is not None and v.get('rpy') is not None # marker poses require entries 'xyz' and 'rpy'
		self.marker_poses_ids = list(self.marker_poses.keys())
			
		# check for target poses
		self.target_marker_poses = {}
		for target, marker_set in self.camera_marker_poses_config.items():
			# target defined
			if 'target' in target:
				assert isinstance(marker_set, dict) # a target marker set is not defined
				assert all( [isinstance(key, int) for key in marker_set.keys()] ) # dict(int: dict(str: list) | None) required
				self.get_logger().info(f"Found target marker set {target}. Performing marker pose detection and optionally transformation into target pose.")
				self.target_marker_poses[target] = marker_set
	
	def write_result(self) -> None:
		if not self.success:
			self.get_logger().warning("No result to write!")
			return
		
		result = {}

		# origin
		result['root'] = {'xyz': self.root_pose['xyz'], 'rpy': self.root_pose['rpy']}

		# camera pose
		result['camera_pose'] = {}
		result['camera_pose']['xyz'] = self.cam_trans.tolist()
		result['camera_pose']['rpy'] = self.cam_rot_ext_xyz_euler.tolist()
		result['camera_pose']['quat'] = self.cam_rot_ext_xyz_quat.tolist()
		result['camera_pose']['mat'] = self.cam_rot_ext_xyz_mat.tolist()
		result['camera_pose']['reprojection_error'] = round(to_serializable(self.err), 6)

		# target poses
		result['target_poses'] = to_serializable(self.target_poses)

		if self.result_file != '':
			with open(self.result_file, 'w') as fw:
				yaml.dump(result, fw)

	def putTextToCorner(self, 
					    id: int,
						label_lines: list, 
						img: cv2.typing.MatLike, 
						px_margin: Optional[int]=50,
						corner: Optional[str]='upleft',
						line_type: Optional[int]=cv2.LINE_AA,
						font: Optional[int]=cv2.FONT_HERSHEY_SIMPLEX,
						scale: Optional[Union[float, None]]=None,
						thckns: Optional[Union[int, None]]=None,
						color: Optional[Union[tuple, None]]=None,
						) -> None:
		
		height, width = img.shape[:2] 
		line_height = int(30 * self.FONT_SCALE)
		scale = self.FONT_SCALE if scale is None else scale
		thckns = self.FONT_THCKNS if thckns is None else thckns
		color = self.det.GREEN if color is None else color

		pos_label_lines, xs = [], []
		for i, line in enumerate(label_lines if 'up' in corner else reversed(label_lines)):
			stack_idx = i if id < 0 else id + i
			(tw, th), baseline = cv2.getTextSize(line, font, scale, thckns)
			y = px_margin + stack_idx * (line_height + baseline) if 'up' in corner else height - px_margin - stack_idx * (line_height + baseline)
			pos_label_lines.append( (line, y) )
			xs.append(width - tw - px_margin)

		x = px_margin if 'left' in corner else min(xs)
		for line, y in pos_label_lines:
			cv2.putText(img, line, (x, y), font, scale, color, thckns, line_type)

	def labelDetection(self, 
					   img: cv2.typing.MatLike, 
					   id: int, 
					   trans: np.ndarray, 
					   rot: np.ndarray, 
					   err: Optional[Union[float, None]]=None, 
					   emphasize_marker_ids: Optional[List[int]]=[],
					   ) -> None:
		
		label_lines = []
		corner = 'upleft'
		color = self.det.GREEN

		if id > -1:
			# marker pose labels
			if emphasize_marker_ids is not None and id in emphasize_marker_ids:
				repr_error = self.cam_reprojection_errors.get(id, -1.0)
				txt = "{} X: {:.3f} Y: {:.3f} Z: {:.3f} R: {:.2f} P: {:.2f} Y: {:.2f}, err {:.2f}".format(id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], repr_error)
				label_lines.append(txt)
			else:
				color = self.det.RED
				txt = "{} X: {:.3f} Y: {:.3f} Z: {:.3f} R: {:.2f} P: {:.2f} Y: {:.2f}".format(id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
				label_lines.append(txt)
		
		elif id == -1:
			# camera pose label
			corner = 'lowright'
			label_lines.append("CAMERA POSE ESTIMATE")
			label_lines.append("X {:.3f}".format(trans[0]))
			label_lines.append("Y {:.3f}".format(trans[1]))
			label_lines.append("Z {:.3f}".format(trans[2]))
			label_lines.append("R {:.3f}".format(rot[0]))
			label_lines.append("P {:.3f}".format(rot[1]))
			label_lines.append("Y {:.3f}".format(rot[2]))
			if err is not None and err is not np.inf:
				label_lines.append("mean reprojection error: {:.2f}".format(err))
		
		elif id == -2:
			# target marker pose label
			corner = 'lowright'
			label_lines.append("Target estimation")
			label_lines.append("X {:.3f} Y {:.3f} Z {:.3f}".format(trans[0], trans[1], trans[2]))
			label_lines.append("R {:.3f} P {:.3f} Y {:.3f}".format(rot[0], rot[1], rot[2]))

		self.get_logger().debug(f"Labeling image: id {id}, label:\n{label_lines}\ntrans: {trans}, rot: {rot}\nemphasized: {emphasize_marker_ids is not None and id in emphasize_marker_ids}")
		self.putTextToCorner(id, label_lines, img, corner=corner, color=color)

	def reprojectionError(self, det_corners: np.ndarray, proj_corners: np.ndarray) -> float:
		error = np.linalg.norm(det_corners - proj_corners, axis=1)
		return np.mean(error)
	
	def projectSingleMarkerCameraFrame(self, 
									   detection: dict, 
									   id: int, 
									   T_camera_marker: np.ndarray, 
									   img: Optional[Union[None, cv2.typing.MatLike]]=None, 
									   emphasize: Optional[bool]=False,
									   ) -> float:
		# compute tag corners wrt camera frame
		cam_corners = self.tagWorldCorners(T_camera_marker, self.det.square_points)

		# project to image
		projected_corners, _ = cv2.projectPoints(cam_corners, T_camera_marker[:3, :3], T_camera_marker[:3, 3], self.det.cmx, self.det.dist)
		projected_corners = np.int32(projected_corners).reshape(-1, 2)

		# draw
		if img is not None:
			cv2.polylines(img, [projected_corners], isClosed=True, color=self.det.BLUE if emphasize else self.det.RED, thickness=2 if emphasize else 1)
			cv2.putText(img, str(id), (projected_corners[0][0]+5, projected_corners[0][1]+5), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE if emphasize else self.FONT_SCALE/2, self.det.BLACK if emphasize else self.det.WHITE, self.FONT_THCKNS, cv2.LINE_AA)
		
		return self.reprojectionError(detection['corners'], projected_corners)
	
	def projectSingleMarkerWorldFrame(self, 
								   	  detection:dict, 
									  id: int, 
									  camera_pose: np.ndarray, 
									  img: Optional[Union[None, cv2.typing.MatLike]]=None, 
									  emphasize: Optional[bool]=False,
									  ) -> float:
		# tf marker corners wrt. world
		T_world_marker = self.getWorldMarkerTF(id)
		world_corners = self.tagWorldCorners(T_world_marker, self.det.square_points)
		
		# project corners to image plane
		projected_corners, _ = cv2.projectPoints(world_corners, camera_pose[:3, :3], camera_pose[:3, 3], self.det.cmx, self.det.dist)
		projected_corners = np.int32(projected_corners).reshape(-1, 2)
		
		if img is not None:
			cv2.polylines(img, [projected_corners], isClosed=True, color=self.det.BLUE if emphasize else self.det.RED, thickness=2 if emphasize else 1)
			cv2.putText(img, str(id), (projected_corners[0][0]+5, projected_corners[0][1]+5), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE if emphasize else self.FONT_SCALE/2, self.det.BLACK if emphasize else self.det.WHITE, self.FONT_THCKNS, cv2.LINE_AA)
		
		return self.reprojectionError(detection['corners'], projected_corners)
	
	def projectMarkersWorldFrame(self, detection:dict, camera_pose: np.ndarray, img: cv2.typing.MatLike=None, emphasize_marker_ids: Optional[List[int]]=[]) -> list:
		err = []
		# invert world to camera tf for reprojection
		tvec_inv, euler_inv = invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_cam_world = pose2Matrix(tvec_inv, euler_inv, RotTypes.EULER)
		# iter measured markers
		for id, det in detection.items():
			# consider only relevant ids
			if id in emphasize_marker_ids:
				# get reprojection error
				e = self.projectSingleMarkerWorldFrame(det, id, T_cam_world, img, True)
				self.cam_reprojection_errors.update({id: e})
				err.append(e)
			else:
				self.projectSingleMarkerWorldFrame(det, id, T_cam_world, img)
		return err
	
	def tagWorldCorners(self, world_tag_tf: np.ndarray, tag_corners: np.ndarray) -> np.ndarray:
		"""Transform marker corners to world frame""" 
		homog_corners = np.hstack((tag_corners, np.ones((tag_corners.shape[0], 1))))
		world_corners = world_tag_tf @ homog_corners.T
		world_corners = world_corners.T 
		return world_corners[:, :3]
	
	def getWorldMarkerTF(self, id: int) -> np.ndarray:
		# marker root tf
		root = self.marker_poses.get('root')
		T_world_root = pose2Matrix(root['xyz'], root['rpy'], RotTypes.EULER) if root is not None else np.eye(4)
		
		# marker tf
		marker = self.marker_poses.get(id)
		if marker is None:	
			self.get_logger().warning(f"id {id} not present in marker poses!")
			return T_world_root
		
		T_root_marker = pose2Matrix(marker['xyz'], marker['rpy'], RotTypes.EULER)

		# worldTmarker
		return T_world_root @ T_root_marker
	
	def camTF(self, detection: dict, id: int) -> np.ndarray:
		tf = np.zeros(6)
		det = detection.get(id)
		if det is None:
			self.get_logger().warning(f"Cannot find id {id} in detection!")
			return tf
		
		# get markerTcamera
		inv_tvec, inv_euler = invPersp(tvec=det['ftrans'], rot=det['frot'], rot_t=RotTypes.EULER)
		T_marker_cam = pose2Matrix(inv_tvec, inv_euler, RotTypes.EULER)

		# get worldTmarker
		T_world_marker = self.getWorldMarkerTF(id)

		# compute worldTcamera
		T_world_cam = T_world_marker @ T_marker_cam
		tf[:3] = T_world_cam[:3, 3]
		tf[3:] = R.from_matrix(T_world_cam[:3, :3]).as_euler('xyz')

		return tf
	
	def initialGuess(self, detection: dict) -> np.ndarray:
		# get pose for id with min detection error
		errs = [val['pose_err'] for val in detection.values()]
		min_err_idx = errs.index(min(errs))

		return self.camTF(detection, min_err_idx)

	def residuals(self, camera_pose: np.ndarray, marker_poses: dict, detection: dict) -> np.ndarray:
		"""Compute the residual (error) between world and detected poses.
			Rotations are extr. xyz euler angles.
		"""
		error = []
		# estimate
		T_world_camera = pose2Matrix(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		# invert for reprojection
		tvec_inv, euler_inv = invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_camera_world = pose2Matrix(tvec=tvec_inv, rot=euler_inv, rot_t=RotTypes.EULER)

		for id in marker_poses:
			det = detection.get(id)

			if det is not None:
				# detected tag pose wrt camera frame
				T_camera_marker = pose2Matrix(det['ftrans'], det['frot'], RotTypes.EULER)
				T_world_marker_est = T_world_camera @ T_camera_marker
				# measured tag pose wrt world 
				T_world_marker = self.getWorldMarkerTF(id)

				# errors
				position_error = np.linalg.norm(T_world_marker_est[:3, 3] - T_world_marker[:3, 3])
				orientation_error = np.linalg.norm(T_world_marker_est[:3, :3] - T_world_marker[:3, :3])
				error.append(position_error)  
				error.append(orientation_error)		

				# reprojection_error
				# repr_err = self.projectSingleMarkerWorldFrame(det, id, T_camera_world, True)
				# error.append(repr_err)
			else:
				self.get_logger().warning(f"Cannot find marker {id} in detection!")

		return np.hstack(error) if len(error) else np.array(error)

	def estimateCamPoseLS(self, img: cv2.typing.MatLike, err: float, est_camera_pose: np.ndarray, detection: dict) -> Tuple[float, np.ndarray]:
		try:
			res = least_squares(self.residuals, 
								est_camera_pose, 
								args=(self.marker_poses, detection),
								method='trf', 
								bounds=(self.lower_bounds, self.upper_bounds),
								max_nfev=5000, # max iterations
								ftol=1e-8,    # tolerance for the cost function
								xtol=1e-8,    # tolerance for the solution parameters
								gtol=1e-8     # tolerance for the gradient
								)
			
			if res.success:
				opt_cam_pose = res.x
				# reproject markers
				errors = self.projectMarkersWorldFrame(detection, opt_cam_pose, img, list(self.marker_poses.keys()))
				reserr = np.mean(errors) if len(errors) else np.inf
				self.get_logger().info(
					f"Result: {res.status} {res.message}\n"
					f"camera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\n"
					f"reprojection error: {reserr}\n"
					f"cost: {res.cost}\n"
					f"evaluations: {res.nfev}\n"
					f"optimality: {res.optimality}\n"
				)

				for id, error in self.cam_reprojection_errors.items():
					if error > self.err_term:
						self.get_logger().info(f"id {id} reprojection error: {error:.2f} > {self.err_term} threshold")

				# put pose label
				self.labelDetection(img, -1, opt_cam_pose[:3], opt_cam_pose[3:], reserr)

				return reserr, opt_cam_pose

			else:
				self.get_logger().warning(f"Least squares failed: {res.status} {res.message}")

		except Exception as e:
			msg = f"Least squares optimization failed: {e}\n:" \
					+ f"estimate: {est_camera_pose}," \
					+ f"\nmarker poses: ".join(f'\n{k}: {v}' for k, v in self.marker_poses.items()) \
					+ f"\nbounds: {(self.lower_bounds, self.upper_bounds)}"
			self.get_logger().error(msg)

		return err, est_camera_pose
	
	def estimateCamPoseFL(self, img: cv2.typing.MatLike, err: float, detection: dict) -> Tuple[float, np.ndarray]:
		filter = None
		filtered_pose = np.zeros(6)
		
		for id in detection:
			T_world_cam = self.camTF(detection, id)
			if filter is None:
				filter = createFilter(self.filter_type, PoseFilterBase.poseToMeasurement(tvec=T_world_cam[:3], rot=T_world_cam[3:], rot_t=RotTypes.EULER), self.f_loop)
			else:
				filter.updateFilter(PoseFilterBase.poseToMeasurement(tvec=T_world_cam[:3], rot=T_world_cam[3:], rot_t=RotTypes.EULER))
		
		if filter is not None:
			filtered_pose[:3] = filter.est_translation
			filtered_pose[3:] = filter.est_rotation_as_euler
			self.labelDetection(img, -1, filtered_pose[:3], filtered_pose[3:])
			err = self.projectMarkersWorldFrame(detection, filtered_pose, img)
			self.get_logger().info(f"camera world pose trans: {filtered_pose[:3]}, rot (extr. xyz euler): {filtered_pose[3:]}")
		
		return err, filtered_pose
	
	def estimateCameraPose(self, det_img: cv2.typing.MatLike, marker_det: dict) -> Tuple[bool, list]:
		if not marker_det:
			return False, []
		
		initial_guess = self.est_camera_pose if self.init else self.initialGuess(marker_det)
		self.init = True
		success = False

		if self.debug:
			msg = ""
			for id, res in marker_det.items():
				msg += f"id: {id}\n"
				for k, v in res.items():
					msg += f"{k}: {v}\n"
			self.get_logger().debug(f"Cam pose estimation input:\n{msg}")

		self.get_logger().info("Running estimation")
		(self.err, self.est_camera_pose) = self.estimateCamPoseLS(det_img, self.err, initial_guess, marker_det)

		if self.last_err is not None:
			if self.last_err <= self.err or self.err <= self.err_term:
				success = True
				self.camera_pose = self.est_camera_pose
				tvec_inv, euler_inv = invPersp(tvec=self.camera_pose[:3], rot=self.camera_pose[3:], rot_t=RotTypes.EULER)
				self.inv_camera_pose = np.append(tvec_inv, euler_inv)
				self.get_logger().info(f"Camera pose estimation terminated by criteria: {'current error >= last error' if self.last_err <= self.err else 'error < threshold'}.\nEstimated camera pose xyz (m): {self.camera_pose[:3]},\nextr. xyz Euler angles (rad): {self.camera_pose[3:]},\nmean reprojection error: {round(self.err, 3)}")
		else:
			self.get_logger().info(f"Camera pose estimation failed, mean reprojection error: {round(self.err, 3)} > threshold: {self.err_term} and error {round(self.err, 3)} < last error {round(self.last_err, 3) if self.last_err is not None else np.inf}")
			self.last_err = self.err

		return success, list(marker_det.keys())
	
	def estimateTargetPoseFL(self, target_poses: List[np.ndarray]) -> np.ndarray:
		filter = None
		filtered_pose = np.zeros(6)
		
		for pose in target_poses:
			if filter is None:
				filter = createFilter(self.filter_type, PoseFilterBase.poseToMeasurement(tvec=pose[:3], rot=pose[3:], rot_t=RotTypes.EULER).tolist(), self.f_loop)
			else:
				filter.updateFilter(PoseFilterBase.poseToMeasurement(tvec=pose[:3], rot=pose[3:], rot_t=RotTypes.EULER))
		
		if filter is not None:
			filtered_pose[:3] = filter.est_translation
			filtered_pose[3:] = filter.est_rotation_as_euler

		return filtered_pose
	
	def drawTargetPose(self, img: cv2.typing.MatLike, target_world_pose: np.ndarray) -> None:
		T_world_target = pose2Matrix(target_world_pose[:3], target_world_pose[3:], RotTypes.EULER)
		T_camera_world = self.inv_cam_tf_matrix
		assert T_camera_world is not None # the camera tf is required
		T_camera_target = T_camera_world @ T_world_target

		self.labelDetection(img, -2, target_world_pose[:3], target_world_pose[3:])
		cv2.drawFrameAxes(img, self.det.cmx, self.det.dist, getRotation(T_camera_target[:3, :3], RotTypes.MAT, RotTypes.RVEC), T_camera_target[:3, 3], self.target_pose_marker_length*self.det.AXIS_LENGTH, self.det.AXIS_THICKNESS)

	def estimateTargetPose(self, det_img: cv2.typing.MatLike, marker_det: dict) -> Tuple[bool, str, list]:
		result = {}
		target_poses = []

		# marker to target tf
		target_key = list(self.target_marker_poses.keys())[0]
		target_vals = list(self.target_marker_poses.values())[0]
		target_ids = list(target_vals.keys())
		# camera extrinsics
		T_world_camera = self.cam_tf_matrix
		assert T_world_camera is not None # the camera tf is required

		for id, pose in target_vals.items():
			result[id] = {}
			det = marker_det.get(id)

			if det is not None:
				self.get_logger().debug(f"\nComputing target pose for {id} with\ntranslation: {det['ftrans']}\nrotation: {det['frot']}")

				# compute tag pose wrt camera frame
				T_camera_marker = pose2Matrix(det['ftrans'], det['frot'], RotTypes.EULER)
				T_world_marker_est = T_world_camera @ T_camera_marker
				result[id]['marker_xyz'] = T_world_marker_est[:3, 3]
				result[id]['marker_rpy'] = getRotation(T_world_marker_est[:3, :3], RotTypes.MAT, RotTypes.EULER)

				self.get_logger().debug(f"World marker pose is\nntranslation: {result[id]['marker_xyz']}\nrotation: {result[id]['marker_rpy']}")

				# transform into target frame
				if pose is not None:
					T_marker_target = pose2Matrix(pose['xyz'], pose['rpy'], RotTypes.EULER)
					T_world_target_est = T_world_marker_est @ T_marker_target
					trans = T_world_target_est[:3, 3]
					rot = getRotation(T_world_target_est[:3, :3], RotTypes.MAT, RotTypes.EULER)
					result[id]['target_xyz'] = trans
					result[id]['target_rpy'] = rot
					target_poses.append(np.append(trans, rot))
					self.get_logger().debug(f"World target pose {id}\nntranslation: {trans}\nrotation: {rot}")
			
			else:
				self.get_logger().warning(f"Cannot find marker {id} in pose detection for target {target_key}, aborting!")
				return False, "", target_ids 
		
		# filter target pose
		filtered_pose = self.estimateTargetPoseFL(target_poses)
		self.get_logger().info(f"Filtered target pose translation: {filtered_pose[:3]}, rotation (extr. xyz euler): {filtered_pose[3:]}")
		result['filtered_target_pose'] = {'xyz': filtered_pose[:3], 'rpy': filtered_pose[3:]}
		self.target_poses[target_key] = result
		self.drawTargetPose(det_img, filtered_pose)
		# remove target
		self.get_logger().info(f"All marker poses for target {target_key} computed, removing target ...")
		self.target_marker_poses.pop(target_key)

		return True, target_key, target_ids 

	def run(self) -> None:
		try:
			res = False
			target_name = ""
			emphasize_marker_ids = []

			# detect markers 
			(marker_det, det_img, proc_img, raw_img) = self.preProcImage()
			# initially show raw and preprocessed images
			if self.vis:
				self.show_images(None, proc_img, raw_img, 1 if self.init else 10000)

			# process detection
			if marker_det is not None and det_img is not None:
				
				if self.camera_pose is None:
					# task: estimate cam pose
					self.get_logger().info("\n\nNo camera pose is present. Running camera pose estimation.", throttle_duration_sec=2)
					camera_marker_det = {k: v for k, v in marker_det.items() if k in self.marker_poses_ids} # TODO: fix this permanently
					(res, emphasize_marker_ids) = self.estimateCameraPose(det_img, camera_marker_det)
				
				elif self.target_marker_poses:
					# task: compute target poses
					self.get_logger().info("\n\nA camera pose is present. Running target transformation.", throttle_duration_sec=2)
					if self.det.marker_length != self.target_pose_marker_length:
						self.createDetector(self.target_pose_marker_length)
						self.get_logger().info("Adapted detector params to target marker length!")
					(res, target_name, emphasize_marker_ids) = self.estimateTargetPose(det_img, marker_det)
			
			# set the result if all tasks are complete
			if self.camera_pose is not None and not self.target_marker_poses:
				self.success = True
				self.write_result()

			# show detections and estimates
			if self.vis and det_img is not None:
				# put frame counter
				cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
				
				if marker_det is not None:
					# label marker pose
					for id, det in marker_det.items():
						self.labelDetection(det_img, id, det['ftrans'], det['frot'], emphasize_marker_ids=emphasize_marker_ids)
					
					# save result images
					if res and self.img_result_path != '':
						cv2.imwrite(os.path.join(self.img_result_path, f"{target_name if target_name else 'camera_pose'}_estimation.jpg"), det_img)
					
				# show result
				if self.vis:
					for _ in range(1 if self.cv_window else 10):
						self.show_images(det_img, None, None, 100000 if self.success else 10000 if res else 1)
						if not self.cv_window:
							self.get_clock().sleep_for(Duration(nanoseconds=int(0.250*1e9)))

			# terminate after showing result
			if self.success:
				self.get_logger().info("All tasks done. Terminating ...")
				self.timer.cancel()
				rclpy.shutdown()
				exit(0)

		except Exception as e:
			self.get_logger().error(f"Error occurred in run: {e}")
			self.timer.cancel()
			rclpy.shutdown()
			raise e

def main():
	rclpy.init()
	node = None

	if "camera_pose" in sys.argv:
		node = CameraPoseDetect()
	else:
		node = DetectBase()

	rclpy.spin(node)

	if node is not None:
		node.clean()
		node.destroy_node()

	if rclpy.ok():
		rclpy.shutdown()

if __name__ == '__main__':
	main()
