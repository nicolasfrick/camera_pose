#!/usr/bin/env python3

import os
import sys
import cv2
import yaml
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_share_directory
from rclpy.wait_for_message import wait_for_message
from typing import Optional, Any, Tuple, Union

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

	FONT_THCKNS = 1
	FONT_SCALE = 0.7
	FONT_CLR =  (0,0,0)
	TXT_OFFSET = 30
	
	def __init__(self) -> None:
		
		super().__init__('camera_pose')

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
		self.declare_parameter('marker_length', 0.010)
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
		self.marker_length = self.get_parameter('marker_length').get_parameter_value().double_value
		self.filter_type = self.get_parameter('filter_type').get_parameter_value().string_value
		self.filter_iters = self.get_parameter('filter_iters').get_parameter_value().integer_value
		self.filter_iters = self.filter_iters if (self.filter_type != 'none' and self.filter_iters > 0) else 1

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
			(success, self.rgb_info) = wait_for_message(msg_type=CameraInfo, node=self, topic=self.camera_info_topic, time_to_wait=25)
			if success:
				self.get_logger().info(f"Camera info received. Camera height: {self.rgb_info.height}, width: {self.rgb_info.width}")
			else:
				self.get_logger().error("Failed to receive camera info. Exiting...")
				rclpy.shutdown()

		# init detector
		self.det = AprilDetector(marker_length=self.marker_length, 
								 K=self.rgb_info.k, 
								 D=self.rgb_info.d,
								 dt=1/self.fps,
								 invert_pose=False,
								 filter_type=self.filter_type,
								 pwd=self.pwd,
								 log_info_fn=self.get_logger().info,
								 )
			
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
		if self.cv_window:
			cv2.destroyAllWindows()

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
				rgb = wait_for_message(msg_type=Image, topic=self.image_topic, node=self, time_to_wait=5)
				raw_img = self.bridge.imgmsg_to_cv2(rgb, self.raw_encoding)

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
		self.declare_parameter('camera_pose_file', '')
		self.declare_parameter('marker_poses_file', '')
	
		self.err_term = self.get_parameter('err_term').get_parameter_value().double_value
		self.camera_pose_file = self.get_parameter('camera_pose_file').get_parameter_value().string_value
		if not self.camera_pose_file:
			self.camera_pose_file = os.path.join(self.pwd, 'config/camera_pose.yaml')
		self.marker_poses_file = self.get_parameter('marker_poses_file').get_parameter_value().string_value
		if not self.marker_poses_file:
			self.marker_poses_file = os.path.join(self.pwd, 'config/marker_poses.yaml')
		
		# optimization bounds
		cartesian_lower_bounds = self.get_parameter('cartesian_bounds_low').get_parameter_value().double_array_value.tolist()
		cartesian_upper_bounds = self.get_parameter('cartesian_bounds_high').get_parameter_value().double_array_value.tolist()
		rotational_lower_bounds = self.get_parameter('rotational_bounds_low').get_parameter_value().double_array_value.tolist()
		rotational_upper_bounds = self.get_parameter('rotational_bounds_high').get_parameter_value().double_array_value.tolist()
		self.lower_bounds = np.array(cartesian_lower_bounds + rotational_lower_bounds, dtype=np.float32)
		self.upper_bounds = np.array(cartesian_upper_bounds + rotational_upper_bounds, dtype=np.float32)

		self.err = np.inf
		self.init = False
		self.success = False
		self.reprojection_errors = {}
		self.est_camera_pose = np.zeros(6)

		dbg_msg = f"camera_ns='{self.camera_ns}'" \
				  + f"\nimage_topic='{self.image_topic}'" \
				  + f"\ncamera_info_topic='{self.camera_info_topic}'" \
				  + f"\nvis={self.vis}" \
				  + f"\ntest={self.test}" \
				  + f"\ncv_window={self.cv_window}" \
				  + f"\nrefine_pose={self.refine_pose}" \
				  + f"\nflip_outliers={self.flip_outliers}" \
				  + f"\nfps={self.fps}" \
				  + f"\nf_loop={self.f_loop}" \
				  + f"\nmarker_length={self.marker_length}" \
				  + f"\nfilter_type={self.filter_type}" \
				  + f"\nfilter_iters={self.filter_iters}" \
				  + f"\nK={self.rgb_info.k}" \
				  + f"\nD={self.rgb_info.d}" \
				  + f"\ndebug={self.debug}" \
				  + f"\nerr_term={self.err_term}" \
				  + f"\ncamera_pose_file={self.camera_pose_file}" \
				  + f"\nmarker_poses_file={self.marker_poses_file}" \
				  + f"\nlower_bounds={self.lower_bounds}" \
				  + f"\nupper_bounds={self.upper_bounds}"
		
		self.get_logger().debug(dbg_msg)

		# load camera poses
		with open(self.camera_pose_file, 'r') as fr:
			self.camera_pose = yaml.safe_load(fr)

		# load marker poses
		with open(self.marker_poses_file, 'r') as fr:
			self.marker_poses = yaml.safe_load(fr)

		self.get_logger().info("Running camera_pose node")
	
	@property
	def has_result(self) -> bool:
		return self.success
	@property
	def cam_trans(self) -> Union[None, np.ndarray]:
		return self.est_camera_pose[:3] if self.success else None
	@property
	def cam_rot_ext_xyz_euler(self) -> Union[None, np.ndarray]:
		return self.est_camera_pose[3:] if self.success else None
	@property
	def cam_rot_ext_xyz_quat(self) -> Union[None, np.ndarray]:
		return getRotation(self.est_camera_pose[3:], RotTypes.EULER, RotTypes.QUAT) if self.success else None
	@property
	def cam_rot_ext_xyz_mat(self) -> Union[None, np.ndarray]:
		return getRotation(self.est_camera_pose[3:], RotTypes.EULER, RotTypes.MAT) if self.success else None

	def write_camera_pose_result(self) -> None:
		if not self.success:
			self.get_logger().warning("No valid camera pose to write!")
			return
		
		self.camera_pose['xyz'] = self.cam_trans.tolist()
		self.camera_pose['rpy'] = self.cam_rot_ext_xyz_euler.tolist()
		self.camera_pose['quat'] = self.cam_rot_ext_xyz_quat.tolist()
		self.camera_pose['mat'] = self.cam_rot_ext_xyz_mat.tolist()
		self.camera_pose['reprojection_error'] = self.err

		with open(self.camera_pose_file, 'w') as fw:
			yaml.dump(self.camera_pose, fw)

	# def labelDetection(self, img: cv2.typing.MatLike, trans: np.ndarray, rot: np.ndarray, corners: np.ndarray) -> None:
	# 		pos_txt = "X: {:.4f} Y:  {:.4f} Z:  {:.4f}".format(trans[0], trans[1], trans[2])
	# 		ori_txt = "R: {:.4f} P:  {:.4f} Y:  {:.4f}".format(rot[0], rot[1], rot[2])
	# 		x_max = int(np.max(corners[:, 0]))
	# 		y_max = int(np.max(corners[:, 1]))
	# 		y_min = int(np.min(corners[:, 1]))
	# 		x_offset = 0 if x_max <= img.shape[1]/2 else -int(len(pos_txt)*20*self.FONT_SCALE)
	# 		y_offset1 = self.TXT_OFFSET if y_max <= img.shape[0]/2 else -self.TXT_OFFSET-(y_max-y_min)
	# 		y_offset2 = y_offset1 + int(self.FONT_SCALE*50) if y_offset1 > 0 else y_offset1 - int(self.FONT_SCALE*50)
	# 		cv2.putText(img, pos_txt, (x_max+x_offset, y_max+(y_offset1 if y_offset1 > 0 else y_offset2)), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
	# 		cv2.putText(img, ori_txt, (x_max+x_offset, y_max+(y_offset2 if y_offset1 > 0 else y_offset1)), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)

	def labelDetection(self, img: cv2.typing.MatLike, id: int, trans: np.ndarray, rot: np.ndarray, err: Optional[Union[float, None]]=None) -> None:
		if id > -1:
			repr_error = self.reprojection_errors.get(id)
			if repr_error is None:
				repr_error = -1.0
			pos_txt = "{} X: {:.4f} Y: {:.4f} Z: {:.4f} R: {:.4f} P: {:.4f} Y: {:.4f}, err {:.2f}".format(id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], repr_error)
			xpos = self.TXT_OFFSET
			ypos = (id+1)*self.TXT_OFFSET
			cv2.putText(img, pos_txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.RED, self.FONT_THCKNS, cv2.LINE_AA)
		else:
			xpos = self.TXT_OFFSET
			ypos = self.CAM_LABEL_YPOS*self.TXT_OFFSET
			cv2.putText(img, "CAMERA", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, "X {:.4f}".format(trans[0]), (xpos, ypos+2*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, "Y {:.4f}".format(trans[1]), (xpos, ypos+3*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, "Z {:.4f}".format(trans[2]), (xpos, ypos+4*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, "roll {:.4f}".format(rot[0]), (xpos, ypos+5*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, "pitch {:.4f}".format(rot[1]), (xpos, ypos+6*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, "yaw {:.4f}".format(rot[2]), (xpos, ypos+7*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
			if err is not None and err is not np.inf:
				cv2.putText(img, "mean reprojection error: {:.4f}".format(err), (xpos, ypos+8*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)

	def reprojectionError(self, det_corners: np.ndarray, proj_corners: np.ndarray) -> float:
		error = np.linalg.norm(det_corners - proj_corners, axis=1)
		return np.mean(error)
	
	def projectSingleMarker(self, detection:dict, id: int, camera_pose: np.ndarray, img: cv2.typing.MatLike=None) -> float:
		if self.marker_poses.get(id) is None:
			self.get_logger().warning(f"id {id} not present in marker poses!")
			return np.inf
		# tf marker corners wrt. world
		T_world_marker = self.getWorldMarkerTF(id)
		world_corners = self.tagWorldCorners(T_world_marker, self.det.square_points)
		# project corners to image plane
		projected_corners, _ = cv2.projectPoints(world_corners, camera_pose[:3, :3], camera_pose[:3, 3], self.det.cmx, self.det.dist)
		projected_corners = np.int32(projected_corners).reshape(-1, 2)
		if img is not None:
			cv2.polylines(img, [projected_corners], isClosed=True, color=self.det.BLUE, thickness=2)
			cv2.putText(img, str(id), (projected_corners[0][0]+5, projected_corners[0][1]+5), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
		return self.reprojectionError(detection['corners'], projected_corners)
	
	def projectMarkers(self, detection:dict, camera_pose: np.ndarray, img: cv2.typing.MatLike=None) -> list:
		err = []
		# invert world to camera tf for reprojection
		tvec_inv, euler_inv = invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_cam_world = pose2Matrix(tvec_inv, euler_inv, RotTypes.EULER)
		# iter measured markers
		for id, det in detection.items():
			# reprojection error
			e = self.projectSingleMarker(det, id, T_cam_world, img)
			self.reprojection_errors.update({id: e})
			err.append(e)
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
				# repr_err = self.projectSingleMarker(det, id, T_camera_world)
				# error.append(repr_err)

		return np.hstack(error) if len(error) else np.array(error)

	def estimatePoseLS(self, img: cv2.typing.MatLike, err: float, est_camera_pose: np.ndarray, detection: dict) -> np.ndarray:
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
				errors = self.projectMarkers(detection, opt_cam_pose, img)
				reserr = np.mean(errors) if len(errors) else np.inf
				self.get_logger().info(
					f"Result: {res.status} {res.message}\n"
					f"camera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\n"
					f"reprojection error: {reserr}\n"
					f"cost: {res.cost}\n"
					f"evaluations: {res.nfev}\n"
					f"optimality: {res.optimality}\n"
				)

				for id, error in self.reprojection_errors.items():
					if error > self.err_term:
						self.get_logger().warning(f"id {id} reprojection error: {error:.2f} > {self.err_term} threshold")

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
	
	def estimatePoseFL(self, img: cv2.typing.MatLike, err: float, detection: dict) -> np.ndarray:
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
			self.labelDetection(img, 30, filtered_pose[:3], filtered_pose[3:])
			err = self.projectMarkers(detection, filtered_pose, img)
			self.get_logger().info(f"camera world pose trans: {filtered_pose[:3]}, rot (extr. xyz euler): {filtered_pose[3:]}")
		
		return err, filtered_pose
		
	def run(self) -> None:
		# try:
			# detect markers 
			(marker_det, det_img, proc_img, raw_img) = self.preProcImage()

			# initially show 
			if self.vis:
				self.show_images(None, proc_img, raw_img, 1 if self.init else 10000)

			# estimate cam pose
			if marker_det is not None and det_img is not None:
				initial_guess = self.est_camera_pose if self.init else self.initialGuess(marker_det)
				self.init = True

				self.get_logger().info("Running estimation")
				(self.err, self.est_camera_pose) = self.estimatePoseLS(det_img, self.err, initial_guess, marker_det)

				if self.err <= self.err_term:
					self.success = True
					self.write_camera_pose_result()
					self.get_logger().info(f"Pose estimation terminated.\nEstimated camera pose xyz (m): {self.est_camera_pose[:3]},\nextr. xyz Euler angles (rad): {self.est_camera_pose[3:]}, mean reprojection error: {self.err}")

			if self.vis and det_img is not None:
				# frame counter
				cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
				
				if marker_det is not None:
					# label marker pose
					for id, det in marker_det.items():
						self.labelDetection(det_img, id, det['ftrans'], det['frot'])

					self.show_images(det_img, None, None, 100000 if self.success else 1)
			
			if self.success:
				rclpy.shutdown()

		# except Exception as e:
		# 	self.get_logger().error(f"Error occurred in run: {e}")
		# 	rclpy.shutdown()

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
