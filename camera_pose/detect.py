#!/usr/bin/env python

import os
import cv2
import yaml
import rospy
import cv_bridge
import numpy as np
import sensor_msgs.msg
import dynamic_reconfigure.server

from typing import Optional, Any, Tuple, Union
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from sensor_msgs.msg import Image

from .util import *
from .pose_filter import *
from .marker_detector import AprilDetector

from camera_pose.cfg import DetectorConfig


class DetectBase():
	"""
		@param camera_ns Camera namespace preceding 'image_raw' and 'camera_info'
		@type str
		@param vis Show detection images
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
		@param use_tags
		@type bool

	"""

	FONT_THCKNS = 1
	FONT_SCALE = 0.7
	FONT_CLR =  (0,0,0)
	TXT_OFFSET = 30
	
	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis: Optional[bool]=True,
				cv_window:Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				filter_iters: Optional[int]=10,
				f_ctrl: Optional[int]=30,
				test: Optional[bool]=False,
				refine_pose: Optional[bool]=False,
				flip_outliers: Optional[bool]=False,
				fps: Optional[float]=30.0,
				use_tags: Optional[bool]=True,
				) -> None:
		
		self.vis = vis
		self.cv_window = cv_window and vis
		self.test = test
		self.f_loop = f_ctrl
		self.filter_type = filter_type
		self.refine_pose = refine_pose
		self.flip_outliers = flip_outliers
		self.filter_iters = filter_iters if (filter_type != 'none' and filter_iters > 0) else 1
		self.frame_cnt = 0

		# dummies
		self.rgb_info= sensor_msgs.msg.CameraInfo()
		self.rgb_info.K = np.array([1396.5938720703125, 0.0, 944.5514526367188, 0.0, 1395.5264892578125, 547.0949096679688, 0.0, 0.0, 1.0], dtype=np.float64)
		self.rgb_info.D = np.array([0,0,0,0,0], dtype=np.float64)
		self.img = cv2.imread(os.path.join(DATA_PTH, 'marker/imgs/test_img.jpg'), cv2.IMREAD_COLOR)
		# init ros
		if not test:
			self.img = None
			self.bridge = cv_bridge.CvBridge()
			self.img_topic = camera_ns + '/image_raw'
			rospy.loginfo("Waiting for camera_info from %s", camera_ns + '/camera_info')
			self.rgb_info = rospy.wait_for_message(camera_ns + '/camera_info', sensor_msgs.msg.CameraInfo, 25)
			print("Camera height:", self.rgb_info.height, "width:", self.rgb_info.width)

		# init detector
		if use_tags:
			self.det = AprilDetector(marker_length=marker_length, 
									K=self.rgb_info.K, 
									D=self.rgb_info.D,
									dt=1/fps,
									invert_pose=False,
									filter_type=filter_type)
			
		# init vis	
		if cv_window and use_tags:
			cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
	
		# init dynamic reconfigure
		if use_reconfigure and use_tags:
			print("Using reconfigure server")
			self.det_config_server = dynamic_reconfigure.server.Server(DetectorConfig, self.det.setDetectorParams)

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
			print(f"Marker {mid} orientation is likely flipped ...", end=" ")
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
					print("fixed")
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
	
	def preProcImage(self, vis: bool=True) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]:
		""" Put num filter_iters images into
			fresh detection filter and get last
			detection.
		"""
		# test img
		raw_img = self.img
		self.det.resetFilters()
		for i in range(self.filter_iters):
			self.frame_cnt += 1
			# real image
			if not self.test:
				rgb = rospy.wait_for_message(self.img_topic, Image)
				raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
			(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy(), vis if (i >= self.filter_iters-1 and self.vis) else False)

		# align rotations by consens
		if self.flip_outliers:
			if not self.flipOutliers(marker_det):
				beep()
		# improve detection
		if self.refine_pose:
			self.refineDetection(marker_det)

		return marker_det, det_img, proc_img, raw_img
	
	def runDebug(self) -> None:
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
				(marker_det, det_img, proc_img, img) = self.preProcImage()
				if self.vis:
					# frame counter
					cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
					if self.cv_window:
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
						if cv2.waitKey(1) == ord("q"):
							break
				try:
					rate.sleep()
				except:
					pass
		except Exception as e:
			rospy.logerr(e)
		finally:
			cv2.destroyAllWindows()

	def detectionRoutine(self, arg: Any) -> Union[Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, int], dict]:
		raise NotImplementedError
	
	def run(self) -> None:
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

	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis :Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				filter_iters: Optional[int]=10,
				f_ctrl: Optional[int]=30,
				err_term: Optional[float]=2.0,
				cart_bound_low: Optional[float]=-3.0,
				cart_bound_high: Optional[float]=3.0,
				flip_outliers: Optional[bool]=True,
				refine_pose: Optional[bool]=True,
				fn: Optional[str]= 'marker_holder_poses.yml',
				fps: Optional[float]=30.0,
				) -> None:
		
		super().__init__(marker_length=marker_length,
						use_reconfigure=use_reconfigure,
						flip_outliers=flip_outliers,
						refine_pose=refine_pose,
						camera_ns=camera_ns,
						filter_type=filter_type,
						filter_iters=filter_iters,
						f_ctrl=f_ctrl,
						test=False,
						vis=vis,
						fps=fps,
						)
		
		self.err_term = err_term
		self.reprojection_errors = {}
		self.lower_bounds = [cart_bound_low, cart_bound_low, cart_bound_low, -np.pi, -np.pi, -np.pi]
		self.upper_bounds = [cart_bound_high, cart_bound_high, cart_bound_high, np.pi, np.pi, np.pi]
		# load marker poses
		self.fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/" + fn)
		with open(self.fl, 'r') as fr:
			self.marker_table_poses = yaml.safe_load(fr)

	def labelDetection(self, img: cv2.typing.MatLike, trans: np.ndarray, rot: np.ndarray, corners: np.ndarray) -> None:
			pos_txt = "X: {:.4f} Y:  {:.4f} Z:  {:.4f}".format(trans[0], trans[1], trans[2])
			ori_txt = "R: {:.4f} P:  {:.4f} Y:  {:.4f}".format(rot[0], rot[1], rot[2])
			x_max = int(np.max(corners[:, 0]))
			y_max = int(np.max(corners[:, 1]))
			y_min = int(np.min(corners[:, 1]))
			x_offset = 0 if x_max <= img.shape[1]/2 else -int(len(pos_txt)*20*self.FONT_SCALE)
			y_offset1 = self.TXT_OFFSET if y_max <= img.shape[0]/2 else -self.TXT_OFFSET-(y_max-y_min)
			y_offset2 = y_offset1 + int(self.FONT_SCALE*50) if y_offset1 > 0 else y_offset1 - int(self.FONT_SCALE*50)
			cv2.putText(img, pos_txt, (x_max+x_offset, y_max+(y_offset1 if y_offset1 > 0 else y_offset2)), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, ori_txt, (x_max+x_offset, y_max+(y_offset2 if y_offset1 > 0 else y_offset1)), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)

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
		if self.marker_table_poses.get(id) is None:
			print(f"id {id} not present in marker poses!")
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
	
	def projectMarkers(self, detection:dict, camera_pose: np.ndarray, img: cv2.typing.MatLike=None) -> float:
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
		root = self.marker_table_poses.get('root')
		T_world_root = pose2Matrix(root['xyz'], root['rpy'], RotTypes.EULER) if root is not None else np.eye(4)
		# marker tf
		marker = self.marker_table_poses.get(id)
		assert(marker) # marker id entry in yaml?
		T_root_marker = pose2Matrix(marker['xyz'], marker['rpy'], RotTypes.EULER)
		# worldTmarker
		return T_world_root @ T_root_marker
	
	def camTF(self, detection: dict, id: int) -> np.ndarray:
		tf = np.zeros(6)
		det = detection.get(id)
		if det is None:
			print(f"Cannot find id {id} in detection!")
			return tf
		# get markerTcamera
		inv_tvec, inv_euler = invPersp(tvec=det['ftrans'], rot=det['frot'], rot_t=RotTypes.EULER)
		T_marker_cam = pose2Matrix(inv_tvec, inv_euler, RotTypes.EULER)
		# get worldTcamera
		T_world_marker = self.getWorldMarkerTF(id)
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
		res = least_squares(self.residuals, 
							est_camera_pose, 
							args=(self.marker_table_poses, detection),
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
			txt = f"Result: {res.status} {res.message}\n"
			txt += f"camera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\n"
			txt += f"reprojection error: {reserr}\n"
			txt += f"cost: {res.cost}\n"
			txt += f"evaluations: {res.nfev}\n"
			txt += f"optimality: {res.optimality}\n"
			print(txt)

			for id, error in self.reprojection_errors.items():
				if error > self.err_term:
					print("id {} reprojection error: {:.2f} > {} threshold".format(id, error, self.err_term))

			# put pose label
			self.labelDetection(img, -1, opt_cam_pose[:3], opt_cam_pose[3:], reserr)

			return reserr, opt_cam_pose
		
		print(f"Least squares failed: {res.status} {res.message}")
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
		print(f"camera world pose trans: {filtered_pose[:3]}, rot (extr. xyz euler): {filtered_pose[3:]}")
		return err, filtered_pose
		
	def run(self) -> None:
		init = True
		success = False
		err = np.inf
		est_camera_pose = np.zeros(6)
		rate = rospy.Rate(self.f_loop)

		try:
			while not rospy.is_shutdown():
					
					# detect markers 
					(marker_det, det_img, proc_img, _) = self.preProcImage()

					# initially show 
					if self.vis and self.cv_window:
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
						if cv2.waitKey(10000 if init else 1) == ord("q"):
							break

					# estimate cam pose
					if marker_det:

						initial_guess = est_camera_pose
						if init:
							init = False
							initial_guess = self.initialGuess(marker_det)
						
						print("Running estimation")
						(err, est_camera_pose) = self.estimatePoseLS(det_img, err, initial_guess, marker_det)

						if err <= self.err_term:
							print(f"Estimated camera pose xyz (m): {est_camera_pose[:3]}, extr. xyz Euler angles (rad): {est_camera_pose[3:]}, mean reprojection error: {err}")
							success = True

					if self.vis:
						# frame counter
						cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
						for id, det in marker_det.items():
							# label marker pose
							self.labelDetection(det_img, id, det['ftrans'], det['frot'])

						if self.cv_window:
							cv2.imshow('Processed', proc_img)
							cv2.imshow('Detection', det_img)
							if cv2.waitKey(100000 if success else 1) == ord("q"):
								break
					
					if success:
						break

					try:
						rate.sleep()
					except:
						pass

		except Exception as e:
			print(e)

		finally:
			cv2.destroyAllWindows()

def main() -> None:
	rospy.init_node('camera_pose')
	if rospy.get_param('~debug', False):
		DetectBase(camera_ns=rospy.get_param('~markers_camera_name', ''),
				   marker_length=rospy.get_param('~marker_length', 0.010),
				   use_reconfigure=rospy.get_param('~use_reconfigure', False),
				   vis=rospy.get_param('~vis', True),
				   filter_type=rospy.get_param('~filter', 'none'),
				   filter_iters=rospy.get_param('~filter_iters', 10),
				   f_ctrl=rospy.get_param('~f_ctrl', 30),
				   test=rospy.get_param('~test', False),
				   fps=rospy.get_param('~fps', 30.0),
				   refine_pose=True,
				   flip_outliers=False,
				   ).runDebug()
	elif rospy.get_param('~camera_pose', False):
		CameraPoseDetect(camera_ns=rospy.get_param('~markers_camera_name', ''),
				   		fn=rospy.get_param('~marker_poses_file', 'marker_holder_poses.yml'),
						marker_length=rospy.get_param('~marker_length', 0.010),
						use_reconfigure=rospy.get_param('~use_reconfigure', False),
						vis=rospy.get_param('~vis', True),
						filter_type=rospy.get_param('~filter', 'none'),
						filter_iters=rospy.get_param('~filter_iters', 10),
						f_ctrl=rospy.get_param('~f_ctrl', 30),
				   		fps=rospy.get_param('~fps', 30.0),
						err_term=rospy.get_param('~err_term', 2.0),
						).run()
		
if __name__ == "__main__":
	main()
