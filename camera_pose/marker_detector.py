#!/usr/bin/env python

import os
import cv2
import yaml
import dataclasses
import numpy as np
import cv2.aruco as aru
import dt_apriltags as apl

from threading import Lock
from typing import Optional, Tuple, Any, Callable

from pose_filter import *
from util import *

@dataclasses.dataclass(eq=False)
class KfParams():
  error_post = 0.0
  process_noise = 1e-8
  measurement_noise = 1e-8
  param_change = False

@dataclasses.dataclass(eq=False)
class ReconfParams():
	aruco_params = aru.DetectorParameters()
	estimate_params = aru.EstimateParameters()
	kf_params = KfParams()

class MarkerDetectorBase():
	"""Base clas for fiducial marker detection.

		@param marker_length:  Physical dimension (sidelength) of the marker in meters!
		@type  float
		@param K: camera intrinsics
		@type  tuple
		@param D: camera distorsion coefficients (plumb bob)
		@type  tuple
		@param dt Time delta for Kalman filter
		@type float
		@param bbox: bounding box ((x_min, x_max),(y_min,y_max)), if provided, image will be cropped
		@type  tuple
		@param print_stats Print detection statistics
		@type bool
		@param filter_type Determine whether the detections are filtered and the type of filter
		@type FilterTypes
		@param invert_pose Invert detected marker pose 
		@type bool
		@param refine_detection
		@type bool
		@param crosshair Draw camera coordinate system.
		@type bool
	"""

	RED = (0,0,255)
	GREEN = (0,255,0)
	BLUE = (255,0,0)
	AXIS_LENGTH = 1.5
	AXIS_THICKNESS = 2
	CIRCLE_SIZE = 3
	CIRCLE_CLR = BLUE
	FONT_THCKNS = 2
	FONT_SCALE = 0.5

	def __init__(self,					
				 K: Tuple,
				 D: Tuple,
				 marker_length: float,
				 dt: Optional[float]=0.1,
				 bbox: Optional[Tuple]=None,
				 crosshair: Optional[bool]=False,
				 print_stats: Optional[bool]=True,
				 invert_pose: Optional[bool]=False,
				 filter_type: Optional[Union[FilterTypes, str]]=FilterTypes.NONE,
				 ) -> None:
		
		# params
		self.params_lock = Lock()
		self.params = ReconfParams()		
		# initialize params
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/detector_params.yaml")
		with open(fl, 'r') as fr:
			self.params_config = yaml.safe_load(fr)
		for _, v in self.params_config.items():
			self.setDetectorParams(v, 0)

		self.print_stats = print_stats
		self.marker_length = marker_length
		self.filter_type = filter_type 
		self.invert_perspective = invert_pose		
		self.crosshair = crosshair
		self.dt = dt
		self.bbox = bbox
		self.filters = {}
		self.cmx = np.asanyarray(K).reshape(3,3)
		self.dist =  np.asanyarray(D)
		self.t_total = 0
		self.it_total = 0
		self._genSquarePoints(marker_length)

	def setDetectorParams(self, config: Any, level: int) -> Any:
		with self.params_lock:
			for k,v in config.items():
				if k in self.params_config['pose_estimation']:
					self.params.estimate_params.__setattr__(k, v)
				elif k in self.params_config['aruco_detection']:
					self.params.aruco_params.__setattr__(k, v)
				elif k in self.params_config['kalman_filter']:
					# set only when value has changed (up to 10 digits)
					if np.round(v, 10) != np.round(self.params_config['kalman_filter'][k], 10):
						self.params.kf_params.__setattr__(k, v)
						self.params.kf_params.param_change = True
				else:
					self.params.__setattr__(k, v)
			self.params_change = True
			return config
	
	def setBBox(self, bbox: Tuple[float, float, float, float]) -> None:
		self.bbox = bbox

	def getFilteredTranslationById(self, id: int) -> Union[np.ndarray, None]:
		f = self.filters.get(id)
		if f is not None:
			return f.est_translation
		return None
	
	def getFilteredRotationEulerById(self, id: int) -> Union[np.ndarray, None]:
		f = self.filters.get(id)
		if f is not None:
			return f.est_rotation_as_euler
		return None
	
	def resetFilters(self) -> None:
		self.filters.clear()
	
	@property
	def square_points(self) -> np.ndarray:
		return self.obj_points
	
	def _genSquarePoints(self, length: float) -> None:
		"""
			https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
			cv::SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation. 
			Number of input points must be 4. Object points must be defined in the following order:
				point 0: [-squareLength / 2, squareLength / 2, 0]
				point 1: [ squareLength / 2, squareLength / 2, 0]
				point 2: [ squareLength / 2, -squareLength / 2, 0]
				point 3: [-squareLength / 2, -squareLength / 2, 0]
		"""
		self.obj_points = np.zeros((4, 3), dtype=np.float32)
		self.obj_points[0,:] = np.array([-length/2, length/2, 0])  # top-left corner
		self.obj_points[1,:] = np.array([length/2, length/2, 0])   # top-right corner
		self.obj_points[2,:] = np.array([length/2, -length/2, 0])  # bottom-right corner
		self.obj_points[3,:] = np.array([-length/2, -length/2, 0]) # bottom-left corner 
		# define the order counter-clockwise to follow the standard 
		# right-handed coordinate system for pose estimation
		# self.obj_points[0,:] = np.array([-length/2, -length/2, 0]) # bottom-left corner 
		# self.obj_points[1,:] = np.array([length/2, -length/2, 0])  # bottom-right corner
		# self.obj_points[2,:] = np.array([length/2, length/2, 0])   # top-right corner
		# self.obj_points[3,:] = np.array([-length/2, length/2, 0])  # top-left corner

	def _projPoints(self, img: cv2.typing.MatLike, obj_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> cv2.typing.MatLike:
		"""Test the solvePnP by projecting the 3D Points to camera"""
		proj, _ = cv2.projectPoints(obj_points, rvec, tvec, self.cmx, self.dist)
		for p in proj:
			cv2.circle(img, (int(p[0][0]), int(p[0][1])), self.CIRCLE_SIZE, self.CIRCLE_CLR, -1)
		return img

	def _cropImage(self, img: np.ndarray) -> np.ndarray:
		return img[self.bbox[0][0]: self.bbox[0][1], self.bbox[1][0]: self.bbox[1][1]]
			
	def _printStats(self, tick: int) -> None:
		t_current = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
		self.t_total += t_current
		self.it_total += 1
		if self.it_total % 10 == 0:
			print("Detection Time = {} ms (Mean = {} ms)".format(int(t_current * 1000), int(1000 * self.t_total / self.it_total)))

	def _drawCamCS(self, img: cv2.typing.MatLike) -> None:
		thckns = 2
		arw_len = 100
		img_center =(int(img.shape[1]/2), int(img.shape[0]/2))
		cv2.arrowedLine(img, img_center, (img_center[0]+arw_len, img_center[1]), self.RED, thckns, cv2.LINE_AA)
		cv2.arrowedLine(img, img_center, (img_center[0], img_center[1]+arw_len), self.GREEN, thckns, cv2.LINE_AA)
		cv2.putText(img, 'X', (img_center[0]-10, img_center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.BLUE, thckns, cv2.LINE_AA)
		cv2.circle(img, img_center, self.CIRCLE_SIZE, self.CIRCLE_CLR, -1)

	def _drawMarkers(self, id: int, corners: np.ndarray, img: cv2.typing.MatLike) -> None:
		cv2.putText(img, str(id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.RED, self.FONT_THCKNS)
		for i in range(4):
			cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), self.GREEN, self.AXIS_THICKNESS)

	def _printSettings(self) -> None:
		raise NotImplementedError
	
	def _detectionRoutine(self):
		raise NotImplementedError
		
	def detMarkerPoses(self, img: cv2.typing.MatLike, subroutine: Callable[[cv2.typing.MatLike, cv2.typing.MatLike], Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]], vis: bool=True) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]:	
		"""Generic marker detection method."""
		with self.params_lock:
			tick = cv2.getTickCount()
			# grasycale image
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# improve contrast
			if self.params.hist_equalization:
				gray = cv2.equalizeHist(gray)
			#  image denoising using Non-local Means Denoising algorithm
			if self.params.denoise:
				gray = cv2.fastNlMeansDenoising(gray, None, self.params.h, self.params.templateWindowSize, self.params.searchWindowSize)

			(marker_poses, out_img, gray) = subroutine(img, gray, vis)

			if vis:
				if self.crosshair:
					self._drawCamCS(out_img)
				for _, pose in marker_poses.items():
					out_img = cv2.drawFrameAxes(out_img, self.cmx, self.dist, pose['rvec'], pose['tvec'], self.marker_length*self.AXIS_LENGTH, self.AXIS_THICKNESS)
					out_img = self._projPoints(out_img, pose['points'], pose['rvec'], pose['tvec'])

			if self.print_stats:
				self._printStats(tick)

			return marker_poses, img, gray
	
class AprilDetector(MarkerDetectorBase):
	"""
		Apriltag marker detector.
		
		@param marker_family Type of Apriltag marker
		@type str

		The tag's coordinate frame is centered at the center of the tag, with x-axis to the right, y-axis down, and z-axis INTO the tag.
													
		Max detection distance in meters = t /(2 * tan( (b* f * p) / (2 * r ) ) )
		t = size of your tag in meters
		b = the number of bits that span the width of the tag (excluding the white border for Apriltag 2). ex: 36h11 = 8, 25h9 = 6, standard41h12 = 9
		f = horizontal FOV of your camera
		r = horizontal resolution of you camera
		p = the number of pixels required to detect a bit. This is an adjustable constant. We recommend 5. Lowest number we recommend is 2 which is the Nyquist Frequency. 
				We recommend 5 to avoid some of the detection pitfalls mentioned above.
	"""
	def __init__(self,
			  				K: Tuple,
							D: Tuple,
							marker_length: float,
							dt: Optional[float]=0.1,
							bbox: Optional[Tuple]=None,
							print_stats: Optional[bool]=True,
							invert_pose: Optional[bool]=False,
							filter_type: Optional[Union[FilterTypes, str]]=FilterTypes.NONE,
							marker_family: Optional[str]='tag16h5',
							debug: Optional[bool]=False,
							) -> None:
		
		super().__init__(K=K, 
				   						D=D, 
										marker_length=marker_length, 
										dt=dt, 
										bbox=bbox, 
										invert_pose=invert_pose,
										print_stats=print_stats, 
										filter_type=filter_type,
										)
		self.det = apl.Detector(families=marker_family, 
													nthreads=self.params.nthreads,
													quad_decimate=self.params.quad_decimate,   
													quad_sigma=self.params.quad_sigma,
													refine_edges=self.params.refine_edges, 
													decode_sharpening=self.params.decode_sharpening,
													debug=False,
													)
		self.debug = debug
		self.params_change = False
		self.camera_params = (K[0], K[4], K[2], K[5])
		self._printSettings()

	def _printSettings(self) -> None:
		txt =   f"Running Apriltag Detector with settings:\n"
		txt += f"Camera params fx: {self.camera_params[0]}, fy: {self.camera_params[1]}, cx: {self.camera_params[2]}, cy: {self.camera_params[3]}\n"
		txt += f"Distorsion: {self.dist}\n"
		txt += f"print_stats: {self.print_stats},\n"
		txt += f"marker_length: {self.marker_length},\n"
		txt += f"delta t: {self.dt},\n"
		txt += f"filter type: {self.filter_type},\n"
		for attr in dir(self.params):
			if not attr.startswith('__'):
				txt += f"{attr}: {self.params.__getattribute__(attr)},\n"
		print(txt)
		print()

	def _validateDetection(self, detection: apl.Detection) -> bool:
		corners = detection.corners.astype(int)
		marker_width = np.linalg.norm(corners[0] - corners[1])
		marker_height = np.linalg.norm(corners[1] - corners[2])
		if self.debug:
			print(detection)
			print(marker_height, marker_width)
		return detection.decision_margin >= self.params.decision_margin \
						and detection.hamming <= self.params.max_hamming \
							and marker_width >= self.params.min_marker_width \
								and marker_height >= self.params.min_marker_height

	def _adaptParams(self):
		self.det.tag_detector_ptr.contents.nthreads = int(self.params.nthreads)
		self.det.tag_detector_ptr.contents.quad_decimate = float(self.params.quad_decimate)
		self.det.tag_detector_ptr.contents.quad_sigma = float(self.params.quad_sigma)
		self.det.tag_detector_ptr.contents.refine_edges = int(self.params.refine_edges)
		self.det.tag_detector_ptr.contents.decode_sharpening = int(self.params.decode_sharpening)
		self.params_change = False
		
	def _detectionRoutine(self, img: cv2.typing.MatLike, gray: cv2.typing.MatLike, vis: bool=True)  -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]:	
		marker_poses = {}
		if self.params_change:
			self._adaptParams()

		detections = self.det.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.marker_length)
		if len(detections) > 0:
			for detection in detections:
				# proc detections
				if self._validateDetection(detection):
					id = detection.tag_id
					tvec = detection.pose_t.flatten()
					rot_mat = detection.pose_R
					# invert pose
					if self.invert_perspective:
						(tvec, rot_mat) = invPersp(tvec=tvec, rot=rot_mat, rot_t=RotTypes.MAT)
					# filtering
					if id in self.filters.keys() and not self.params.kf_params.param_change:
						self.filters[id].updateFilter(PoseFilterBase.poseToMeasurement(tvec=tvec, rot=rot_mat, rot_t=RotTypes.MAT))
					else:
						# new filter
						self.filters.update( {id: createFilter(self.filter_type, 
																							PoseFilterBase.poseToMeasurement(tvec=tvec, rot=rot_mat, rot_t=RotTypes.MAT),
																							dt_kalman=self.dt, # applies only for kalman filter
																							process_noise_kalman=self.params.kf_params.process_noise, # applies only for kalman filter
																							measurement_noise_kalman=self.params.kf_params.measurement_noise, # applies only for kalman filter 
																							error_post_kalman=self.params.kf_params.error_post)} ) # applies only for kalman filter
					# result
					marker_poses.update({id: {'rvec': cv2.Rodrigues(rot_mat)[0].flatten(), 
							   											'rot_mat': rot_mat,
							   											'tvec': tvec, 
																		'points': self.obj_points, 
																		'corners': detection.corners.astype(int), 
																		'ftrans': self.getFilteredTranslationById(id), 
																		'frot': self.getFilteredRotationEulerById(id),
																		'center': detection.center,
																		'pose_err': detection.pose_err}})
					if vis:
						self._drawMarkers(id, detection.corners.astype(int), img)
				elif vis:
					self._drawMarkers(detection.tag_id, detection.corners.astype(int), gray)

			# reset flag for next kalman update
			self.params.kf_params.param_change = False 

		return marker_poses, img, gray

	def detMarkerPoses(self, img: np.ndarray, vis: bool=True) -> Tuple[dict, np.ndarray]:
		"""Detect Apriltag marker in bgr image.
			@param img Input image with 'bgr' encoding
			@type np.ndarray
			@return Detected marker poses, marker detection image, processed image
		"""
		return super().detMarkerPoses(img, self._detectionRoutine, vis)
