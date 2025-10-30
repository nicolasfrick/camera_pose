#!/usr/bin/env python

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enum import Enum
from typing import Tuple, Optional, Union
from scipy.spatial.transform import Rotation as R

from .util import *

class FilterTypes(Enum):
	NONE='none'
	MEAN='mean'
	MEDIAN='median'
	KALMAN_SIMPLE='kalman_simple'
	KALMAN='kalman'
FILTER_TYPES_MAP={ FilterTypes.NONE.value : FilterTypes.NONE,
				                            FilterTypes.MEAN.value : FilterTypes.MEAN, 
											FilterTypes.MEDIAN.value : FilterTypes.MEDIAN, 
											FilterTypes.KALMAN_SIMPLE.value : FilterTypes.KALMAN_SIMPLE, 
											FilterTypes.KALMAN.value : FilterTypes.KALMAN }

class PoseFilterBase():
	""" Filter base class.

		@param state_init Initialize the states of the filter (x,y,z,r,p,y), 
						  repr. initial return value.
		@type Tuple[float,float,float,float,float,float]
	"""

	def __init__(self,
			  	 state_init: Optional[Tuple[float,float,float,float,float,float]]=tuple(6*[0]),
				 ) -> None:
		# init return values
		self.translation_estimated = np.array(state_init[:3], dtype=np.float32)
		self.rotation_estimated = np.array(state_init[3:], dtype=np.float32)

	@property
	def est_rotation_as_euler(self) -> np.ndarray:
		return self.rotation_estimated
	@property
	def est_rotation_as_matrix(self) -> np.ndarray:
		return R.from_euler('xyz', self.rotation_estimated) 
	@property
	def est_translation(self) -> np.ndarray:
		return self.translation_estimated
	
	def initFilter(self, filter: cv2.KalmanFilter) -> None:
		raise NotImplementedError()
	
	def updateFilter(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		self.translation_estimated = measurement[:3]
		self.rotation_estimated = measurement[3:]
		return self.translation_estimated, self.rotation_estimated
	
	@classmethod
	def poseToMeasurement(self, tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
		"""Convert pose vectors to filter input.

			@param tvec Translation vector
			@type np.ndarray
			@param rot Rotation vector or matrix
			@type np.ndarray
			@param rot_t Type of rotation information
			@type RotTypes
			@return Measurement vector [x,y,z,r,p,y] with orientation converted to extr. xyz Euler-angles
			@type np.ndarray
		"""
		measurement = np.zeros(6, dtype=np.float32)
		measurement[:3] = tvec[:] 
		measured_euler = getRotation(rot, rot_t, RotTypes.EULER)
		measurement[3:] = measured_euler[:]
		return measurement
	
class PoseFilterMean(PoseFilterBase):
	""" Compute the mean for each value

		@param use_median Use the median instead of mean as filter
		@type bool
	"""

	COLS = ['x','y','z','roll','pitch','yaw']

	def __init__(self,
			  	state_init: Optional[Tuple[float,float,float,float,float,float]]=tuple(6*[0]),
				use_median: Optional[bool]=False) -> None:
		super().__init__(state_init)
		self.use_median = use_median
		self.filter = pd.DataFrame(np.array([state_init]),columns=self.COLS, dtype=pd.Float32Dtype)

	def updateFilter(self, measurement: np.ndarray) -> Tuple[np.ndarray]:
		"""Process filter
			@param measurement Measured values in order [x,y,z,r,p,y] with orientation as extr. xyz Euler-angles
			@type np.ndarray
			@return translation, rotation as extr. xyz Euler-angles
			@type Tuple[np.ndarray, np.ndarray]
			"""
		self.filter.loc[len(self.filter)] = measurement
		mean = self.filter.median() if self.use_median else self.filter.mean()
		self.translation_estimated = np.array([mean.x, mean.y, mean.z])
		self.rotation_estimated = np.array([mean.roll, mean.pitch, mean.yaw])
		return self.translation_estimated, self.rotation_estimated
	
class PoseFilterKalman(PoseFilterBase):
	"""
		Kalman Filter implementation for 6D pose tracking as 
		depicted in https://campar.in.tum.de/Chair/KalmanFilter and 
		https://docs.opencv.org/3.3.0/dc/d2c/tutorial_real_time_pose.html
		and https://pieriantraining.com/kalman-filter-opencv-python-example/
		the latter is using a linear motion model that is suitable for slow movements.

		@param dt Time delta for updates
		@type float
		@param state_init Initialize the states of the filter (x,y,z,r,p,y)
		@type Tuple[float,float,float,float,float,float]
		@param process_noise Noise model initialization for process covariance
		@type float
		@param measurement_noise Noise model initialization for measurement covariance
		@type float
		@param error_post Initialization for post state error covariance 
		@type float
		@param lin_motion Use linear motion model 
		@type bool
	"""

	num_states_per_model = [18, 6]
	num_measurements_per_model = [6, 3]

	def __init__(self,
			  	dt: float,
				state_init: Optional[Tuple[float,float,float,float,float,float]]=tuple(6*[0]),
				process_noise: Optional[float]=1e-10,
				measurement_noise: Optional[float]=1e-10,
				error_post: Optional[float]=0.0,
				lin_motion: Optional[bool]=True) -> None:
		
		super().__init__(state_init)
		self.dt = dt
		self.state_init = state_init
		self.process_noise = process_noise
		self.measurement_noise = measurement_noise
		self.error_post = error_post
		self.lin_motion = lin_motion
		self.translation_predicted = np.zeros(3, dtype=np.float32)
		self.rotation_predicted = np.zeros(3, dtype=np.float32)
		self.num_states = self.num_states_per_model[int(lin_motion)]
		self.num_measurements = self.num_measurements_per_model[int(lin_motion)]

		self.filter1 = cv2.KalmanFilter(self.num_states, self.num_measurements)
		self.initFilter(self.filter1)
		if lin_motion:
			self.filter2 = cv2.KalmanFilter(self.num_states, self.num_measurements)
			self.initFilter(self.filter2)

	@property
	def pred_rotation_as_euler(self) -> np.ndarray:
		return self.rotation_predicted
	@property
	def pred_rotation_as_matrix(self) -> np.ndarray:
		return R.from_euler('xyz', self.rotation_predicted) 
	@property
	def pred_translation(self) -> np.ndarray:
		return self.translation_predicted

	def initFilter(self, filter: cv2.KalmanFilter) -> None:
		"""        
			Initialize the filter matrices.

			DYNAMIC MODEL 
			[1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
			[0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
			[0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
			[0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
			[0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
			[0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
			[0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

			MEASUREMENT MODEL 
			[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
			[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
			[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
			[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
			[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
			[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
		"""

		# covariance matrices
		filter.errorCovPost = cv2.setIdentity(filter.errorCovPost, self.error_post)
		filter.processNoiseCov = cv2.setIdentity(filter.processNoiseCov, self.process_noise)
		filter.measurementNoiseCov = cv2.setIdentity(filter.measurementNoiseCov, self.measurement_noise)

		# mapping matrices
		transition_matrix = np.eye(self.num_states, dtype=np.float32)
		measurement_matrix = np.zeros((self.num_measurements, self.num_states), dtype=np.float32)

		# linear motion model
		if self.lin_motion:
			transition_matrix[0, 3] = 1
			transition_matrix[1, 4] = 1
			transition_matrix[2, 5] = 1

			measurement_matrix[0, 0] = 1
			measurement_matrix[1, 1] = 1
			measurement_matrix[2, 2] = 1

		# nonlinear motion model
		else:
			# position
			transition_matrix[0, 3] = self.dt
			transition_matrix[1, 4] = self.dt
			transition_matrix[2, 5] = self.dt
			transition_matrix[3, 6] = self.dt
			transition_matrix[4, 7] = self.dt
			transition_matrix[5, 8] = self.dt
			transition_matrix[0, 6] = 0.5*pow(self.dt, 2)
			transition_matrix[1, 7] = 0.5*pow(self.dt, 2)
			transition_matrix[2, 8] = 0.5*pow(self.dt, 2)
			# orientation
			transition_matrix[9, 12] = self.dt
			transition_matrix[10, 13] = self.dt
			transition_matrix[11, 14] = self.dt
			transition_matrix[12, 15] = self.dt
			transition_matrix[13, 16] = self.dt
			transition_matrix[14, 17] = self.dt
			transition_matrix[9, 15] = 0.5*pow(self.dt, 2)
			transition_matrix[10, 16] = 0.5*pow(self.dt, 2)
			transition_matrix[11, 17] = 0.5*pow(self.dt, 2)
			# measurement matrix
			measurement_matrix[0, 0] = 1  # x
			measurement_matrix[1, 1] = 1  # y
			measurement_matrix[2, 2] = 1  # z
			measurement_matrix[3, 9] = 1  # roll
			measurement_matrix[4, 10] = 1 # pitch
			measurement_matrix[5, 11] = 1 # yaw

		filter.transitionMatrix = transition_matrix
		filter.measurementMatrix = measurement_matrix

		# states
		init_state = np.zeros((self.num_states, 1), dtype=np.float32)
		init_state[0][0] = self.state_init[0]
		init_state[1][0] = self.state_init[1]
		init_state[2][0] = self.state_init[2]
		if not self.lin_motion:
			init_state[9][0] = self.state_init[3]
			init_state[10][0] = self.state_init[4]
			init_state[11][0] = self.state_init[5]
		filter.statePre = init_state
		filter.statePost = init_state

	def updateFilter(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Process filter
			@param measurement Measured values in order [x,y,z,r,p,y] with orientation as extr. xyz Euler-angles
			@type np.ndarray
			@return translation, rotation as extr. xyz Euler-angles
			@type Tuple[np.ndarray, np.ndarray]
			"""
		# predict, to update the internal statePre variable
		prediction1 = self.filter1.predict()
		prediction1 = prediction1.flatten()
		# correct phase uses predicted value and measurement
		estimated1 = self.filter1.correct(measurement[:3] if self.lin_motion else measurement)
		estimated1 = estimated1.flatten()
		self.translation_predicted= prediction1[:3]
		self.translation_estimated = estimated1[:3]

		if self.lin_motion:
			prediction2 = self.filter2.predict()
			prediction2 = prediction2.flatten()
			estimated2 = self.filter2.correct(measurement[3:])
			estimated2 = estimated2.flatten()
			self.rotation_predicted = prediction2[:3] 
			self.rotation_estimated = estimated2[:3]
		else:
			self.rotation_predicted = prediction1[9:12] 
			self.rotation_estimated = estimated1[9:12]

		return self.translation_estimated, self.rotation_estimated
	
def createFilter(filter_type: Union[FilterTypes, str], 
				init_state: Optional[Tuple[float, float, float, float, float, float]]=6*[0],
				dt_kalman: Optional[float]=0.1,
				process_noise_kalman: Optional[float]=1e-10, 
				measurement_noise_kalman: Optional[float]=1e-10, 
				error_post_kalman: Optional[float]=0.0
				) -> Union[PoseFilterMean, PoseFilterKalman, PoseFilterBase]:
	ft = filter_type if isinstance(filter_type, FilterTypes) else FILTER_TYPES_MAP[filter_type]
	if ft in [FilterTypes.MEAN, FilterTypes.MEDIAN]:
		return PoseFilterMean(state_init=init_state, 
							use_median=ft==FilterTypes.MEDIAN)  
	elif ft in [FilterTypes.KALMAN, FilterTypes.KALMAN_SIMPLE]:
		return PoseFilterKalman(dt=dt_kalman, 
						  		state_init=init_state, 
								measurement_noise=measurement_noise_kalman, 
								process_noise=process_noise_kalman, 
								error_post=error_post_kalman,
								lin_motion=ft==FilterTypes.KALMAN_SIMPLE)
	else:
		# no filtering
		return PoseFilterBase()
	
def testKalmanFilter(lin):
	fq, size = 30, 200
	init = (19, -0.1, 3.142, 1, 1, 0)
	proc_noise = 1e-10
	meas_noise = 1e-9
	error = 0
	error_2 = 0.01

	for var in range(6):
		f = PoseFilterKalman(fq, process_noise=proc_noise, measurement_noise=meas_noise, error_post=error, lin_motion=lin)
		est, inp = np.zeros((size, 6), dtype=np.float32), np.zeros((size, 6), dtype=np.float32)
		for i in range(size):
			meas = np.random.uniform(init[var]-error_2, init[var]+error_2, 6).astype(np.float32)
			(trans, rot) = f.updateFilter(meas)
			inp[i] = meas
			est[i,:3] = trans
			est[i,3:] = rot

		xs = [x/fq for x in range(size)]
		mean = sum(inp[:, var])/len(inp[:, var])
		print("Last estimation: ", est[-1,var], ", mean: ", mean, ", true value: ", init[var])
		plt.plot(xs, inp[:, var], label='measurement')
		plt.plot(xs, [init[var] for _ in range(size)], label='init')
		plt.plot(xs, [mean for _ in range(size)], label='mean')
		plt.plot(xs, est[:, var], label='estimate', linewidth=2)
		plt.legend()
		plt.show()

def testMeanFilter(median):
	size = 200
	init = (19, -0.1, 3.142, 1, 1, 0)
	error_2 = 0.01

	for var in range(6):
		f = PoseFilterMean(init)
		est, inp = np.zeros((size, 6), dtype=np.float32), np.zeros((size, 6), dtype=np.float32)
		for i in range(size):
			meas = np.random.uniform(init[var]-error_2, init[var]+error_2, 6).astype(np.float32)
			(trans, rot) = f.updateFilter(meas)
			inp[i] = meas
			est[i,:3] = trans
			est[i,3:] = rot

		xs = [x for x in range(size)]
		mean = sum(inp[:, var])/len(inp[:, var])
		print("Last estimation: ", est[-1,var], ", mean: ", mean, ", true value: ", init[var])
		plt.plot(xs, inp[:, var], label='measurement')
		plt.plot(xs, [init[var] for _ in range(size)], label='init')
		plt.plot(xs, [mean for _ in range(size)], label='mean')
		plt.plot(xs, est[:, var], label='estimate', linewidth=2)
		plt.legend()
		plt.show()

if __name__=="__main__":
	# testKalmanFilter(True)
	testMeanFilter(True)
