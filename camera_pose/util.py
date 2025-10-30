import os
import cv2
import yaml
import numpy as np

from enum import Enum
from cv2 import Rodrigues
from typing import Tuple, Union
from scipy.spatial.transform import Rotation as R

class RotTypes(Enum):
	RVEC = 'rvec'
	EULER = 'xyz_euler'
	MAT = 'matrix'
	QUAT = 'quaternions'

class NormalTypes(Enum):
	XY = 'xy'
	XZ = 'xz'
	YZ = 'yz'
NORMAL_TYPES_MAP = {NormalTypes.XY.value: NormalTypes.XY,
					NormalTypes.XZ.value: NormalTypes.XZ,
					NormalTypes.YZ.value: NormalTypes.YZ,
					}
NORMAL_IDX_MAP = {NormalTypes.XY: 2,
				  NormalTypes.XZ: 1,
				  NormalTypes.YZ: 0,
				  }

def parseIntTuple(value: str) -> Union[None, Tuple]:
	if not ',' in value:
		return None
	t = tuple(map(int, value.split(',')))
	return t

def parseFloatTuple(value: str) -> Union[None, Tuple]:
	if not ',' in value:
		return None
	t = tuple(map(float, value.split(',')))
	return t

def getRotation(rot: np.ndarray, rot_type: RotTypes, out_type: RotTypes) -> np.ndarray:
	if rot_type == out_type:
		return rot

	# convert to mat
	mat = rot
	if rot_type == RotTypes.RVEC:
		(mat, _) = Rodrigues(rot)
	elif rot_type == RotTypes.EULER:
		# gimble lock?
		if np.abs(np.abs(rot[1]) - np.pi*0.5) < 1e-6:
			print("Incoming gimble lock detected")
		mat = R.from_euler('xyz', rot).as_matrix()
	elif rot_type == RotTypes.QUAT:
		# x y z w
		mat = R.from_quat(rot).as_matrix()
	elif rot_type == RotTypes.MAT:
		pass
	else:
		raise NotImplementedError

	# convert to output format
	res = mat
	if out_type == RotTypes.RVEC:
		(res, _) = Rodrigues(res)
		res = res.flatten()
	elif out_type == RotTypes.EULER:
		res = R.from_matrix(res).as_euler('xyz')
		res = res.flatten()
		# gimble lock?
		if np.abs(np.abs(res[1]) - np.pi*0.5) < 1e-6:
			print("Outgoing gimble lock detected")
	elif out_type == RotTypes.QUAT:
		res = R.from_matrix(mat).as_quat()
	elif out_type == RotTypes.MAT:
		pass
	else:
		raise NotImplementedError

	return res

def rotDiff(rot1: np.ndarray, rot2: np.ndarray) -> float:
	diff = rot1.T @ rot2  # relative rotation
	angle_diff = np.arccos((np.trace(diff) - 1) / 2)  # angle difference
	return angle_diff

def avgRotMat(rotation_matrices: np.ndarray) -> np.ndarray:
	quats = np.array([getRotation(mat, RotTypes.MAT, RotTypes.QUAT)
					  for mat in rotation_matrices])
	avg_quat = np.mean(quats, axis=0) 	# average
	avg_quat /= np.linalg.norm(avg_quat)    # normalize
	return getRotation(avg_quat, RotTypes.QUAT, RotTypes.MAT)

def pose2Matrix(tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
	transformation_matrix = np.eye(4)
	transformation_matrix[:3, :3] = getRotation(rot, rot_t, RotTypes.MAT)
	transformation_matrix[:3, 3] = tvec
	return transformation_matrix

def invPersp(tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> Tuple[np.ndarray, np.ndarray]:
	"""Apply the inversion to the given vectors [[R^-1 -R^-1*d][0 0 0 1]]"""
	mat = getRotation(rot, rot_t, RotTypes.MAT)
	mat = np.matrix(mat).T  # orth. matrix: A.T = A^-1
	inv_tvec = -mat @ tvec  # -R^-1*d
	inv_rot = getRotation(mat, RotTypes.MAT, rot_t)
	if rot_t != RotTypes.MAT:
		inv_rot = inv_rot.flatten()
	return np.array(inv_tvec.flat), inv_rot

def consensusRot(rot_mats: np.ndarray, n_iterations: int = 100, threshold: float = 0.2) -> Tuple[np.ndarray, list]:
	"""Find the rotation that has a minimal distance to all rotations."""
	best_inliers = []
	best_rotation = None
	for _ in range(n_iterations):
		# random sample two rotations
		sample_indices = np.random.choice(len(rot_mats), 2, replace=False)
		sample_rotations = [rot_mats[i] for i in sample_indices]

		# find consensus rotation
		consens_rot = avgRotMat(sample_rotations)
		# find inliers
		current_inliers = []
		for idx, mat in enumerate(rot_mats):
			if rotDiff(consens_rot, mat) < threshold:
				current_inliers.append(idx)

		# update best inliers
		if len(current_inliers) > len(best_inliers):
			best_inliers = current_inliers
			best_rotation = consens_rot

	return best_rotation, best_inliers

def refRotZ(rot_mat: np.ndarray, ref_rotations: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
	"""Rotate the z axis of rot_mat by the angular difference to ref_rotations average."""
	# compute average z-axis from the reference rotations
	avg_z_axis = np.mean([r[:3, 2] for r in ref_rotations], axis=0)
	avg_z_axis /= np.linalg.norm(avg_z_axis)
	# get z-axis of target rotation
	current_z_axis = rot_mat[:3, 2]
	current_z_axis /= np.linalg.norm(current_z_axis)

	# compute the axis- angle to rotate the current z-axis to the average z-axis
	rotation_axis = np.cross(current_z_axis, avg_z_axis)  # axis of rotation
	if np.linalg.norm(rotation_axis) < threshold:  # already aligned
		return rot_mat
	rotation_axis /= np.linalg.norm(rotation_axis)
	# compute angle of rotation
	rotation_angle = np.arccos(np.dot(current_z_axis, avg_z_axis))

	# compute the correction rotation matrix around the rotation_axis
	correction_rotation = R.from_rotvec(
		rotation_angle * rotation_axis).as_matrix()
	# align the z-axis while keeping x and y as intact as possible
	return rot_mat @ correction_rotation

def findAxisOrientOutliers(rot_mats: np.ndarray, tolerance: float = 1e-6, axis_idx: int = ord('x')) -> Tuple[list, np.ndarray]:
	"""Average the axes of all matrices and find the index
									for matrices that do not match the avg orientation.
	"""
	# compute average axes
	axs_avg = np.mean([r[:3, axis_idx] for r in rot_mats], axis=0)
	axs_avg /= np.linalg.norm(axs_avg)

	# compare each axis to the average
	outliers = []
	for idx, mat in enumerate(rot_mats):
		# normalize
		axs = mat[:, axis_idx] / np.linalg.norm(mat[:, axis_idx])
		# check degree of alignment
		if abs(np.dot(axs, axs_avg)) < tolerance:
			outliers.append(idx)

	return outliers, axs_avg

def ransacPose(tvec: np.ndarray, rvec: np.ndarray, corners: np.ndarray, obj_points: np.ndarray, cmx: np.ndarray, dist: np.ndarray, solver_flag: int = cv2.SOLVEPNP_IPPE_SQUARE) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
	"""RANSAC over given pose can improve the accuracy."""
	(success, out_rvec, out_tvec, inliers) = cv2.solvePnPRansac(objectPoints=obj_points,
																imagePoints=np.array(
																	corners, dtype=np.float32),
																cameraMatrix=cmx,
																distCoeffs=dist,
																rvec=rvec,
																tvec=tvec,
																useExtrinsicGuess=False,
																flags=solver_flag,
																)
	if success:
		out_tvec = out_tvec.reshape(3)
		out_rvec = out_rvec.reshape(3)
		inliers = inliers.flatten()
		return success, out_tvec, out_rvec,  inliers

	return success, tvec, rvec,  None

def refinePose(tvec: np.ndarray, rvec: np.ndarray, corners: np.ndarray, obj_points: np.ndarray, cmx: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	""" Non-linear Levenberg-Marquardt minimization scheme and the current implementation 
																	computes the rotation update as a perturbation and not on SO(3).
	"""
	# TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 20, FLT_EPSILON)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	(out_rvec, out_tvec) = cv2.solvePnPRefineLM(objectPoints=obj_points,
												imagePoints=np.array(
													corners, dtype=np.float32),
												cameraMatrix=cmx,
												distCoeffs=dist,
												rvec=rvec,
												tvec=tvec,
												criteria=criteria,
												)
	return out_tvec.flatten(), out_rvec.flatten()

def loadMarkerConfig() -> dict:
	# load marker configuration
	fl = os.path.join(os.path.dirname(os.path.dirname(
		os.path.abspath(__file__))), "config/marker_config.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
		marker_config = config['marker_config']
		return marker_config
