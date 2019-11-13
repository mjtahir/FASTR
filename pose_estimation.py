import cv2 as cv
import numpy as np
import config

def estimatePosePnP(image_points):

	# Sort points from top left corner CCW to top right corner. Change to float
	# as solvePnP requires floats
	sorted_points = orderPoints(image_points)
	sorted_points = sorted_points.astype(np.float64)

	# Note GATE_COORDS are also in CCW order from top-left to top-right (for 2D
	# 3D point correspondence)
	ret, rvec, tvec = cv.solvePnP(config.GATE_COORDS, sorted_points, 
									config.CAMERA_MATRIX, 0)#, config.DISTORTION)

	# Convert Rodrigues vector to rotation matrix to extract Euler angles
	rotation_matrix, _ = cv.Rodrigues(rvec)

	# Proj_matrix = np.matmul(config.CAMERA_MATRIX, np.concatenate((rotation_matrix, tvec), axis=1))
	# eulers = cv.decomposeProjectionMatrix(Proj_matrix)[-1]

	# Convert rotation matrix to Euler angles. If statement checks for gimbal lock
	# order: Rx(theta_x) * Ry(theta_y) * Rz(theta_z)
	if -1 < rotation_matrix[0, 2] < 1:
		theta_y = -np.arcsin(rotation_matrix[0, 2])
		theta_x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[2, 2])
		theta_z = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[0, 0])
		euler = np.array([theta_y, theta_x + np.pi/2, theta_z + np.pi/2])
	else:
		euler = np.array([0, 0, 0])
		print('Non unique orientation')

	# Transpose (or invert) to find pose of gate w.r.t. the drone
	rotation_matrix_wrt_drone = np.transpose(rotation_matrix)

	# Position of drone w.r.t to the object coordinates
	tvec_wrt_camera = -rotation_matrix_wrt_drone @ tvec
	tvec_wrt_camera = tvec_wrt_camera.ravel()	# Change to 1D array

	return (sorted_points, rvec, tvec), (euler, tvec_wrt_camera)


def orderPoints(points):
	'''This is a helper function for estimatePosePnP. It orders the contour
	points from 'cv.findContours()' or 'cv.approxPolyDP()' into an order from 
	top-left corner CCW.'''

	# Convert shape (4,1,2) to (4,2) - remove unnecessary dimension
	points = np.squeeze(points)
	num_of_points = np.size(points, axis=0)

	# Sort from lowest to highest x coord, then separate to left and right
	x_sorted = points[np.argsort(points[:, 0]), :]
	left_points = x_sorted[:num_of_points // 2, :]	# two left points
	right_points = x_sorted[num_of_points // 2:, :]	# two right points

	# Sort from lowest to highest y coord, note origin is top left of image
	left_points = left_points[np.argsort(left_points[:, 1]), :]
	right_points = right_points[np.argsort(right_points[:, 1]), :]

	# CCW order is wanted therefore flip the right_points to go from bottom to 
	# top of image
	right_points = np.flip(right_points, 0)

	# Concatenate left and right points to yield 1 matrix
	sorted_points = np.concatenate((left_points, right_points), axis=0)
	return sorted_points


def simpleDistCalibration(img, GATE_WIDTH, DISTANCE_TO_GATE):
	'''
	The calibration image is assumed to contain a detectable gate which is 
	upright and not angled.
	'''
	from gate_detection import colourSegmentation

	# Width of the gate from colourSegmentation function
	_, cs_coords, _ = colourSegmentation(img)
	_, _, PIXEL_WIDTH, _, _ = cs_coords
	
	# Throw error if gate is not detected
	assert (PIXEL_WIDTH is not None), ("colourSegmentation() returned pixel "
		"width of None. Use a different calibration image where the gate can "
		"be detected.")

	# Constant focal length per pixel 
	FOCAL_LENGTH_PER_PIXEL = PIXEL_WIDTH * DISTANCE_TO_GATE / GATE_WIDTH
	return FOCAL_LENGTH_PER_PIXEL


def simpleDist(FOCAL_LENGTH_PER_PIXEL, GATE_WIDTH, pixel_width):
	''' Calculates the distance given the calibration and current pixel width'''
	# Calculate distance to the gate
	if pixel_width is not None:
		return GATE_WIDTH * FOCAL_LENGTH_PER_PIXEL / pixel_width
	else:
		return None
