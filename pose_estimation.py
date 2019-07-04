def simpleDistCalibration(img, GATE_WIDTH, DISTANCE_TO_GATE):
	'''
	The calibration image is assumed to contain a detectable gate which is 
	upright and not angled.
	'''
	from gate_detection import colourSegmentation

	# Width of the gate from colourSegmentation function
	_, cs_coords = colourSegmentation(img)
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
