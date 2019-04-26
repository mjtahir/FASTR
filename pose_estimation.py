
def simpleDistCalibration(img, GATE_WIDTH, DISTANCE_TO_GATE):
	'''
	The calibration image is assumed to contain a detectable gate which is upright
	and not angled.
	'''
	from gate_detection import colourSegmentation

	# Width of the gate from colourSegmentation function
	_, _, _, PIXEL_WIDTH = colourSegmentation(img)
	
	assert (PIXEL_WIDTH is not None), "colourSegmentation() returned pixel width \
		of {}".format(PIXEL_WIDTH)

	# Constant focal length per pixel 
	FOCAL_LENGTH_PER_PIXEL = PIXEL_WIDTH * DISTANCE_TO_GATE / GATE_WIDTH
	return FOCAL_LENGTH_PER_PIXEL


def simpleDist(FOCAL_LENGTH_PER_PIXEL, GATE_WIDTH, pixel_width):
	# Calibration image: name.png
	# Distance to gate = 100 cm
	# Width of gate = 52 cm

	# measure 1: 324 cm
	# measure 2: 201 cm


	# Calculate distance to the gate
	try:
		return GATE_WIDTH * FOCAL_LENGTH_PER_PIXEL / pixel_width
	except:
		return None