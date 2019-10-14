# This module was created to share constants across files.
import numpy as np

# Gate's width and height in cm
GATE_WIDTH = 50
GATE_HEIGHT = 50

# Body coordinates of gate assuming y right, z down
GATE_COORDS = np.array([[0, -GATE_WIDTH/2, -GATE_HEIGHT/2],	# top-left
						[0, -GATE_WIDTH/2, GATE_HEIGHT/2],	# bottom-left
						[0, GATE_WIDTH/2, GATE_HEIGHT/2],	# bottom-right
						[0, GATE_WIDTH/2, -GATE_HEIGHT/2]])	# top-right

# Load the camera matrix and distortion coefficients from the .txt file
f = open('Images/calibration/calibration_results.txt','r')
CAMERA_MATRIX = np.loadtxt(f, max_rows=3)	# read first 3 rows
DISTORTION = np.loadtxt(f)	# file has reached 3rd line, no need to skip lines
f.close()

# Reshape array from 1D (5) to 2D (1,5). This shape is the same as returned 
# from the calibration. Note the -1 in the tuple allows whatever length needed.
DISTORTION.shape = (1, -1)
