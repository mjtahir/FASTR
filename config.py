# This module was created to share information across files.
import numpy as np
import logging
import cv2 as cv

# Gate's width and height in cm
GATE_WIDTH = 50
GATE_HEIGHT = 50

# Down scaled resolution from (960, 720)
RESOLUTION = (480, 360)

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

# Data logger
# Sets up the logger file, write to file, debug level, only include message
logging.basicConfig(filename='Outputs/data_log.txt', filemode='w+', 
	level=logging.INFO, format='%(message)s')

data_log = dict.fromkeys(["time (1)", "x_true (cm) (2)", "y_true (cm) (3)", 
	"z_true (cm) (4)", "q0_true (5)", "q1_true (6)", "q2_true (7)", "q3_true (8)", 
	"x_est (cm) (9)", 
	"y_est (cm) (10)", "z_est (cm) (11)", "phi_est (rad) (12)", "theta_est (rad) (13)", 
	"psi_est (rad) (14)", "mode (15)", "distance (cm) (16)", "PID_edot_lr (17)", 
	"PID_edot_fb (18)",
	"PID_edot_ud (19)", "PID_edot_yaw (20)", "PID_cmd_lr (21)", "PID_cmd_fb (22)", 
	"PID_cmd_ud (23)", "PID_cmd_yaw (24)","x_true_gate (25)","y_true_gate (26)",
	"z_true_gate (27)"])
for current_key in data_log:
	data_log[current_key] = "NaN"	# Set default value to NaN so MATLAB can read

# Create the header in the file
header_string = "#" + ', '.join(list(data_log.keys()))
logging.info('%s', header_string)

# The H264 codec is automatically changed to AVC1, results in small file size
fourcc = cv.VideoWriter_fourcc('H','2','6','4')
video_fps = 30	# Tello's FPS is 30
video_out = cv.VideoWriter('Outputs/video.mp4', fourcc, video_fps, RESOLUTION)