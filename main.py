import cv2 as cv
import time
import numpy as np
import logging

import config
from tello_methods import Tello
from pose_estimation import simpleDistCalibration, simpleDist, estimatePosePnP
from gate_detection import colourSegmentation, gateEdgeDetector, PlotFrames
from controller import runPID, runEdgeCont, runWaypoint
from localisation import (startTracking, trueState, waypointGeneration, 
		waypointUpdate, estimatedState)


def logData(data_dict):
	data_dict_string = str(list(data_dict.values()))
	data_dict_string = data_dict_string.replace("'", '')
	data_dict_string = data_dict_string[1:-1]
	data_dict_string = data_dict_string.replace('None', 'NaN')

	# Log the data to file on a new line
	logging.info('%s', data_dict_string)

# Initialise Tello
tello = Tello()
tello.getBattery()
tello.startVideoCapture(method="H264Decoder")
tello.startStateCapture()
tello.rc()	# reset controls to zeros

fly = True
if fly:
	tello.takeoff()
	time.sleep(8)
	tello.move('up',100)
	time.sleep(5)

# Setup connection with optitrack and the waypoints
optitrack = False
if optitrack:
	streamingClient = startTracking(4, 3)	# Get rigid body ID from Motive
	waypoint = waypointGeneration(streamingClient)

# Initialise the plots object. Constructs and positions the required plots.
plots = PlotFrames(plot_frame=True, plot_blur=False, plot_mask=True)

# Find focal length per pixel using the calibration image
cali_image = cv.imread('Images/cali_100cm.png', 1)
focal_length = simpleDistCalibration(cali_image, config.GATE_WIDTH, 100)

# Initialise variables and flags
screenshot_count = 1		# tracks number of screenshots
track_gate_user = False
controller_on_user = True
log_data_user = False

cs_find_time = 0
time_initialise = time.time()
time_datalog = 0
data_log_rate = 0.05	# Rate at which data is logged to prevent large log files

try:
	while True:
		# Read latest frame from Tello method
		frame = tello.readFrame()

		# Reset variables
		mode = None
		true_state = None
		estimated_state = None
		distance = None

		########################### TRACKING ###################################
		if track_gate_user and tello.new_frame:
			# Find the gate and unpack the returned variables
			cs_frames, cs_coords, cs_features = colourSegmentation(frame, config.RESOLUTION)
			img, blur, mask_hsv = cs_frames
			cs_x, cs_y, cs_width, cs_height, cs_offset = cs_coords
			cs_contour, cs_bounding_box = cs_features
			width = cs_width

			cs_time_since_gate = time.time() - cs_find_time

			# If colourSegmentation does not find gate but it did find the gate
			# recentlyq, then switch to gateEdgeDetector
			if cs_offset is not None:
				# Solve the PnP problem using the 4 contour points
				(cs_contour_sorted, rvec, tvec), (euler, tvec_camera) = estimatePosePnP(cs_contour)

				# Update distance, plots and last gate find time
				distance = np.linalg.norm(tvec_camera)
				mode = 1
				plots.updatePlots(img, blur, mask_hsv, mode, distance=distance, 
					gate_centre=(int(cs_x), int(cs_y)), rvec=rvec, tvec=tvec, 
					contour=cs_contour, bounding_box=cs_bounding_box, sorted_contour=cs_contour_sorted)
				cs_find_time = time.time()

			# Gate not found using colourSegmentation
			elif cs_time_since_gate < 1.5:
				ed_offset, ed_width, num_of_edges = gateEdgeDetector(img, mask_hsv)
				width = ed_width

				mode = 2
				distance = simpleDist(focal_length, config.GATE_WIDTH, width)
				plots.updatePlots(img, blur, mask_hsv, mode, distance=distance)
			else:
				mode = 3
				plots.updatePlots(img, blur, mask_hsv, mode)

			# New frame processed therefore flag back to False
			tello.new_frame = False
			processed_frame = True		# This is just for the controller below

			if log_data_user:
				config.video_out.write(img)
		# Just display the raw frame if user is not tracking gate
		elif not track_gate_user:
			cv.imshow('Raw Frame', frame)

		########################### CONTROLLER #################################
		if track_gate_user and controller_on_user:
			# Update current waypoint and get vector to current waypoint
			if optitrack:
				id_tello = trueState(streamingClient)[0][0]
				pos_tello = trueState(streamingClient)[1][0,:]
				rot_tello = trueState(streamingClient)[2][:,0].T
				euler_tello = trueState(streamingClient)[3][0,:]
				true_state = np.array([id_tello, pos_tello, rot_tello, euler_tello])
				true_position_gate = trueState(streamingClient)[1][1,:]

				relative_vector = waypointUpdate(streamingClient, true_state, waypoint)

			end_time = time.time()
			try:
				dt = end_time - start_time
			except NameError:
				dt = 0.005	# Approximate dt for first iteration

			# Run PID for gate tracking if gate was found (mode 1)
			if mode == 1 and processed_frame:
				estimated_state = estimatedState(rvec, tvec_camera)
				runPID(cs_width, cs_height, cs_offset, tvec_camera, distance, dt, tello, 
					euler, cs_contour_sorted)
			# Run EdgeCont if edges were found within (mode 2)
			elif mode == 2 and processed_frame:
				runEdgeCont(ed_width, ed_offset, num_of_edges, distance, dt, tello)
			# Run Waypoint tracking if optitrack is running
			elif optitrack and processed_frame:
				runWaypoint(true_state, streamingClient, relative_vector, dt, tello)

		start_time = time.time()

		############################### MISC ###################################
		# Log data if time elapsed is more than the log_rate
		if start_time - time_datalog >= data_log_rate and log_data_user and processed_frame:
			config.data_log['time (1)'] = start_time
			if true_state is not None:
				config.data_log['x_true (cm) (2)'] = true_state[1][0]
				config.data_log['y_true (cm) (3)'] = true_state[1][1]
				config.data_log['z_true (cm) (4)'] = true_state[1][2]
				config.data_log['q0_true (5)'] = true_state[2][3]
				config.data_log['q1_true (6)'] = true_state[2][0]
				config.data_log['q2_true (7)'] = true_state[2][1]
				config.data_log['q3_true (8)'] = true_state[2][2]
				# Gate's position
				config.data_log['x_true_gate (25)'] = true_position_gate[0]
				config.data_log['y_true_gate (26)'] = true_position_gate[1]
				config.data_log['z_true_gate (27)'] = true_position_gate[2]
			else:
				config.data_log['x_true (cm) (2)'] = None
				config.data_log['y_true (cm) (3)'] = None
				config.data_log['z_true (cm) (4)'] = None
				config.data_log['q0_true (5)'] = None
				config.data_log['q1_true (6)'] = None
				config.data_log['q2_true (7)'] = None
				config.data_log['q3_true (8)'] = None
			
			if mode == 1:
				config.data_log['x_est (cm) (9)'] = tvec_camera[0]
				config.data_log['y_est (cm) (10)'] = tvec_camera[1]
				config.data_log['z_est (cm) (11)'] = tvec_camera[2]

			config.data_log['mode (15)'] = mode
			config.data_log['distance (cm) (16)'] = distance
			logData(config.data_log)
			time_datalog = time.time()

		processed_frame = False	# Done processing therefore reset flag

		# Display frame and check for user input. Note these user inputs only 
		# work if a frame is displayed and selected.
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			tello.rc()
			break

		elif key == ord('s'):
			# cv.imwrite('Outputs/rawFrame'+str(screenshot_count)+'.png', frame)
			cv.imwrite('Outputs/processedFrame'+str(screenshot_count)+'.png', img)
			# cv.imwrite('Outputs/maskFrame'+str(screenshot_count)+'.png', mask_hsv)
			# cv.imwrite('Outputs/blurFrame'+str(screenshot_count)+'.png', blur)
			screenshot_count += 1

		elif key == ord('t'):
			track_gate_user = not track_gate_user
			tello.rc()
			if track_gate_user:
				cv.destroyWindow('Raw Frame')

		elif key == ord('c'):
			controller_on_user = not controller_on_user
			tello.rc()

		elif key == ord('l'):
			log_data_user = not log_data_user
			if log_data_user:
				print("Logging data and recording video")
			else:
				print("Stopped logging data and recording video")

		elif key == ord('m'):
			tello.rc()
			# Implement function dict
			# https://stackoverflow.com/questions/12495218/using-user-input-to-call-functions
			pass # manual tello command

	config.video_out.release()		# release the video file
	tello.land()
	tello.shutdown()
	cv.destroyAllWindows()

# This except block executes when the main loop above crashes. The crash can
# leave the tello with some control input leading it to crash and suffer damage.
except Exception:
	tello.rc()
	print("Main loop crashed. Exception Handling")
	raise


# implement user control into code,

# If command causes error or "error Not Joystick", then reset rc and land
# or reset rc and continue with rest of script