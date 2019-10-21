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

# USE Z, Y DISTANCE FROM PnP AS ERROR FOR LEFT/RIGHT AND UP/DOWN

def initialiseDataLogger():
	f = open('Outputs/data_log.txt', 'w+')
	header_labels = ["Timestamp", "x_true (cm)", "y_true (cm)", "z_true (cm)", 
					 "q0_true", "q1_true", "q2_true", "q3_true", 
					 "x_est (cm)", "y_est (cm)", "z_est (cm)", "phi_est (rad)",
					 "theta_est (rad)", "psi_est (rad)", "Mode", "Distance (cm)"]
	header_string = "#" + ', '.join(header_labels)
	f.write(header_string)
	f.write('\n')	# new line after header

	# The H264 codec is automatically changed to AVC1, results in small file size
	fourcc = cv.VideoWriter_fourcc('H','2','6','4')
	video_fps = 30	# Tello's FPS is 30
	video_out = cv.VideoWriter('Outputs/video.mp4', fourcc, video_fps, config.RESOLUTION)

	return f, video_out


def logData(file_handle, time, true_state, estimated_state, mode, distance):
	# If no optitrack data then set arrays to None
	if true_state is None:
		true_position = ["NaN","NaN","NaN"]
		true_quat = ["NaN","NaN","NaN","NaN"]
	else:
		true_position = true_state[1]
		true_quat = true_state[2]

	# if no state was estimated
	if estimated_state is None:
		estimated_position = ["NaN","NaN","NaN"]
		estimated_euler = ["NaN","NaN","NaN"]
	else:
		estimated_position = estimated_state[0]
		estimated_euler = estimated_state[1]

	if mode is None:
		mode = "NaN"

	if distance is None:
		distance = "NaN"

	# Form a list of the data in the appropriate order as the header_labels
	data = [time] + list(true_position) + list(true_quat) + list(estimated_position) + \
		list(estimated_euler) + [mode, distance]

	# Convert to string
	data_str = str(data)
	data_str = data_str.replace("'", '')	# remove the quote symbols from string
	data_str = data_str[1:-1]	# remove the list brackets at both ends

	# Write to file and add new line for next iteration of data log
	file_handle.write(data_str)
	file_handle.write('\n')

# Initialise Tello
tello = Tello()
tello.getBattery()
tello.startVideoCapture(method="H264Decoder")
tello.startStateCapture()
tello.rc()	# reset controls to zeros

fly = False
if fly:
	tello.takeoff()
	time.sleep(8)
	# tello.move('up',100)
	# time.sleep(5)

# Setup connection with optitrack and the waypoints
optitrack = False
if optitrack:
	streamingClient = startTracking(14)	# Get rigid body ID from Motive
	waypoint = waypointGeneration(streamingClient)

log_file, video_out = initialiseDataLogger()

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
				(cs_contour_sorted, rvec, tvec), euler = estimatePosePnP(cs_contour)

				# Update distance, plots and last gate find time
				distance = tvec[2, 0]	# 3rd value in array (perpendicular)
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
				video_out.write(img)
		# Just display the raw frame if user is not tracking gate
		elif not track_gate_user:
			cv.imshow('Raw Frame', frame)

		########################### CONTROLLER #################################
		if track_gate_user and controller_on_user:
			# Update current waypoint and get vector to current waypoint
			if optitrack:
				true_state = trueState(streamingClient)
				relative_vector = waypointUpdate(streamingClient, true_state, waypoint)

			end_time = time.time()
			try:
				dt = end_time - start_time
			except NameError:
				dt = 0.005	# Approximate dt for first iteration

			# Run PID for gate tracking if gate was found (mode 1)
			if mode == 1 and processed_frame:
				estimated_state = estimatedState(rvec, tvec)
				runPID(cs_width, cs_height, cs_offset, tvec, distance, dt, tello, 
					euler, cs_contour_sorted)
			# Run EdgeCont if edges were found within (mode 2)
			elif mode == 2 and processed_frame:
				runEdgeCont(ed_width, ed_offset, num_of_edges, distance, dt, tello)
			# Run Waypoint tracking if optitrack is running
			elif optitrack and processed_frame:
				print('about to run waypoint')
				runWaypoint(true_state, streamingClient, relative_vector, dt, tello)

			processed_frame = False	# Done processing there for reset flag
		start_time = time.time()

		############################### MISC ###################################
		# Log data if time elapsed is more than the log_rate
		if start_time - time_datalog >= data_log_rate and log_data_user:
			logData(log_file, start_time-time_initialise, true_state, 
				estimated_state, mode, distance)
			time_datalog = time.time()

		# Display frame and check for user input. Note these user inputs only 
		# work if a frame is displayed and selected.
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			tello.rc()
			break

		elif key == ord('s'):
			cv.imwrite('Outputs/rawFrame'+str(screenshot_count)+'.png', frame)
			# cv.imwrite('Outputs/processedFrame'+str(screenshot_count)+'.png', img)
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

	log_file.close()		# close the logging file
	video_out.release()		# release the video file
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
# Try lower gain on yaw from PnP

# pt1 = [100, 100, 100]
# pt2 = [150, 0, 100]
# speed = 60
# tello.curve(pt1, pt2, speed)

# # If command causes error or "error Not Joystick", then reset rc and land
# # or reset rc and continue with rest of script