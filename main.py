import cv2 as cv
import time
import numpy as np

import config
from tello_methods import Tello
from pose_estimation import simpleDistCalibration, simpleDist
from gate_detection import colourSegmentation, gateEdgeDetector, PlotFrames
from controller import runPID, runEdgeCont, runWaypoint
from localisation import (startTracking, trueState, waypointGeneration, 
		waypointUpdate)

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

# Setup connection with OptiTrack and the waypoints
streamingClient = startTracking()
waypoint = waypointGeneration(streamingClient)

# Initialise the plots object. Constructs and positions the required plots.
plots = PlotFrames(plot_frame=True, plot_blur=True, plot_mask=True)

# Find focal length per pixel using the calibration image
cali_image = cv.imread('Images/cali_100cm.png', 1)
focal_length = simpleDistCalibration(cali_image, config.GATE_WIDTH, 100)

# Initialise variables and flags
screenshot_count = 1		# tracks number of screenshots
track_gate_user = False
controller_on_user = True
cs_find_time = 0

try:
	while True:
		# Read latest frame from Tello method
		frame = tello.readFrame()

		if track_gate_user:
			# Find the gate and its distance
			cs_frames, cs_coords = colourSegmentation(frame)
			img, blur, mask_hsv = cs_frames
			_, _, cs_width, cs_height, cs_offset = cs_coords
			width = cs_width

			cs_time_since_gate = time.time() - cs_find_time

			# If colourSegmentation does not find gate but it did find the gate
			# recently, then switch to gateEdgeDetector
			edge_detector_used = False
			if cs_offset is None and cs_time_since_gate < 1.5:
				ed_offset, ed_width, num_of_edges = gateEdgeDetector(img, mask_hsv)
				width = ed_width
				edge_detector_used = True
			elif cs_offset is not None:
				cs_find_time = time.time()	# update last gate find time

			distance = simpleDist(focal_length, config.GATE_WIDTH, width)
			plots.updatePlots(img, blur, mask_hsv, distance)
		else:
			cv.imshow('Raw Frame', frame)

		end_time = time.time()
		try:
			dt = end_time - start_time
		except NameError:
			dt = 0.005	# Approximate dt for first iteration

		if track_gate_user and controller_on_user:
			# Update current waypoint and get vector to current waypoint
			relative_vector = waypointUpdate(streamingClient, waypoint)

			if cs_offset is not None:
				runPID(cs_width, cs_height, cs_offset, distance, dt, tello)
			elif edge_detector_used:
				runEdgeCont(ed_width, ed_offset, num_of_edges, distance, dt, tello)
			else:
				runWaypoint(streamingClient, relative_vector, dt, tello)
		start_time = time.time()

		# Display frame and check for user input. Note these user inputs only 
		# work if a frame is displayed and selected.
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			tello.rc()
			break

		elif key == ord('s'):
			cv.imwrite('screenshotFrame'+str(screenshot_count)+'.png', frame)
			screenshot_count += 1

		elif key == ord('t'):
			track_gate_user = not track_gate_user
			tello.rc()
			if track_gate_user:
				cv.destroyWindow('Raw Frame')

		elif key == ord('c'):
			controller_on_user = not controller_on_user
			tello.rc()

		elif key == ord('m'):
			tello.rc()
			# Implement function dict
			# https://stackoverflow.com/questions/12495218/using-user-input-to-call-functions
			pass # manual tello command

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
# dilation for gate detection, currently flickers <---
# PnP

# pt1 = [100, 100, 100]
# pt2 = [150, 0, 100]
# speed = 60
# tello.curve(pt1, pt2, speed)

# # If command causes error or "error Not Joystick", then reset rc and land
# # or reset rc and continue with rest of script