import cv2 as cv
import time
import numpy as np
from timeit import default_timer as timer
from tello_methods import Tello
from pose_estimation import simpleDistCalibration, simpleDist
import config
from gate_detection import colourSegmentation, plotFrame
from controller import run

# Instantiate a tello object and start video capture
tello = Tello()
tello.getBattery()
tello.startVideoCapture()
tello.rc()	# reset controls to zereos
tello.takeoff()
time.sleep(8)
tello.move('up',70)
# time.sleep(3)

screenshot_count = 1
track_gate_user = False
controller_on_user = True

# Find focal length per pixel using the calibration image
cali_image = cv.imread('Images/cali_100cm.png', 1)
focal_length = simpleDistCalibration(cali_image, config.GATE_WIDTH, 100)
while True:

	# Read latest frame from Tello method
	frame = tello.readFrame()

	if track_gate_user:
		# Find the gate and its distance then plot
		cs_frames, cs_coords = colourSegmentation(frame)
		img, blur, mask_hsv = cs_frames
		_, _, w, h, offset = cs_coords
		distance = simpleDist(focal_length, config.GATE_WIDTH, w)

		if controller_on_user:
			end_time = timer()
			try:
				dt = end_time - start_time
			except NameError:
				dt = 0.005	# Approximate dt for first iteration
			run(w, h, offset, distance, dt, tello)
			start_time = timer()

		plotFrame(img, blur, mask_hsv, distance)
	else:
		cv.imshow("Frame", frame)

	# Display frame and check for user input. Note these user inputs only work
	# if a frame is displayed and selected.
	key = cv.waitKey(1) & 0xFF
	if key == ord('q'):
		tello.rc()
		break

	elif key == ord('s'):
		cv.imwrite('screenshotFrame'+str(screenshot_count)+'.png', frame)
		screenshot_count += 1

	elif key == ord('t'):
		track_gate_user = not track_gate_user

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

# implement user control into code, different key for controller on.

#----------------------------------------------
# tello.getBattery()
# tello.rc(fb=0)
# tello.takeoff()
# time.sleep(7)

# # start = time.time()
# # while time.time() < start+3:
# # 	tello.rc(lr=5)

# # start = time.time()
# # while time.time() < start+3:
# # 	tello.rc(fb=5)

# # start = time.time()
# # while time.time() < start+3:
# # 	tello.rc(ud=5)


# tello.rc(fb=100)
# time.sleep(3)

# tello.rc(fb=0)
# time.sleep(3)
# tello.rotate('ccw',185)
# time.sleep(5)
# tello.rc(fb=100)
# time.sleep(3)

# # start = time.time()
# # while time.time() < start+4:
# # 	tello.rc(lr=0,fb=0,ud=0,yaw=30)

# tello.rc(fb=0)
# tello.land()
# tello.shutdown()

# # If command causes error or "error Not Joystick", then reset rc and land
# # or reset rc and continue with rest of script