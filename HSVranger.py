import cv2 as cv
import numpy as np
from tello_methods import Tello
from gate_detection import colourSegmentation, PlotFrames

# This does nothing. It was created because createTrackbar requires a null arg
def nothing(x):
	pass

def contourFinding(img, mask_hsv):

	# Find the contours in the HSV mask. RETR_LIST does not compute heirachy
	# APPROX_SIMPLE saves memory by reducing number of points on straight line
	contours, _ = cv.findContours(mask_hsv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

	# Sort contours from largest to smallest based on area
	contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

	# Define min area for contour and loop through each contour.
	threshold_area = 10
	for cnt in contours:

		# Find current contour's area
		area_cnt = cv.contourArea(cnt)

		# If area is bigger than the min (cnt_area)
		if area_cnt > threshold_area:

			# Find the perimeter of cnt, closed loop contour.
			perimeter_cnt = cv.arcLength(cnt, True)

			# Find max deviation from contour allowed to occur for approximation.
			# Smaller number defines stricter error requirement
			epsilon = 0.015 * perimeter_cnt

			# Finds an approx contour for cnt, epsilon fitting factor and 
			# closed
			approx_cnt = cv.approxPolyDP(cnt, epsilon, True)

			# If quadrilateral shape
			if len(approx_cnt) == 4:

				# Find x, y (top left point) and w, h (width and height) 
				# of the bounding rect. Calculate aspect ratio.
				x, y, w, h = cv.boundingRect(cnt)
				aspect_ratio = float(w) / h

				# If aspect ratio is ~= 1 then its a square (gate)
				#if 0.90 <= aspect_ratio <= 1.10:

				# Draw contours on img, biggest one, all of them, color, thickness
				cv.drawContours(img, approx_cnt, -1, (0, 255, 0), 6)
				print("drawing")

				# Plot rectangle, coords, color, thickness
				cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)

				# Compute the centre of the bounding rect and plot (on img, 
				# using centre coords, radius, color, filled)
				centre = np.array([int(x + w/2), int(y + h/2)])
				cv.circle(img, tuple(centre), 5, (255, 255, 255), -1)

				# Largest cnt meeting requirements found therefore break
				break

# Start taking webcam input
#vid = cv.VideoCapture(0)

tello = Tello()
tello.getBattery()
tello.startVideoCapture()

# Create a window, and create 6 track bars (bar name, output to result frame,
# start value, max value, conduct nothing everytime value is changed)
cv.namedWindow('Result')
cv.createTrackbar('Lower h', 'Result', 10, 180, nothing)
cv.createTrackbar('Upper h', 'Result', 170, 180, nothing)
cv.createTrackbar('Lower s', 'Result', 120, 255, nothing)
cv.createTrackbar('Upper s', 'Result', 255, 255, nothing)
cv.createTrackbar('Lower v', 'Result', 70, 255, nothing)
cv.createTrackbar('Upper v', 'Result', 255, 255, nothing)

# Loop through each frame
while True:
	# ret, img = vid.read()

	# Read latest frame from Tello method
	img = tello.readFrame()

	# Blur and convert to HSV color space
	blur = cv.GaussianBlur(img, (5,5), 0)
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	# Retrieve the position on the trackbar
	lower_h = cv.getTrackbarPos('Lower h', 'Result')
	upper_h = cv.getTrackbarPos('Upper h', 'Result')
	lower_s = cv.getTrackbarPos('Lower s', 'Result')
	upper_s = cv.getTrackbarPos('Upper s', 'Result')
	lower_v = cv.getTrackbarPos('Lower v', 'Result')
	upper_v = cv.getTrackbarPos('Upper v', 'Result')

	# Lower and upper HSV values from the retrieved inputs. Create mask
	hsv_lower = np.array([0, lower_s, lower_v])
	hsv_upper = np.array([lower_h, upper_s, upper_v])
	mask_hsv1 = cv.inRange(hsv, hsv_lower, hsv_upper)
	hsv_lower = np.array([upper_h, lower_s, lower_v])
	hsv_upper = np.array([180, upper_s, upper_v])
	mask_hsv2 = cv.inRange(hsv, hsv_lower, hsv_upper)

	mask = mask_hsv1 + mask_hsv2

	kernel = np.ones((5,5), np.uint8)
	dilation = cv.dilate(mask, kernel)
	contourFinding(img, dilation)

	# mask_hsv is binary hence AND with original image only returns RGB colors in the detected mask range otherwise black.
	#Result = cv.bitwise_and(img, img, mask=mask)
	cv.imshow('Result', img)
	cv.imshow('Mask', mask)
	cv.imshow('Dilated', dilation)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv.destroyAllWindows()
