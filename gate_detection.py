import cv2 as cv
import numpy as np

#src = cv.imread('red_gate.jpg', 1)
#img = np.copy(src)

vid = cv.VideoCapture(0)

while(True):
	ret, img = vid.read()

	# Blur the image to reduce noise then convert to HSV color space
	blur = cv.GaussianBlur(img, (5, 5), 0)
	hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

	#cv.imshow("Blurred", blur)

	# Red color hue values range from approx 167 to 10 hence 2 separate masks 
	# are required and combined
	red_lower = np.array([0, 90, 89])
	red_upper = np.array([10, 255, 255])
	mask_hsv1 = cv.inRange(hsv, red_lower, red_upper)
	red_lower = np.array([167, 90, 89])
	red_upper = np.array([179, 255, 255])
	mask_hsv2 = cv.inRange(hsv, red_lower, red_upper)

	# Combine the masks
	mask_hsv = mask_hsv1 + mask_hsv2

	# Find the contours in the HSV mask
	contours, _ = cv.findContours(mask_hsv, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	# If at least 1 contour found
	if len(contours) != 0:

		# Find the biggest contour from contours array
		max_cnt = max(contours, key = cv.contourArea)

		# Draw contours on img, biggest one, all of them, color, thickness
		cv.drawContours(img, max_cnt, -1, (0, 255, 0), 2)

		# Find x, y (top left box coords) and w, h (width and height) of the bounding box
		# Plot rectangle, coords, color, thickness
		x, y, w, h = cv.boundingRect(max_cnt)
		cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 1)

	cv.namedWindow("Frame", cv.WINDOW_NORMAL)
	cv.resizeWindow("Frame", 600, 400)
	#cv.imshow("Source", src)
	cv.imshow("Frame", img)

	#cv.waitKey(0)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv.destroyAllWindows()