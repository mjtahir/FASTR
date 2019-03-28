import cv2 as cv
import numpy as np

# This does nothing. It was created because createTrackbar requires a null arg
def nothing(x):
	pass

# Start taking webcam input
vid = cv.VideoCapture(0)

# Create a window, and create 6 track bars (bar name, output to result frame,
# start value, max value, conduct nothing everytime value is changed)
cv.namedWindow('Result')
cv.createTrackbar('Lower h', 'Result', 0, 180, nothing)
cv.createTrackbar('Upper h', 'Result', 0, 180, nothing)
cv.createTrackbar('Lower s', 'Result', 0, 255, nothing)
cv.createTrackbar('Upper s', 'Result', 0, 255, nothing)
cv.createTrackbar('Lower v', 'Result', 0, 255, nothing)
cv.createTrackbar('Upper v', 'Result', 0, 255, nothing)

# Loop through each frame
while True:
	ret, img = vid.read()

	# Blur and convert to HSV color space
	blur = cv.GaussianBlur(img, (5,5), 0)
	hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

	# Retrieve the position on the trackbar
	lower_h = cv.getTrackbarPos('Lower h', 'Result')
	upper_h = cv.getTrackbarPos('Upper h', 'Result')
	lower_s = cv.getTrackbarPos('Lower s', 'Result')
	upper_s = cv.getTrackbarPos('Upper s', 'Result')
	lower_v = cv.getTrackbarPos('Lower v', 'Result')
	upper_v = cv.getTrackbarPos('Upper v', 'Result')

	# Lower and upper HSV values from the retrieved inputs. Create mask
	hsv_lower = np.array([lower_h, lower_s, lower_v])
	hsv_upper = np.array([upper_h, upper_s, upper_v])
	mask_hsv = cv.inRange(hsv, hsv_lower, hsv_upper)

	# mask_hsv is binary hence AND with original image only returns RGB colors in the detected mask range otherwise black.
	Result = cv.bitwise_and(img, img, mask = mask_hsv)
	cv.imshow('Result', Result)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv.destroyAllWindows()