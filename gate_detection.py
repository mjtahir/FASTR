import cv2 as cv
import numpy as np
import argparse
from timeit import default_timer as timer

# Read an image and create a copy
#src = cv.imread('red_gate.jpg', 1)
#img = np.copy(src)

# Parse arguments to allow for video file input from terminal
# format: python3 [filename].py --video [video_file_name].mp4
parser = argparse.ArgumentParser(description='Run video, image or webcam.')
parser.add_argument('-v', '--video', help="Path to the (optional) video file.")
args = parser.parse_args()

# Use webcam if no video file input else use the video
if not args.video:
	vid = cv.VideoCapture(0)	# webcam
else:
	vid = cv.VideoCapture(args.video)	# video

#start = timer()
# Loop through each frame
while True:

	# ret (return) is true if img successfully found otherwise false.
	# Type is tuple.
	ret, img = vid.read()

	# Terminates video if img is empty (e.g. end of video file)
	if img is None:
		break

	# Resize image to reduce resolution for quicker processing
	# Blur the image to reduce noise then convert to HSV colour space
	img = cv.resize(img, None, fx=0.4, fy=0.4, interpolation = cv.INTER_NEAREST)
	blur = cv.GaussianBlur(img, (5, 5), 0)
	hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

	# Red color hue values range from approx 167 to 10 hence 2 separate masks 
	# are required and combined
	red_lower = np.array([0, 120, 70])
	red_upper = np.array([10, 255, 255])
	mask_hsv1 = cv.inRange(hsv, red_lower, red_upper)
	red_lower = np.array([170, 120, 70])
	red_upper = np.array([180, 255, 255])
	mask_hsv2 = cv.inRange(hsv, red_lower, red_upper)

	# Combine the masks
	mask_hsv = mask_hsv1 + mask_hsv2

	# Find the contours in the HSV mask. RETR_LIST does not compute heirachy
	# APPROX_SIMPLE saves memory by reducing number of points on straight line
	contours, _ = cv.findContours(mask_hsv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	
	# Sort contours from largest to smallest based on area
	contours = sorted(contours, key = cv.contourArea, reverse = True)

	# If at least 1 contour found
	if len(contours) != 0:
		
		# Define min area for contour and loop through each contour
		cnt_area = 20
		for cnt in contours:
			
			# Find current contour's area
			area_cnt = cv.contourArea(cnt)

			# If area is bigger than the min (cnt_area)
			if area_cnt > cnt_area:

				# Find the perimeter of cnt, closed loop contour.
				perimeter_cnt = cv.arcLength(cnt, True)
				
				# Find max deviation from contour allowed to occur for approx.
				# Smaller number defines stricter error requirement
				epsilon = 0.015 * perimeter_cnt

				# Finds an approx contour for cnt, epsilon fitting factor and 
				# closed
				approx_cnt = cv.approxPolyDP(cnt, epsilon, True)

				# If quadrilateral shape
				if len(approx_cnt) == 4:

					# Find x, y (top left box coords) and w, h (width and height) 
					# of the bounding box. Calculate aspect ratio.
					x, y, w, h = cv.boundingRect(cnt)
					aspect_ratio = float(w) / h

					# If aspect ratio is ~= 1 then its a square (gate) therefore
					# store it as the current largest contour
					#if aspect_ratio >= 0.90 and aspect_ratio <= 1.1:
					cnt_area = area_cnt
			
					# Draw contours on img, biggest one, all of them, color, thickness
					cv.drawContours(img, cnt, -1, (0, 255, 0), 2)
				
					# Plot rectangle, coords, color, thickness
					cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 1)

					# Compute the centre of the bounding box and plot (on img, 
					# using centre coords, radius, color, filled)
					centre = (int(x + w/2), int(y + h/2))
					cv.circle(img, centre, 5, (255, 255, 255), -1)

					# Largest cnt meeting requirements found therefore break
					break

		# Compute distance from centre of image to centre of box

	cv.namedWindow("Frame", cv.WINDOW_NORMAL)
	cv.resizeWindow("Frame", 600, 400)
	cv.imshow("Frame", img)
	#cv.imshow("Source", src)
	#cv.imshow("Blurred", blur)
	#cv.imshow("Mask HSV", mask_hsv)

	#cv.waitKey(0)
	# Set to waitKey(33) for nearly 30 fps
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

#end = timer()
#print(end - start)
vid.release()
cv.destroyAllWindows()