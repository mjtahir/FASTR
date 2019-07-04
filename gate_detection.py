import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def colourSegmentation(img, down_resolution=(480,360)):

	# Initialise variables
	x = None
	y = None
	w = None
	h = None
	offset = None

	# Resize image to reduce resolution for quicker processing
	# Blur the image to reduce noise then convert to HSV colour space
	img = cv.resize(img, down_resolution, interpolation=cv.INTER_NEAREST)
	img_shape = img.shape[:2]	# (rows, cols)
	img_centre = np.array([int(img_shape[1]/2), int(img_shape[0]/2)]) # (x, y)
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

	# Combine the masks. Outputs 0 or 255.
	mask_hsv = mask_hsv1 + mask_hsv2

	# Find the contours in the HSV mask. RETR_LIST does not compute heirachy
	# APPROX_SIMPLE saves memory by reducing number of points on straight line
	contours, _ = cv.findContours(mask_hsv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

	# Sort contours from largest to smallest based on area
	contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

	# Define min area for contour and loop through each contour.
	# If no contours found, then the loop will not start
	cnt_area = 1000
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

				# Find x, y (top left point) and w, h (width and height) 
				# of the bounding rect. Calculate aspect ratio.
				x, y, w, h = cv.boundingRect(cnt)
				aspect_ratio = float(w) / h

				# If aspect ratio is ~= 1 then its a square (gate) therefore
				# store it as the current largest contour
				#if aspect_ratio >= 0.90 and aspect_ratio <= 1.1:
				cnt_area = area_cnt		# <-- not needed

				# Draw contours on img, biggest one, all of them, color, thickness
				cv.drawContours(img, cnt, -1, (0, 255, 0), 3)

				# Plot rectangle, coords, color, thickness
				cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)

				# Compute the centre of the bounding rect and plot (on img, 
				# using centre coords, radius, color, filled)
				centre = np.array([int(x + w/2), int(y + h/2)])
				cv.circle(img, tuple(centre), 5, (255, 255, 255), -1)

				# Find horizontal and vertical distance from centre of frame to
				# centre of bounding rect. The [-1, 1] corrects coordinates to
				# align camera rotation with aircraft coordinates.
				offset = (centre - img_centre) * np.array([-1, 1])
				cv.arrowedLine(img, tuple(img_centre), (tuple(centre)[0], 
					tuple(img_centre)[1]), (255, 255, 255))
				cv.arrowedLine(img, tuple(img_centre), (tuple(img_centre)[0], 
					tuple(centre)[1]), (255, 255, 255))

				# Largest cnt meeting requirements found therefore break
				break

	return (img, blur, mask_hsv), (x, y, w, h, offset)


def gateEdgeDetector(frame, binary_mask):

	hori_offset = None
	width = None
	num_of_edges = 0

	# Sum each column of the mask and form a single row vector.
	# (image, column-wise, sum, datatype)
	column_sum = cv.reduce(binary_mask, 0, cv.REDUCE_SUM, dtype=cv.CV_32S)
	column_sum = np.ravel(column_sum)	# Change to 1D array

	img_shape = frame.shape[:2]	# (rows, cols)
	img_centre = np.array([int(img_shape[1]/2), int(img_shape[0]/2)]) # (x, y)
	max_height = img_shape[0] * 255

	# Find peaks of the row vector with min height and distance requirements
	peaks, properties = find_peaks(column_sum, height=0.2*max_height,
		distance=0.15*img_shape[1])

	if len(peaks) > 2:

		# Sort all detected peak heights in descending order (with their index)
		peak_heights = properties['peak_heights']
		sort_index = peak_heights.argsort()[::-1]

		# Starting from the highest peak, compare its height to the 2nd highest
		# and use these two peaks if their heights are similar
		for i in range(len(peaks) - 1):
			if (peak_heights[sort_index[i]] - peak_heights[sort_index[i+1]]) <= 0.2*max_height:

				# Similar heights therefore use peak i and i + 1
				peaks = np.array([peaks[sort_index[i]], peaks[sort_index[i+1]]])

				print('Breaking: ' + str(peaks))
				break

			# Last iteration, no suitable peaks therefore set to empty list
			elif i == (len(peaks) - 2):
				print('Too much height difference')
				peaks = []

	elif len(peaks) == 1:

		# One peak detected hence only 1 edge of the gate is detected.
		# This offset is with respect to the detected edge and not the centre
		# of the gate since the centre cannot be determined with only 1 edge.
		hori_offset = peaks - img_centre[0]
		num_of_edges = 1

	# If the original num of peaks was 2, or if more than 2 were found but
	# reduced to 2 with the above if statement.
	if len(peaks) == 2:
		centre = (peaks[0] + peaks[1]) / 2

		# Find horizontal offset and width. -1 corrects coordinates
		hori_offset = (centre - img_centre[0]) * -1
		width = abs(peaks[0] - peaks[1])
		num_of_edges = 2

	# Draw the peak columns on the frame
	print("About to draw: " + str(peaks))
	for column_index in peaks:
		cv.rectangle(frame, (column_index, 0), (column_index, img_shape[1]), 
			(255, 0, 0), 2)

	# Live plot of the peaks, requires Matplotlib.pyplot.
	# plt.plot(column_sum)
	# plt.plot(peaks, column_sum[peaks], "x")
	# plt.ylim(0, max_height)
	# plt.pause(0.00001)
	# plt.cla()

	# num_of_edges returns 0, 1 or 2 (number of post-processed peaks)
	return (hori_offset, width, num_of_edges)


def plotFrame(frame, blur, mask_hsv, distance):

	frame_plot = True
	blur_plot = False
	mask_hsv_plot = False

	if frame_plot:
		# Text on frame. (frame, text, bottom-left position, font, font size,
		# colour, thickness)
		if distance is not None:
			text = 'Distance: {:.4g} cm'.format(distance)
			cv.putText(frame, text, (0, 350), cv.FONT_HERSHEY_PLAIN, 1, 
				(255,255,255), 1)

		cv.namedWindow("Frame", cv.WINDOW_NORMAL)
		cv.resizeWindow("Frame", 480, 360)
		cv.imshow("Frame", frame)

	if blur_plot:
		cv.imshow("Blurred", blur)

	if mask_hsv_plot:
		cv.imshow("Mask HSV", mask_hsv)

################################################################################

if __name__ == "__main__":

	import argparse
	import cProfile
	from timeit import default_timer as timer

	from pose_estimation import simpleDistCalibration, simpleDist
	import config

	# Read an image and create a copy
	# src = cv.imread('Images/cali_100cm.png', 1)
	# img = np.copy(src)

	# Parse arguments to allow for video & image file input from terminal
	# format: python3 [filename].py --video [video_file_name].mp4
	parser = argparse.ArgumentParser(description='Run video, image or webcam.')
	parser.add_argument('-v', '--video', help="Path to the (optional) video file.")
	parser.add_argument('-i', '--image', help="Path to the (optional) image file.")
	args = parser.parse_args()

	# Find focal length per pixel using the calibration image
	cali_image = cv.imread('Images/cali_100cm.png', 1)
	focal_length = simpleDistCalibration(cali_image, config.GATE_WIDTH, 100)

	# If video present use that, otherwise try image, else use webcam
	if args.video:
		vid = cv.VideoCapture(args.video)	# video file
	elif args.image:
		# Read the image, segment it, find distance and plot
		image = cv.imread(args.image, 1)
		cs_frames, cs_coords = colourSegmentation(image)
		img, blur, mask_hsv = cs_frames
		_, _, w, _, _ = cs_coords

		distance = simpleDist(focal_length, config.GATE_WIDTH, w)
		plotFrame(img, blur, mask_hsv, distance)
		cv.waitKey(0)

		# Since image has been analysed, no need for rest of code
		raise SystemExit(0)
	else:
		vid = cv.VideoCapture(0)	# webcam

	screenshot_count = 1
	start = timer()
	while True:

		# ret (return) is true if img successfully found otherwise false.
		# Type is tuple.
		_, src_img = vid.read()

		# Terminates video if img is empty (e.g. end of video file)
		if src_img is None:
			break

		# Find the gate, then its distance then plot
		cs_frames, cs_coords = colourSegmentation(src_img)
		img, blur, mask_hsv = cs_frames
		_, _, w, _, offset = cs_coords
		distance = simpleDist(focal_length, config.GATE_WIDTH, w)

		gateEdgeDetector(img, mask_hsv)
		plotFrame(img, blur, mask_hsv, distance)

		# Set to waitKey(33) for nearly 30 fps
		# Display frame and check for user input.
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		elif key == ord('s'):
			#cv.imwrite('screenshotFrame'+str(screenshot_count)+'.png', frame)
			cv.imwrite('screenshotHSV'+str(screenshot_count)+'.png', mask_hsv)
			screenshot_count += 1

	end = timer()
	print('Time: ', end - start)

	vid.release()
	cv.destroyAllWindows()
