import numpy as np
import cv2 as cv
import glob

# NOTE: THIS CALIBRATION HAS BEEN CONDUCTED WITH A RESOLUTION OF (960, 720), 
# BUT THE CAMERA MATRIX WAS SCALED DOWN FOR (480, 360) IMAGES. 
down_resolution = (480, 360)
scale_factor = 0.5

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
objp *= 2	# 2 cm square size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.png')

for fname in images:
    print(fname)
    img = cv.imread(fname)
    # img = cv.resize(img, down_resolution, interpolation=cv.INTER_NEAREST)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,9), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, 
                                gray.shape[::-1], None, None)#, flags=cv.CALIB_ZERO_TANGENT_DIST)
print('Camera matrix: \n', mtx)
print('Distortion matrix: \n', dist)

# img = cv.imread('screenshotFrame4.png')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 
mtx = mtx * scale_factor
mtx[2, 2] = mtx[2, 2] / scale_factor    # should be back to 1
print(mtx)

# undistort
for fname in images:
    img = cv.imread(fname)
    img = cv.resize(img, down_resolution, interpolation=cv.INTER_NEAREST)
    # dist[0][2:6] = 0  # use only k1 and k2
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    dst = cv.undistort(img, mtx, dist)
    cv.imshow('Undistorted', dst)
    cv.waitKey(0)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    tot_error += error

print("mean error: ", tot_error/len(objpoints))

# Save the results
f = open('calibration_results.txt', 'w+')
np.savetxt(f, mtx)
f.write('\n')
np.savetxt(f, dist)
f.write('\n')
f.close()

# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imshow('Undistorted', dst)
# cv.waitKey(0)

# Visualisations of the calibration (in MATLAB):
# http://amroamroamro.github.io/mexopencv/opencv/calibration_demo.html
