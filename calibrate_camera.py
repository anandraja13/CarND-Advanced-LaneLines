import numpy as np
import cv2
import os
import pickle

# prepare object points
nx = 9
ny = 6

# Make a list of calibration images
dir_name = 'camera_cal/'
items = os.listdir(dir_name)

# Fix a directory for writing chessboard images
out_dir = 'output_images/calib_img/'

imgpoints = [] # 2D image points
objpoints = [] # 3D world points

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for name in items:
    fname = dir_name + name
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw the corners, write to image
        print('Corners found')
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imwrite(out_dir+name, img)

        # Append image and world points
        imgpoints.append(corners)
        objpoints.append(objp)

# Calibrate camera with computed image and world points of checkerboard corners
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistort a test image with these parameters
img = cv2.imread(dir_name + items[0])
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Write out undistorted images
cv2.imwrite(out_dir+'undistorted_'+items[0], dst)

# Save calibration info
calib_pickle = {}
calib_pickle["mtx"]   = mtx
calib_pickle["dist"]  = dist
calib_pickle["rvecs"] = rvecs
calib_pickle["tvecs"] = tvecs
pickle.dump(calib_pickle, open('calib_pickle.p', 'wb'))
