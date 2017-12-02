import numpy as np
import cv2

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # NOTE: This assumes use of cv2.imread to read in images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0)
    else:
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    abs_sobel_scaled = np.uint8(255 * abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(abs_sobel_scaled)
    binary_output[(abs_sobel_scaled>=thresh_min) & (abs_sobel_scaled<=thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # NOTE: This assumes use of cv2.imread to read in images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    mag = np.sqrt( sobelx*sobelx + sobely*sobely )

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    mag_scaled = np.uint8( 255 * mag/np.max(mag) )

    # 5) Create a binary mask where mag thresholds are met
    binary_mask = np.zeros_like(mag_scaled)
    binary_mask[ (mag_scaled>=mag_thresh[0]) & (mag_scaled<=mag_thresh[1]) ] = 1

    # 6) Return this mask as your binary_output image
    return binary_mask


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # NOTE: This assumes use of cv2.imread to read in images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir)
    binary_output[ (dir>=thresh[0]) & (dir<=thresh[1]) ] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hls[:,:,2]

    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s)
    binary_output[ (s>thresh[0]) & (s<=thresh[1]) ] = 1

    # 3) Return a binary image of threshold result
    return binary_output

# Combined segmentation pipeline
def segmentation_pipeline(ipm_img):
    # Compute individual thresholded images
    sobel_abs = abs_sobel_thresh(ipm_img, 'x', 30, 255)
    sobel_mag = mag_thresh(ipm_img, 15, (58, 255))
    sobel_dir = dir_threshold(ipm_img, 15, (0,0.2))
    color_hsl = hls_select(ipm_img, (180,255))

    # Compute combined threshold
    segmented_img = np.zeros_like(sobel_abs)
    segmented_img[((sobel_abs==1) | (sobel_mag==1)) | ((color_hsl==1) & (sobel_dir==1))] = 1
    return segmented_img
