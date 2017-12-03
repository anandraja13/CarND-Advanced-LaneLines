import numpy as np
import cv2
import matplotlib.pyplot as plt

def window_mask(width, height, img_ref, center, level):

    output = np.zeros_like(img_ref)

    ymin = int(img_ref.shape[0]-(level+1)*height)
    ymax = int(img_ref.shape[0]-level*height)
    xmin = max(0,int(center-width/2))
    xmax = min(int(center+width/2),img_ref.shape[1])
    output[ymin:ymax,xmin:xmax] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolution

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the windpw template

    ## Sum quarter bottom of image to get slice
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(image.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # Convolve the window into the vertical slice of the image
        ymin = int(image.shape[0]-(level+1)*window_height)
        ymax = int(image.shape[0]-level*window_height)
        image_layer = np.sum(image[ymin:ymax,:], axis=0)
        conv_signal = np.convolve(window, image_layer)

        # Find the best left centroid by using the past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

        # Add found window_centroids
        window_centroids.append((l_center,r_center))

    return window_centroids

def plot_window_centroids(warped, window_centroids, window_width, window_height):

    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0],level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1],level)

            # Add the graphic points from window mask to total pixels found
            l_points[ (l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[ (r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
        warpage = np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results

    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)), np.uint8)

    return output

def find_lane_fit(binary_warped, nwindows=10, margin=40, minpix = 50):

    midy = binary_warped.shape[0]//2

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[midy:,:], axis=0)

    # Create an output image
    out = np.dstack((binary_warped,binary_warped,binary_warped))*255

    # Find the peak for left and right halves of histogram
    midx = np.int(histogram.shape[0]/2)
    leftx_base  = np.argmax(histogram[:midx])
    rightx_base = np.argmax(histogram[midx:]) + midx

    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Find non-zero pixels
    nz = binary_warped.nonzero()
    nzy = np.array(nz[0])
    nzx = np.array(nz[1])

    leftx_current  = leftx_base
    rightx_current = rightx_base

    left_idx  = []
    right_idx = []

    for window in range(nwindows):
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - (window  )*window_height
        win_xleft_low   = leftx_current - margin
        win_xleft_high  = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_idx = ((nzy >= win_y_low) & (nzy < win_y_high) &
        (nzx >= win_xleft_low) &  (nzx < win_xleft_high)).nonzero()[0]

        good_right_idx = ((nzy >= win_y_low) & (nzy < win_y_high) &
        (nzx >= win_xright_low) &  (nzx < win_xright_high)).nonzero()[0]

        left_idx.append(good_left_idx)
        right_idx.append(good_right_idx)

        # Recenter window if new signal is strong enough
        if len(good_left_idx) > minpix:
            leftx_current = np.int(np.mean(nzx[good_left_idx]))
        if len(good_right_idx) > minpix:
            rightx_current = np.int(np.mean(nzy[good_right_idx]))


    left_idx  = np.concatenate(left_idx)
    right_idx = np.concatenate(right_idx)

    leftx  = nzx[left_idx]
    lefty  = nzy[left_idx]
    rightx = nzx[right_idx]
    righty = nzy[right_idx]

    # Fit a second order polynomial
    left_fit  = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_idx, right_idx, nzx, nzy

def plot_lane_fit(binary_warped, left_fit, right_fit, left_idx, right_idx, nzx, nzy, nwindows=9, margin=100):

    # Create image for visualization
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255

    # Mark x and y points - left lane in red, right in blue
    out_img[nzy[left_idx], nzx[left_idx]]   = [255, 0, 0]
    out_img[nzy[right_idx], nzx[right_idx]] = [0, 0, 255]

    # Plot line fit
    ploty = np.linspace(0, binary_warped.shape[0]-1,binary_warped.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0,binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)
