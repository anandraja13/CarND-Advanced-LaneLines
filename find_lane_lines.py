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

def find_lane_fit(binary_warped, nwindows=10, margin=40, minpix=50):

    ystart = binary_warped.shape[0]//3

    # Take a histogram of the bottom two-thirds of the image
    histogram = np.sum(binary_warped[ystart:,:], axis=0)

    # Create an output image
    out = np.dstack((binary_warped,binary_warped,binary_warped))*255

    # Find the peak for left and right halves of histogram
    midx = np.int(histogram.shape[0]/2)
    leftx_base  = np.argmax(histogram[:midx])
    rightx_base = midx + np.argmax(histogram[midx:])

    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Find non-zero pixels
    nz = binary_warped.nonzero()
    nzy = np.array(nz[0])
    nzx = np.array(nz[1])

    leftx_current  = leftx_base
    rightx_current = rightx_base

    left_idx  = []
    right_idx = []

    rect_corners = []
    for window in range(nwindows):
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - (window  )*window_height
        win_xleft_low   = leftx_current - margin
        win_xleft_high  = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        rect_corners.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

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
            rightx_current = np.int(np.mean(nzx[good_right_idx]))


    left_idx  = np.concatenate(left_idx)
    right_idx = np.concatenate(right_idx)

    leftx  = nzx[left_idx]
    lefty  = nzy[left_idx]
    rightx = nzx[right_idx]
    righty = nzy[right_idx]

    # Fit a second order polynomial
    left_fit  = None
    right_fit = None
    if len(leftx) != 0:
        left_fit  = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_idx, right_idx, nzx, nzy, rect_corners

def update_lane_fit(binary_warped, left_fit, right_fit, nwindows=10, margin=40, minpix=50):

    nz  = binary_warped.nonzero()
    nzy = np.array(nz[0])
    nzx = np.array(nz[1])

    left_idx = ((nzx > (left_fit[0]*(nzy**2) + left_fit[1]*nzy +
    left_fit[2] - margin)) & (nzx < (left_fit[0]*(nzy**2) +
    left_fit[1]*nzy + left_fit[2] + margin)))

    right_idx = ((nzx > (right_fit[0]*(nzy**2) + right_fit[1]*nzy +
    right_fit[2] - margin)) & (nzx < (right_fit[0]*(nzy**2) +
    right_fit[1]*nzy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nzx[left_idx]
    lefty = nzy[left_idx]
    rightx = nzx[right_idx]
    righty = nzy[right_idx]

    # Fit a second order polynomial to each
    left_fit  = None
    right_fit = None
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_idx, right_idx, nzx, nzy

def plot_lane_fit(binary_warped, left_fit, right_fit, left_idx, right_idx, nzx, nzy, rect_corners, nwindows=10, margin=40):

    # Create image for visualization
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255

    # Mark x and y points - left lane in red, right in blue
    out_img[nzy[left_idx], nzx[left_idx]]   = [255, 0, 0]
    out_img[nzy[right_idx], nzx[right_idx]] = [0, 0, 255]

    # Draw rectangles
    for rect in rect_corners:
        cv2.rectangle(out_img, (rect[2],rect[0]), (rect[3],rect[1]), (0,255,0), 2)
        cv2.rectangle(out_img, (rect[4],rect[0]), (rect[5],rect[1]), (0,255,0), 2)

    # Plot line fit
    ploty = np.linspace(0, binary_warped.shape[0]-1,binary_warped.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0,binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)

def plot_lane_fit_poly(binary_warped, left_fit, right_fit, left_idx, right_idx, nzx, nzy, nwindows=10, margin=40):

    # Create image for visualization
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
    win_img = np.zeros_like(out_img)

    # Mark x and y points - left lane in red, right in blue
    out_img[nzy[left_idx], nzx[left_idx]]   = [255, 0, 0]
    out_img[nzy[right_idx], nzx[right_idx]] = [0, 0, 255]

    # Plot line fit
    ploty = np.linspace(0, binary_warped.shape[0]-1,binary_warped.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate search areas
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(win_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(win_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, win_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)

def radius_of_curvature(poly_fit, y_eval):
    radius = ((1 + (2*poly_fit[0]*y_eval + poly_fit[1])**2)**1.5) / np.absolute(2*poly_fit[0])

def compute_radius_and_center_dist(ipm_img, left_fit, right_fit):
    # Extrinsics
    xm_per_pix = 3.7/350 # meters per pixel (the lane width seems to be 350 pixels, and is typically 3.7 meters)
    ym_per_pix = 3.048/180 # meters per pixel ( the dashed line seems to be 180 pixels, and is typically 3.048 meters)

    y_eval = ipm_img.shape[0]-1

    if left_fit is None or right_fit is None:
        return None,None,None

    # Fit new polynomials to x,y in world space
    ploty = np.linspace(0, ipm_img.shape[0]-1, ipm_img.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_radius = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    vehicle_center = ipm_img.shape[1] / 2

    left_x_pix  = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_pix = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x_pix + right_x_pix) / 2

    dist = (vehicle_center - lane_center) * xm_per_pix

    return left_radius, right_radius, dist

def draw_nice_lane(img, orig_warped, seg_img, left_fit, right_fit, M_inv):

    if left_fit is None or right_fit is None:
        return orig_warped

    ipm_img = np.copy(orig_warped)

    warp_zero = np.zeros_like(ipm_img).astype(np.uint8)
    color_warp = warp_zero

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, ipm_img.shape[0]-1, ipm_img.shape[0])
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    poly_pts = np.squeeze(np.int_([pts]))
    cv2.fillPoly(color_warp, np.int32([poly_pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (seg_img.shape[1], seg_img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result

def draw_lane_details(orig_img, radius, dist):

    if radius is None or dist is None:
        return orig_img

    img = np.copy(orig_img)

    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Radius of Curvature: ' + '{:04.3f}'.format(radius) + 'm'

    cv2.putText(img, text, (50,50), font, 2, (0,255,0), 2, cv2.LINE_AA)

    text = 'Distance from Center:' + '{:04.2f}'.format(dist) + 'm'

    cv2.putText(img, text, (50, 100), font, 2, (0,255,0), 2, cv2.LINE_AA)
    return img

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #recent fits
        self.recent_fits = []

    def update(self, fit, idx):
        if fit is not None:
            if self.best_fit is not None:
                # update diffs
                self.diffs = abs(fit - self.best_fit)

            # If fit is too different, reject it
            if (self.diffs[0]>0.002 or self.diffs[1] > 1 or self.diffs[2] > 100) and len(self.recent_fits)>0:
                self.detected = False
            else:
                self.detected = True
                # update current fit
                self.current_fit = fit
                # add fit to recent fits
                self.recent_fits.append(fit)
                if len(self.recent_fits) > 5:
                    self.recent_fits = self.recent_fits[len(self.recent_fits)-5:]
                # update best fit with new fit
                self.best_fit = np.average(self.recent_fits, axis=0)
        else:
            self.detected = False

            if len(self.recent_fits) > 0:
                # move sliding window
                self.recent_fits = self.recent_fits[:len(self.recent_fits)-1]
                # update best fit
                self.best_fit = np.average(self.recent_fits, axis=0)
