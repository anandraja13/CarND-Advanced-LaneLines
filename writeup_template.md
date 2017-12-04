
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calib_img/undistorted_calibration5.jpg "Undistorted"
[image2]: ./test_images/straight_lines1.jpg "Road"
[image3]: ./output_images/ipm_images/straight_lines1.jpg "Road Transformed"
[image4]: ./output_images/seg_image.png "Segmented image"
[image5]: ./output_images/lane_fits.png "Fit Visual"
[image6]: ./output_images/nice_lane_viz.png "Output"
[video1]: ./project_video_track.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the python file located in "./calibrate.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The distortion-corrected image for the above image is like this:

![alt text][image3]

I created an experimental IPython notebook "./get_ipm.ipynb" to accomplish this step.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I initially used a combination of color and gradient thresholds to generate a binary image. After some experiments, I settled on using just color thresholds (using both lab and hsl color spaces). I created the experimental notebook located at "./Lane Segmentation Pipeline.ipynb" to play with the thresholds and select a final pipeline. The final segmentation pipeline after these experiments is located in "./segment_image.py" as part of the `segmentation_pipeline` function. Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `inverse_perspective_mapping()`, which appears at the top of the file `ipm.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `inverse_perspective_mapping()` function takes as inputs an image (`img`). It uses the function `get_ipm_transform` with chosen source (`src`) and destination (`dst`) points to derive the transform.  I chose the hardcode the source and destination points by manually selecting points in an image with straight lane lines:

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 536, 488      | 450, 0        |
| 751, 488      | 1280-450, 0   |
| 241, 687      | 450, 720      |
| 1071, 687     | 1280-450, 720 |

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to find the lane lines, I started with the convolution-based approach, as seen in the `find_window_centroids()` function in "./find_lane_lines.py". After some experiments, I settled on the approach using the basic histogram, as seen in the `find_lane_fit()` function. I use the bottom two-thirds of the image to compute a histogram (summing along columns). This is used to estimate initial position for the lane centers, which are updated over 9 windows. A 2nd degree polygon is fit to these points. These experiments are demonstrated in "./Lane Finding Pipeline.ipynb". A sample result is shown here:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I prototyped this in "./Drawing and Radius of Curvature Pipeline.ipynb" and finally implemented it in the function `compute_radius_and_center_dist()` in `find_lane_lines.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I prototyped this step in implemented this step in "./Drawing and Radius of Curvature Pipeline.ipynb" and finally implemented it in the function `draw_nice_lane()` in `find_lane_lines.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [https://github.com/anandraja13/CarND-Advanced-LaneLines/blob/master/project_video_track.mp4](./project_video_track.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I noticed that the use of Sobel gradients was very noisy. I wasn't able to reliably get the thresholds right. Also, a more robust technique for finding lane lines like RANSAC based polynomial fitting can be used. The segmentation pipeline is finicky to requiring a good, clean road with very well marked lanes.  
