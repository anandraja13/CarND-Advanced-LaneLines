import numpy as np
import cv2

# Read in an image with straight lines
img = cv2.imread('test_images/straight_lines1.jpg')

src = np.float32([(536,488),
                  (751,488),
                  (241,687),
                  (1071,687)])
dst = np.float32([(450,0),
                  (1280-450,0),
                  (450,720),
                  (1280-450,720)])
