import numpy as np
import cv2

def get_ipm_transform():
    src = np.float32([(536,488),
                    (751,488),
                    (241,687),
                    (1071,687)])
    dst = np.float32([(450,0),
                    (1280-450,0),
                    (450,720),
                    (1280-450,720)])
    M    = cv2.getPerspectiveTransform(src, dst)

    return M

def inverse_perspective_mapping(img):
    M = get_ipm_transform()
    h,w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped

img = cv2.imread('test_images/straight_lines1.jpg')
img_ipm = inverse_perspective_mapping(img)

cv2.imwrite('output_images/ipm_images/straight_lines1.jpg', img_ipm)
