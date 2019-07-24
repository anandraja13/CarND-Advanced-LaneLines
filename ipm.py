import numpy as np
import cv2

def get_ipm_transform():
    """Compute inverse-perspective mapping transformation matrix"""
    
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

# Transform image to IPM
def inverse_perspective_mapping(img):
    """Transform original image to inverse perspective mapping(IPM)"""
    
    # Get the IPM transformation matrix
    M = get_ipm_transform()
    
    # Warp the image using this transformation matrix
    h,w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    
    return warped

def get_inv_ipm_transform():
    """Compute inverse of inverse-perspective mapping transform transformation matrix"""
    
    src = np.float32([(536,488),
                    (751,488),
                    (241,687),
                    (1071,687)])
    dst = np.float32([(450,0),
                    (1280-450,0),
                    (450,720),
                    (1280-450,720)])

    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    return M_inv


if __name__ == "__main__":
    # Read a test image
    img = cv2.imread('test_images/straight_lines1.jpg')
    
    # Compute the birds-eye-view image
    img_ipm = inverse_perspective_mapping(img)

    # Write the output image
    cv2.imwrite('output_images/ipm_images/straight_lines1.jpg', img_ipm)
