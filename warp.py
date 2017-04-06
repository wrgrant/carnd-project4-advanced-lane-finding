import cv2
import numpy as np
import myplot
import cam_correct




# Source (in original format
# camera_space_transform_points = np.float32([[0, 720], [512, 500], [790, 500], [1280, 720]])
camera_space_transform_points = np.float32([[0, 720], [571, 450], [700, 450], [1280, 720]])
warped_space_transform_points = np.float32([[0, 720], [0, 0], [1280, 0], [1280, 720]])




# Functions to warp between camera and top-down view perspectives.
def warp_to_top_down(img):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(camera_space_transform_points, warped_space_transform_points)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    #myplot.plot_double(img, warped)
    return warped


def warp_from_top_down(img):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(warped_space_transform_points, camera_space_transform_points)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped




def examine_single():
    img = cv2.imread('./test_images/straight_lines1.jpg')
    img = img[:, :, [2, 1, 0]]
    # myplot.plot(img)
    img = cam_correct.undistort_single(img)
    warped = warp_to_top_down(img)
    myplot.plot_double(img, warped)




# examine_single()
