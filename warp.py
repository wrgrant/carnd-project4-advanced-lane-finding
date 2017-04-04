import cv2
import numpy as np




# Source (in original format
camera_space_transform_points = np.float32([[0, 720], [512, 500], [790, 500], [1280, 720]])
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
