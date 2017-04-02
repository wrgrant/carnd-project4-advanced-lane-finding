from moviepy.editor import VideoFileClip
import pickle
import cam_correct
import myplot
import cv2
import numpy as np
import threshold

f = open('results.p', 'rb')
results = pickle.load(f)


def process(img):
    processed = cam_correct.undistort(img, results)
    #myplot.plot_double(img, processed, 'original', 'undistorted')
    return processed


def undistort_movie():
    clip = VideoFileClip('challenge_video.mp4')
    corrected = clip.fl_image(process)
    corrected.write_videofile('challenge_corrected.mp4', audio=False)


# Debugging function for inspecting specific frames of a video.
def load_frame_at_time(time):
    clip = VideoFileClip('challenge_video.mp4')
    frame = clip.get_frame(time)
    #warped = apply_warp(frame)
    #myplot.plot(frame)
    #myplot.plot_double(frame, warped)


def apply_warp(img):
    src = np.float32([[0, 720], [512, 500], [790, 500], [1280, 720]])
    dst = np.float32([[0, 720], [0, 0], [1280, 0], [1280, 720]])
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    #myplot.plot_double(img, warped)
    return warped


def apply_threshold(img):
    return threshold.do_it(img)



def process_images(img):
    img = apply_warp(img)
    img = apply_threshold(img)
    return img


def do_it(output=''):
    clip = VideoFileClip('challenge_corrected.mp4')
    processed = clip.fl_image(process_images)
    processed.write_videofile(output, audio=False)









# Algorithm...
#undistort_movie()
#load_frame_at_time(16)
do_it(output='transformed.mp4')


# Perspective transform
