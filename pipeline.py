from moviepy.editor import VideoFileClip
import pickle
import myplot
import cv2
import numpy as np
import threshold
import lane_find
import cam_correct
import warp




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
    #warped = warp_to_top_down(frame)
    #myplot.plot(frame)
    #myplot.plot_double(frame, warped)




def process_images(orig_img):
    img = warp.warp_to_top_down(orig_img)
    img = threshold.do_it(img)
    img = lane_find.process(img)
    img = cv2.addWeighted(orig_img, 1, img, 0.3, 0)
    img = lane_find.add_info_overlay(img)
    myplot.timed_plot(img)
    return img




def do_it(input, output):
    clip = VideoFileClip(input)
    processed = clip.fl_image(process_images)
    processed.write_videofile(output, audio=False)




# Algorithm...
#undistort_movie()
#load_frame_at_time(16)
do_it(input='harder_challenge_video.mp4', output='pipeline_extra_challenge.mp4')


# Perspective transform
