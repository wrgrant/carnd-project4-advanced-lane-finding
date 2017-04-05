from moviepy.editor import VideoFileClip
import pickle
import myplot
import cv2
import numpy as np
import lane_pixels
import lane_find
import cam_correct
import warp
import pprofile




f = open('results.p', 'rb')
results = pickle.load(f)




def undistort_image(img):
    undist = cam_correct.undistort(img, results)
    #myplot.plot_double(img, processed, 'original', 'undistorted')
    return undist




def undistort_movie():
    clip = VideoFileClip('project_video.mp4')
    corrected = clip.fl_image(undistort_image)
    corrected.write_videofile('project_corrected.mp4', audio=False)




# Debugging function for inspecting specific frames of a video.
def load_frame_at_time(time):
    clip = VideoFileClip('challenge_video.mp4')
    frame = clip.get_frame(time)
    undist = cam_correct.undistort(frame, results)
    warped = warp.warp_to_top_down(frame)
    #myplot.plot(frame)
    # myplot.plot_double(frame, warped, 'original', 'top-down')




def process_images(orig_img):
    # orig_img = cam_correct.undistort(orig_img, results)
    img = warp.warp_to_top_down(orig_img)
    img = lane_pixels.find(img)
    img = lane_find.process(img)
    img = cv2.addWeighted(orig_img, 1, img, 0.3, 0)
    img = lane_find.add_info_overlay(img)
    # myplot.timed_plot(img)
    return img




def do_it(input, output):
    clip = VideoFileClip(input)
    clip.start = 20
    clip.duration = 10
    processed = clip.fl_image(process_images)
    processed.write_videofile(output, audio=False)





#cam_correct.undistort_single()
# undistort_movie()
#load_frame_at_time(16)
#do_it(input='harder_challenge_video.mp4', output='pipeline_extra_challenge.mp4')
prof = pprofile.Profile()
with prof():
    do_it(input='project_corrected.mp4', output='./temp_output/project_pipeline.mp4')
# prof.print_stats()

f = open('cachegrind.out', 'w')
prof.callgrind(f)
