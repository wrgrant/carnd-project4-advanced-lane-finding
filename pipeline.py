from moviepy.editor import VideoFileClip
import moviepy
import matplotlib.pyplot as plt
import pickle
import cam_correct
import myplot


f = open('results.p', 'rb')
results = pickle.load(f)


def process(img):
    processed = cam_correct.undistort(img, results)
    #myplot.plot(img, processed, 'original', 'undistorted')
    return processed

clip = VideoFileClip("challenge_video.mp4")
corrected = clip.fl_image(process)

corrected.write_videofile('challenge_corrected.mp4', audio=False)


# Un-distort and save as a movie again.


# Perspective transform
