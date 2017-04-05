import numpy as np
import cv2
import myplot




def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    thresh_min, thresh_max = thresh
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output




# Function to return the magnitude of the gradient for a given sobel kernel size
# and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output




# Applies Sobel x and y, then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, 180)):
    # Convert threshold from degrees to radians
    thresh = np.float32(thresh) * np.pi / 180
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output




def find(image):
    # Choose a Sobel kernel size
    ksize = 11 # Choose a larger odd number to smooth gradient measurements

    yel_bin = get_yellow_binary(image)
    white_bin = get_white_binary(image)
    combined = yel_bin | white_bin

    #myplot.plot_double(image, combined, 'original image', 'yellow+white binary image')
    #myplot.timed_plot_double(image, combined, 'original image', 'thresholded image')

    return combined




def get_white_binary(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_chan = hls[:, :, 1]
    s_chan = hls[:, :, 2]

    # White is basically high Lightness and low color Saturation
    l_thresh = 200
    l_mask = l_chan > l_thresh

    s_mask = (s_chan < 50) | (s_chan > 220)

    white_bin = np.zeros_like(l_chan)
    white_bin[l_mask & s_mask] = 1

    return white_bin




def get_yellow_binary(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]

    # Define range for yellow in Hue channel. Remember these are 0-180 degrees.
    h_thresh_low = 10
    h_thresh_high = 26
    h_mask = (h_chan > h_thresh_low) & (h_chan < h_thresh_high)

    # Only want to look for high-saturation yellow though
    s_thresh = 80
    s_mask = s_chan > s_thresh

    yellow_bin = np.zeros_like(h_chan)
    yellow_bin[h_mask & s_mask] = 1

    return yellow_bin




def get_white_old(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    s_chan = hls[:, :, 2]

    s_thresh = (15, 255)
    s_bin = np.zeros_like(s_chan)
    s_mask = (s_chan > s_thresh[0]) & (s_chan <= s_thresh[1])
    return s_mask




def get_gradient_mask(image, ksize):
    # Choose a Sobel kernel size
    ksize = 11 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(200, 255))
    #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, 80))

    combined = np.zeros_like(dir_binary)
    #combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    combined = gradx
    return combined