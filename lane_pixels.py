import numpy as np
import cv2
import myplot
import matplotlib.pyplot as plt




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
    # ksize = 11 # Choose a larger odd number to smooth gradient measurements
    yCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    yel_mask, yel_thresh = get_yellow_mask(yCrCb)
    white_mask, white_thresh = get_white_mask(yCrCb)

    yel_bin = np.zeros_like(image[:, :, 0])
    yel_bin[yel_mask] = 1

    white_bin = np.zeros_like(image[:, :, 0])
    white_bin[white_mask] = 1

    out_bin = yel_bin | white_bin

    # myplot.plot(out_bin)
    # myplot.plot_double(yel_bin, yCrCb[:, :, 2], 'yellow detect - bottom percent', 'thresh={}'.format(yel_thresh))
    myplot.plot_double(white_bin, yCrCb[:, :, 0], 'white detect - top percent', 'thresh={}'.format(white_thresh))


    #myplot.timed_plot_double(image, combined, 'original image', 'thresholded image')

    return out_bin




def get_yellow_mask(yCrCb):
    cb_chan = yCrCb[:, :, 2]
    # myplot.plot(cb_chan, 'yellow detect - bottom percent')

    max = np.max(cb_chan)
    min = np.min(cb_chan)

    range = max - min
    bottom_percent = range * .6
    thresh = min + bottom_percent
    mask = cb_chan < thresh

    return mask, thresh




def get_white_mask(yCrCb):
    y_chan = yCrCb[:, :, 0]
    # myplot.plot(y_chan, 'white_detect - top percent')

    max_val = np.max(y_chan)
    thresh = max_val * 0.85
    mask = y_chan > thresh

    return mask, thresh




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