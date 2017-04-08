import numpy as np
import cv2
import myplot




def find_white_and_yellow(image):
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
    # myplot.plot_double(white_bin, yCrCb[:, :, 0], 'white detect - top percent', 'thresh={}'.format(white_thresh))

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
