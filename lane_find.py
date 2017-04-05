import numpy as np
import cv2
import matplotlib.pyplot as plt
import warp
import line




# module variables -------------------

# Incoming image  (warped to birds-eye and thresholded).
img = None
# Image for temporary visualization during development.
out_img = None
# Final outgoing image with the color fill over the
un_warped = None


left_fit_x = None
right_fit_x = None
ploty = None
midpoint = None
nonzero_x = None
nonzero_y = None
left_lane_idxs = None
right_lane_idxs = None

# Left and right line objects.
left_line_obj = line.Line(True)
right_line_obj = line.Line(False)




# Number of windows to use over height of image for moving window lane finding.
n_windows = 9
# Set the width of the windows +/- margin
margin = 300
# Set minimum number of pixels found to recenter window
minpix = 50




def process(in_img):
    global img
    img = in_img
    pre_process()

    if not right_line_obj.detected:
        find_lines_initial()
        fit_curves()
        # plot_window_find_initial()
        # print('doing initial search')

    else:
        find_lines_update()
        fit_curves()
        # plot_line_find_update()

    warp_back_to_original()
    return un_warped





def pre_process():
    # Dump all data in bottom pixels as there are reflections from the car hood.
    global img
    img[-50:-1, :] = 0



# noinspection PyTypeChecker
def find_lines_initial():
    global img
    global out_img
    global n_windows
    global nonzero_x
    global nonzero_y
    global midpoint

    x_dim = img.shape[1]
    midpoint = np.int(x_dim / 2)

    histogram = np.sum(img[int(midpoint/2):,:], axis=0)

    # Find the highest histogram count location for left and right halves of image.
    leftx_bottom = np.argmax(histogram[:midpoint])
    rightx_bottom = np.argmax(histogram[midpoint:]) + midpoint

    # plt.plot(histogram)
    # plt.show()

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255

    # Set height of windows
    window_height = np.int(img.shape[0] / n_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current window centers. To be updated for each window. Initialize with bottom centers
    leftx_current = leftx_bottom
    rightx_current = rightx_bottom

    global left_lane_idxs, right_lane_idxs
    # Create empty lists to receive left and right lane pixel indices
    left_lane_idxs = []
    right_lane_idxs = []

    # Step through the windows one by one...
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Within the windows, identify any nonzero pixels.
        nonzero_left_idxs = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
            nonzero_x < win_xleft_high)).nonzero()[0]
        nonzero_right_idxs = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
            nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_idxs.append(nonzero_left_idxs)
        right_lane_idxs.append(nonzero_right_idxs)

        # If you found > minpix pixels, center window on their mean position
        if len(nonzero_left_idxs) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[nonzero_left_idxs]))
        if len(nonzero_right_idxs) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[nonzero_right_idxs]))

    # Concatenate the arrays of indices
    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)





def find_lines_update():
    global nonzero_x, nonzero_y, left_lane_idxs, right_lane_idxs
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_fit = left_line_obj.get_fit()
    right_fit = right_line_obj.get_fit()

    left_lane_idxs = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - margin)) & (
        nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + margin)))

    right_lane_idxs = (
        (nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - margin)) & (
            nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + margin)))




def fit_curves():
    # Extract left and right line pixel positions and feed into line objects

    leftx = nonzero_x[left_lane_idxs]
    lefty = nonzero_y[left_lane_idxs]
    left_line_obj.update(leftx, lefty)

    rightx = nonzero_x[right_lane_idxs]
    righty = nonzero_y[right_lane_idxs]
    right_line_obj.update(rightx, righty)




def plot_window_find_initial():
    generate_plot_points()

    out_img[nonzero_y[left_lane_idxs], nonzero_x[left_lane_idxs]] = [255, 0, 0]    # Left =red
    out_img[nonzero_y[right_lane_idxs], nonzero_x[right_lane_idxs]] = [0, 0, 255]  # Right=blue
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if True:
        plt.show(block=False)
        plt.pause(.00001)
        plt.close()
    else:
        plt.show()




def generate_plot_points():
    global ploty, left_fitx, right_fitx

    left_fit = left_line_obj.get_fit()
    right_fit = right_line_obj.get_fit()

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]




def plot_line_find_update():
    generate_plot_points()

    # Create an image to draw on and an image to show the selection window
    global out_img
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzero_y[left_lane_idxs], nonzero_x[left_lane_idxs]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_idxs], nonzero_x[right_lane_idxs]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if True:
        plt.show(block=False)
        plt.pause(.00001)
        plt.close()
    else:
        plt.show()




def warp_back_to_original():
    generate_plot_points()
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    global un_warped
    # Warp the blank back to original camera perspective.
    un_warped = warp.warp_from_top_down(color_warp)

    # plt.imshow(un_warped)
    # plt.show()




def add_info_overlay(img):

    curvature = calculate_line_curvature()
    offset = calculate_center_offset()

    color = (255, 255, 255)
    str = 'radius of curvature {:.2f}m'.format(curvature)
    cv2.putText(img, str, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    str = 'center offset {:.2f}m'.format(offset)
    cv2.putText(img, str, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    return img




def calculate_line_curvature():
    generate_plot_points()
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/850 # meters per pixel in x dimension
    y_eval = img.shape[0]

    # We are taking the top-down perspective x points from the polynomial fit
    # and applying this meters-per-pixel offset. So what I need here are
    # the left and right X pixels from the top-down perspective.

    # Fit new polynomials to x,y in original camera perspective.
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    left_line_obj.update_curvature(left_curverad)
    right_line_obj.update_curvature(right_curverad)

    return np.average([left_line_obj.get_curvatuve(),
                       right_line_obj.get_curvatuve()])
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m




def calculate_center_offset():
    # Find centerpoint of lane lines
    centerpoint = np.average([left_line_obj.get_x_pos(), right_line_obj.get_x_pos()])

    # Offset in pixels
    offset = centerpoint - midpoint

    # Apply conversion factor to get offset in meters
    xm_per_pix = 3.7 / 850
    return offset * xm_per_pix
