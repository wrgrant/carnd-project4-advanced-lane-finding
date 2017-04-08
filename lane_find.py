import numpy as np
import cv2
import matplotlib.pyplot as plt
import warp
import line
import myplot
import lane_pixels




# Incoming image  (warped to birds-eye and color pixels found).
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
left_line_obj = line.Line(True, 'left')
right_line_obj = line.Line(False, 'right')




# Number of windows to use over height of image for moving window lane finding.
n_windows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50




def process(warped, orig_img):
    global img
    img = lane_pixels.find_white_and_yellow(warped)

    pre_process()

    if not right_line_obj.detected:
        find_lines_initial()
        fit_curves()
        # plot_window_find_initial()

    else:
        find_lines_update()
        fit_curves()
        # plot_line_find_update()

    out_img = overlay_binary_pixels(img, orig_img)
    out_img = overlay_line_fit(out_img)
    out_img = add_info_overlay(out_img)
    return out_img




def pre_process():
    # Dump all data in bottom pixels as there are reflections from the car hood.
    global img
    img[-10:-1, :] = 0




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

    # Get the current fits
    left_fit = left_line_obj.get_fit()
    right_fit = right_line_obj.get_fit()

    left_lane_idxs = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - margin)) & (
        nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + margin)))

    right_lane_idxs = (
        (nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - margin)) & (
            nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + margin)))




def fit_curves():
    # Extract left and right line pixel positions and feed into line objects
    # myplot.plot(img)

    leftx = nonzero_x[left_lane_idxs]
    lefty = nonzero_y[left_lane_idxs]
    left_line_obj.update(leftx, lefty)

    rightx = nonzero_x[right_lane_idxs]
    righty = nonzero_y[right_lane_idxs]
    right_line_obj.update(rightx, righty)

    # print('left fit: {}'.format(left_line_obj.get_fit()))
    # print('right fit {}'.format(right_line_obj.get_fit()))




def plot_window_find_initial():
    generate_plot_points()

    out_img[nonzero_y[left_lane_idxs], nonzero_x[left_lane_idxs]] = [255, 0, 0]    # Left =red
    out_img[nonzero_y[right_lane_idxs], nonzero_x[right_lane_idxs]] = [0, 100, 255]  # Right=blue
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if False:
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
    out_img[nonzero_y[right_lane_idxs], nonzero_x[right_lane_idxs]] = [0, 100, 255]

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

    if False:
        plt.show(block=False)
        plt.pause(.00001)
        plt.close()
    else:
        plt.show()




# noinspection PyTypeChecker
def calculate_line_curvature():
    generate_plot_points()
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
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

    return np.average([left_line_obj.get_curvature(),
                       right_line_obj.get_curvature()])




def calculate_center_offset():
    # Find centerpoint of lane lines
    centerpoint = np.average([left_line_obj.eval_at_y_point(720), right_line_obj.eval_at_y_point(720)])

    # Offset in pixels
    offset = midpoint - centerpoint

    # Apply conversion factor to get offset in meters
    xm_per_pix = 3.7 / 850
    return offset * xm_per_pix




def overlay_binary_pixels(binary, orig_img):
    # Create an RGB image from the binary.
    out_r = img * 1
    out_g = img * 1
    out_b = img * 255

    binary = np.dstack((out_r, out_g, out_b))
    binary_warped = warp.warp_from_top_down(binary)

    new = replace_colors(binary_warped, orig_img)
    return new




def replace_colors(new, orig):
    out = np.copy(orig)
    mask = np.nonzero(new)
    out[mask] = new[mask]

    return out




def overlay_line_fit(orig_img):
    generate_plot_points()

    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original camera perspective.
    un_warped = warp.warp_from_top_down(color_warp)

    return cv2.addWeighted(orig_img, 1, un_warped, 0.2, 0)




def add_info_overlay(out_img):
    curvature = calculate_line_curvature()
    offset = calculate_center_offset()

    str = 'radius of curvature {:.2f}m'.format(curvature)
    add_text_helper(str, (20, 100), 2, 5, out_img)

    str = 'center offset {:.2f}m'.format(offset)
    add_text_helper(str, (20, 200), 2, 5, out_img)

    fit = left_line_obj.get_fit()
    str = 'L poly: {:.6f}  {:.4f}  {:.2f}'.format(fit[0], fit[1], fit[2])
    add_text_helper(str, (20, 300), 1, 2, out_img)

    fit = right_line_obj.get_fit()
    str = 'R poly: {:.6f}  {:.4f}  {:.2f}'.format(fit[0], fit[1], fit[2])
    add_text_helper(str, (20, 350), 1, 2, out_img)
    #
    # str = 'L good {}'.format(left_line_obj.this_frame_good)
    # add_text_helper(str, (20, 400), 1, 2, out_img)
    #
    # str = 'R good {}'.format(right_line_obj.this_frame_good)
    # add_text_helper(str, (20, 450), 1, 2, out_img)

    return out_img



def add_text_helper(str, pos, size, thickness, img):
    color = (255, 255, 255)
    cv2.putText(img, str, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
