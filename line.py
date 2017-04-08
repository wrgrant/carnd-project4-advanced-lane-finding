import numpy as np
import collections


# def is_within_margin(new, existing, margin, label):
#     # Return true if no existing data, as there is nothing to compare to.
#     if len(existing) == 0:
#         return True
#
#     existing = np.abs(np.average(existing))
#     new = np.abs(new)
#
#     diff = np.abs((new - existing))
#     rel_change = diff / (1 + existing)
#
#     if rel_change < margin:
#         return True
#     else:
#         if label is 'x':
#             hello = 1
#         return False


class Line:
    def __init__(self, is_print, name):
        # was the line detected in the last iteration?
        self.name = name
        self.detected = False
        self.n_bad_frames = 0
        self.n_good_frames = 0
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = collections.deque(maxlen=10)
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.is_print = is_print
        self.minpix = 500
        self.this_frame_good = False

        self.reset_buffers()


    def reset_buffers(self):
        self.x_pos_buffer = collections.deque(maxlen=10)
        self.c1_buffer = collections.deque(maxlen=10)
        self.c2_buffer = collections.deque(maxlen=10)
        # print('resetting buffers')




    def update(self, x_pts, y_pts):

        if len(x_pts) < self.minpix:
            self.n_bad_frames += 1
            self.this_frame_good = False
            return

        if self.n_bad_frames > 10:
            self.reset_buffers()
            self.n_good_frames = 0
            self.n_bad_frames = 0
            self.detected = False

        # At this point, arrays are not bad, so continue on.
        fit = np.polyfit(y_pts, x_pts, 2)
        # Add the coefficients to the circular buffers.
        self.update_fit(fit)
        # self.n_bad_frames = 0
        self.n_good_frames += 1
        self.this_frame_good = True




    def update_fit(self, fit):
        c1 = fit[0]
        c2 = fit[1]
        x_pos = fit[2]

        self.x_pos_buffer.append(x_pos)
        self.c1_buffer.append(c1)
        self.c2_buffer.append(c2)

        # Once we build up the buffer, set detected to true.
        if self.n_good_frames > 20:
            self.detected = True




    def get_fit(self):
        x_pos = np.average(self.x_pos_buffer)
        c1 = np.average(self.c1_buffer)
        c2 = np.average(self.c2_buffer)

        return np.array([c1, c2, x_pos])




    def get_x_pos(self):
        return np.average(self.x_pos_buffer)




    def update_curvature(self, value):
        self.radius_of_curvature.append(value)




    def get_curvature(self):
        return np.average(self.radius_of_curvature)




    def eval_at_y_point(self, y=720):
        c1 = np.average(self.c1_buffer)
        c2 = np.average(self.c2_buffer)
        c3 = np.average(self.x_pos_buffer)

        return c1*(y**2) + c2*(y) + c3