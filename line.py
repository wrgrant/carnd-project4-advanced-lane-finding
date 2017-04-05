import numpy as np
import collections


class Line:
    def __init__(self, is_print):
        # was the line detected in the last iteration?
        self.detected = False
        self.n_bad_frames = 0
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
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.is_print = is_print

        # Circular buffers to hold base x-position.
        self.x_pos_buffer = collections.deque(maxlen=30)
        self.c1_buffer = collections.deque(maxlen=30)
        self.c2_buffer = collections.deque(maxlen=30)

        self.reset_buffers()


    def reset_buffers(self):
        # Give some initial values so we don't get NaN's floating around.
        self.x_pos_buffer.append(600)
        self.c1_buffer.append(0)
        self.c2_buffer.append(0)
        print('resetting buffers')




    def update(self, x_pts, y_pts):
        # Put this in a try-except so it will just not update if bad data comes in.
        try:
            fit = np.polyfit(y_pts, x_pts, 2)

            # Add the coefficients to the circular buffers.

            self.update_fit(fit)

            if self.is_print:
                hello = 1

        except:
            hello = 1




    def update_fit(self, fit):
        c1 = fit[0]
        c2 = fit[1]
        x_pos = fit[2]

        # if self.is_print:
        #     print(self.get_fit())

        # If we're not detected just append whatever comes in.
        if not self.detected:
            self.x_pos_buffer.append(x_pos)

        elif self.is_within_margin(x_pos, self.x_pos_buffer, 0.4):
            # Get more picky when detected though.
            self.x_pos_buffer.append(x_pos)
        # else:
        #     # If incoming value doesn't look within reasonable range,
        #     # return, skipping setting other values as well.
        #     self.n_bad_frames += 1
        #     if self.n_bad_frames > 5:
        #         self.detected = False
        #         self.reset_buffers()
        #     return


        if not self.detected:
            self.c1_buffer.append(c1)
        elif self.is_within_margin(c1, self.c1_buffer, 1):
            self.c1_buffer.append(c1)

        if not self.detected:
            self.c2_buffer.append(c2)

        elif self.is_within_margin(c2, self.c2_buffer, 1):
            self.c2_buffer.append(c2)

        # Once we build up the buffer, set detected to true and start being picky.
        if len(self.x_pos_buffer) > 10:
            self.detected = True

        # if self.n_bad_frames > 10:
        #     self.detected = False
        #     self.n_bad_frames = 0







    def is_within_margin(self, new, existing, margin):
        existing = np.abs(np.average(existing))
        new = np.abs(new)

        # if (new > existing * (1+margin)) | (new < existing * (1-margin)):
        #     return False
        # else:
        #     return True

        diff = np.abs((new - existing))
        rel_change = diff / existing

        if rel_change < margin:
            return True
        else:
            return False




    def get_fit(self):
        x_pos = np.average(self.x_pos_buffer)
        c1 = np.average(self.c1_buffer)
        c2 = np.average(self.c2_buffer)

        return np.array([c1, c2, x_pos])




    def get_x_pos(self):
        return np.average(self.x_pos_buffer)




    def update_curvature(self, value):
        self.radius_of_curvature.append(value)




    def get_curvatuve(self):
        return np.average(self.radius_of_curvature)