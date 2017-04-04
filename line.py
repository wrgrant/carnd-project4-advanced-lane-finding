import numpy as np
import collections


class Line:
    def __init__(self):
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
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # Circular buffers to hold base x-position.
        self.x_pos_buffer = collections.deque(maxlen=10)
        self.c1_buffer = collections.deque(maxlen=10)
        self.c2_buffer = collections.deque(maxlen=10)



    def update(self, x_pts, y_pts):


        # Put this in a try-except so it will just not update if bad data comes in.
        try:
            fit = np.polyfit(y_pts, x_pts, 2)

            # Add the coefficients to the circular buffers.
            self.update_fit(fit)

        except:
            hello = 1




    def update_fit(self, fit):
        # Only update any of the parameters if the change is within
        # 10% of the existing average.
        margin = 0.1

        c1 = fit[0]
        c2 = fit[1]
        x_pos = fit[2]

        # if self.is_within_margin(c1, self.c1_buffer, 0.8):
            # print('update c1')
        self.c1_buffer.append(c1)
        # else:
            #

        # if self.is_within_margin(c2, self.c2_buffer, 0.8):
            # print('update c2')
        self.c2_buffer.append(c2)

        if self.is_within_margin(x_pos, self.x_pos_buffer, 0.2):
            # print('update x pos')
            self.x_pos_buffer.append(x_pos)




    def is_within_margin(self, new, existing, margin):
        existing = np.average(existing)

        if (new > existing * (1+margin)) | (new < existing * (1-margin)):
            return False
        else:
            return True




    def get_fit(self):
        x_pos = np.average(self.x_pos_buffer)
        c1 = np.average(self.c1_buffer)
        c2 = np.average(self.c2_buffer)

        return np.array([c1, c2, x_pos])



    def get_x_pos(self):
        return np.average(self.x_pos_buffer)