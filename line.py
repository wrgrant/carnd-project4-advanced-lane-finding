import numpy as np
import collections


class Line:
    def __init__(self, is_print):
        # was the line detected in the last iteration?
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

        # Circular buffers to hold base x-position.
        self.x_pos_buffer = collections.deque(maxlen=50)
        self.c1_buffer = collections.deque(maxlen=50)
        self.c2_buffer = collections.deque(maxlen=50)

        self.reset_buffers()


    def reset_buffers(self):
        # Give some initial values so we don't get NaN's floating around.
        self.x_pos_buffer.append(600)
        self.c1_buffer.append(0)
        self.c2_buffer.append(0)
        print('resetting buffers')




    def update(self, x_pts, y_pts):

        if self.n_bad_frames > 10:
            self.n_good_frames = 0
            self.n_bad_frames = 0
            self.detected = False

        if len(x_pts) == 0 or len(y_pts) == 0:
            self.n_bad_frames += 1
            return

        # At this point, arrays are not bad, so continue on.
        fit = np.polyfit(y_pts, x_pts, 2)
        # Add the coefficients to the circular buffers.
        self.update_fit(fit)
        # self.n_bad_frames = 0
        self.n_good_frames += 1

        # if self.is_print:
        #     print('good frames: {}, bad frames: {}'.format(self.n_good_frames, self.n_bad_frames))




    def update_fit(self, fit):
        c1 = fit[0]
        c2 = fit[1]
        x_pos = fit[2]

        self.worker(self.x_pos_buffer, x_pos, 0.9, 'x')
        self.worker(self.c1_buffer, c1, 0.5, 'c1')
        self.worker(self.c2_buffer, c2, 0.5, 'c2')

        # if self.is_print:
        #     print(self.get_fit())

        # Once we build up the buffer, set detected to true and start being picky.
        if self.n_good_frames > 20:
            self.detected = True



    def worker(self, buffer, value, margin, label):
        if not self.detected:
            # If we're not detected just append whatever comes in.
            buffer.append(value)

        elif self.is_within_margin(value, buffer, margin, label):
            # Get more picky when detected though.
            buffer.append(value)
        else:
            self.n_bad_frames += 1
            print('dropped' + label)




    def is_within_margin(self, new, existing, margin, label):
        existing = np.abs(np.average(existing))
        new = np.abs(new)

        diff = np.abs((new - existing))
        rel_change = diff / (1 + existing)

        if rel_change < margin:
            return True
        else:
            if label is 'x':
                hello = 1
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