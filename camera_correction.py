import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import argparse
import pickle


def calculate_correction(image_path):

    points_per_row = 9
    points_per_col = 6

    nx = points_per_row
    ny = points_per_col

    checker_dims = (nx, ny)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((points_per_col * points_per_row, 3), np.float32)
    objp[:,:2] = np.mgrid[0:points_per_row, 0:points_per_col].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(image_path)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checker_dims, None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners. This section just for plotting
            cv2.drawChessboardCorners(img, checker_dims, corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #plt.imshow(img)
            #plt.show()


    # Now that all images have been calculated, calculate calibration constants.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Store for later use.
    results = {'mtx': mtx, 'dist': dist}
    pickle.dump(results, 'results.p')


# Undistorts an image using the previously saved parameters
def undistort(img, results):

    return cv2.undistort(img, results.mtx, results.dist)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correct image distortion.')
    parser.add_argument('-p', '--path', required=True,
                        help='Relative path the correction images are stored in with a glob matching pattern.')
    args = parser.parse_args()

    calculate_correction(args.path)
