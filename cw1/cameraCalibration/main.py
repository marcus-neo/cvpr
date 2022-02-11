# import argparse

from utilities import camera_calib_cv2

cam_mat, dist = camera_calib_cv2.camera_calibration()

camera_calib_cv2.undistort("./data/9.JPG", cam_mat, dist)