import numpy as np
import cv2
import glob

# define constants
# board size
BOARD_SIZE = (6,8)
# termination criteria
STOP_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# set image height, width
WIDTH = 4608
HEIGHT= 3456
def main():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_SIZE[1],0:BOARD_SIZE[0]].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('data_new/*.JPG')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (BOARD_SIZE[1],BOARD_SIZE[0]), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), STOP_CRITERIA)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (BOARD_SIZE[1],BOARD_SIZE[0]), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h,  w = HEIGHT, WIDTH
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return mtx, newcameramtx, dist, rvecs, tvecs, roi

if __name__ == "__main__":
    mtx, newcameramtx, dist, rvecs, tvecs, roi = main()
    print("newcameramtx", newcameramtx)
    print("dist", dist)

    # Proceed with test
    test_img = cv2.imread("test/test_real.JPG")
    # undistort
    dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('test/calibresult.png', dst)