import numpy as np
import cv2


def camera_calibration():
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ...
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = [f"./data_new/{num}.JPG" for num in range(13)]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("1")
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            print(2)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(mtx)
    exit()
    # print("mtx", mtx)
    # print("dist", dist)
    # print("rvecs", rvecs)
    # print("tvecs", tvecs)
    # Undistortion
    test_img = cv2.imread("./data/calibration/test.jpg")
    h, w = test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )
    dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
    print(dst)
    # print(roi)
    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    # print(dst)
    cv2.imwrite("./data/calibration/result.jpg", dst)
    cv2.imshow("img", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_calibration()
