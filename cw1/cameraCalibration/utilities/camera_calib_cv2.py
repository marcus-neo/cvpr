import numpy as np
import cv2

# Defining constants
CHECKERBOARD_SIZE = (8, 6)


def camera_calibration():
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ...
    objp = np.zeros(
        (CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32
    )
    objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = [f"./data/{num}.JPG" for num in range(10)]
    for index, fname in enumerate(images):

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray, (CHECKERBOARD_SIZE[0], CHECKERBOARD_SIZE[1]), None
        )
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(
                img,
                (CHECKERBOARD_SIZE[0], CHECKERBOARD_SIZE[1]),
                corners2,
                ret,
            )
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist


def undistort(image, cam_mat, dist):
    # Undistortion
    test_img = cv2.imread(image)
    h, w = test_img.shape[:2]
    print(h, w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        cam_mat, dist, (w, h), 1, (w, h)
    )
    dst = cv2.undistort(test_img, cam_mat, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    cv2.imwrite("./output/result.jpg", dst)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    undistort(
        "./data/9.JPG",
        np.array(
            [
                [32373.8623, 0, 483.720506],
                [0, 13767.2822, 2301.1021],
                [0, 0, 1],
            ]
        ),
        np.array(
            [[49.1927923, -3236.37433, -0.530820443, -1.75069662, 71948.7919]]
        ),
    )
    # camera_calibration()
