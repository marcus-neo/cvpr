import glob
from functools import reduce
import numpy as np
import cv2

from utils import utils, estimator

# REAL_POINTS = np.array(
#     [
#         [0.0, 1.0, 1.0],
#         [0.0, 2.0, 1.0],
#         [0.0, 3.0, 1.0],
#         [0.0, 4.0, 1.0],
#         [0.0, 5.0, 1.0],
#         [0.0, 6.0, 1.0],
#         [1.0, 1.0, 0.0],
#         [1.0, 2.0, 0.0],
#         [1.0, 3.0, 0.0],
#         [1.0, 4.0, 0.0],
#         [1.0, 5.0, 0.0],
#         [1.0, 6.0, 0.0],
#     ],
#     dtype=np.float32,
# )
# REAL_POINTS = np.array(
#     [
#         [0.0, 0.028, 0.028],
#         [0.0, 0.056, 0.028],
#         [0.0, 0.084, 0.028],
#         [0.0, 0.112, 0.028],
#         [0.0, 0.14, 0.028],
#         [0.0, 0.168, 0.028],
#         [0.028, 0.028, 0.0],
#         [0.028, 0.056, 0.0],
#         [0.028, 0.084, 0.0],
#         [0.028, 0.112, 0.0],
#         [0.028, 0.14, 0.0],
#         [0.028, 0.168, 0.0],
#     ],
#     dtype=np.float32,
# )
REAL_POINTS_2D = np.array(
    [
        # [0.0, 0.028, 0.028],
        # [0.0, 0.056, 0.028],
        # [0.0, 0.084, 0.028],
        # [0.0, 0.112, 0.028],
        # [0.0, 0.14, 0.028],
        # [0.0, 0.168, 0.028],
        [0.028, 0.028, 0.0],
        [0.028, 0.056, 0.0],
        [0.028, 0.084, 0.0],
        [0.028, 0.112, 0.0],
        [0.028, 0.14, 0.0],
        [0.028, 0.168, 0.0],
    ],
    dtype=np.float32,
)
REAL_POINTS_ALL = np.array(
    [
        [0.0, 0.028, 0.028],
        [0.0, 0.056, 0.028],
        [0.0, 0.084, 0.028],
        [0.0, 0.112, 0.028],
        [0.0, 0.14, 0.028],
        [0.0, 0.168, 0.028],
        [0.028, 0.028, 0.0],
        [0.028, 0.056, 0.0],
        [0.028, 0.084, 0.0],
        [0.028, 0.112, 0.0],
        [0.028, 0.14, 0.0],
        [0.028, 0.168, 0.0],
    ],
    dtype=np.float32,
)


def main():
    images = glob.glob("data/*.jpg")
    # get image size
    img_size = (cv2.imread(images[1]).shape)[1:3]
    # estimate calibration matrix
    # img_pt = np.array([])
    # obj_pt = np.array([])
    # for index, image in enumerate(images):
    #     img_pt_one = utils.get_points_from_image(image, "all")
    #     if index == 0:
    #         img_pt = img_pt_one
    #         obj_pt = REAL_POINTS_ALL
    #     else:
    #         img_pt = np.r_[img_pt, img_pt_one]
    #         obj_pt = np.r_[obj_pt, REAL_POINTS_ALL]

    img_pt = utils.get_points_from_image(images[0], "all")
    initial_matrix = estimator.estimate_calibration_matrix(
        REAL_POINTS_ALL, img_pt
    )

    print("initial_matrix", initial_matrix)
    img_points = list(
        map(lambda x: utils.get_points_from_image(x, "all"), images)
    )
    obj_points = list(
        map(
            lambda _: REAL_POINTS_ALL,
            range(len(images)),
        )
    )

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        img_size,
        initial_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    print(mtx)
    exit()
    x = cv2.initCameraMatrix2D(obj_points, img_points, img_size)

    all_img_points = img_points = list(
        map(lambda x: utils.get_points_from_image(x, "all"), images)
    )
    obj_points_all = list(
        map(
            lambda _: REAL_POINTS_ALL,
            range(len(images)),
        )
    )
    # cv2.calibrateCameraExtended
    # print(len(obj_points))
    # print(len(img_points))
    # print("hi")
    # print(obj_points)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_all,
        all_img_points,
        img_size,
        x,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    print(mtx)
    # print(green_centers)
    # print(red_centers)


if __name__ == "__main__":
    main()


# green_contours = sorted(
#     cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2],
#     key=lambda x: cv2.contourArea(x),
# )[-6:]
# red_contours = sorted(
#     cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2],
#     key=lambda x: cv2.contourArea(x),
# )[-6:]
# contours = cv2.findContours(full_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
#     -2
# ]

# cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
# cnts_sorted = cnts_sorted[-6:]
# cv2.drawContours(img, cnts_sorted, -1, [0, 0, 0], 10)
# cv2.imwrite("mask.jpg", full_mask)
# cv2.imwrite("contours.jpg", img)
