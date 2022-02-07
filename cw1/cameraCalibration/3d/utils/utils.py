import numpy as np
import cv2
import random


def color_detector(image: np.ndarray, color: str) -> list:
    """Transforms all of the requested color to white, and all else black.

    :param image: np.ndarray of image, in BGR color format.
    :param color: either "red" or "green"
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == "red":
        # Lower boundary of red
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # Upper boundary of red
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([180, 255, 255])
        lower_mask = cv2.inRange(hsv_img, lower1, upper1)
        upper_mask = cv2.inRange(hsv_img, lower2, upper2)
        full_mask = upper_mask + lower_mask
    elif color == "green":
        lower = np.array([40, 40, 40])
        upper = np.array([70, 255, 255])
        full_mask = cv2.inRange(hsv_img, lower, upper)

    return full_mask


def point_centers(mask: np.ndarray) -> list:
    """Generate point centers of circles of given binary mask of circles.

    :param mask: Binary mask image containing white circles, with dims [H, W].
    :return: Point centers of the circles in [Y, X] coordinates.
    """

    def cont_center(contour: np.ndarray):
        """Generate contour center of a particular contour.

        :param contour: The contour with dimensions [None, 1, 2].
        :return: The contour center in [Y, X] Coordinates.
        """
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [[cY, cX]]

    contours = sorted(
        cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2],
        key=lambda x: cv2.contourArea(x),
    )[-6:]

    xcenters = list(map(cont_center, contours))
    xcenters.sort(key=lambda x: x[0][0], reverse=True)
    return xcenters


def get_points_from_image(image_path: str, dims: str = "2d") -> np.ndarray:
    img = cv2.imread(image_path)
    red_mask = color_detector(img, "red")
    red_centers = point_centers(red_mask)

    green_mask = color_detector(img, "green")
    green_centers = point_centers(green_mask)
    # for index, point in enumerate(green_centers):

    #     temp = point[0][0]
    #     point[0][0] = point[0][1]
    #     point[0][1] = temp
    #     cv2.circle(img, tuple(point[0]), 5, (255, 255, 255))
    #     cv2.putText(
    #         img,
    #         str(index),
    #         (point[0][0] + 15, point[0][1] + 15),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1,
    #         (255, 255, 255),
    #     )
    # for index, point in enumerate(red_centers):
    #     temp = point[0][0]
    #     point[0][0] = point[0][1]
    #     point[0][1] = temp
    #     cv2.circle(img, tuple(point[0]), 5, (0, 0, 0))
    #     cv2.putText(
    #         img,
    #         str(index),
    #         (point[0][0] + 15, point[0][1] + 15),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1,
    #         (0, 0, 0),
    #     )
    # cv2.imwrite(f"testoutput/{random.randint(0,10000)}.jpg", img)
    # exit()
    # if dims == "2d":
    #     return np.array(red_centers, dtype=np.float32)

    return np.array(green_centers + red_centers, dtype=np.float32)
