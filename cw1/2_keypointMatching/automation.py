from typing import Tuple, Union
from cv2 import CV_8SC2
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


def correspondences_automation(
    img1: np.ndarray, img2: np.ndarray
) -> Tuple[
    bool,
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
    """Using OpenCV to find the matches between two images.

    :param filepath1: The first image.
    :param filepath2: The second image.
    :return: A tuple containing the following:
        - A boolean indicating whether or not the images were matched.
        (And if images were matched,)
        - The source points and destination points of good matches.
        - The source points and destination points of bad matches.
    """
    print(img1)
    # Convert the images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # queryImage
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # trainImage

    # Initiate the SIFT detector
    sift = cv2.SIFT_create()

    # Using SIFT, identify the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Initiate the FlannBasedMatcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the best matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    # Also store all the bad matches, i.e. distance > 0.99
    good = []
    bad = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
        elif m.distance > 0.99 * n.distance:
            bad.append(m)

    if len(good) < MIN_MATCH_COUNT:
        return False, None, None, None, None

    # Store the source and destination points of the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Store the source and destination points of the bad matches
    src_pts_bad = np.float32([kp1[m.queryIdx].pt for m in bad]).reshape(
        -1, 1, 2
    )
    dst_pts_bad = np.float32([kp2[m.trainIdx].pt for m in bad]).reshape(
        -1, 1, 2
    )

    return True, src_pts, dst_pts, src_pts_bad, dst_pts_bad


if __name__ == "__main__":
    image_1 = "data/LEFT_IMG.JPG"
    image_2 = "data/RIGHT_IMG.JPG"
    ogimg1 = cv2.imread(image_1)
    ogimg2 = cv2.imread(image_2)
    ret, src, dst, src_bad, dst_bad = correspondences_automation(
        ogimg1, ogimg2
    )
    print(src, dst, src_bad, dst_bad)
