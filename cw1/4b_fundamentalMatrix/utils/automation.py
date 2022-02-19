from time import time
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


def corrs_automation(
    img1: np.ndarray,
    img2: np.ndarray,
    find_h: Optional[bool] = False,
) -> Tuple[
    bool,
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
    """Using OpenCV to find the matches between two images.

    :param img1: The first image.
    :param img2: The second image.
    :param find_h: Using the source points and destination points,
        find the homography matrix.
    :return: A dictionary containing the following:
        - A boolean indicating whether or not the images were matched.
        (And if images were matched,)
        - The source points and destination points of good matches.
        - The source points and destination points of bad matches.
        - The transformed homography matrix (if find_h is True).
    """
    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # queryImage
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # trainImage

    # Initiate the SIFT detector
    sift = cv2.SIFT_create()

    # Using SIFT, identify the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

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
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) < MIN_MATCH_COUNT:
        return False, None, None

    # Store the source and destination points of the good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if not find_h:
        return True, src_pts, dst_pts, None

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1_gray.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2
    )
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params_1 = dict(
        matchesThickness=5,
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )
    img3 = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
        kp1,
        cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
        kp2,
        good,
        None,
        **draw_params_1,
    )

    plt.imshow(img3, "gray")
    plt.axis("off")
    plt.savefig("goodcorr.png")
    
    return True, src_pts, dst_pts, M


if __name__ == "__main__":
    start_time = time()
    image_1 = "data/LEFT_IMG.JPG"
    image_2 = "data/RIGHT_IMG.JPG"
    ogimg1 = cv2.imread(image_1)
    ogimg2 = cv2.imread(image_2)
    ret, src, dst, _ = corrs_automation(
        ogimg1, ogimg2
    )
    end_time = time()
    time_taken = end_time - start_time
    good_points = len(src)
    rate = good_points / time_taken
    print("good points", good_points)
    print("time:", time_taken)
    print("points per second", rate)
