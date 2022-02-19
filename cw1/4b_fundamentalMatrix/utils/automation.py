from time import time
from typing import Optional, Tuple, Union
import numpy as np
import cv2

MIN_MATCH_COUNT = 10

def corrs_automation(
    img1: np.ndarray,
    img2: np.ndarray,
    return_outliers: Optional[bool]=False,
) -> Tuple[
    bool,
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
    """Using OpenCV to find the matches between two images.

    :param img1: The first image.
    :param img2: The second image.
    :param return_outliers: Whether or not to return outlier matches.
    :return: A tuple containing the following:
        - A boolean indicating whether or not the images were matched.
        (And if images were matched,)
        - The sorted source points of inlier matches.
        - The sorted destination points of inlier matches.
        (And if return_bad is True,)
        - The sorted source points of outlier matches.
        - The sorted destination points of outlier matches.
    """
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
    inliers_tup = []
    outliers_tup = [] if return_outliers else None
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            inliers_tup.append((m, m.distance/n.distance))
        elif return_outliers:
            outliers_tup.append((m, m.distance/n.distance))
    
    inliers_tup.sort(key=lambda tup: tup[1])
    inliers, _ = list(zip(*inliers_tup))

    if len(inliers) < MIN_MATCH_COUNT:
        return False, None, None

    # Store the source and destination points of the inlier matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in inliers]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in inliers]).reshape(-1, 1, 2)
    
    src_pts_outliers = None
    dst_pts_outliers = None

    if return_outliers:
        outliers_tup.sort(key=lambda tup: tup[1])
        outliers, _ = list(zip(*outliers_tup))
        # Store the source and destination points of the outlier matches
        src_pts_outliers = np.float32([kp1[m.queryIdx].pt for m in outliers]).reshape(-1, 1, 2)
        dst_pts_outliers = np.float32([kp2[m.trainIdx].pt for m in outliers]).reshape(-1, 1, 2)

    return True, src_pts, dst_pts, src_pts_outliers, dst_pts_outliers


if __name__ == "__main__":
    start_time = time()
    image_1 = "data/LEFT_IMG.JPG"
    image_2 = "data/RIGHT_IMG.JPG"
    ogimg1 = cv2.imread(image_1)
    ogimg2 = cv2.imread(image_2)
    ret, src, dst, _, _ = corrs_automation(
        ogimg1, ogimg2
    )
    end_time = time()
    time_taken = end_time - start_time
    good_points = len(src)
    rate = good_points / time_taken
    print("good points", good_points)
    print("time:", time_taken)
    print("points per second", rate)
