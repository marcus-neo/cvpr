from copy import deepcopy
import csv
import math

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
FILE_1 = f"./data/0.JPG"
FILE_2 = f"./data/2.JPG"

PERM_LUT = [
    (0.5, 0.9),
    (0.5, 0.84),
    (0.5, 0.86),
    (0.5, 0.88),
    (0.49, 0.9),
    (0.49, 0.84),
    (0.49, 0.86),
    (0.49, 0.88),
    (0.48, 0.9),
    (0.48, 0.84),
    (0.48, 0.86),
    (0.48, 0.88),
    (0.47, 0.9),
    (0.47, 0.84),
    (0.47, 0.86),
    (0.47, 0.88),
    (0.46, 0.9),
    (0.46, 0.84),
    (0.46, 0.85),
    (0.46, 0.84),
    (0.45, 0.84),
    (0.44, 0.82),
    (0.45, 0.82),
    (0.46, 0.82),
    (0.47, 0.82),
]

def threshold_search(sorted_good, sorted_bad, ratio_bad, upper_limit_bad=1.0):
    sorted_bad = [item for item in sorted_bad if item[1]<=upper_limit_bad]
    good_length = len(sorted_good)
    bad_length = len(sorted_bad)
    required_bad = math.ceil(ratio_bad * good_length)
    if required_bad > bad_length:
        raise ValueError(
            "Required Length of bad points is smaller than total number of bad points"
        )
    utilised_good = sorted_good[:good_length-required_bad]
    _, u_good_ratio = list(zip(*utilised_good))
    utilised_bad = sorted_bad[-required_bad:]
    _, u_bad_ratio = list(zip(*utilised_bad))
    average_distance_good = np.mean(u_good_ratio)
    average_distance_bad = np.mean(u_bad_ratio)

    final_utilised = utilised_good
    final_utilised[-required_bad:] = utilised_bad
    final_pts, final_ratios = list(zip(*final_utilised))
    average_distance_final = np.mean(final_ratios)

    return_dict = {
        "total_initial": good_length,
        "total_good": len(utilised_good),
        "average_good": average_distance_good,
        "total_bad": len(utilised_bad),
        "average_bad": average_distance_bad,
        "final_pts": final_pts,
        "average_final": average_distance_final
    }
    return return_dict

if __name__ == "__main__":
    img1_og = cv.imread(FILE_1)
    img2_og = cv.imread(FILE_2)

    img1 = cv.cvtColor(img1_og, cv.COLOR_BGR2GRAY)  # queryImage
    img2 = cv.cvtColor(img2_og, cv.COLOR_BGR2GRAY)  # trainImage

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print("Length of KPs:")
    print(len(kp1), len(kp2))


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    print(len(matches))
    exit()
    good_pts = []
    bad_pts = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_pts.append((m, m.distance/n.distance))
        else:
            bad_pts.append((m, m.distance/n.distance))

    good_pts.sort(key=lambda tup: tup[1])
    bad_pts.sort(key=lambda tup: tup[1])

    if len(good_pts) <= MIN_MATCH_COUNT:
        raise ValueError("Insufficient Matches.")

    for num, (ratio_bad, upper_limit_bad) in enumerate(PERM_LUT):
        print(f"Performing calculations for ratio {ratio_bad} and bad upper limit {upper_limit_bad}.")
        output_dict = threshold_search(good_pts, bad_pts, ratio_bad, upper_limit_bad)
        final_pts = output_dict["final_pts"]
        avg_good = output_dict["average_good"]
        avg_bad = output_dict["average_bad"]
        avg_final = output_dict["average_final"]
        total_good = output_dict["total_good"]
        total_bad = output_dict["total_bad"]
        init_length = output_dict["total_initial"]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in final_pts]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in final_pts]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        #######################################################################
        ### APPLY HOMOGRAPHY MATRIX BACK TO IMAGE
        img_test = cv.cvtColor(cv.imread(FILE_2), cv.COLOR_BGR2RGB)
        img_OG = cv.cvtColor(cv.imread(FILE_1), cv.COLOR_BGR2RGB)
        (h, w) = img_test.shape[:2]

        transformed_img = cv.warpPerspective(img_test, np.linalg.inv(M), (w, h))
        # create figure
        fig = plt.figure(num, figsize=(10, 3))

        # Adds a subplot at the 1st position
        fig.add_subplot(1, 3, 1)

        # showing image
        plt.imshow(img_OG)
        plt.axis("off")
        plt.title("Image 1")
        # Adds a subplot at the 2nd position
        fig.add_subplot(1, 3, 2)

        # showing image
        plt.imshow(img_test)
        plt.axis("off")
        plt.title("Image 2")

        # Adds a subplot at the 3rd position
        fig.add_subplot(1, 3, 3)

        # showing image
        plt.imshow(transformed_img)
        plt.axis('off')
        plt.title("Homography applied to Image 2")
        output_path = f"output/{ratio_bad}_{upper_limit_bad}.png"
        plt.savefig(output_path)
        plt.close()
        with open("result.csv", "a" , newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")
            csv_writer.writerow([ratio_bad, upper_limit_bad, init_length, total_good, total_bad, avg_good, avg_bad, avg_final, output_path])

    print("Completed.")
