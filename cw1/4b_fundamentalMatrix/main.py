from copy import deepcopy

import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils.automation import corrs_automation

def drawlines(img1src, img2src, lines, pts2src):
    """img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines"""
    img1_copy = deepcopy(img1src)
    img2_copy = deepcopy(img2src)
    r, c, _ = img1src.shape
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt2 in tuple(zip(lines, pts2src))[:10]:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_copy = cv2.line(img1_copy, (x0, y0), (x1, y1), color, 40)
        img2_copy = cv2.circle(img2_copy, tuple(np.int32(pt2[0])), 50, color, -1)
    return img1_copy, img2_copy


img1 = cv2.imread(f"./data/93.jpg")  #queryimage # left image
img2 = cv2.imread(f"./data/91.jpg") #trainimage # right image

ret, src_pts, dst_pts, _, _ = corrs_automation(img1, img2) # Refer to appendix D

[F, mask] = cv2.findFundamentalMat(src_pts,dst_pts,cv2.RANSAC, 5.0)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(
    dst_pts.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, src_pts, dst_pts)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(
    src_pts.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, dst_pts, src_pts)

# Convert to RGB to be saved
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)

# Saving Output Images
plt.figure()
plt.subplot(121), plt.axis('off'), plt.imshow(img6)
plt.subplot(122), plt.axis('off'), plt.imshow(img5)
plt.suptitle("Epipolar Points (Left) and Corresponding Epipolar Lines (Right)", y = 0.75)
plt.savefig("outputs/A_points_to_B_lines.png")


plt.figure()
plt.subplot(121), plt.axis('off'), plt.imshow(img4)
plt.subplot(122), plt.axis('off'), plt.imshow(img3)
plt.suptitle("Epipolar Points (Left) and Corresponding Epipolar Lines (Right)", y = 0.75)
plt.savefig("outputs/B_points_to_A_lines.png")

plt.figure()
plt.subplot(121), plt.axis('off'), plt.imshow(img5)
plt.subplot(122), plt.axis('off'), plt.imshow(img3)
plt.suptitle("Epilines in both images", y = 0.75)
plt.savefig('outputs/epilines.png')