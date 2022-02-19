# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from utils.automation import corrs_automation
img1 = cv.imread(f"./data/93.jpg",0)  #queryimage # left image
img2 = cv.imread(f"./data/91.jpg",0) #trainimage # right image

# img1 = cv.imread(f"F:/DCIM/147___02/IMG_9099.JPG",0)  #queryimage # left image
# img2 = cv.imread(f"F:/DCIM/147___02/IMG_9102.JPG",0) #trainimage # right image

# a = corrs_automation(img1, img2, True)
# exit()
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print('Length of KPs:')
print(len(kp1), len(kp2))

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

pts1 = []
pts2 = []
good = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append((m, m.distance/n.distance))
        # pts2.append(kp2[m.trainIdx].pt)
        # pts1.append(kp1[m.queryIdx].pt)

good.sort(key=lambda tup: tup[1])
sorted_pts, _ = list(zip(*good))
pts1 = np.int32([kp1[m.queryIdx].pt for m in sorted_pts])
pts2 = np.int32([kp2[m.trainIdx].pt for m in sorted_pts])
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# [F, mask] = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
[F, mask] = cv.findFundamentalMat(pts1,pts2,cv.RANSAC, 5.0)
print('Fundamental Matrix:')
print(F)

# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1] 

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in tuple(zip(lines, pts1src, pts2src))[:10]:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 40)
        # img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 50, color, -1)
    return img1color, img2color

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

plt.figure()
plt.subplot(121), plt.axis('off'), plt.imshow(img6)
plt.subplot(122), plt.axis('off'), plt.imshow(img5)
plt.suptitle("Epipolar Points (Left) and Corresponding Epipolar Lines (Right)", y = 0.75)
plt.savefig("outputs/A_points_to_B_lines.png")

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.figure()
plt.subplot(121), plt.axis('off'), plt.imshow(img4)
plt.subplot(122), plt.axis('off'), plt.imshow(img3)
plt.suptitle("Epipolar Points (Left) and Corresponding Epipolar Lines (Right)", y = 0.75)
plt.savefig("outputs/B_points_to_A_lines.png")

cv.imwrite("./outputs/left.jpg", img5)
cv.imwrite("./outputs/right.jpg", img3)
plt.figure()
plt.subplot(121), plt.axis('off'), plt.imshow(img5)
plt.subplot(122), plt.axis('off'), plt.imshow(img3)
plt.suptitle("Epilines in both images", y = 0.75)
plt.savefig('Q4_epi.png')
plt.show()