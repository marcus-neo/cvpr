# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread(f"./rectified_1.png",0)  #queryimage # left image
img2 = cv.imread(f"./rectified_2.png",0) #trainimage # right image

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

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
[F, mask] = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
print('Fundamental Matrix:')
print(F)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1] 

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.axis('off'), plt.imshow(img5)
plt.subplot(122), plt.axis('off'), plt.imshow(img3)
plt.suptitle("Epilines in both images (After Rectification)", y = 0.75)
plt.savefig('epi(aft_rect).png')
plt.show()

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1,img2)
plt.imshow(disparity,'gray')
plt.show()

####################################################
# h1, w1 = img1.shape
# h2, w2 = img2.shape
# _, H1, H2 = cv.stereoRectifyUncalibrated(
#     np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
# )

# # Undistort (rectify) the images and save them
# # Adapted from: https://stackoverflow.com/a/62607343
# img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
# img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
# cv.imwrite("rectified_1.png", img1_rectified)
# cv.imwrite("rectified_2.png", img2_rectified)
