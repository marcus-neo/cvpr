import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
file1 = f"./HG/0.jpg"
file2 = f"./HG/9.jpg"

img1_og = cv.imread(file1)
img2_og = cv.imread(file2)

img1 = cv.cvtColor(img1_og, cv.COLOR_BGR2GRAY)  # queryImage
img2 = cv.cvtColor(img2_og, cv.COLOR_BGR2GRAY)  # trainImage

# Initiate SIFT detector
# n_kp = 100
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
good = []
bad = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)
    elif m.distance > 0.99 * n.distance:
        bad.append(m)

src_pts_bad = np.float32([kp1[m.queryIdx].pt for m in bad]).reshape(-1, 1, 2)
dst_pts_bad = np.float32([kp2[m.trainIdx].pt for m in bad]).reshape(-1, 1, 2)
M_bad, mask_bad = cv.findHomography(src_pts_bad, dst_pts_bad, cv.RANSAC, 5.0)
matchesMask_bad = mask_bad.ravel().tolist()
print("good", len(good))
print("bad", len(bad))
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    print("Homography Matrix:")
    print(M)

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2
    )
    dst = cv.perspectiveTransform(pts, M)

    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print(
        "Not enough matches are found - {}/{}".format(
            len(good), MIN_MATCH_COUNT
        )
    )
    matchesMask = None

draw_params_1 = dict(
    matchColor=(0, 255, 0),  # draw matches in green color
    singlePointColor=None,
    matchesMask=matchesMask,  # draw only inliers
    flags=2,
)
draw_params_2 = dict(
    matchColor=(255, 0, 0),  # draw matches in red color
    singlePointColor=None,
    matchesMask=matchesMask_bad,  # draw only outliers
    flags=2,
)
img3 = cv.drawMatches(
    cv.cvtColor(img1_og, cv.COLOR_BGR2RGB),
    kp1,
    cv.cvtColor(img2_og, cv.COLOR_BGR2RGB),
    kp2,
    good,
    None,
    **draw_params_1,
)
img4 = cv.drawMatches(
    cv.cvtColor(img1_og, cv.COLOR_BGR2RGB),
    kp1,
    cv.cvtColor(img2_og, cv.COLOR_BGR2RGB),
    kp2,
    bad,
    None,
    **draw_params_2,
)

plt.imshow(img3, "gray")
plt.axis("off")
plt.savefig("goodcorr.png")
plt.show()


plt.imshow(img4, "gray")
plt.axis("off")
plt.savefig("badcorr.png")
plt.show()


#######################################################################
### APPLY HOMOGRAPHY MATRIX BACK TO IMAGE
img_test = cv.cvtColor(cv.imread(file2), cv.COLOR_BGR2RGB)
img_OG = cv.cvtColor(cv.imread(file1), cv.COLOR_BGR2RGB)
(h, w) = img_test.shape[:2]

transformed_img = cv.warpPerspective(img_test, np.linalg.inv(M), (w, h))
# create figure
fig = plt.figure(figsize=(10, 3))

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
plt.axis("off")
plt.title("H applied to Image 2")

plt.savefig("H_applied.png")
plt.show(block=True)
