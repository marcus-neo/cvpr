import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from utils.automation import corrs_automation

MIN_MATCH_COUNT = 10
file1 = f"./data/0.JPG"
file2 = f"./data/2.JPG"

img1_og = cv.imread(file1)
img2_og = cv.imread(file2)

ret, src_pts, dst_pts, M = corrs_automation(
    img1_og, img2_og, find_h=True
)

### APPLY HOMOGRAPHY MATRIX BACK TO IMAGE
img_test = cv.cvtColor(img2_og, cv.COLOR_BGR2RGB)
img_OG = cv.cvtColor(img1_og, cv.COLOR_BGR2RGB)
(h, w) = img_test.shape[:2]

transformed_img = cv.warpPerspective(img_test, np.linalg.inv(M), (w, h))
# create figure
fig = plt.figure(3, figsize=(10, 3))

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

plt.savefig("H_applied.png")
