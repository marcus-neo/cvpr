# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
from copy import deepcopy
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

# Reuse camera matrix and distortion coefficients from part 3
camera_mat = np.array([
    [2.94953174e3, 0, 2.22933329e3],
    [0, 2.91936279e3, 1.74526873e3],
    [0,0,1]
])

dist = np.array([[0.02299966, -0.17932223, 0.00227571, -0.00475911, 0.26282244]])
img1 = cv2.imread(f"data/5.jpg")  #queryimage # left image
img2 = cv2.imread(f"data/6.jpg") #trainimage # right image

sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
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
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# Find essential matrix. This is required to estimate the rotation and translation coefficients.
E, mask = cv2.findEssentialMat(pts1, pts2, camera_mat, cv2.RANSAC, threshold=5.0)

# Estimate the rotation and translation matrix
points, r_est, t_est, mask_pose = cv2.recoverPose(E, pts1, pts2, camera_mat)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1] 

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img1_copy = deepcopy(img1src)
    img2_copy = deepcopy(img2src)
    r, c, _ = img1src.shape
    img1color = img1_copy
    img2color = img2_copy
    # img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    # img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.figure(1)
plt.subplot(121), plt.axis('off'), plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.axis('off'), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.suptitle("Epilines in both images (Before Rectification)", y = 0.75)
plt.savefig('outputs/epi.png')

h1, w1, _ = img1.shape
h2, w2, _ = img2.shape

r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(camera_mat, dist, camera_mat, dist, (w1, h1), r_est, t_est)

map1x, map1y = cv2.initUndistortRectifyMap(camera_mat, dist, r1, camera_mat, (w1, h1), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_mat, dist, r2, camera_mat, (w2, h2), cv2.CV_32FC1)


# Rectify the images and save them
img1_rectified = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

cv2.imwrite("outputs/rectified_1.png", img1_rectified)
cv2.imwrite("outputs/rectified_2.png", img2_rectified)

img1_rectified_show = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2RGB)
img2_rectified_show = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2RGB)

plt.figure(2)
plt.subplot(121), plt.axis('off'), plt.imshow(img1_rectified_show)
plt.subplot(122), plt.axis('off'), plt.imshow(img2_rectified_show)
plt.suptitle("Rectified images")
plt.savefig('outputs/rectified_images.png')

imgL = img1_rectified # downsample_image(img1_rectified, 3)
imgR = img2_rectified # downsample_image(img2_rectified, 3)
cv2.imshow("", imgL)
cv2.waitKey()
imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgLgray,imgRgray)
plt.figure(3)
plt.imshow(disparity, "gray")
plt.savefig("outputs/depth_map.png")
disparity = disparity.astype(np.float32)
disparity /= 16.0
# Convert disparity map to float32 and divide by 16 as shown in documentation
# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity, q, handleMissingValues=False)

# Get colour of the reprojected points
colors = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

# Get rid of points with value 0 (no depth)
mask_map = disparity > disparity.min()


# Mask colors and points
# print(points_3D)
# print(mask_map)
output_points = points_3D[mask_map]
output_colors = colors[mask_map]


def create_point_cloud_file(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])
    
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, "w") as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        for item in vertices:
            if not np.any(np.isinf(item)):
                f.write(f"{item[0]} {item[1]} {item[2]} {int(item[3])} {int(item[4])} {int(item[5])}\n")


output_file = "outputs/pointCloud.ply"

# Generate point cloud file
create_point_cloud_file(output_points, output_colors, output_file)

print(points_3D)