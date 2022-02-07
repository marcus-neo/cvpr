import math
from copy import deepcopy
import numpy as np
import cv2
from sympy import true

# Define Color Boundaries
RED = (np.array([0, 0, 110]), np.array([100, 100, 255]))
BLUE = (np.array([100, 100, 0]), np.array([255, 255, 100]))
GREEN = (np.array([100, 180, 100]), np.array([150, 255, 150]))
# Read Image
img = cv2.imread("data/201.jpg")


def axis_line(img: np.ndarray, col: tuple) -> tuple:
    # Finding the Red Line (x-y plane and y-z plane separator)
    mask_img = cv2.inRange(img, col[0], col[1])
    line_cont_candidates = cv2.findContours(
        mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )[-2]

    line_contour = max(line_cont_candidates, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, line_contour, -1, [0, 0, 0], 10)
    cv2.imwrite("test.jpg", img)
    left_most = min(line_contour, key=lambda x: x[0][0])[0]
    right_most = max(line_contour, key=lambda x: x[0][0])[0]
    bottom_most = max(line_contour, key=lambda x: x[0][1])[0]
    top_most = min(line_contour, key=lambda x: x[0][1])[0]
    distance_0_1 = np.linalg.norm(top_most - left_most)
    distance_1_2 = np.linalg.norm(right_most - top_most)
    if distance_0_1 >= distance_1_2:
        pt1 = np.array(
            [
                int((bottom_most[0] + left_most[0]) / 2),
                int((bottom_most[1] + left_most[1]) / 2),
            ]
        )
        pt2 = np.array(
            [
                int((top_most[0] + right_most[0]) / 2),
                int((top_most[1] + right_most[1]) / 2),
            ]
        )
    else:
        pt1 = np.array(
            [
                int((top_most[0] + left_most[0]) / 2),
                int((top_most[1] + left_most[1]) / 2),
            ]
        )
        pt2 = np.array(
            [
                int((right_most[0] + bottom_most[0]) / 2),
                int((right_most[1] + bottom_most[1]) / 2),
            ]
        )
    return pt1, pt2


def shortest_distance(
    point1: np.ndarray, point2: np.ndarray, point3: np.ndarray
) -> float:
    """calculate shortest distance from point1 to line created from points2,3."""

    line = point2 - point3
    normal_vect = deepcopy(line)
    temp = normal_vect[1]
    normal_vect[1] = -normal_vect[0]
    normal_vect[0] = temp
    norm_factor = np.linalg.norm(normal_vect)
    unit_norm_vect = normal_vect / norm_factor
    point_to_line = point2 - point1
    output = abs(np.dot(point_to_line, unit_norm_vect))
    return output


def intersect_vects(
    point1: np.ndarray,
    point2: np.ndarray,
    point3: np.ndarray,
    point4: np.ndarray,
) -> np.ndarray:
    line_1_2 = point2 - point1
    line_3_4 = point4 - point3
    line_3_1 = point3 - point1
    mat = np.c_[line_1_2, line_3_4]
    inv_mat = np.linalg.inv(mat)
    vars = np.dot(inv_mat, line_3_1)
    output = point1 + vars[0] * line_1_2
    return output


red_pt1, red_pt2 = axis_line(img, RED)
print("here")
blue_pt1, blue_pt2 = axis_line(img, BLUE)
green_pt1, green_pt2 = axis_line(img, GREEN)

# Finding all Calibration Dots
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
output = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
dot_cont_candidates = cv2.findContours(
    output[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS
)[-2]
s1 = 300
s2 = 2500
dot_conts = [
    cnt for cnt in dot_cont_candidates if s1 < cv2.contourArea(cnt) < s2
]
xcenters = []
for c in dot_conts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    # cv2.putText(
    #     img,
    #     f"({cX},{cY})",
    #     (cX + 10, cY + 10),
    #     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    #     1,
    #     (0, 0, 0),
    # )
    xcenters.append(np.array([cX, cY]))

# Find 6-nearest points on the y-z plane to the x-y/y-z separator

# Exclude all dots not to the left of the x-y/y-z plane separator
y_z_dots = list(
    filter(
        lambda x: (
            x[0] < max(red_pt1[0], red_pt2[0])
            and x[1] < min(blue_pt1[1], blue_pt2[1])
        ),
        xcenters,
    )
)
list.sort(y_z_dots, key=lambda x: shortest_distance(x, red_pt1, red_pt2))
y_z_dots = y_z_dots[:6]
list.sort(y_z_dots, key=lambda x: x[1], reverse=True)
y_z_dots = np.array(y_z_dots)
print("yz", y_z_dots.T)

x_y_dots = list(
    filter(
        lambda x: (
            x[0] > min(red_pt1[0], red_pt2[0])
            and x[1] < min(green_pt1[1], green_pt2[1])
        ),
        xcenters,
    )
)
list.sort(x_y_dots, key=lambda x: shortest_distance(x, red_pt1, red_pt2))
x_y_dots = x_y_dots[:6]
list.sort(x_y_dots, key=lambda x: x[1], reverse=True)
x_y_dots = np.array(x_y_dots)
print("xy", x_y_dots.T)
origin = intersect_vects(red_pt1, red_pt2, green_pt1, green_pt2)
x_z_dots = list(
    filter(
        lambda x: (x[0] > origin[1]),
        xcenters,
    )
)
list.sort(x_z_dots, key=lambda x: abs(np.linalg.norm(x - origin)))

x_z_dots = x_z_dots[0]

i_mat = np.c_[x_y_dots.T, y_z_dots.T, x_z_dots.T]
# for item in list(y_z_dots):
#     cv2.circle(img, item, 3, (0, 255, 0))
# for item in list(x_y_dots):
#     cv2.circle(img, item, 3, (0, 0, 255))

# cv2.circle(img, x_z_dots, 3, (255, 0, 0))
# print(list(dot_conts))
# cv2.drawContours(img, dot_conts, -1, (34, 177, 76), 2)

cv2.line(img, tuple(red_pt1), tuple(red_pt2), color=(255, 255, 255))
cv2.line(img, tuple(blue_pt1), tuple(blue_pt2), color=(255, 255, 255))
cv2.line(img, tuple(green_pt1), tuple(green_pt2), color=(255, 255, 255))
cv2.imwrite("output.jpg", img)


p_mat = np.array(
    [
        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0, 0, 0, 0, 0, 0, 0.03],
        [
            0.03,
            0.06,
            0.09,
            0.12,
            0.15,
            0.18,
            0.03,
            0.06,
            0.09,
            0.12,
            0.15,
            0.18,
            0,
        ],
        [0, 0, 0, 0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    ]
)
for id in range(p_mat.shape[1]):
    r_point = np.r_[p_mat[:, id], 1]
    i_point = i_mat[:, id]
    corr_1 = np.array(
        np.r_[(r_point * -1), np.zeros(4), (r_point * i_point[0])]
    )
    corr_1 = np.expand_dims(corr_1, axis=0)
    corr_2 = np.array(
        np.r_[np.zeros(4), (r_point * -1), (r_point * i_point[1])]
    )
    corr_2 = np.expand_dims(corr_2, axis=0)
    if id == 0:
        output = corr_1
        output = np.r_[output, corr_2]
    else:
        output = np.r_[output, corr_1, corr_2]

svd_matrix = np.linalg.svd(output)
u, d, v_h = svd_matrix
m_vect = v_h[-1, :]
m_mat = np.reshape(m_vect, (3, 4))

# split into a_matrix, and v_vector
a_mat = m_mat[:, :-1]
b_vect = m_mat[:, -1]

rho = 1 / np.linalg.norm(a_mat[2, :])

principal_x = (rho ** 2) * (np.dot(a_mat[0, :].T, a_mat[2, :]))
principal_y = (rho ** 2) * (np.dot(a_mat[1, :].T, a_mat[2, :]))

cross_1 = np.cross(a_mat[0, :], a_mat[2, :])
norm_1 = np.linalg.norm(cross_1)
cross_2 = np.cross(a_mat[1, :], a_mat[2, :])
norm_2 = np.linalg.norm(cross_2)
skew = np.dot(cross_1.T, cross_2) / (norm_1 * norm_2)
theta = math.acos(skew)
focal_alpha = (rho ** 2) * norm_1 * math.sin(theta)
focal_beta = (rho ** 2) * norm_2 * math.sin(theta)

camera_params = np.array(
    [
        [focal_alpha, -focal_alpha / math.tan(theta), principal_x],
        [0, focal_beta / math.sin(theta), principal_y],
        [0, 0, 1],
    ]
)
cam_inverse = np.linalg.inv(camera_params)

rotate_1 = cross_2 / norm_2
rotate_2 = a_mat[2, :] / np.linalg.norm(a_mat[2, :])
rotate_3 = np.cross(rotate_1, rotate_2)
rotate = np.r_[[rotate_1], [rotate_2], [rotate_3]]
translate = rho * np.dot(cam_inverse, b_vect)
print(camera_params)
print(rotate)
print(translate)
