from cgitb import small
import math
import numpy as np
import cv2

img = cv2.imread("./data/7.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
output = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite("./output2.jpg", output[1])
cnts = cv2.findContours(output[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[
    -2
]
s1 = 40
s2 = 2500
xcnts = [cnt for cnt in cnts if s1 < cv2.contourArea(cnt) < s2]
xcenters = []
for c in xcnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(
        img,
        f"({cX},{cY})",
        (cX + 10, cY + 10),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1,
        (0, 0, 0),
    )
    xcenters.append((cX, cY))
print(xcnts)
cv2.drawContours(img, xcnts, -1, (34, 177, 76), 2)
print(xcenters)
xcenters.sort(key=lambda a: (a[1], a[0]))
print(xcenters)
cv2.imwrite("./output.jpg", img)

# # print(len(xcnts))

# # P Matrix
# p_matrix = np.array(
#     [
#         [
#             0,
#             1,
#             2,
#             3,
#             0,
#             1,
#             2,
#             3,
#             0,
#             1,
#             2,
#             3,
#             0,
#             1,
#             2,
#             3,
#             0,
#             0,
#             1,
#             1,
#             2,
#             2,
#             3,
#             3,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#         ],
#         [
#             0,
#             0,
#             0,
#             0,
#             1,
#             1,
#             1,
#             1,
#             2,
#             2,
#             2,
#             2,
#             3,
#             3,
#             3,
#             3,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             1,
#             1,
#             2,
#             2,
#             3,
#             3,
#         ],
#         [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             1,
#             2,
#             1,
#             2,
#             1,
#             2,
#             1,
#             2,
#             1,
#             2,
#             1,
#             2,
#             1,
#             2,
#         ],
#     ]
# )

# i_matrix = np.array(
#     [
#         [
#             410,
#             625,
#             847,
#             1079,
#             398,
#             620,
#             849,
#             1089,
#             384,
#             612,
#             852,
#             1103,
#             367,
#             603,
#             855,
#             1117,
#             304,
#             559,
#             826,
#             1105,
#             148,
#             459,
#             792,
#             1142,
#             290,
#             122,
#             264,
#             83,
#             238,
#             38,
#         ],
#         [
#             848,
#             852,
#             863,
#             875,
#             635,
#             640,
#             647,
#             653,
#             402,
#             405,
#             407,
#             408,
#             151,
#             147,
#             144,
#             141,
#             496,
#             963,
#             971,
#             990,
#             1088,
#             1102,
#             1134,
#             1162,
#             691,
#             777,
#             412,
#             428,
#             106,
#             39,
#         ],
#     ]
# )

# for id in range(p_matrix.shape[1]):
#     r_point = np.r_[p_matrix[:, id], 1]
#     i_point = i_matrix[:, id]
#     z = np.array(np.r_[(r_point * -1), np.zeros(4), (r_point * i_point[0])])
#     z = np.expand_dims(z, axis=0)
#     if id == 0:
#         output = z
#     else:
#         output = np.r_[output, z]


# svd_matrix = np.linalg.svd(output)
# u, d, v_h = svd_matrix
# m_vect = v_h[-1, :]
# m_mat = np.reshape(m_vect, (3, 4))

# # split into a_matrix, and v_vector
# a_mat = m_mat[:, :-1]
# b_vect = m_mat[:, -1]

# rho = 1 / np.linalg.norm(a_mat[2, :])

# principal_x = (rho * rho) * (np.dot(a_mat[0, :].T, a_mat[2, :]))
# principal_y = (rho * rho) * (np.dot(a_mat[1, :].T, a_mat[2, :]))
# cross_1 = np.cross(a_mat[0, :], a_mat[2, :])
# norm_1 = np.linalg.norm(cross_1)
# cross_2 = np.cross(a_mat[1, :], a_mat[2, :])
# norm_2 = np.linalg.norm(cross_2)
# skew = np.dot(cross_1.T, cross_2) / (norm_1 * norm_2)
# theta = math.acos(skew)
# focal_alpha = rho * rho * norm_1 * math.sin(theta)
# focal_beta = rho * rho * norm_2 * math.sin(theta)
# camera_params = np.array(
#     [
#         [focal_alpha, -focal_alpha / math.tan(theta), principal_x],
#         [0, focal_beta / math.sin(theta), principal_y],
#         [0, 0, 1],
#     ]
# )
# cam_inverse = np.linalg.inv(camera_params)

# rotate_1 = cross_2 / norm_2
# rotate_2 = a_mat[2, :] / np.linalg.norm(a_mat[2, :])
# rotate_3 = np.cross(rotate_1, rotate_2)
# rotate = np.r_[[rotate_1], [rotate_2], [rotate_3]]
# translate = rho * np.dot(cam_inverse, b_vect)
# print(rotate)
# print(translate)