import math
import cv2
import numpy as np

# points = np.array(
#     [
#         [0, 6, 5, 1133, 613],
#         [0, 6, 4, 1413, 593],
#         [0, 6, 3, 1645, 565],
#         [0, 6, 2, 1837, 545],
#         [0, 6, 1, 2001, 1197],
#         [0, 5, 5, 1197, 1081],
#         [0, 5, 4, 1457, 1001],
#         [0, 5, 3, 1669, 937],
#         [0, 5, 3, 1845, 885],
#         [0, 5, 1, 1993, 849],
#         [0, 4, 5, 1257, 1497],
#         [0, 4, 4, 1489, 1377],
#         [0, 4, 3, 1685, 1277],
#         [0, 4, 2, 1853, 1197],
#         [0, 4, 1, 1993, 1137],
#         [0, 3, 5, 1301, 1869],
#         [0, 3, 4, 1521, 1709],
#         [0, 3, 3, 1705, 1589],
#         [0, 3, 2, 1857, 1485],
#         [0, 3, 1, 1989, 1401],
#         [0, 2, 5, 1357, 2193],
#         [0, 2, 4, 1561, 2013],
#         [0, 2, 3, 1729, 1873],
#         [0, 2, 2, 1873, 1749],
#         [0, 2, 1, 1997, 1649],
#         [0, 1, 5, 1409, 2485],
#         [0, 1, 4, 1601, 2289],
#         [0, 1, 3, 1757, 2125],
#         [0, 1, 2, 1885, 1993],
#         [0, 1, 1, 2005, 1881],
#         [1, 6, 0, 2377, 529],
#         [2, 6, 0, 2661, 537],
#         [3, 6, 0, 2957, 561],
#         [4, 6, 0, 3273, 577],
#         [5, 6, 0, 3265, 597],
#         [1, 5, 0, 2365, 825],
#         [2, 5, 0, 2633, 849],
#         [3, 5, 0, 2926, 877],
#         [4, 5, 0, 3221, 901],
#         [5, 5, 0, 3549, 941],
#         [1, 4, 0, 2357, 1105],
#         [2, 4, 0, 2617, 1133],
#         [3, 4, 0, 2881, 1169],
#         [4, 4, 0, 3173, 1201],
#         [5, 4, 0, 3489, 1249],
#         [1, 3, 0, 2337, 1361],
#         [2, 3, 0, 2589, 1397],
#         [3, 3, 0, 2853, 1441],
#         [4, 3, 0, 3133, 1489],
#         [5, 3, 0, 3245, 1529],
#         [1, 2, 0, 2333, 1601],
#         [2, 2, 0, 2573, 1637],
#         [3, 2, 0, 2821, 1685],
#         [4, 2, 0, 3089, 1733],
#         [5, 2, 0, 3369, 1785],
#         [1, 1, 0, 2317, 1829],
#         [2, 1, 0, 2553, 1865],
#         [3, 1, 0, 2801, 1917],
#         [4, 1, 0, 3067, 1973],
#         [5, 1, 0, 3325, 2025],
#     ]
# )

points = np.array(
    [
        [0, 6, 2, 1543, 716],
        [0, 5, 5, 1099, 686],
        [0, 4, 4, 1307, 1401],
        [1, 3, 0, 2013, 1844],
        [5, 2, 0, 3243, 2132],
        [3, 1, 0, 2637, 2448],
    ]
)


def homogeneous_matrix(real_points, image_points):
    output = []
    for [u, v], [x, y, z] in zip(image_points, real_points):
        output.append([-x, -y, -z, -1, 0, 0, 0, 0, u * x, u * y, u * z, u])
        output.append([0, 0, 0, 0, -x, -y, -z, -1, v * x, v * y, v * z, v])
    return np.array(output)


def direct_linear_transformation(hom_mat):
    u, s, v_T = np.linalg.svd(hom_mat)
    test_2 = v_T[-1, :]
    return test_2.reshape((3, 4))


def compute_focal_length(f_px, sensor_size):
    f_world = f_px * sensor_size
    return f_world


def main():
    # 1. Real Points (Homogenous Coordinate Form)
    # real_points = np.array(
    #     [[1, 6, 0], [0, 5, 3], [2, 4, 0], [0, 3, 1], [5, 2, 0], [0, 1, 2]]
    # )
    # image_points = np.array(
    #     [
    #         [619, 129],
    #         [399, 49],
    #         [746, 395],
    #         [451, 522],
    #         [1151, 671],
    #         [417, 902],
    #     ]
    # )
    # image_points = np.array(
    #     [
    #         [1941, 405],
    #         [1257, 175],
    #         [2353, 1235],
    #         [1421, 1637],
    #         [2641, 2109],
    #         [1309, 2851],
    #     ]
    # )
    # hom_mat = homogeneous_matrix(real_points, image_points)
    output = []
    for [x, y, z, u, v] in points:
        x = x
        y = y
        z = z
        output.append([-x, -y, -z, -1, 0, 0, 0, 0, u * x, u * y, u * z, u])
        output.append([0, 0, 0, 0, -x, -y, -z, -1, v * x, v * y, v * z, v])
    hom_mat = np.array(output)
    m_mat = direct_linear_transformation(hom_mat)
    m_mat = m_mat / m_mat[-1, -1]

    a_mat = m_mat[:, :-1]

    b_mat = m_mat[:, -1]
    rho = 1 / np.linalg.norm(a_mat[-1])
    principal_x = (rho ** 2) * (np.dot(a_mat[0, :], a_mat[-1, :]))
    principal_y = (rho ** 2) * (np.dot(a_mat[1, :], a_mat[2, :]))
    cross_1 = np.cross(a_mat[0, :].T, a_mat[2, :].T)
    norm_1 = np.linalg.norm(cross_1)
    cross_2 = np.cross(a_mat[1, :].T, a_mat[2, :].T)
    norm_2 = np.linalg.norm(cross_2)
    skew = np.dot(cross_1.T, cross_2) / (norm_1 * norm_2)
    theta = math.acos(skew)
    print(math.sin(theta))
    focal_alpha = rho * rho * norm_1 * math.sin(theta)
    focal_beta = rho * rho * norm_2 * math.sin(theta)
    camera_params = np.array(
        [
            [focal_alpha, -focal_alpha / math.tan(theta), principal_x],
            [0, focal_beta / math.sin(theta), principal_y],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    print(camera_params)
    # print(compute_focal_length(focal_alpha, 6.17 * (10 ** -3)))
    # print(compute_focal_length(focal_beta, 4.55 * (10 ** -3)))


if __name__ == "__main__":
    main()