import math
import numpy as np


def get_normalization_matrix(pts: np.ndarray) -> np.ndarray:

    pts = pts.astype(np.float64)
    x_mean, y_mean = np.mean(pts, axis=1)
    var_x, var_y = np.var(pts, axis=1)
    s_x, s_y = np.sqrt(2 / var_x), np.sqrt(2 / var_y)
    n = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])

    return n.astype(np.float64)


def normalize_points(img_points: np.ndarray):
    N_u = get_normalization_matrix(img_points)
    _, cols = img_points.shape
    hom_img_pts = np.r_[img_points, np.ones((1, cols))]
    norm_hom_img_pts = np.dot(N_u, hom_img_pts)
    norm_hom_img_pts = norm_hom_img_pts[:-1]
    return norm_hom_img_pts


def estimate_calibration_matrix(
    obj_points: np.ndarray, img_points: np.ndarray
) -> np.ndarray:
    # Obtain image points vector form
    img_points = np.array([item[0] for item in img_points]).T
    # Obtain object point vector
    obj_points = obj_points.T
    # img_points = normalize_points(img_points)
    for index in range(obj_points.shape[1]):
        real_point = np.r_[obj_points[:, index], 1]
        image_point = img_points[:, index]
        corr_1 = np.array(
            [
                np.r_[
                    (-real_point),
                    np.zeros(4),
                    (real_point * image_point[1]),
                ]
            ]
        )
        corr_2 = np.array(
            [
                np.r_[
                    np.zeros(4),
                    (real_point * -1),
                    (real_point * image_point[0]),
                ]
            ]
        )
        if index == 0:
            output = corr_1
            output = np.r_[output, corr_2]
        else:
            output = np.r_[output, corr_1, corr_2]
    # output = output[5:]
    svd_matrix = np.linalg.svd(output)
    u, d, v_h = svd_matrix
    print(d)
    # exit()
    # print(v_h[:, -1])
    # print(np.linalg.norm(v_h[:, -1]))

    m_vect = v_h[-3, :]  # / v_h[-1, -1]
    print(m_vect)
    m_mat = np.reshape(m_vect, (3, 4))
    print(m_mat)
    # h_1 = m_mat[:, :-1]
    # inv_h_1 = np.linalg.inv(h_1)
    # q, r = np.linalg.qr(inv_h_1)
    # cam_mat = np.linalg.inv(r)
    # cam_mat = cam_mat / cam_mat[-1, -1]
    # print(cam_mat)
    # return cam_mat

    # r = r / np.linalg.norm(r[-1])
    # r[-1, -1] = abs(r[-1, -1])
    # print(r)
    # return r
    # print(r)
    # exit()
    # q, r = np.linalg.qr(m_mat[:, :-1])
    # print(r)
    # exit()
    # return r
    # print(q)
    print(m_mat)
    a_mat = m_mat[:, :-1]
    b_vect = m_mat[:, -1]
    rho = 1 / np.linalg.norm(a_mat[-1].T)
    # print(rho)
    # exit()
    principal_x = (rho ** 2) * (np.dot(a_mat[0, :], a_mat[-1, :].T))
    principal_y = (rho ** 2) * (np.dot(a_mat[1, :], a_mat[2, :].T))
    # print(a_mat[0])
    # print(a_mat[-1])
    # print(a_mat[1])
    # print(np.dot(a_mat[0, :], a_mat[-1, :].T))
    # print((np.dot(a_mat[1, :], a_mat[2, :].T)))
    # print(principal_x)
    # print(principal_y)
    # exit()
    # print("principal_x:", principal_x)
    # print("principal_y:", principal_y)
    cross_1 = np.cross(a_mat[0, :].T, a_mat[2, :].T)
    # print(a_mat[0, :])
    # print(a_mat[2, :])
    # print(cross_1)
    # exit()
    norm_1 = np.linalg.norm(cross_1)
    cross_2 = np.cross(a_mat[1, :].T, a_mat[2, :].T)
    norm_2 = np.linalg.norm(cross_2)
    skew = np.dot(cross_1.T, cross_2) / (norm_1 * norm_2)
    theta = math.acos(skew)
    focal_alpha = rho * rho * norm_1 * math.sin(theta)
    focal_beta = rho * rho * norm_2 * math.sin(theta)

    # print(rho)
    principal_x = np.dot(a_mat[0, :].T, a_mat[2, :])
    principal_y = np.dot(a_mat[1, :].T, a_mat[2, :])
    focal_alpha = norm_1 * math.sin(theta)
    focal_beta = norm_2 * math.sin(theta)
    # exit()
    camera_params = np.array(
        [
            [focal_alpha, -focal_alpha / math.tan(theta), principal_x],
            [0, focal_beta / math.sin(theta), principal_y],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # camera_params = np.array(
    #     [
    #         [focal_alpha, 0, principal_x],
    #         [0, focal_beta / math.sin(theta), principal_y],
    #         [0, 0, 1],
    #     ],
    #     dtype=np.float32,
    # )

    return camera_params


def calibration_dlt(xyz, uv):
    transform_xyz, norm_xyz = normalize(3, xyz)
    uv = np.array([item[0] for item in uv])
    print(uv)
    transform_uv, norm_uv = normalize(2, uv)

    A = []
    for i in range(xyz.shape[0]):
        x, y, z = norm_xyz[i, 0], norm_xyz[i, 1], norm_xyz[i, 2]
        u, v = norm_uv[i, 0], norm_uv[i, 1]
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    print(S)
    L = Vh[-4, :] / Vh[-1, -3]
    H = L.reshape(3, 4)
    H = np.dot(np.dot(np.linalg.pinv(transform_uv), H), transform_xyz)
    H = H / H[-1, -1]
    print(H)
    H_mat = np.reshape(H, (3, 4))
    h_1 = H_mat[:, :-1]
    inv_h_1 = np.linalg.inv(h_1)
    q, r = np.linalg.qr(inv_h_1)
    cam_mat = np.linalg.inv(r)
    cam_mat = cam_mat / cam_mat[-1, -1]
    print(cam_mat)
    exit()


def normalize(dims, data):
    m, s = np.mean(data, 0), np.std(data)
    if dims == 2:
        transform = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        transform = np.array(
            [[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]]
        )

    transform = np.linalg.inv(transform)
    data = np.dot(
        transform, np.concatenate((data.T, np.ones((1, data.shape[0]))))
    )
    data = data[0:dims, :].T
    return transform, data
