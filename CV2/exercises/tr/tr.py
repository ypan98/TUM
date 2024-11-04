import numpy as np

# read input for camera parameters
def read_camera_params():
    input_ = input().split()
    input_list = list(map(float, input_[1:]))
    cam_res = (input_list[0], input_list[1])
    cam_k = np.array([
        [input_list[2], 0, input_list[4]],
        [0, input_list[3], input_list[5]],
        [0, 0, 1]
    ])
    return cam_res, cam_k

# read points until EOF
def read_points():
    points = []
    while True:
        try:
            input_ = input().split()
            input_list = list(map(float, input_))
            points.append(input_list)
        except EOFError:
            return np.array(points)


def normalize_cam_points(pixels, k):
    pixels = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    pixels = np.linalg.inv(k) @ pixels.T
    pixels = pixels.T
    pixels /= pixels[:, 2].reshape(-1, 1)
    pixels = pixels[:, 0:2]
    return pixels

# normalize pixel coordinates to normalized coordinates
def normalize_points(points: np.array, cam_1_k: np.array, cam_2_k: np.array) -> np.array:
    normalize_cam_1_points = normalize_cam_points(points[:, 0:2], cam_1_k)
    normalize_cam_2_points = normalize_cam_points(points[:, 2:4], cam_2_k)
    normalized_points = np.hstack((normalize_cam_1_points, normalize_cam_2_points))
    return normalized_points

# construct Q matrix
def construct_Q(points: np.array) -> np.array:
    Q = np.zeros((points.shape[0], 9))
    for i in range(points.shape[0]):
        u1, v1, u2, v2 = points[i]
        Q[i] = np.array([u1*u2, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1])
        # Q[i] = np.array([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1])
    return Q


# decompose essential matrix into 4 possible R and T
def decompose_essential_matrix(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    Rz_90 = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
    Rz_minus90 = np.array([[0, 1, 0],
                           [-1, 0, 0],
                           [0, 0, 1]])
    R1 = U @ Rz_90 @ Vt
    R2 = U @ Rz_90.T @ Vt
    R3 = U @ Rz_minus90 @ Vt
    R4 = U @ Rz_minus90.T @ Vt

    T1 = U[:, 2]
    T2 = -U[:, 2]
    T3 = U[:, 2]
    T4 = -U[:, 2]

    return (R1, T1), (R2, T2), (R3, T3), (R4, T4)

# triangulation to select correct R and T
def select_correct_RT(rt_matrices, points, cam_1_k, cam_2_k):
    cam_1_points = points[:, 0:2]
    cam_2_points = points[:, 2:4]
    for R, T in rt_matrices:
        RT = np.hstack((R, T.reshape(3, 1)))
        found = True
        for i in range(points.shape[0]):
            # p_skew_1 = np.array([[0, -1, cam_1_points[i, 1]],
            #                         [1, 0, -cam_1_points[i, 0]],
            #                         [-cam_1_points[i, 1], cam_1_points[i, 0], 0]])
            # M1 = cam_1_k @ np.hstack((np.eye(3), np.zeros((3, 1))))
            # cam_1_eqs = p_skew_1 @ M1
            # p_skew_2 = np.array([[0, -1, cam_2_points[i, 1]],
            #                    [1, 0, -cam_2_points[i, 0]],
            #                    [-cam_2_points[i, 1], cam_2_points[i, 0], 0]])
            # M2 = cam_2_k @ RT
            # cam_2_eqs = p_skew_2 @ M2
            # A = np.vstack((cam_1_eqs[:2], cam_2_eqs[:2]))

            M1 = cam_1_k @ np.hstack((np.eye(3), np.zeros((3, 1))))
            M2 = cam_2_k @ RT
            A = np.vstack((cam_1_points[i, 1] * M1[2] - M1[1],
                                M1[0] - cam_1_points[i, 0] * M1[2],
                                cam_2_points[i, 1] * M2[2] - M2[1],
                                M2[0] - cam_2_points[i, 0] * M2[2]))

            _, _, V = np.linalg.svd(A.T@A)
            P_1_homo = V[-1].reshape(4, 1)
            P_1_homo /= P_1_homo[3]
            P_2_homo = RT @ P_1_homo

            if P_1_homo[2] < 0 or P_2_homo[2] < 0:
                found = False
                break
        if found:
            return R, T
    return None


# eight point algorithm
def compute_transformation(cam_1_k, cam_2_k, points):
    valid_points = points[(points > 0).all(axis=1)]
    normalized_points = normalize_points(valid_points, cam_1_k, cam_2_k)
    Q = construct_Q(normalized_points)
    U, _, V = np.linalg.svd(Q.T@Q)
    E = V[-1].reshape(3, 3)
    decompositions = decompose_essential_matrix(E)
    R, t = select_correct_RT(decompositions, valid_points, cam_1_k, cam_2_k)
    return R, t

def invert(R, t):
    return R.T, -R.T@t

# recover the scale from a visible point from all 3 cameras
def recover_scales(points, M1, M2, M3):
    A = np.vstack((points[1] * M1[2] - M1[1],
                    M1[0] - points[0] * M1[2],
                    points[3] * M2[2] - M2[1],
                    M2[0] - points[2] * M2[2],
                    points[5] * M3[2] - M3[1],
                    M3[0] - points[4] * M3[2]))
    _, _, V = np.linalg.svd(A)
    if np.linalg.det(V) < 0:
        V *= -1
    sol = V[-1]
    if sol[2] < 0:
        sol *= -1
    s1 = sol[3]
    s2 = sol[4]
    return s1, s2

if __name__ == "__main__":
    cam_1_res, cam_1_k = read_camera_params()
    cam_2_res, cam_2_k = read_camera_params()
    cam_3_res, cam_3_k = read_camera_params()
    points = read_points()

    R_1_2, t_1_2 = compute_transformation(cam_1_k, cam_2_k, points[:, 0:4])
    R_2_3, t_2_3 = compute_transformation(cam_2_k, cam_3_k, points[:, 2:6])


    all_visiable_points = points[(points > 0).all(axis=1)]
    R_2_1, t_2_1 = invert(R_1_2, t_1_2)
    M1 = cam_1_k @ np.hstack((R_2_1, t_2_1.reshape(3, 1), np.zeros((3, 1))))
    M2 = cam_2_k @ np.hstack((np.eye(3), np.zeros((3, 2))))
    M3 = cam_3_k @ np.hstack((R_2_3, np.zeros((3, 1)), t_2_3.reshape(3, 1)))

    s1, s2 = recover_scales(all_visiable_points[0], M1, M2, M3)
    RT_2_3 = np.vstack((np.hstack((R_2_3, s2*t_2_3.reshape(3, 1))), np.array([0, 0, 0, 1])))
    RT_2_1 = np.vstack((np.hstack((R_2_1, s1*t_2_1.reshape(3, 1))), np.array([0, 0, 0, 1])))

    RT_3_1 = RT_2_1 @ np.linalg.inv(RT_2_3)
    norm = np.linalg.norm(RT_3_1[:3,3])
    RT_3_1[:3,3] /= norm

    RT_3_1 = RT_3_1[:3,:]
    for i in range(3):
        output = ' '.join(map(str, RT_3_1[i].tolist()))
        print(output)
