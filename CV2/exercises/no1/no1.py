import numpy as np

np.set_printoptions(precision=5, suppress=True)
delta = 1e-7

def read_pixel_coords():
    points = []
    for _ in range(8):
        input_ = input().split()
        input_list = list(map(float, input_))
        points.append(input_list)
    return np.array(points)

def read_K():
    _ = input() # skip line
    K = np.zeros((3, 3))
    for i in range(3):
        input_ = input().split()
        input_list = list(map(float, input_))
        K[i] = np.array(input_list)
    return K

def read_E():
    _ = input() # skip line
    E = np.zeros((3, 4))
    for i in range(3):
        input_ = input().split()
        input_list = list(map(float, input_))
        E[i] = np.array(input_list)
    return E

# convert rotation matrix to rodrigues vector (ref cv2.Rodrigues)
def so3_log_map(rotation_matrix):
    trace = np.trace(rotation_matrix)
    theta = np.arccos((trace - 1) / 2)

    if np.isclose(theta, 0.0):
        return np.array([0, 0, 0])
    axis = (1 / (2 * np.sin(theta))) * np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                                                rotation_matrix[0, 2] - rotation_matrix[2, 0],
                                                rotation_matrix[1, 0] - rotation_matrix[0, 1]])
    rodrigues_vector = theta * axis
    return rodrigues_vector

# convert rodrigues vector to rotation matrix (ref cv2.Rodrigues)
def so3_exp_map(rodrigues_vector):
    theta = np.linalg.norm(rodrigues_vector)
    if np.isclose(theta, 0.0):         
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vector / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = np.cos(theta) * I + (1 - np.cos(theta)) * r_rT + np.sin(theta) * r_cross
    return rotation_mat 

# compute residual (reprojection error) given E, K, pixel coordinates and world coordinates points
def compute_residual(E, K, pixel_coords, points_world):
    points_world_homo = np.hstack(((points_world), np.ones((len(points_world),1)))) # 8x4
    points_cam = E @ points_world_homo.T    # 4x8
    points_cam = points_cam[:3]
    points_pixel_homo = points_cam / points_cam[2]  # 3x8
    points_pixel = K @ points_pixel_homo  # 3x8
    points_pixel = points_pixel[:2].T    # 8x2
    return (points_pixel - pixel_coords).flatten()  # 16

def skew_symm_vec2mat(skew_vec):
    return np.array([
        [0, -skew_vec[2], skew_vec[1]],
        [skew_vec[2], 0, -skew_vec[0]],
        [-skew_vec[1], skew_vec[2], 0]
    ])

# jacobian used for translation part of mapping between SE(3)-se(3)
def left_jacobian(phi):
    angle = np.linalg.norm(phi)
    if np.isclose(angle, 0.):
        return np.identity(3) + 0.5 * skew_symm_vec2mat(phi)
    axis = phi / angle
    s = np.sin(angle)
    c = np.cos(angle)
    return (s / angle) * np.identity(3) + \
        (1 - s / angle) * np.outer(axis, axis) + \
        ((1 - c) / angle) * skew_symm_vec2mat(axis)

# convert EK to the 10 dof vector representation with log map of E = SE(3)
def EK_to_dof(E, K):
    phi = so3_log_map(E[:3, :3])    # rot
    left_jac = left_jacobian(phi)
    rho = np.linalg.inv(left_jac)@E[:3, 3]  # transl
    k = K[[0, 1, 0, 1], [0, 1, 2, 2]]
    return np.concatenate((k, rho, phi))

# get EK back from 10 dof vector with exp map of e = se(3)
def dof_to_EK(dof10):
    K = np.eye(3)
    K[[0, 1, 0, 1], [0, 1, 2, 2]] = dof10[:4]
    E = np.eye(4)
    rho = dof10[4:7]    # transl
    phi = dof10[7:]      # rot
    E[:3, 3] = left_jacobian(phi) @ rho
    E[:3, :3] = so3_exp_map(phi)
    return E, K

if __name__ == "__main__":
    points_world = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    pixel_coords = read_pixel_coords()
    K = read_K()
    E_w2c = read_E()
    E_w2c = np.vstack((E_w2c, np.array([0, 0, 0, 1])))
    E = np.eye(4)   
    
    E_ = E@E_w2c
    r = compute_residual(E_, K, pixel_coords, points_world)
    for i in range(r.shape[0]):
        print(r[i], end=' ')
    print()

    dof10 = EK_to_dof(E, K)
    Jac = np.zeros((len(r), len(dof10)))
    for i in range(len(dof10)):
        newDOF10 = dof10.copy()
        newDOF10[i] += delta
        newE, newK = dof_to_EK(newDOF10)
        newE_ = newE@E_w2c
        new_r = compute_residual(newE_, newK, pixel_coords, points_world)
        grad = (new_r - r) / delta
        Jac[:, i] = grad
    
    for i in range(Jac.shape[0]):
        for j in range(Jac.shape[1]):
            print(Jac[i, j], end=' ')
        print()
        





