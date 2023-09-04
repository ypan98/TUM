import numpy as np

def rq_decomposition(A):
    Q, R = np.linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T
    return R[:, ::-1], Q[::-1, :]

points_3d = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
num_points = points_3d.shape[0]
points_2d = np.zeros((num_points, 2))
Q = np.zeros((2 * num_points, 12))

for i in range(num_points):
    input_ = input().split()
    point_2d = list(map(float, input_))
    points_2d[i] = point_2d
    X, Y, Z = points_3d[i][0], points_3d[i][1], points_3d[i][2]
    u, v = point_2d[0], point_2d[1]
    Q[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
    Q[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

# get projection matrix M
_, _, Vh = np.linalg.svd(Q)
M = Vh[-1].reshape((3, 4))
if M[2, 3] < 0:
    M = -M
# get K, R, t from M
K, R = rq_decomposition(M[:, :3])
# to fix sign issue
T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R
# tackle unknown scaling
K[0, 1] = 0
t = np.linalg.inv(K) @ M[:, 3]
K = K / K[2, 2]
E = np.hstack((R, t.reshape(3, 1)))

# print out result
for i in range(3):
    for j in range(3):
        print(K[i, j], end=" ")
    print()

for i in range(3):
    for j in range(4):
        print(E[i, j], end=" ")
    print()