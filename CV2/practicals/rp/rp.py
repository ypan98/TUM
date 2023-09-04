import numpy as np

class Camera:

    def __init__(self, type, w, h, K, fov_w):
        self.type = type
        self.w = w
        self.h = h
        self.K = K
        self.fov_w = fov_w

    def calculate_rd(self, ru: np.ndarray) -> np.ndarray:
        return (1/self.fov_w) * np.arctan(2*ru*np.tan(self.fov_w/2))

    def calculate_ru(self, rd: np.ndarray) -> np.ndarray:
        return np.tan(rd*self.fov_w) / (2*np.tan(self.fov_w/2))

    def backproject_points_2_cam2_coords(self, points: np.ndarray) -> np.ndarray:
        # image to cam1 space
        distances = points[:, 2].T  # 1 x N
        points = points[:, :2]  # N x 2
        points_homo = np.hstack((points, np.ones((len(points), 1)))).T  # 3 x N
        projected_coords = np.linalg.inv(self.K) @ points_homo  # 3 x N: (x/z, y/z, 1)
        # undistort
        if self.type == 'fov':
            rd = np.linalg.norm(projected_coords[:2, :], axis=0)
            ru = self.calculate_ru(rd)
            projected_coords[:2, :] *= ru/rd
        depths = distances / np.linalg.norm(projected_coords, axis=0)  # 1 x N : z
        cam_coords = depths * projected_coords  # 3 x N: (xc, yc, zc)
        # cam1 to cam2 space
        cam_coords_homo = np.vstack((cam_coords, np.ones((1, len(points)))))  # 4 x N
        cam2_coords = (self.cam1_2_cam2[:3, :] @ cam_coords_homo).T  # 3 x N
        return cam2_coords

    def project_points_to_img_space(self, points: np.ndarray) -> np.ndarray:
        # cam2 to image space
        cam_coords_homo = np.hstack((points, np.ones((len(points), 1)))).T  # 4 x N
        cam_coords = cam_coords_homo[:3, :]  # 3 x N: (x, y, z)
        projected_coords = cam_coords / cam_coords[2, :]  # (x/z, y/z, 1)
        # distort on projected coords
        if self.type == 'fov':
            ru = np.linalg.norm(projected_coords[:2, :], axis=0)  # 1 x N
            rd = self.calculate_rd(ru)  # 1 x N
            projected_coords[:2, :] *= rd/ru  # 3 x N :  (rd/ru * x/z, rd/ru * y/z, 1)
            # ar = self.K[0,0] / self.K[1,1]
            # projected_coords[0, :] *= ar

        image_coords = self.K @ projected_coords  # 3 x N
        image_coords = image_coords[:2, :]  # 2 x N
        return image_coords.T  # N x 2

def read_cam():
    # read cams
    cam_input = input().split()
    cam_type = cam_input[0]
    cam_w, cam_h = map(int, cam_input[1:3])
    params_list = list(map(float, cam_input[3:]))
    cam_intrinsics = np.array([
        [params_list[0], 0, params_list[2]],
        [0, params_list[1], params_list[3]],
        [0, 0, 1]
    ])
    cam_fov_w = float(params_list[4]) if cam_type == "fov" else None
    return Camera(cam_type, cam_w, cam_h, cam_intrinsics, cam_fov_w)

def read_input():
    # read cams
    cam_1 = read_cam()
    cam_2 = read_cam()

    cam_1_to_cam_2 = []
    for _ in range(3):
        cam_1_to_cam_2.append(list(map(float, input().split())))
    cam_1_to_cam_2.append([0, 0, 0, 1])
    cam_2_pos = np.array(cam_1_to_cam_2)
    cam_1.cam1_2_cam2 = cam_2_pos

    # read points
    points = []
    while True:
        try:
            points.append(list(map(float, input().split())))
        except EOFError:
            break
    points = np.array(points)

    return cam_1, cam_2, points

if __name__ == '__main__':
    cam_1, cam_2, points = read_input()
    points_cam2 = cam_1.backproject_points_2_cam2_coords(points)
    points_image = cam_2.project_points_to_img_space(points_cam2)

    neg_z = points_cam2[:, 2] < 0

    for i in range(len(points_image)):
        point = points_image[i]
        if not neg_z[i] and point[0] >= 0 and point[0] <= cam_2.w and point[1] >= 0 and point[1] <= cam_2.h:
            print(point[0], point[1])
        else:
            print('OB')










