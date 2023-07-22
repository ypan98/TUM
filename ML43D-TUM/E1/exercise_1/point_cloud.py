"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    faces_area = np.zeros(faces.shape[0])
    # calculate area of each face
    # Herons formula a = sqrt(s(s-a)(s-b)(s-c)) s=(a+b+c)/2
    for i in range(len(faces)):
        v1 = vertices[faces[i][0]]
        v2 = vertices[faces[i][1]]
        v3 = vertices[faces[i][2]]
        a = np.linalg.norm(v1-v2)
        b = np.linalg.norm(v2-v3)
        c = np.linalg.norm(v3-v1)
        s = (a+b+c)/2
        faces_area[i] = np.sqrt(s*(s-a)*(s-b)*(s-c))

    faces_idx = list(range(len(faces)))
    sampled_faces_idx = np.random.choice(faces_idx, n_points, p=faces_area/faces_area.sum())
    # from sampled triangle faces to points
    sampled_points = np.zeros((n_points, 3))
    for i in range(len(sampled_faces_idx)):
        face = faces[sampled_faces_idx[i]]
        A = vertices[face[0]]
        B = vertices[face[1]]
        C = vertices[face[2]]
        r = np.random.random(2)
        u = 1-np.sqrt(r[0])
        v = np.sqrt(r[0])*(1-r[1])
        w = np.sqrt(r[0])*r[1]
        sampled_points[i] = u*A+v*B+w*C
    return sampled_points
    raise NotImplementedError
    # ###############
