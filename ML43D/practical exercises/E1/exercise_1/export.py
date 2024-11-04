"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    with open(path, 'w') as file:
        for v in vertices:
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for f in faces:
            # index starts from 1
            file.write("f {} {} {}\n".format(f[0]+1, f[1]+1, f[2]+1))
    return
    raise NotImplementedError
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    with open(path, 'w') as file:
        for v in pointcloud:
            # axis flipping
            file.write("v {} {} {}\n".format(v[0], -v[2], v[1]))
    return
    raise NotImplementedError
    # ###############
