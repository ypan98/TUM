"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    # TODO: Implement
    grid_x = np.zeros((resolution, resolution, resolution))
    grid_y = np.zeros((resolution, resolution, resolution))
    grid_z = np.zeros((resolution, resolution, resolution))

    # Assuming unit cube centered at (0,0,0)
    step = 1 / resolution
    start_point = -resolution * step / 2
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid_x[i][j][k] = start_point + i * step
                grid_y[i][j][k] = start_point + j * step
                grid_z[i][j][k] = start_point + k * step
    grid_x_flatted = grid_x.flatten()
    grid_y_flatted = grid_y.flatten()
    grid_z_flatted = grid_z.flatten()
    grid_flatted = sdf_function(grid_x_flatted, grid_y_flatted, grid_z_flatted)
    grid = grid_flatted.reshape((resolution, resolution, resolution))
    return grid <= 0
    raise NotImplementedError
    # ###############
