from typing import Tuple

import numpy as np


def signed_distance_sphere(x, y, z, r, x_0, y_0, z_0):
    return np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2 + (z - z_0) ** 2) - r


def signed_distance_torus(x, y, z, R, r, x_0, y_0, z_0):
    a = np.sqrt((x - x_0) ** 2 + (z - z_0) ** 2) - R
    return np.sqrt(a ** 2 + (y - y_0) ** 2) - r


def sdf_grid(sdf_function, resolution):
    x_range = y_range = z_range = np.linspace(-0.5, 0.5, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    sdf_values = sdf_function(grid_x, grid_y, grid_z).reshape((resolution, resolution, resolution))
    return sdf_values


def generate_toy_data(num_samples: int) -> Tuple[np.array, np.array]:
    print(f"Generating toy data ...")
    sphere_radii = np.random.rand(num_samples // 2)
    sphere_sdfs = np.stack([sdf_grid(lambda x, y, z: signed_distance_sphere(x, y, z, sphere_radius, 0, 0, 0), resolution=32) for sphere_radius in sphere_radii])

    torus_major_radii = np.random.rand(num_samples - num_samples // 2)
    torus_minor_radii = np.random.rand(num_samples - num_samples // 2)
    torus_sdfs = np.stack([sdf_grid(lambda x, y, z: signed_distance_torus(x, y, z, torus_major_radius, torus_minor_radius, 0, 0, 0), resolution=32) for (torus_major_radius, torus_minor_radius) in zip(torus_major_radii, torus_minor_radii)])

    return np.concatenate([sphere_sdfs, torus_sdfs]).astype(np.float32), np.concatenate([np.zeros(shape=[num_samples // 2], dtype=np.int64), np.ones(shape=[num_samples - num_samples // 2], dtype=np.int64)])
