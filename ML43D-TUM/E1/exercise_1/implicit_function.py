"""Definitions for Signed Distance Fields"""
import numpy as np


def signed_distance_sphere(x, y, z, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a sphere of radius r, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param r: radius of the sphere
    :param x_0: x coordinate of the center of the sphere
    :param y_0: y coordinate of the center of the sphere
    :param z_0: z coordinate of the center of the sphere
    :return: signed distance from the surface of the sphere
    """
    # ###############
    # TODO: Implement
    x_c = x-x_0
    y_c = y-y_0
    z_c = z-z_0
    return np.sqrt(np.square(x_c)+np.square(y_c)+np.square(z_c))-r

    raise NotImplementedError
    # ###############


def signed_distance_torus(x, y, z, R, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a torus of minor radius r and major radius R, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param R: major radius of the torus
    :param r: minor radius of the torus
    :param x_0: x coordinate of the center of the torus
    :param y_0: y coordinate of the center of the torus
    :param z_0: z coordinate of the center of the torus
    :return: signed distance from the surface of the torus
    """
    # ###############
    # TODO: Implement
    x_c = x - x_0
    y_c = y - y_0
    z_c = z - z_0
    a = np.sqrt(np.square(x_c)+np.square(z_c))-R
    res = np.sqrt(np.square(a)+np.square(y_c))-r
    return res
    raise NotImplementedError
    # ###############


def signed_distance_atom(x, y, z):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a hydrogen atom consisting of a spherical proton, a torus orbit, and one spherical electron
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :return: signed distance from the surface of the hydrogen atom
    """
    proton_center = (0, 0, 0)
    proton_radius = 0.1
    orbit_radius = 0.35  # The major radius of the orbit torus
    orbit_thickness = 0.01  # The minor radius of the orbit torus
    electron_center = (orbit_radius, 0, 0)
    electron_radius = 0.05
    # ###############
    # TODO: Implement
    proton = signed_distance_sphere(x, y, z, proton_radius, proton_center[0], proton_center[1], proton_center[2])
    orbit = signed_distance_torus(x, y, z, orbit_radius, orbit_thickness, proton_center[0], proton_center[1], proton_center[2])
    electron = signed_distance_sphere(x, y, z, electron_radius, electron_center[0], electron_center[1], electron_center[2])
    return np.min([proton, orbit, electron], axis=0)
    raise NotImplementedError
    # ###############
