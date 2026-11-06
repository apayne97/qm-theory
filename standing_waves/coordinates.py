"""Polar (2D) and spherical (3D) coordinates: conversions and the lines of the coordinate grid. numpy only.

Conventions (physics): the polar angle theta is measured from the z axis, 0 <= theta <= pi; the azimuth phi goes
around the z axis, 0 <= phi < 2 pi. In two dimensions the angle is phi, 0 <= phi < 2 pi, and r >= 0.
"""

import numpy as np


def polar_to_cartesian(r, phi):
    """(x, y) of the point at distance r and angle phi."""
    r, phi = np.asarray(r, dtype=float), np.asarray(phi, dtype=float)
    return r * np.cos(phi), r * np.sin(phi)


def cartesian_to_polar(x, y):
    """(r, phi) with r >= 0 and 0 <= phi < 2 pi (the angle wraps around)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.hypot(x, y), np.mod(np.arctan2(y, x), 2 * np.pi)


def spherical_to_cartesian(r, theta, phi):
    """(x, y, z) of the point at distance r, polar angle theta (from the z axis) and azimuth phi."""
    r, theta, phi = (np.asarray(a, dtype=float) for a in (r, theta, phi))
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)


def cartesian_to_spherical(x, y, z):
    """(r, theta, phi) with r >= 0, 0 <= theta <= pi and 0 <= phi < 2 pi."""
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.ones_like(r), where=r > 0))
    return r, theta, np.mod(np.arctan2(y, x), 2 * np.pi)


def polar_grid_lines(n_circles=4, n_rays=12, r_max=2.0):
    """The lines of the polar grid: radii of the circles of constant r (r_max / n_circles, ..., r_max) and the angles of
    the rays of constant phi (0, 2 pi / n_rays, ...)."""
    radii = r_max * np.arange(1, n_circles + 1) / n_circles
    angles = 2 * np.pi * np.arange(n_rays) / n_rays
    return radii, angles


# The one view of every 3D figure with axes: a right-handed system seen from the first octant, z up, y to the right,
# x towards the viewer (down and to the left), as in the classic textbook figure of spherical coordinates.
VIEW_ELEV, VIEW_AZIM = 20, 35
