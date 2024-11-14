from __future__ import annotations

import numpy as np


def shrink(data, new_size):
    """
    A simple function to shrink a 3D array by summing over blocks of new_size.
    """
    xs = new_size
    ys = new_size
    zs = new_size
    return (
        data.reshape(
            xs, data.shape[0] // xs, ys, data.shape[1] // ys, zs, data.shape[2] // zs
        )
        .sum(axis=1)
        .sum(axis=2)
        .sum(axis=3)
    )


def wall(n):
    """
    Builds a wall out of an nxnxn array. All values in the wall
    are set to 1. Other points are set to 0.
    """
    array = np.zeros((n, n, n))
    array[:, :, :] = 0.1
    index = n // 2
    array[index, :, :] = 1
    return array


def cylinder(n, r):
    """
    Returns an nxnxn array with a cylinder in the center with
    radius r. All points within the radius are set to 1, the rest
    are set to 0.
    """
    array = np.zeros((n, n, n), dtype=np.float64)
    array[:, :, :] = 0.1
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - (n // 2)) ** 2 + (y - (n // 2)) ** 2 < r**2:
                    array[x, y, z] = 1
    return array


def sphere(n, r):
    """
    Returns an nxnxn array with a sphere in the center with
    radius r. All points within the radius are set to 1, the rest
    are set to 0.
    """
    array = np.zeros((n, n, n), dtype=np.float64)
    array[:, :, :] = 0.1
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - (n // 2)) ** 2 + (y - (n // 2)) ** 2 + (
                    z - (n // 2)
                ) ** 2 < r**2:
                    array[x, y, z] = 1
    return array
