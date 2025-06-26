from __future__ import annotations

import numpy as np


def shrink(data, new_size):
    """
    A simple function to shrink a 3D array by summing over blocks of new_size.

    Args:
        data (:obj:`3D float np.ndarray`):
            The input 3D array to be shrunk.
        new_size (:obj:`int`):
            The new size for each dimension after shrinking.

    Returns:
        (:obj:`3D float np.ndarray`): The shrunk 3D array.
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
    are set to 10. Other points are set to 0.

    Args:
        n (:obj:`int`):
            The size of the 3D array to create.

    Returns:
        (:obj:`3D float np.ndarray`): An nxnxn array with a wall in the center.
    """
    array = np.zeros((n, n, n))
    array[:, :, :] = 0.1
    index = n // 2
    array[index, :, :] = 10.0
    return array


def cylinder(n, r):
    """
    Returns an nxnxn array with a cylinder in the center with
    radius r. All points within the radius are set to 10, the rest
    are set to 0.

    Args:
        n (:obj:`int`):
            The size of the 3D array to create.
        r (:obj:`int`):
            The radius of the cylinder.

    Returns:
        (:obj:`3D float np.ndarray`): An nxnxn array with a cylinder in the center.
    """
    array = np.zeros((n, n, n), dtype=np.float64)
    array[:, :, :] = 0.1
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - (n // 2)) ** 2 + (y - (n // 2)) ** 2 < r**2:
                    array[x, y, z] = 10.0
    return array


def sphere(n, r):
    """
    Returns an nxnxn array with a sphere in the center with
    radius r. All points within the radius are set to 10, the rest
    are set to 0.

    Args:
        n (:obj:`int`):
            The size of the 3D array to create.
        r (:obj:`int`):
            The radius of the sphere.

    Returns:
        (:obj:`3D float np.ndarray`): An nxnxn array with a sphere in the center.
    """
    array = np.zeros((n, n, n), dtype=np.float64)
    array[:, :, :] = 0.1
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (x - (n // 2)) ** 2 + (y - (n // 2)) ** 2 + (
                    z - (n // 2)
                ) ** 2 < r**2:
                    array[x, y, z] = 10.0
    return array
