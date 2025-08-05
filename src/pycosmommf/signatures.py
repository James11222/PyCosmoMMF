from __future__ import annotations

import numba as nb
import numpy as np

from .filter import smooth_gauss, smooth_loggauss, wavevectors3D
from .hessian import fast_hessian_from_smoothed

jit_compiler = nb.njit(parallel=True, fastmath=True)


@jit_compiler
def signatures_from_hessian(hessian):  # pragma: no cover
    """
    Function to calculate the signatures from a given hessian.

    Args:
        hessian (:obj:`4D float np.ndarray`):
            The hessian matrix of the smoothed field, with shape ``(nx, ny, nz, 6)``.

    Returns:
        (:obj:`4D float np.ndarray`): The signatures array with shape ``(nx, ny, nz, 3)``. Each signature corresponds to a different type of structure:
            - ``sigs[..., 0]``: Cluster signature
            - ``sigs[..., 1]``: Filament signature
            - ``sigs[..., 2]``: Wall signature
    """

    hsize = hessian.shape[:3]
    sigs = np.zeros((hsize[0], hsize[1], hsize[2], 3), dtype=np.float32)

    # helper functions
    def θ(x):
        """
        Heaviside step function. Returns one if the argument is positive, zero otherwise.
        """
        if x >= 0:
            return 1
        return 0

    def xθ(x):
        """
        Heaviside step times x. Returns x if the argument is positive, zero otherwise.
        """
        if x >= 0:
            return x
        return 0

    for i in nb.prange(hsize[0]):
        for j in range(hsize[1]):
            for k in range(hsize[2]):
                hes_slice = np.array(
                    [
                        [hessian[i, j, k, 0], hessian[i, j, k, 1], hessian[i, j, k, 2]],
                        [hessian[i, j, k, 1], hessian[i, j, k, 3], hessian[i, j, k, 4]],
                        [hessian[i, j, k, 2], hessian[i, j, k, 4], hessian[i, j, k, 5]],
                    ]
                )

                eigenvalues = np.sort(
                    np.real(np.linalg.eigvals(hes_slice))
                )  # 3 eigenvalues, sorted

                if eigenvalues[0] == 0:
                    sigs[i, j, k, 0] = 0
                    sigs[i, j, k, 1] = 0
                    sigs[i, j, k, 2] = 0
                else:
                    sigs[i, j, k, 0] = (
                        np.abs(eigenvalues[2] / eigenvalues[0]) * np.abs(eigenvalues[2])
                    ) * (θ(-eigenvalues[0]) * θ(-eigenvalues[1]) * θ(-eigenvalues[2]))

                    sigs[i, j, k, 1] = (
                        np.abs(eigenvalues[1] / eigenvalues[0])
                        * xθ(1 - np.abs(eigenvalues[2] / eigenvalues[0]))
                    ) * (
                        np.abs(eigenvalues[1]) * θ(-eigenvalues[0]) * θ(-eigenvalues[1])
                    )

                    sigs[i, j, k, 2] = (
                        xθ(1 - np.abs(eigenvalues[1] / eigenvalues[0]))
                        * xθ(1 - np.abs(eigenvalues[2] / eigenvalues[0]))
                    ) * (np.abs(eigenvalues[0]) * θ(-eigenvalues[0]))

    return sigs


@jit_compiler
def signatures_from_hessian_tidal(hessian):  # pragma: no cover
    """
    Function to calculate the signatures from a hessian of the tidal field.

    Args:
        hessian (:obj:`4D float np.ndarray`):
            The hessian matrix of the smoothed field, with shape ``(nx, ny, nz, 6)``.

    Returns:
        (:obj:`4D float np.ndarray`): The signatures array with shape ``(nx, ny, nz, 3)``. Each signature corresponds to a different type of structure:
            - ``sigs[..., 0]``: Cluster signature
            - ``sigs[..., 1]``: Filament signature
            - ``sigs[..., 2]``: Wall signature
    """
    hsize = hessian.shape[:3]
    sigs = np.zeros((hsize[0], hsize[1], hsize[2], 3), dtype=np.float32)

    # helper functions
    def θ(x):
        """
        Heaviside step function. Returns one if the argument is positive, zero otherwise.
        """
        if x >= 0:
            return 1
        return 0

    for i in nb.prange(hsize[0]):
        for j in range(hsize[1]):
            for k in range(hsize[2]):
                hes_slice = np.array(
                    [
                        [hessian[i, j, k, 0], hessian[i, j, k, 1], hessian[i, j, k, 2]],
                        [hessian[i, j, k, 1], hessian[i, j, k, 3], hessian[i, j, k, 4]],
                        [hessian[i, j, k, 2], hessian[i, j, k, 4], hessian[i, j, k, 5]],
                    ]
                )

                eigenvalues = np.sort(
                    np.real(np.linalg.eigvals(hes_slice))
                )  # 3 eigenvalues, sorted

                sigs[i, j, k, 0] = eigenvalues[0] * θ(eigenvalues[0])
                sigs[i, j, k, 1] = eigenvalues[1] * θ(eigenvalues[1])
                sigs[i, j, k, 2] = eigenvalues[2] * θ(eigenvalues[2])

    return sigs


def maximum_signature(Rs, density_cube, algorithm="NEXUSPLUS", eps=1e-16):
    """
    Compute the maximum signatures across all scales Rs.

    Args:
        Rs (:obj:`list` of :obj:`float`):
            List of smoothing scales in units of voxels.
        density_cube (:obj:`3D float np.ndarray`):
            The 3D density field to analyze.
        algorithm (:obj:`str`, optional):
            The algorithm to use for smoothing. Can be either "NEXUS_DEN", "NEXUS_TIDAL", or "NEXUSPLUS". Defaults to "NEXUSPLUS".
        eps (:obj:`float`, optional):
            Small value to avoid division by zero. Defaults to ``1e-16``.

    Returns:
        (:obj:`4D float np.ndarray`):
            An array of signatures with shape ``(nx, ny, nz, 3)``, where ``nx, ny, nz`` are the dimensions of the input ``density_cube``.
            The last dimension contains the signatures for clusters, filaments, and walls respectively.
    """

    if algorithm not in ["NEXUS_DEN", "NEXUSPLUS", "NEXUS_TIDAL"]:  # pragma: no cover
        msg = "algorithm must be either 'NEXUS' or 'NEXUSPLUS' or 'NEXUS_TIDAL'"
        raise ValueError(msg)

    nx, ny, nz = density_cube.shape

    # Make sure the field has no 0 values
    mean_density = np.mean(density_cube)
    overdensity_field = (density_cube / mean_density) - 1.0
    density_cube = density_cube + eps

    # Calculate wave vectors for our field
    wave_vecs = wavevectors3D((nx, ny, nz))

    # Calculate signatures at each scale Rn, determine max
    sigmax = np.ones((nx, ny, nz, 3), dtype=np.float32) * eps

    @jit_compiler
    def perform_loop(sigmax, sigs_Rn, nx, ny, nz):  # pragma: no cover
        for ix in nb.prange(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for sigtype in range(3):
                        sigmax[ix, iy, iz, sigtype] = max(
                            sigmax[ix, iy, iz, sigtype], sigs_Rn[ix, iy, iz, sigtype]
                        )

    for R in Rs:
        if algorithm == "NEXUSPLUS":
            f_Rn = smooth_loggauss(density_cube, R, wave_vecs)
        elif algorithm in ("NEXUS_DEN", "NEXUS_TIDAL"):
            f_Rn = smooth_gauss(overdensity_field, R, wave_vecs)
        else:  # pragma: no cover
            msg = "algorithm must be either 'NEXUS_DEN', 'NEXUS_TIDAL' or 'NEXUSPLUS'"
            raise ValueError(msg)

        H_Rn = fast_hessian_from_smoothed(f_Rn, R, wave_vecs, algorithm)

        if algorithm == "NEXUS_TIDAL":
            sigs_Rn = signatures_from_hessian_tidal(H_Rn)
        else:
            sigs_Rn = signatures_from_hessian(H_Rn)

        perform_loop(sigmax, sigs_Rn, nx, ny, nz)

    return sigmax
