from __future__ import annotations

import numba as nb
import numpy as np

jit_compiler = nb.njit(parallel=True, fastmath=True)


def fast_hessian_from_smoothed(f_Rn, R_S, kv, algorithm):
    """
    Compute the hessian matrix of the smoothed field f_Rn.

    Args:
        f_Rn (:obj:`3D float np.ndarray`):
            The smoothed field in real space.
        R_S (:obj:`float`):
            The smoothing scale in units of voxels.
        kv (:obj:`tuple`):
            The wavevectors in each dimension.
        algorithm (:obj:`str`):
            The algorithm to use. Currently only "NEXUSPLUS" is supported.
            This parameter is included for compatibility with other functions.

    Returns:
        (:obj:`4D float np.ndarray`): The hessian matrix of the smoothed field.
        The shape is ``(nx, ny, nz, 6)``, where ``nx, ny, nz`` are the dimensions of ``f_Rn``.
    """

    f_Rn_hat = np.fft.fftn(f_Rn)
    dims = f_Rn_hat.shape
    hessian = np.zeros((dims[0], dims[1], dims[2], 6), dtype=np.complex64)
    kx, ky, kz = kv[0], kv[1], kv[2]
    tidal_factor = (kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2) * R_S**2

    @jit_compiler
    def perform_loop(hessian):  # pragma: no cover
        """
        This function is a loop that calculates the hessian matrix
        for each point in the 3D array.
        """
        for x in nb.prange(len(kv[0])):
            for y in range(len(kv[1])):
                for z in range(len(kv[2])):
                    # (1,1)
                    hessian[x, y, z, 0] = -kx[x] * kx[x] * R_S**2 * f_Rn_hat[x, y, z]
                    # (1,2)
                    hessian[x, y, z, 1] = -kx[x] * ky[y] * R_S**2 * f_Rn_hat[x, y, z]
                    # (1,3)
                    hessian[x, y, z, 2] = -kx[x] * kz[z] * R_S**2 * f_Rn_hat[x, y, z]
                    # (2,2)
                    hessian[x, y, z, 3] = -ky[y] * ky[y] * R_S**2 * f_Rn_hat[x, y, z]
                    # (2,3)
                    hessian[x, y, z, 4] = -ky[y] * kz[z] * R_S**2 * f_Rn_hat[x, y, z]
                    # (3,3)
                    hessian[x, y, z, 5] = -kz[z] * kz[z] * R_S**2 * f_Rn_hat[x, y, z]

        return hessian

    hessian = perform_loop(hessian)

    for j in range(6):
        if algorithm == "NEXUS_TIDAL":
            hessian[:, :, :, j] = hessian[:, :, :, j] / (tidal_factor + 1e-16)  # avoid division by zero
            
        hessian[:, :, :, j] = np.fft.ifftn(hessian[:, :, :, j])

    return np.real(hessian)
