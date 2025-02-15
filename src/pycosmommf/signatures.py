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

    Parameters
    ----------
    hessian : 4D array
        The hessian matrix.

    Returns
    -------
    sigs : 4D array
        The signatures.
    """

    hsize = hessian.shape[:3]
    sigs = np.zeros((hsize[0], hsize[1], hsize[2], 3), dtype=np.float32)

    # helper functions
    def θ(x):
        """
        Heaviside step function. Returns one if the argument is positive, zero otherwise.
        """
        if x > 0:
            return 1
        return 0

    def xθ(x):
        """
        Heaviside step times x. Returns x if the argument is positive, zero otherwise.
        """
        if x > 0:
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

                e1, e2, e3 = np.sort(np.real(np.linalg.eigvals(hes_slice)))

                if e1 == 0:
                    sigs[i, j, k, 0] = 0
                    sigs[i, j, k, 1] = 0
                    sigs[i, j, k, 2] = 0
                else:
                    sigs[i, j, k, 0] = (np.abs(e3 / e1) * np.abs(e3)) * (
                        θ(-e1) * θ(-e2) * θ(-e3)
                    )

                    sigs[i, j, k, 1] = (np.abs(e2 / e1) * xθ(1 - np.abs(e3 / e1))) * (
                        np.abs(e2) * θ(-e1) * θ(-e2)
                    )

                    sigs[i, j, k, 2] = (
                        xθ(1 - np.abs(e2 / e1)) * xθ(1 - np.abs(e3 / e1))
                    ) * (np.abs(e1) * θ(-e1))

    return sigs


def maximum_signature(Rs, field, alg="NEXUSPLUS", eps=1e-8):
    """
    Compute the maximum signatures across all scales Rs.

    Parameters
    ----------
    Rs : array
        The scales.
    field : 3D array
        The field.
    alg : string, optional
        The algorithm to use. Can be either 'NEXUS' or 'NEXUSPLUS'.
        Default is 'NEXUSPLUS'.
    eps : float, optional
        A small number to avoid division by zero errors and whatnot. Default is 1e-8.

    Returns
    -------
    sigmax : 4D array
        The maximum signatures.
    """
    if alg not in ["NEXUS", "NEXUSPLUS"]:  # pragma: no cover
        msg = "alg must be either 'NEXUS' or 'NEXUSPLUS'"
        raise ValueError(msg)

    nx, ny, nz = field.shape

    # Make sure the field has no 0 values
    field = field + eps

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
        if alg == "NEXUS":
            f_Rn = smooth_gauss(field, R, wave_vecs)
        elif alg == "NEXUSPLUS":
            f_Rn = smooth_loggauss(field, R, wave_vecs)

        H_Rn = fast_hessian_from_smoothed(f_Rn, R, wave_vecs)

        sigs_Rn = signatures_from_hessian(H_Rn)

        perform_loop(sigmax, sigs_Rn, nx, ny, nz)

    return sigmax
