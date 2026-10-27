from __future__ import annotations

import numba as nb
import numpy as np

from .filter import smooth_gauss, smooth_loggauss, wavevectors3D
from .hessian import fast_hessian_from_smoothed

jit_compiler = nb.njit(parallel=True, fastmath=True)


@nb.njit(fastmath=True, inline="always")
def _eigvalsh_3x3_sym(a, b, c, d, e, f):  # pragma: no cover
    """
    Eigenvalues of the real symmetric 3×3 matrix

        [[a, b, c],
         [b, d, e],
         [c, e, f]]

    for scalar entries. Returns ``(e_min, e_mid, e_max)`` sorted ascending,
    as float32.

    Uses the closed-form trigonometric solution of Smith (1961)
    (Communications of the ACM 4, 168). This is the same solver as
    ``_jax_backend._eigvalsh_3x3_sym``, so both backends share one
    eigenvalue method. All arithmetic is done in float32 to match the JAX
    backend.

    Numerical notes:
      * The matrix is rescaled by its max-abs element before the trig
        formula runs, which keeps the determinant-over-p³ quotient
        well-conditioned for near-zero Hessians.
      * The argument to ``arccos`` is clipped to [-1, 1].
      * The all-zero matrix and the scalar*I case are short-circuited.
    """
    one = np.float32(1.0)
    two = np.float32(2.0)
    three = np.float32(3.0)
    six = np.float32(6.0)
    zero = np.float32(0.0)

    a = np.float32(a)
    b = np.float32(b)
    c = np.float32(c)
    d = np.float32(d)
    e = np.float32(e)
    f = np.float32(f)

    scale = max(abs(a), abs(d), abs(f), abs(b), abs(c), abs(e))
    if scale == zero:
        return zero, zero, zero
    inv_scale = one / scale

    a_s = a * inv_scale
    b_s = b * inv_scale
    c_s = c * inv_scale
    d_s = d * inv_scale
    e_s = e * inv_scale
    f_s = f * inv_scale

    p1 = b_s * b_s + c_s * c_s + e_s * e_s
    q = (a_s + d_s + f_s) / three
    a_q = a_s - q
    d_q = d_s - q
    f_q = f_s - q
    p2 = a_q * a_q + d_q * d_q + f_q * f_q + two * p1
    p = np.sqrt(p2 / six)

    if p == zero:
        eig = q * scale
        return eig, eig, eig

    det_shifted = (
        a_q * (d_q * f_q - e_s * e_s)
        - b_s * (b_s * f_q - e_s * c_s)
        + c_s * (b_s * e_s - d_q * c_s)
    )

    r = det_shifted / (two * p * p * p)
    r = min(max(r, -one), one)

    phi = np.arccos(r) / three
    two_p = two * p
    eig_max = q + two_p * np.cos(phi)
    eig_min = q + two_p * np.cos(phi + np.float32(2.0 * np.pi / 3.0))
    # Trace conservation gives the middle eigenvalue without another trig call.
    eig_mid = three * q - eig_max - eig_min

    return eig_min * scale, eig_mid * scale, eig_max * scale


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
    one = np.float32(1.0)
    zero = np.float32(0.0)

    # helper functions (float32 constants keep Numba from promoting to float64)
    def θ(x):
        """
        Heaviside step function. Returns one if the argument is positive, zero otherwise.
        """
        if x > 0:
            return one
        return zero

    def xθ(x):
        """
        Heaviside step times x. Returns x if the argument is positive, zero otherwise.
        """
        if x > 0:
            return x
        return zero

    for i in nb.prange(hsize[0]):
        for j in range(hsize[1]):
            for k in range(hsize[2]):
                e1, e2, e3 = _eigvalsh_3x3_sym(
                    hessian[i, j, k, 0],
                    hessian[i, j, k, 1],
                    hessian[i, j, k, 2],
                    hessian[i, j, k, 3],
                    hessian[i, j, k, 4],
                    hessian[i, j, k, 5],
                )

                if e1 == 0:
                    sigs[i, j, k, 0] = 0
                    sigs[i, j, k, 1] = 0
                    sigs[i, j, k, 2] = 0
                else:
                    sigs[i, j, k, 0] = (np.abs(e3 / e1) * np.abs(e3)) * (
                        θ(-e1) * θ(-e2) * θ(-e3)
                    )

                    sigs[i, j, k, 1] = (np.abs(e2 / e1) * xθ(one - np.abs(e3 / e1))) * (
                        np.abs(e2) * θ(-e1) * θ(-e2)
                    )

                    sigs[i, j, k, 2] = (
                        xθ(one - np.abs(e2 / e1)) * xθ(one - np.abs(e3 / e1))
                    ) * (np.abs(e1) * θ(-e1))

    return sigs


def _validate_density_for_algorithm(density_cube, algorithm):
    """
    Raise ``ValueError`` if ``density_cube`` is not a valid input for the
    chosen ``algorithm``. Specifically, ``"NEXUSPLUS"`` takes ``log10`` of
    the density, so the input must be strictly positive — passing an
    overdensity δ (which is negative in voids) silently produces NaN
    throughout the pipeline. This check converts that failure mode into a
    clear, immediate error.

    ``"NEXUS"`` does not log-transform the field and so accepts any
    finite values.
    """
    if algorithm == "NEXUSPLUS" and np.any(density_cube < 0):
        n_neg = int(np.sum(density_cube < 0))
        min_val = float(np.min(density_cube))
        msg = (
            "NEXUSPLUS uses log10(density) and requires a strictly positive "
            f"density field. Got {n_neg} negative voxels (min = {min_val:.4g}). "
            "If you passed an overdensity δ = ρ/⟨ρ⟩ - 1, convert it to "
            "δ+1 = ρ/⟨ρ⟩ by adding 1 before calling: "
            "maximum_signature(Rs, density + 1, ...)."
        )
        raise ValueError(msg)


def maximum_signature(
    Rs, density_cube, algorithm="NEXUSPLUS", eps=1e-16, backend="cpu", return_numpy=True
):
    """
    Compute the maximum signatures across all scales Rs.

    Args:
        Rs (:obj:`list` of :obj:`float`):
            List of smoothing scales in units of voxels.
        density_cube (:obj:`3D float np.ndarray`):
            The 3D density field to analyze. For ``algorithm="NEXUSPLUS"``
            this must be strictly positive (i.e. ``δ+1 = ρ/⟨ρ⟩``); a
            ``ValueError`` is raised otherwise. It is converted to float32,
            and every computation runs in float32 on both backends.
        algorithm (:obj:`str`, optional):
            The algorithm to use for smoothing. Can be either "NEXUS" or "NEXUSPLUS". Defaults to "NEXUSPLUS".
        eps (:obj:`float`, optional):
            Small value to avoid division by zero. Defaults to ``1e-16``.
        backend (:obj:`str`, optional):
            ``"cpu"`` uses NumPy, SciPy and Numba. ``"gpu"`` uses JAX on the
            GPU; requires ``pip install 'pycosmommf[gpu]'``. Defaults to
            ``"cpu"``.
        return_numpy (:obj:`bool`, optional):
            Only used with ``backend="gpu"``. ``False`` returns the result as a
            JAX array that stays on the GPU, which ``calc_structure_bools(...,
            backend="gpu")`` can use without a copy. Defaults to ``True``.

    Returns:
        (:obj:`4D float32 np.ndarray`):
            An array of signatures with shape ``(nx, ny, nz, 3)``, where ``nx, ny, nz`` are the dimensions of the input ``density_cube``.
            The last dimension contains the signatures for clusters, filaments, and walls respectively.
    """

    if algorithm not in ["NEXUS", "NEXUSPLUS"]:  # pragma: no cover
        msg = "algorithm must be either 'NEXUS' or 'NEXUSPLUS'"
        raise ValueError(msg)

    if backend not in ["cpu", "gpu"]:
        msg = "backend must be either 'cpu' or 'gpu'"
        raise ValueError(msg)

    density_cube = np.asarray(density_cube, dtype=np.float32)
    _validate_density_for_algorithm(density_cube, algorithm)

    if backend == "gpu":
        from ._jax_backend import maximum_signature_jax

        return maximum_signature_jax(
            Rs, density_cube, algorithm=algorithm, eps=eps, return_numpy=return_numpy
        )

    nx, ny, nz = density_cube.shape

    # Make sure the field has no 0 values
    eps = np.float32(eps)
    density_cube = density_cube + eps

    # Calculate wave vectors for our field
    wave_vecs = wavevectors3D((nx, ny, nz))

    # Calculate signatures at each scale Rn, determine max
    sigmax = np.full((nx, ny, nz, 3), eps, dtype=np.float32)

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
        if algorithm == "NEXUS":
            f_Rn = smooth_gauss(density_cube, R, wave_vecs)
        elif algorithm == "NEXUSPLUS":
            f_Rn = smooth_loggauss(density_cube, R, wave_vecs)

        H_Rn = fast_hessian_from_smoothed(f_Rn, R, wave_vecs)

        sigs_Rn = signatures_from_hessian(H_Rn)

        perform_loop(sigmax, sigs_Rn, nx, ny, nz)

    return sigmax
