from __future__ import annotations

import numba as nb
import numpy as np
import scipy.fft

jit_compiler = nb.njit(parallel=True, fastmath=True)


def wavevectors3D(dims, box_size=(2 * np.pi, 2 * np.pi, 2 * np.pi)):
    """
    Returns the float32 wavevectors for a 3D grid of dimensions ``dims`` and box size ``box_size``.

    Args:
        dims (:obj:`tuple`):
            The dimensions of the grid ``(nx, ny, nz)``.
        box_size (:obj:`tuple`, optional):
            The size of the box in each dimension ``(Lx, Ly, Lz)``. Defaults to ``(2π, 2π, 2π)``.

    Returns:
        (tuple): a tuple containing:
            - **kx** (:obj:`np.ndarray`): The wavevector in the x-direction.
            - **ky** (:obj:`np.ndarray`): The wavevector in the y-direction.
            - **kz** (:obj:`np.ndarray`): The wavevector in the z-direction.
    """
    return tuple(_wavevector(n, L) for n, L in zip(dims, box_size))


def _wavevector(n, box_length):
    """
    float32 wavevector along one axis: the FFT frequency index (``0, 1, ...,
    -1`` as in ``np.fft.fftfreq``) times ``(2π)² / (box_length · n)``.
    """
    index = np.arange(n, dtype=np.int64)
    index[index >= (n + 1) // 2] -= n
    spacing = np.float32((2 * np.pi) ** 2 / (box_length * n))
    return index.astype(np.float32) * spacing


@jit_compiler
def kspace_gaussian_filter(R_S, kv):  # pragma: no cover
    """
    create a Gaussian filter in k-space.

    Args:
        R_S (:obj:`float`):
            The smoothing scale.
        kv (:obj:`tuple`):
            The wavevectors in each dimension, each a ``1D float np.ndarray``.

    Returns:
        (:obj:`3D float32 np.ndarray`): The filter in k-space.
    """
    kx, ky, kz = kv
    nx, ny, nz = len(kx), len(ky), len(kz)
    filter_k = np.zeros((nx, ny, nz), dtype=np.float32)
    half_R2 = np.float32(0.5) * np.float32(R_S) * np.float32(R_S)
    for ix in nb.prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                k2 = kx[ix] * kx[ix] + ky[iy] * ky[iy] + kz[iz] * kz[iz]
                filter_k[ix, iy, iz] = np.exp(-k2 * half_R2)
    return filter_k


@jit_compiler
def kspace_top_hat_filter(R_S, kv):  # pragma: no cover
    """
    create a top-hat filter in k-space.

    Args:
        R_S (:obj:`float`):
            The smoothing scale.
        kv (:obj:`tuple`):
            The wavevectors in each dimension.

    Returns:
        (:obj:`3D float32 np.ndarray`): The filter in k-space.
    """

    kx, ky, kz = kv
    nx, ny, nz = len(kx), len(ky), len(kz)
    filter_k = np.zeros((nx, ny, nz), dtype=np.float32)
    R = np.float32(R_S)
    three = np.float32(3.0)

    for ix in nb.prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                k = np.sqrt(kx[ix] * kx[ix] + ky[iy] * ky[iy] + kz[iz] * kz[iz])
                if k == 0.0:
                    filter_k[ix, iy, iz] = 1.0
                else:
                    kR = k * R
                    filter_k[ix, iy, iz] = (
                        three * (np.sin(kR) - kR * np.cos(kR)) / (kR * kR * kR)
                    )

    return filter_k


def smooth_top_hat(f, R_S, kv):
    """
    apply a top-hat filter to a field f.

    Args:
        f (:obj:`3D float np.ndarray`):
            The field to be smoothed.
        R_S (:obj:`float`):
            The smoothing scale in units of voxels.
        kv (:obj:`tuple`):
            The wavevectors in each dimension.

    Returns:
        (:obj:`3D float32 np.ndarray`): The smoothed field.
    """
    f = np.asarray(f, dtype=np.float32)
    GF = kspace_top_hat_filter(np.float32(R_S), kv)
    f_Rn = np.real(scipy.fft.ifftn(GF * scipy.fft.fftn(f)))
    return f_Rn * (np.sum(f) / np.sum(f_Rn))


def smooth_gauss(f, R_S, kv):
    """
    apply a Gaussian filter to a field f.

    Args:
        f (:obj:`3D float np.ndarray`):
            The field to be smoothed.
        R_S (:obj:`float`):
            The smoothing scale in units of voxels.
        kv (:obj:`tuple`):
            The wavevectors in each dimension.

    Returns:
        (:obj:`3D float32 np.ndarray`): The smoothed field.
    """
    f = np.asarray(f, dtype=np.float32)
    GF = kspace_gaussian_filter(np.float32(R_S), kv)
    f_Rn = np.real(scipy.fft.ifftn(GF * scipy.fft.fftn(f)))
    return f_Rn * (np.sum(f) / np.sum(f_Rn))


def smooth_loggauss(f, R_S, kv):
    """
    apply a Gaussian filter to the log of a field f.

    Args:
        f (:obj:`3D float np.ndarray`):
            The field to be smoothed.
        R_S (:obj:`float`):
            The smoothing scale in units of voxels.
        kv (:obj:`tuple`):
            The wavevectors in each dimension.

    Returns:
        (:obj:`3D float32 np.ndarray`): The smoothed field.
    """
    f = np.asarray(f, dtype=np.float32)

    # Get filter in Fourier space
    GF = kspace_gaussian_filter(np.float32(R_S), kv)

    # Convert f to log scale
    f_log = np.log10(f)

    # Perform FFT
    f_fft = scipy.fft.fftn(f_log)

    # Convolve with the filter
    f_conv = GF * f_fft

    # Perform IFFT
    f_ifft = scipy.fft.ifftn(f_conv)

    # Convert back to linear scale
    f_result = np.power(np.float32(10.0), np.real(f_ifft))

    # Normalize the result
    norm = np.sum(f) / np.sum(f_result)
    f_result *= norm

    return f_result
