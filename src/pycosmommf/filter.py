from __future__ import annotations

import numba as nb
import numpy as np

jit_compiler = nb.njit(parallel=True, fastmath=True)


def wavevectors3D(dims, box_size=(2 * np.pi, 2 * np.pi, 2 * np.pi)):
    """
    Returns the wavevectors for a 3D grid of dimensions `dims` and box size `box_size`.

    Parameters
    ----------
    dims : tuple
        The dimensions of the grid.
    box_size : tuple
        The size of the box in each dimension.

    Returns
    -------
    kx, ky, kz : tuple
        The wavevectors in each dimension.
    """
    sample_rate = 2 * np.pi * (np.array(dims) / box_size)
    kx = (np.fft.fftfreq(dims[0], 1 / sample_rate[0]) * (2 * np.pi / dims[0])).astype(
        np.float32
    )
    ky = (np.fft.fftfreq(dims[1], 1 / sample_rate[1]) * (2 * np.pi / dims[1])).astype(
        np.float32
    )
    kz = (np.fft.fftfreq(dims[2], 1 / sample_rate[2]) * (2 * np.pi / dims[2])).astype(
        np.float32
    )
    return kx, ky, kz


@jit_compiler
def kspace_gaussian_filter(R_S, kv):  # pragma: no cover
    """
    create a Gaussian filter in k-space.

    Parameters
    ----------
    R_S : float
        The smoothing scale.
    kv : tuple
        The wavevectors in each dimension.

    Returns
    -------
    filter_k : array
        The filter in k-space.
    """
    kx, ky, kz = kv
    nx, ny, nz = len(kx), len(ky), len(kz)
    filter_k = np.zeros((nx, ny, nz), dtype=np.float32)
    for ix in nb.prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                filter_k[ix, iy, iz] = np.exp(
                    -(kx[ix] ** 2 + ky[iy] ** 2 + kz[iz] ** 2) * R_S**2 / 2
                )
    return filter_k


@jit_compiler
def kspace_top_hat_filter(R_S, kv):  # pragma: no cover
    """
    create a top-hat filter in k-space.

    Parameters
    ----------
    R_S : float
        The smoothing scale.
    kv : tuple
        The wavevectors in each dimension.

    Returns
    -------
    filter_k : array
        The filter in k-space.
    """

    kx, ky, kz = kv
    nx, ny, nz = len(kx), len(ky), len(kz)
    filter_k = np.zeros((nx, ny, nz), dtype=np.float32)

    for ix in nb.prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                k = np.sqrt(kx[ix] ** 2 + ky[iy] ** 2 + kz[iz] ** 2)
                if k == 0.0:
                    filter_k[ix, iy, iz] = 1.0
                else:
                    filter_k[ix, iy, iz] = (
                        3
                        * (np.sin(k * R_S) - (k * R_S) * np.cos(k * R_S))
                        / (k * R_S) ** 3
                    )

    return filter_k


def smooth_top_hat(f, R_S, kv):
    """
    apply a top-hat filter to a field f.

    Parameters
    ----------
    f : array
        The field to be smoothed.
    R_S : float
        The smoothing scale.
    kv : tuple
        The wavevectors in each dimension.

    Returns
    -------
    f_Rn : array
        The smoothed field.
    """
    GF = kspace_top_hat_filter(R_S, kv)
    f_Rn = np.real(np.fft.ifftn(GF * np.fft.fftn(f)))
    f_Rn = f_Rn * (np.sum(f) / np.sum(f_Rn))
    return f_Rn.astype(np.float32)


def smooth_gauss(f, R_S, kv):
    """
    apply a Gaussian filter to a field f.

    Parameters
    ----------
    f : array
        The field to be smoothed.
    R_S : float
        The smoothing scale.
    kv : tuple
        The wavevectors in each dimension.

    Returns
    -------
    f_Rn : array
        The smoothed field.
    """
    GF = kspace_gaussian_filter(R_S, kv)
    f_Rn = np.real(np.fft.ifftn(GF * np.fft.fftn(f)))
    f_Rn = f_Rn * (np.sum(f) / np.sum(f_Rn))
    return f_Rn.astype(np.float32)


def smooth_loggauss(f, R_S, kv):
    """
    apply a Gaussian filter to the log of a field f.

    Parameters
    ----------
    f : array
        The field to be smoothed.
    R_S : float
        The smoothing scale.
    kv : tuple
        The wavevectors in each dimension.

    Returns
    -------
    f_Rn : array
        The smoothed field.
    """
    # Get filter in Fourier space
    GF = kspace_gaussian_filter(R_S, kv)

    # Convert f to log scale
    f_log = np.log10(f)

    # Perform FFT
    f_fft = np.fft.fftn(f_log)

    # Convolve with the filter
    f_conv = GF * f_fft

    # Perform IFFT
    f_ifft = np.fft.ifftn(f_conv)

    # Convert back to linear scale
    f_result = 10 ** np.real(f_ifft)

    # Normalize the result
    norm = np.sum(f) / np.sum(f_result)
    f_result *= norm

    return f_result.astype(np.float32)
