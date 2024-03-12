import numpy as np
from numba import njit

def wavevectors3D(dims, box_size=(2*np.pi, 2*np.pi, 2*np.pi)):
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
    sample_rate = 2*np.pi * (np.array(dims) / box_size)
    kx = np.fft.fftfreq(dims[0], 1/sample_rate[0]) * (2*np.pi / dims[0])
    ky = np.fft.fftfreq(dims[1], 1/sample_rate[1]) * (2*np.pi / dims[1])
    kz = np.fft.fftfreq(dims[2], 1/sample_rate[2]) * (2*np.pi / dims[2])
    return kx, ky, kz

@njit
def kspace_gaussian_filter(R_S, kv):
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
    filter_k = np.zeros((nx, ny, nz))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                filter_k[ix, iy, iz] = np.exp(
                    -(kx[ix]**2 + ky[iy]**2 + kz[iz]**2) * R_S**2 / 2)
    return filter_k

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
    return f_Rn


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

    return f_result

