from __future__ import annotations

import numpy as np

import pycosmommf as m

test_field = (
    np.roll(m.sphere(32, 5), -10, axis=0)
    + np.roll(m.sphere(32, 5), 10, axis=0)
    + m.cylinder(32, 5)
)


def test_wavevectors3D():
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    assert kx.shape == (32,)
    assert ky.shape == (32,)
    assert kz.shape == (32,)


def test_kspace_gaussian_filter():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    filter_k = m.kspace_gaussian_filter(R_S, (kx, ky, kz))
    assert filter_k.shape == (32, 32, 32)


def test_kspace_top_hat_filter():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    filter_k = m.kspace_top_hat_filter(R_S, (kx, ky, kz))
    assert filter_k.shape == (32, 32, 32)


def test_smooth_top_hat():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    f = test_field
    f_smooth = m.smooth_top_hat(f, R_S, (kx, ky, kz))
    assert f_smooth.shape == (32, 32, 32)  # correct shape
    assert np.any(f_smooth != f)  # not equal to original


def test_smooth_gauss():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    f = test_field
    f_smooth = m.smooth_gauss(f, R_S, (kx, ky, kz))
    assert f_smooth.shape == (32, 32, 32)  # correct shape
    assert np.any(f_smooth != f)  # not equal to original


def test_smooth_loggauss():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    f = test_field
    f_smooth = m.smooth_loggauss(f, R_S, (kx, ky, kz))
    assert f_smooth.shape == (32, 32, 32)  # correct shape
    assert np.any(f_smooth != f)  # not equal to original


def test_fast_hessian_from_smoothed():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    f = test_field
    f_smooth = m.smooth_gauss(f, R_S, (kx, ky, kz))
    hessian = m.fast_hessian_from_smoothed(f_smooth, R_S, (kx, ky, kz))
    assert hessian.shape == (32, 32, 32, 6)
    for i in range(6):
        assert np.all(hessian[:, :, :, i] != 0.0)


def test_signatures_from_hessian():
    R_S = 1.0
    dims = (32, 32, 32)
    box_size = (2 * np.pi, 2 * np.pi, 2 * np.pi)
    kx, ky, kz = m.wavevectors3D(dims, box_size)
    f = test_field
    f_smooth = m.smooth_gauss(f, R_S, (kx, ky, kz))
    hessian = m.fast_hessian_from_smoothed(f_smooth, R_S, (kx, ky, kz))
    sigs = m.signatures_from_hessian(hessian)
    assert sigs.shape == (32, 32, 32, 3)


def test_shrink():
    data = test_field
    new_size = 16
    shrunk = m.shrink(data, new_size)
    assert shrunk.shape == (16, 16, 16)
