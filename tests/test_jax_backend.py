"""
Tests for the JAX backend.

All tests in this module are skipped automatically when JAX is not installed.
They verify that:
  1. The JAX backend produces arrays with the correct shape.
  2. The JAX backend results are numerically close to the CPU backend results
     (within float32 precision — FFT implementations differ slightly).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import pycosmommf as m  # noqa: E402 (import after skip guard)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Synthetic field: two spheres + a cylinder, same as test_integration.py
_field = (
    np.roll(m.sphere(32, 2), -10, axis=0)
    + np.roll(m.sphere(32, 2), 10, axis=0)
    + m.cylinder(32, 5)
    + m.wall(32)
)
_Rs = [np.sqrt(2) ** n for n in range(5)]

# Pre-computed CPU reference so we run the (slow) Numba compile once
_cpu_nexusplus = m.maximum_signature(_Rs, _field, algorithm="NEXUSPLUS", backend="cpu")
_cpu_nexus = m.maximum_signature(_Rs, _field, algorithm="NEXUS", backend="cpu")


# ---------------------------------------------------------------------------
# Individual JAX pipeline function tests
# ---------------------------------------------------------------------------


def test_smooth_gauss_jax_shape():
    from pycosmommf._jax_backend import smooth_gauss_jax

    kv = m.wavevectors3D((32, 32, 32))
    out = smooth_gauss_jax(_field, 1.0, kv)
    assert out.shape == (32, 32, 32)


def test_smooth_loggauss_jax_shape():
    from pycosmommf._jax_backend import smooth_loggauss_jax

    kv = m.wavevectors3D((32, 32, 32))
    out = smooth_loggauss_jax(_field, 1.0, kv)
    assert out.shape == (32, 32, 32)


def test_smooth_gauss_jax_matches_cpu():
    from pycosmommf._jax_backend import smooth_gauss_jax

    kv = m.wavevectors3D((32, 32, 32))
    jax_out = np.asarray(smooth_gauss_jax(_field, 1.0, kv))
    cpu_out = m.smooth_gauss(_field, 1.0, kv)
    # JAX (XLA FFT) and NumPy (FFTW) accumulate float32 rounding differently;
    # observed max relative difference is ~1.2e-4, so 5e-4 gives 4× headroom.
    np.testing.assert_allclose(jax_out, cpu_out, rtol=5e-4, atol=1e-5)


def test_smooth_loggauss_jax_matches_cpu():
    from pycosmommf._jax_backend import smooth_loggauss_jax

    kv = m.wavevectors3D((32, 32, 32))
    jax_out = np.asarray(smooth_loggauss_jax(_field, 1.0, kv))
    cpu_out = m.smooth_loggauss(_field, 1.0, kv)
    np.testing.assert_allclose(jax_out, cpu_out, rtol=5e-4, atol=1e-5)


def test_fast_hessian_jax_shape():
    from pycosmommf._jax_backend import fast_hessian_from_smoothed_jax

    kv = m.wavevectors3D((32, 32, 32))
    f_smooth = m.smooth_gauss(_field, 1.0, kv)
    H = fast_hessian_from_smoothed_jax(f_smooth, 1.0, kv)
    assert H.shape == (32, 32, 32, 6)


def test_fast_hessian_jax_matches_cpu():
    from pycosmommf._jax_backend import fast_hessian_from_smoothed_jax

    kv = m.wavevectors3D((32, 32, 32))
    f_smooth = m.smooth_gauss(_field, 1.0, kv)
    jax_H = np.asarray(fast_hessian_from_smoothed_jax(f_smooth, 1.0, kv))
    cpu_H = m.fast_hessian_from_smoothed(f_smooth, 1.0, kv)
    np.testing.assert_allclose(jax_H, cpu_H, rtol=1e-4, atol=1e-5)


def test_signatures_from_hessian_jax_shape():
    from pycosmommf._jax_backend import (
        fast_hessian_from_smoothed_jax,
        signatures_from_hessian_jax,
    )

    kv = m.wavevectors3D((32, 32, 32))
    f_smooth = m.smooth_gauss(_field, 1.0, kv)
    H = fast_hessian_from_smoothed_jax(f_smooth, 1.0, kv)
    sigs = signatures_from_hessian_jax(H)
    assert sigs.shape == (32, 32, 32, 3)


def test_signatures_from_hessian_jax_matches_cpu():
    from pycosmommf._jax_backend import (
        fast_hessian_from_smoothed_jax,
        signatures_from_hessian_jax,
    )

    kv = m.wavevectors3D((32, 32, 32))
    f_smooth = m.smooth_gauss(_field, 1.0, kv)

    # Use the same hessian for both so differences come only from eigensolver
    cpu_H = m.fast_hessian_from_smoothed(f_smooth, 1.0, kv)
    jax_sigs = np.asarray(signatures_from_hessian_jax(cpu_H))
    cpu_sigs = m.signatures_from_hessian(cpu_H)

    # eigh (JAX symmetric solver) vs eigvals+real (NumPy general solver) may
    # differ slightly; we tolerate relative error up to 1e-3 in float32.
    np.testing.assert_allclose(jax_sigs, cpu_sigs, rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end maximum_signature tests via the backend= parameter
# ---------------------------------------------------------------------------


def test_maximum_signature_jax_nexusplus_shape():
    sigs = m.maximum_signature(_Rs, _field, algorithm="NEXUSPLUS", backend="jax")
    assert sigs.shape == (32, 32, 32, 3)


def test_maximum_signature_jax_nexus_shape():
    sigs = m.maximum_signature(_Rs, _field, algorithm="NEXUS", backend="jax")
    assert sigs.shape == (32, 32, 32, 3)


def test_maximum_signature_jax_nexusplus_matches_cpu():
    jax_sigs = m.maximum_signature(_Rs, _field, algorithm="NEXUSPLUS", backend="jax")
    # Float32 FFT differences compound across the multi-scale loop; near-zero
    # "background" voxels can differ enormously in relative terms but are
    # irrelevant for structure tagging.  We use atol=5e-3 to cover the worst
    # observed absolute difference (~1.3e-3) with ~4× headroom.
    np.testing.assert_allclose(jax_sigs, _cpu_nexusplus, rtol=1e-2, atol=5e-3)


def test_maximum_signature_jax_nexus_matches_cpu():
    jax_sigs = m.maximum_signature(_Rs, _field, algorithm="NEXUS", backend="jax")
    np.testing.assert_allclose(jax_sigs, _cpu_nexus, rtol=1e-2, atol=5e-3)


def test_maximum_signature_returns_numpy():
    """Output must be a plain numpy array regardless of backend."""
    sigs = m.maximum_signature(_Rs, _field, backend="jax")
    assert isinstance(sigs, np.ndarray)


# ---------------------------------------------------------------------------
# Invalid backend
# ---------------------------------------------------------------------------


def test_maximum_signature_invalid_backend():
    with pytest.raises(ValueError, match="backend must be"):
        m.maximum_signature(_Rs, _field, backend="cuda")
