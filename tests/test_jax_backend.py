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

    # The JAX backend uses an analytic Smith-1961 closed-form solver for the
    # 3×3 symmetric eigenvalue problem (so it can skip the (N,N,N,3,3) tensor
    # and the eigenvector workspace that ``jnp.linalg.eigh`` would allocate).
    # On near-degenerate Hessians (background voxels with ~constant field)
    # the analytic formula and NumPy's general ``eigvals`` differ by a few
    # ULP in float32, which the signature step's Heaviside-threshold cutoffs
    # can amplify to ~5e-5 absolute. These voxels are all well below the
    # structure-tagging cutoff, so atol=1e-4 (still ~10× tighter than the
    # end-to-end tolerance below) is the right floor here.
    np.testing.assert_allclose(jax_sigs, cpu_sigs, rtol=1e-3, atol=1e-4)


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


# ---------------------------------------------------------------------------
# Input validation for the log-Gauss (NEXUSPLUS) path
# ---------------------------------------------------------------------------


def test_jax_backend_nexusplus_rejects_negative_input():
    """JAX path must raise the same clear error as the CPU path on δ."""
    delta = _field / np.mean(_field) - 1.0
    with pytest.raises(ValueError, match="NEXUSPLUS"):
        m.maximum_signature(_Rs, delta, algorithm="NEXUSPLUS", backend="jax")


def test_jax_backend_direct_call_rejects_negative_input():
    """Same check fires when ``maximum_signature_jax`` is called directly."""
    from pycosmommf._jax_backend import maximum_signature_jax

    delta = _field / np.mean(_field) - 1.0
    with pytest.raises(ValueError, match="NEXUSPLUS"):
        maximum_signature_jax(_Rs, delta, algorithm="NEXUSPLUS")


def test_jax_backend_nexus_allows_negative_input():
    """NEXUS path does not log-transform, so negative input must work."""
    delta = _field / np.mean(_field) - 1.0
    sigs = m.maximum_signature(_Rs, delta, algorithm="NEXUS", backend="jax")
    assert sigs.shape == delta.shape + (3,)
