"""
Tests for the GPU (CuPy) structure-tagging backend.

All tests in this module are skipped automatically when CuPy is not installed
or no CUDA GPU is available. They verify that:
  1. ``cupyx.scipy.ndimage.label`` with the 3×3×3 structure reproduces the
     26-connected clumps of ``skimage.measure.label``.
  2. ``calc_structure_bools(..., backend="gpu")`` gives the same thresholds
     and boolean filters as the CPU backend, up to float32 summation-order
     differences.
  3. NumPy, CuPy and JAX inputs give the same result.
"""

from __future__ import annotations

import numpy as np
import pytest
import skimage.measure
from scipy.ndimage import gaussian_filter

cp = pytest.importorskip("cupy")
try:
    _n_gpus = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    _n_gpus = 0
if _n_gpus == 0:
    pytest.skip("no CUDA GPU available", allow_module_level=True)

from cupyx.scipy import ndimage  # noqa: E402

import pycosmommf as m  # noqa: E402
from pycosmommf._cupy_tagging import _make_the_clusbool_cupy  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_N = 48
# Lognormal field with clumps up to δ ~ 4000, so the virialized fractions span
# 0 to 1 across the cluster thresholds.
_rng = np.random.default_rng(1)
_gauss = gaussian_filter(_rng.standard_normal((_N,) * 3), 1.0)
_density = np.exp(2.5 * _gauss / _gauss.std())
_density = (_density / _density.mean()).astype(np.float32)
_Rs = [np.sqrt(2) ** n for n in range(4)]
_sigs = m.maximum_signature(_Rs, _density, algorithm="NEXUSPLUS", backend="cpu")


def _assert_same_partition(labels_a, labels_b):
    """Both label arrays describe the same clumps (numbering may differ)."""
    np.testing.assert_array_equal(labels_a > 0, labels_b > 0)
    fg = labels_a > 0
    pairs = np.unique(np.stack([labels_a[fg], labels_b[fg]]), axis=1)
    assert pairs.shape[1] == labels_a.max() == labels_b.max()


def _assert_same_tagging(cpu_out, gpu_out):
    """
    Thresholds must match exactly. Both backends sum in float32 but in a
    different order, so curves and fractions agree to float32 precision and a
    voxel sitting exactly on a threshold may in principle flip.
    """
    for cpu_bool, gpu_bool in zip(cpu_out[:4], gpu_out[:4]):
        assert gpu_bool.dtype == np.bool_
        assert np.count_nonzero(cpu_bool != gpu_bool) <= 1e-4 * cpu_bool.size
    if len(cpu_out) == 5:
        cpu_sum, gpu_sum = cpu_out[4], gpu_out[4]
        for key in ("S_clus", "f_vir_clus", "S_fil", "S_fil_thresh", "S_wall"):
            np.testing.assert_array_equal(cpu_sum[key], gpu_sum[key])
        assert cpu_sum["S_wall_thresh"] == gpu_sum["S_wall_thresh"]
        for key in ("dM2_fil", "dM2_wall"):
            np.testing.assert_allclose(
                cpu_sum[key], gpu_sum[key], rtol=1e-4, atol=1e-5 * cpu_sum[key].max()
            )
        for key in ("mass_fractions", "volume_fractions"):
            np.testing.assert_allclose(cpu_sum[key], gpu_sum[key], rtol=1e-5)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("sigma", "quantile"), [(0.0, 0.75), (1.0, 0.6), (2.0, 0.3)])
def test_label_matches_skimage(sigma, quantile):
    field = np.random.default_rng(0).standard_normal((40, 40, 40))
    if sigma:
        field = gaussian_filter(field, sigma)
    mask = field > np.quantile(field, quantile)
    # Two cubes that touch only at a corner: one clump under 26-connectivity.
    mask[:4, :4, :4] = True
    mask[4:8, 4:8, 4:8] = True

    expected = skimage.measure.label(mask, connectivity=3)
    labels, n = ndimage.label(cp.asarray(mask), structure=np.ones((3, 3, 3), bool))
    assert n == expected.max()
    _assert_same_partition(expected, cp.asnumpy(labels))


@pytest.mark.parametrize("overdensity_threshold", [5, 50, 370])
@pytest.mark.parametrize("sig_scale", [1.0, 0.05])
def test_make_the_clusbool_matches_cpu(overdensity_threshold, sig_scale):
    # sig_scale=0.05 leaves the upper thresholds without clumps (NaN fractions).
    delta = _density - 1.0
    sigs = (_sigs * sig_scale).astype(np.float32)
    cpu = m.make_the_clusbool(delta, sigs, overdensity_threshold)
    gpu = _make_the_clusbool_cupy(
        cp.asarray(delta), cp.asarray(sigs)[..., 0], overdensity_threshold
    )
    if sig_scale < 1:
        assert np.isnan(cpu[3]).any()

    np.testing.assert_array_equal(cpu[0], cp.asnumpy(gpu[0]))
    assert cpu[1] == gpu[1]  # both None when no clump is virialized
    np.testing.assert_array_equal(cpu[2], gpu[2])
    np.testing.assert_array_equal(cpu[3], gpu[3])  # NaN where no clumps


# ---------------------------------------------------------------------------
# Full tagging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verbose_flag", [True, False])
@pytest.mark.parametrize("given_clusbool", [False, True])
def test_calc_structure_bools_matches_cpu(verbose_flag, given_clusbool):
    clusbool = _sigs[..., 0] > 1.0 if given_clusbool else None
    kwargs = {"verbose_flag": verbose_flag, "clusbool": clusbool}
    cpu_out = m.calc_structure_bools(_density, _sigs, backend="cpu", **kwargs)
    gpu_out = m.calc_structure_bools(_density, _sigs, backend="gpu", **kwargs)
    _assert_same_tagging(cpu_out, gpu_out)


def test_calc_structure_bools_accepts_cupy_input():
    cpu_out = m.calc_structure_bools(_density, _sigs, True)
    gpu_out = m.calc_structure_bools(
        cp.asarray(_density), cp.asarray(_sigs), True, backend="gpu"
    )
    _assert_same_tagging(cpu_out, gpu_out)


def test_calc_structure_bools_accepts_jax_input():
    pytest.importorskip("jax")
    sigs_jax = m.maximum_signature(
        _Rs, _density, algorithm="NEXUSPLUS", backend="gpu", return_numpy=False
    )
    from_numpy = m.calc_structure_bools(
        _density, np.asarray(sigs_jax), True, backend="gpu"
    )
    from_jax = m.calc_structure_bools(_density, sigs_jax, True, backend="gpu")
    _assert_same_tagging(from_numpy, from_jax)


def test_calc_structure_bools_cupy_rejects_overdensity():
    with pytest.raises(ValueError, match="overdensity"):
        m.calc_structure_bools(_density - 1.0, _sigs, False, backend="gpu")


def test_calc_structure_bools_gpu_unresolved_clusters_is_empty(capsys):
    # A huge overdensity threshold: no clump is virialized at any threshold.
    out = m.calc_structure_bools(
        _density, _sigs, False, overdensity_threshold=1e9, backend="gpu"
    )
    assert not out[0].any()
    assert "not sufficiently high to resolve virialized" in capsys.readouterr().out
