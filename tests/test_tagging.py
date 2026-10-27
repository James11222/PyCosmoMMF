from __future__ import annotations

import numpy as np
import pytest
import skimage.measure

import pycosmommf as m
from pycosmommf.tagging import (
    _clump_mean_overdensity,
    _cluster_signature_thresholds,
    _log10S_grid,
)

N = 48


def _clump_masks():
    """
    Three cubes of voxels: ``big`` spans every x-plane, ``corner`` touches
    ``big`` only at a single corner (so it merges with it under 26-connectivity),
    and ``small`` is isolated.
    """
    big = np.zeros((N, N, N), dtype=bool)
    big[:, 10:20, 10:20] = True
    corner = np.zeros_like(big)
    corner[20:24, 20:24, 20:24] = True
    small = np.zeros_like(big)
    small[30:34, 30:34, 2:6] = True
    return big, corner, small


def _clump_fields(signature_value=50.0):
    big, corner, small = _clump_masks()
    rng = np.random.default_rng(42)
    delta = np.zeros((N, N, N), dtype=np.float32)
    delta[big] = rng.uniform(0.0, 1000.0, big.sum()).astype(np.float32)
    delta[corner] = 400.0
    delta[small] = 100.0
    max_sigs = np.zeros((N, N, N, 3), dtype=np.float32)
    max_sigs[big | corner | small, 0] = signature_value
    return delta, max_sigs, (big | corner, small)


@pytest.mark.parametrize(("Smin", "Smax"), [(-3, 2), (-2.5, 1.5), (-1, 3)])
def test_log10S_grid_matches_arange(Smin, Smax):
    grid = _log10S_grid(Smin, Smax)
    reference = np.arange(Smin, Smax + 0.1, 0.1)
    assert grid.dtype == np.float32
    assert grid.shape == reference.shape
    np.testing.assert_allclose(grid, reference, atol=1e-6)


def test_float32_throughout():
    """float64 input is converted; every output array and number is float32."""
    rng = np.random.default_rng(3)
    density = np.exp(rng.standard_normal((24, 24, 24)))
    density /= density.mean()
    assert density.dtype == np.float64

    sigs = m.maximum_signature([1.0, 2.0], density, algorithm="NEXUSPLUS")
    assert sigs.dtype == np.float32

    structure_filter, summary = m.calc_structure_bools(
        density, sigs.astype(np.float64), True
    )
    assert structure_filter.dtype == np.uint8
    for value in summary.values():
        if isinstance(value, list):
            assert all(isinstance(v, np.float32) for v in value)
        elif value is not None:
            assert np.asarray(value).dtype == np.float32


def test_clump_mean_overdensity_large_clump_is_exact():
    delta, max_sigs, (merged, small) = _clump_fields()
    clumps, n_clumps = skimage.measure.label(
        max_sigs[..., 0] > 0, connectivity=3, return_num=True
    )
    assert n_clumps == 2  # the corner-touching cube merges with the big one

    means = _clump_mean_overdensity(clumps, n_clumps, delta)
    for label in range(1, n_clumps + 1):
        mask = clumps == label
        expected = delta[mask].astype(np.float64).mean()
        assert means.dtype == np.float32
        np.testing.assert_allclose(means[label - 1], expected, rtol=1e-4)
    assert {int(np.sum(clumps == lab)) for lab in (1, 2)} == {
        int(merged.sum()),
        int(small.sum()),
    }


def test_make_the_clusbool_virialized_fractions():
    delta, max_sigs, (merged, small) = _clump_fields()
    expected_fraction = (np.mean(delta[merged].astype(np.float64)) > 370) / 2 + (
        np.mean(delta[small]) > 370
    ) / 2

    clusbool, S_th, thresholds, fractions = m.make_the_clusbool(delta, max_sigs, 370)

    np.testing.assert_array_equal(thresholds, _cluster_signature_thresholds())
    np.testing.assert_array_equal(fractions, np.full(10, expected_fraction))
    assert S_th == thresholds[0]
    np.testing.assert_array_equal(clusbool, merged | small)


def test_make_the_clusbool_thresholds_without_clumps():
    # Signatures of 5 are below thresholds > 5, which then contain no clumps.
    delta, max_sigs, _ = _clump_fields(signature_value=5.0)
    clusbool, S_th, thresholds, fractions = m.make_the_clusbool(delta, max_sigs, 370)

    empty = thresholds >= 5.0
    assert np.all(np.isnan(fractions[empty]))
    assert np.all(np.isfinite(fractions[~empty]))
    assert S_th < 5.0
    assert clusbool.any()


def test_make_the_clusbool_no_clumps_is_empty(capsys):
    # No clumps at any threshold: every fraction is NaN and clusbool is empty.
    delta, max_sigs, _ = _clump_fields()
    clusbool, S_th, _, fractions = m.make_the_clusbool(
        delta, np.zeros_like(max_sigs), 370
    )
    assert np.all(np.isnan(fractions))
    assert S_th is None
    assert not clusbool.any()
    assert "not sufficiently high to resolve virialized" in capsys.readouterr().out


def test_make_the_clusbool_nothing_virialized_is_empty(capsys):
    # Clumps exist at every threshold, but none has a mean δ above 1e6.
    delta, max_sigs, _ = _clump_fields()
    clusbool, S_th, _, fractions = m.make_the_clusbool(delta, max_sigs, 1e6)
    np.testing.assert_array_equal(fractions, np.zeros(10))
    assert S_th is None
    assert not clusbool.any()
    assert "not sufficiently high to resolve virialized" in capsys.readouterr().out


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_calc_structure_bools_finds_clusters(verbose_flag):
    """Runs the full tagging including the cluster step (``clusbool=None``)."""
    rng = np.random.default_rng(1)
    density = np.exp(2.5 * rng.standard_normal((32, 32, 32))).astype(np.float32)
    density /= density.mean()
    Rs = [np.sqrt(2) ** n for n in range(3)]
    sigs = m.maximum_signature(Rs, density, algorithm="NEXUSPLUS")

    out = m.calc_structure_bools(density, sigs, verbose_flag=verbose_flag)
    structure_filter = out[0] if verbose_flag else out

    assert structure_filter.dtype == np.uint8
    assert structure_filter.shape == density.shape
    # every voxel gets exactly one tag: 1 cluster, 2 filament, 3 wall, 4 void
    assert set(np.unique(structure_filter)) == {1, 2, 3, 4}
    if verbose_flag:
        summary = out[1]
        volume = [np.mean(structure_filter == k) for k in (1, 2, 3, 4)]
        np.testing.assert_allclose(summary["volume_fractions"], volume, rtol=1e-6)
        np.testing.assert_allclose(sum(summary["volume_fractions"]), 1.0)
        np.testing.assert_allclose(sum(summary["mass_fractions"]), 1.0, rtol=1e-5)


@pytest.mark.parametrize("backend", ["cupy", "jax"])
def test_calc_structure_bools_rejects_unknown_backend(backend):
    density = np.ones((8, 8, 8), dtype=np.float32)
    sigs = np.zeros((8, 8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="backend"):
        m.calc_structure_bools(density, sigs, False, backend=backend)
