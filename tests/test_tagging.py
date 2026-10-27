from __future__ import annotations

import numpy as np
import pytest
import skimage.measure

import pycosmommf as m
from pycosmommf.tagging import (
    _cast_threshold,
    _clump_mean_overdensity,
    _cluster_signature_thresholds,
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


def test_cast_threshold_matches_float64_comparison():
    rng = np.random.default_rng(0)
    thresholds = [0.1, 1.0, 3.5, -0.7, 10**-2.95, 10**1.85, float(np.float32(0.1))]
    for t in thresholds:
        t32 = np.float32(t)
        near = [t32]
        for _ in range(3):
            near.append(np.nextafter(near[-1], np.float32(np.inf)))
        near.append(np.nextafter(t32, np.float32(-np.inf)))
        x = np.concatenate(
            [rng.uniform(-50, 50, 1000).astype(np.float32), np.array(near)]
        )
        exact_gt = x.astype(np.float64) > t
        exact_lt = x.astype(np.float64) < t
        assert np.array_equal(x > _cast_threshold(t, x.dtype, ">"), exact_gt)
        assert np.array_equal(x < _cast_threshold(t, x.dtype, "<"), exact_lt)
        assert _cast_threshold(t, x.dtype, ">").dtype == np.float32

    assert _cast_threshold(0.1, np.float64, ">") == 0.1
    assert _cast_threshold(0.1, np.float64, ">").dtype == np.float64


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
        np.testing.assert_allclose(means[label - 1], expected, rtol=1e-12)
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

    # No clumps at any threshold: every fraction is NaN and clusbool is empty.
    clusbool, S_th, _, fractions = m.make_the_clusbool(
        delta, np.zeros_like(max_sigs), 370
    )
    assert np.all(np.isnan(fractions))
    assert S_th == thresholds[0]
    assert not clusbool.any()


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_calc_structure_bools_finds_clusters(verbose_flag):
    """Runs the full tagging including the cluster step (``clusbool=None``)."""
    rng = np.random.default_rng(1)
    density = np.exp(2.5 * rng.standard_normal((32, 32, 32))).astype(np.float32)
    density /= density.mean()
    Rs = [np.sqrt(2) ** n for n in range(3)]
    sigs = m.maximum_signature(Rs, density, algorithm="NEXUSPLUS")

    out = m.calc_structure_bools(density, sigs, verbose_flag=verbose_flag)
    clusbool, filbool, wallbool, voidbool = out[:4]

    total = (
        clusbool.astype(int)
        + filbool.astype(int)
        + wallbool.astype(int)
        + voidbool.astype(int)
    )
    assert np.all(total == 1)  # every voxel gets exactly one tag
    if verbose_flag:
        summary = out[4]
        np.testing.assert_allclose(sum(summary["volume_fractions"]), 1.0)
        np.testing.assert_allclose(sum(summary["mass_fractions"]), 1.0, rtol=1e-6)


def test_calc_structure_bools_rejects_unknown_backend():
    density = np.ones((8, 8, 8), dtype=np.float32)
    sigs = np.zeros((8, 8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="backend"):
        m.calc_structure_bools(density, sigs, False, backend="gpu")
