from __future__ import annotations

import numpy as np
import pytest

import pycosmommf as m

test_field = (
    np.roll(m.sphere(32, 2), -10, axis=0)
    + np.roll(m.sphere(32, 2), 10, axis=0)
    + m.cylinder(32, 5)
    + m.wall(32)
)


def test_maximum_signature():
    Rs = [np.sqrt(2) ** n for n in range(5)]
    field = test_field
    sigs = m.maximum_signature(Rs, field, algorithm="NEXUSPLUS")
    assert sigs.shape == (32, 32, 32, 3)

    sigs = m.maximum_signature(Rs, field, algorithm="NEXUS")
    assert sigs.shape == (32, 32, 32, 3)


def test_calc_structure_bools():
    Rs = [np.sqrt(2) ** n for n in range(5)]
    field = test_field
    sigs = m.maximum_signature(Rs, field, algorithm="NEXUS")
    clusbool = (
        np.roll(m.sphere(32, 2), -10, axis=0) + np.roll(m.sphere(32, 2), 10, axis=0)
    ) >= 1
    structure_filter, summary_stats = m.calc_structure_bools(
        field,
        max_sigs=sigs,
        verbose_flag=True,
        clusbool=clusbool,
        Smin=-3,
        Smax=2,
        overdensity_threshold=370,
    )
    assert structure_filter.shape == (32, 32, 32)
    assert structure_filter.dtype == np.uint8
    assert set(np.unique(structure_filter)) <= {1, 2, 3, 4}
    np.testing.assert_array_equal(structure_filter == 1, clusbool)
    assert type(summary_stats) is dict

    structure_filter_quiet = m.calc_structure_bools(
        field,
        max_sigs=sigs,
        verbose_flag=False,
        clusbool=clusbool,
        Smin=-3,
        Smax=2,
        overdensity_threshold=370,
    )
    np.testing.assert_array_equal(structure_filter_quiet, structure_filter)

    delta = field / np.mean(field) - 1.0

    with pytest.raises(ValueError):  # noqa: PT011
        m.calc_structure_bools(
            density_cube=delta,
            max_sigs=sigs,
            verbose_flag=False,
            clusbool=clusbool,
            Smin=-3,
            Smax=2,
            overdensity_threshold=370,
        )


def test_maximum_signature_nexusplus_rejects_negative_input():
    """
    NEXUSPLUS smooths log10(density), so negative input would silently
    produce NaN throughout the pipeline. The validation should turn that
    into a clear ValueError, regardless of backend.
    """
    Rs = [np.sqrt(2) ** n for n in range(3)]
    delta = test_field / np.mean(test_field) - 1.0  # has negative voxels

    with pytest.raises(ValueError, match="NEXUSPLUS"):
        m.maximum_signature(Rs, delta, algorithm="NEXUSPLUS", backend="cpu")


def test_maximum_signature_nexus_allows_negative_input():
    """NEXUS does not log-transform, so negative input is fine."""
    Rs = [np.sqrt(2) ** n for n in range(3)]
    delta = test_field / np.mean(test_field) - 1.0
    sigs = m.maximum_signature(Rs, delta, algorithm="NEXUS", backend="cpu")
    assert sigs.shape == delta.shape + (3,)


@pytest.mark.parametrize("backend", ["jax", "cupy"])
def test_maximum_signature_rejects_old_backend_names(backend):
    with pytest.raises(ValueError, match="backend must be"):
        m.maximum_signature([1.0], test_field, backend=backend)
