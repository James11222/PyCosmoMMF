from __future__ import annotations

import numpy as np

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
    sigs = m.maximum_signature(Rs, field, alg="NEXUSPLUS")
    assert sigs.shape == (32, 32, 32, 3)

    sigs = m.maximum_signature(Rs, field, alg="NEXUS")
    assert sigs.shape == (32, 32, 32, 3)


def test_calc_structure_bools():
    Rs = [np.sqrt(2) ** n for n in range(5)]
    field = test_field
    sigs = m.maximum_signature(Rs, field, alg="NEXUS")
    clusbool = (
        np.roll(m.sphere(32, 2), -10, axis=0) + np.roll(m.sphere(32, 2), 10, axis=0)
    ) >= 1
    clusbool, filbool, wallbool, voidbool, summary_stats = m.calc_structure_bools(
        field,
        max_sigs=sigs,
        verbose=True,
        clusbool=clusbool,
        Smin=-3,
        Smax=2,
        Δ=370,
    )
    assert clusbool.shape == (32, 32, 32)
    assert filbool.shape == (32, 32, 32)
    assert wallbool.shape == (32, 32, 32)
    assert voidbool.shape == (32, 32, 32)
    assert type(summary_stats) is dict

    clusbool, filbool, wallbool, voidbool = m.calc_structure_bools(
        field,
        max_sigs=sigs,
        verbose=False,
        clusbool=clusbool,
        Smin=-3,
        Smax=2,
        Δ=370,
    )
    assert clusbool.shape == (32, 32, 32)
    assert filbool.shape == (32, 32, 32)
    assert wallbool.shape == (32, 32, 32)
    assert voidbool.shape == (32, 32, 32)
