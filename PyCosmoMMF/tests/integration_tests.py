import PyCosmoMMF
import numpy as np

test_field = PyCosmoMMF.test_field

def test_import():
    try:
        PyCosmoMMF.test_print()
        assert True
    except:
        assert False

def test_maximum_signature():
    Rs = [np.sqrt(2)**n for n in range(5)]
    field = test_field
    sigs = PyCosmoMMF.maximum_signature(Rs, field, alg='NEXUSPLUS')
    assert sigs.shape == (32, 32, 32, 3)

    sigs = PyCosmoMMF.maximum_signature(Rs, field, alg='NEXUS')
    assert sigs.shape == (32, 32, 32, 3)
    # assert np.all(sigs != 0.0)

def test_calc_structure_bools():
    Rs = [np.sqrt(2)**n for n in range(5)]
    field = test_field
    sigs = PyCosmoMMF.maximum_signature(Rs, field, alg='NEXUS')
    clusbool, filbool, wallbool, voidbool, S_fil, dM2_fil, S_wall, dM2_wall = PyCosmoMMF.calc_structure_bools(data=field, 
                                            max_sigs=sigs, 
                                            verbose=True, 
                                            clusbool=None, 
                                            Smin=-3, 
                                            Smax=2, 
                                            Δ=370)
    assert clusbool.shape == (32, 32, 32)
    assert filbool.shape == (32, 32, 32)
    assert wallbool.shape == (32, 32, 32)
    assert voidbool.shape == (32, 32, 32)

    assert (np.sum(clusbool) / np.prod(clusbool.shape)) >= 0.0
    assert (np.sum(clusbool) / np.prod(clusbool.shape)) < 0.5

    assert (np.sum(filbool) / np.prod(filbool.shape)) >= 0.0
    assert (np.sum(filbool) / np.prod(filbool.shape)) < 0.8

    assert (np.sum(wallbool) / np.prod(wallbool.shape)) >= 0.0
    assert (np.sum(wallbool) / np.prod(wallbool.shape)) < 0.8

    assert (np.sum(voidbool) / np.prod(voidbool.shape)) >= 0.0
    assert (np.sum(voidbool) / np.prod(voidbool.shape)) < 0.8




    