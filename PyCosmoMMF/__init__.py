import numpy as np
import numba as nb

from PyCosmoMMF.filter import *
from PyCosmoMMF.hessian import *  
from PyCosmoMMF.tagging import *
from PyCosmoMMF.utils import *
from PyCosmoMMF.signatures import *
import pkg_resources

test_field = np.load(pkg_resources.resource_filename(__name__, 'test_density_cube.npy'))

def version():
    """
    Print the version of the package.
    """
    __version__ = '0.0.8'

    print("PyCosmoMMF version: ", __version__)







