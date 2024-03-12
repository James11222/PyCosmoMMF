import numpy as np
import numba as nb

from PyCosmoMMF.filter import *
from PyCosmoMMF.hessian import *  
from PyCosmoMMF.tagging import *
from PyCosmoMMF.utils import *
from PyCosmoMMF.signatures import *


import pkg_resources
test_field = np.load(pkg_resources.resource_filename(__name__, 'test_density_cube.npy'))



