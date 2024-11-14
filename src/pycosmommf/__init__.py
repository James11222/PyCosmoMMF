"""
Copyright (c) 2024 James Sunseri. All rights reserved.

pycosmommf: A package for identifying structures in the cosmic web.
"""

from __future__ import annotations

from ._version import version as __version__
from .filter import *
from .hessian import *
from .signatures import *
from .tagging import *
from .utils import *

__all__ = ["__version__"]
