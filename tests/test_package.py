from __future__ import annotations

import importlib.metadata

import pycosmommf as m


def test_version():
    assert importlib.metadata.version("pycosmommf") == m.__version__
