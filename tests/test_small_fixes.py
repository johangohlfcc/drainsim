"""Tests for the small fixes from the code review of drainsim 6.3
(one test per fix). Run: python -m pytest -q tests/test_small_fixes.py"""
import warnings

import numpy as np
import pytest

from drainsim import shapes as sh
from drainsim.grid import Grid


def test_cylinder_thin_wall_and_zero_length_axis():
    """A cylinder thinner than the leak-free band is widened to it, like
    segment(), so its wall is continuous; a zero-length axis gives a disc
    (no division by zero, no silently empty mask)."""
    g = Grid.empty([0, 0, 0], [0.2, 0.2, 0.2], 0.01)
    # axis along x, half a cell off every cell centre line: nothing is within
    # 0.1 dx of a cell centre, so an unclamped radius would give an empty mask
    m = sh.cylinder(g, [0.03, 0.105, 0.105], [0.17, 0.105, 0.105], 0.001)
    slab = m[4:16].reshape(12, -1).any(axis=1)
    assert slab.all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # a RuntimeWarning would fail
        d = sh.cylinder(g, [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 0.02)
    assert d.any() and not np.isnan(d.sum())
