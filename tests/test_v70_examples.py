"""drainsim 7.0: the movie changes give the same pictures, faster.
Run: python -m pytest -q tests/test_v70_examples.py"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)


def test_film_shells_are_built_once_and_read_back(tmp_path):
    """The film shells of a recording are built by the first render and read
    from the cache by the next ones (the same arrays); a cache made for other
    inputs is not used."""
    pytest.importorskip("vtk")
    from door_dip_film_movie import FilmScene
    rng = np.random.default_rng(1)
    built = []
    sides = [(rng.random((50, 3)), rng.random((50, 3)), rng.integers(-1, 9, 50))
             for _ in range(2)]
    faces = rng.integers(0, 50, (80, 3))
    meta = ["inputs A"]

    def arrays(mesh):
        built.append(1)
        return sides, faces
    stub = SimpleNamespace(_film_shell_arrays=arrays, _shell_meta=lambda mesh: meta[0])
    path = str(tmp_path / "film_shells.npz")
    for n_built in (1, 1):                          # built, then read back
        s, f = FilmScene._cached_film_shells(stub, None, path)
        assert len(built) == n_built and np.array_equal(f, faces)
        for (v, n, m), (v0, n0, m0) in zip(s, sides):
            assert np.array_equal(v, v0) and np.array_equal(n, n0) and np.array_equal(m, m0)
    assert not os.path.exists(path + ".lock")
    meta[0] = "inputs B"                            # another recording: rebuild
    FilmScene._cached_film_shells(stub, None, path)
    assert len(built) == 2
