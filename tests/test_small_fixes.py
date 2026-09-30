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


def test_drop_count_is_a_running_sum():
    """History.n_drips (a value per step) comes from a running counter of the
    film model, not from re-summing the ever growing list of drip events, and
    equals that sum."""
    from drainsim import cases
    from drainsim.model import Simulation
    g = Grid.empty([0, 0], [0.6, 0.3], 0.003)
    sh.add(g, sh.box(g, [0.1, 0.12], [0.5, 0.15]))          # a horizontal plate
    sim = Simulation(g, cases.static(t_end=20), dt_max=0.25, film=True)
    sim.film.s.h[sim.film.c.normal[:, 1] < -0.7] = 3e-3      # thick film underneath
    H = sim.run()
    f = sim.film
    assert f.s.drips and f.n_drops > 0                       # it does drip
    assert f.n_drops == sum(d[3] for d in f.s.drips)
    assert H.n_drips[-1] == f.n_drops
    assert np.all(np.diff(H.n_drips) >= 0)


def test_render_animation_removes_its_frames_when_ffmpeg_fails(tmp_path, monkeypatch):
    """A failing ffmpeg (or VTK) call used to leave the folder of PNG frames
    in the temp directory."""
    import subprocess
    import tempfile
    pytest.importorskip("vtk")
    from drainsim import cases, render3d
    from drainsim.model import Simulation
    from drainsim.motion import Keyframes
    from drainsim.worldviz import export_world
    g, interior = cases.box3d(dx=0.03, hole=0.06, vent=0.06)
    m = Keyframes([0, 1.0], np.array([[0, 0, 0], [0, 20, 0]]), np.zeros((2, 3)), 3)
    sim = Simulation(g, m, initial_L=interior.astype(float), dt_max=0.25)
    H = sim.run(snapshot_every=0.5)
    export_world(sim, H, str(tmp_path / "vtk"), prefix="box", volume=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(render3d.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        render3d.render_animation(str(tmp_path / "vtk"), "box", str(tmp_path / "box.mp4"),
                                  size=(160, 120))
    assert list(scratch.iterdir()) == []


def test_mark_surface_sampling_is_shared_with_sample_triangles():
    """_mark_surface(exact=False) draws its sample points with
    grid.sample_triangles (it used to carry its own copy of that code) and
    marks the same cells as that copy did."""
    import trimesh
    from drainsim.grid import _mark_surface
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.09)
    mesh.apply_translation([0.1, 0.1, 0.1])
    tri = np.asarray(mesh.triangles, float)
    g = Grid.empty([0, 0, 0], [0.2, 0.2, 0.2], 0.01)
    _mark_surface(g, tri, exact=False)
    # the former inline implementation, as the reference
    ref = np.zeros(g.shape, bool)
    dx, spacing = g.dx, 0.45
    emax = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1)
    nsub = np.maximum(1, np.ceil(emax / (spacing * dx))).astype(int)
    for n in np.unique(nsub):
        T = tri[nsub == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        ok = i + j <= n
        w1, w2 = (i[ok] / n)[:, None], (j[ok] / n)[:, None]
        w0 = 1.0 - w1 - w2
        P = (w0[None] * T[:, None, 0] + w1[None] * T[:, None, 1]
             + w2[None] * T[:, None, 2]).reshape(-1, 3)
        idx = np.floor((P - g.origin) / dx).astype(np.int64)
        good = np.all((idx >= 0) & (idx < np.array(g.shape)), axis=1)
        idx = idx[good]
        ref[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    assert g.solid.any() and np.array_equal(g.solid, ref)
    import inspect
    import drainsim.grid as gm
    assert "np.meshgrid" not in inspect.getsource(gm._mark_surface)
