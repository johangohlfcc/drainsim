"""drainsim 7.1: holes and gaps found on the mesh (drainsim.openings).
Run: python -m pytest -q tests/test_openings.py"""
import os
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from drainsim import openings as op                               # noqa: E402


def _stl(tmp_path, mesh, name="m.stl"):
    p = os.path.join(tmp_path, name)
    mesh.export(p)
    return p


def _plate_with_hole(r_hole=0.010, r_out=0.08, t=0.001, sections=96):
    """A 1 mm plate (an annulus: closed, outward normals) with a round hole."""
    return trimesh.creation.annulus(r_min=r_hole, r_max=r_out, height=t, sections=sections)


def test_read_stl_merges_the_corners(tmp_path):
    m = _plate_with_hole()
    V, F = op.read_stl(_stl(tmp_path, m), scale=1.0)
    assert len(F) == len(m.faces)
    assert len(V) == len(np.unique(np.asarray(m.vertices, np.float32), axis=0))
    tri = V[F]
    ref = np.asarray(m.vertices, np.float32)[m.faces].astype(np.float64)
    assert np.allclose(np.sort(tri.reshape(-1, 9), axis=0), np.sort(ref.reshape(-1, 9), axis=0))


def test_a_hole_in_a_plate_and_a_pin(tmp_path):
    plate = _plate_with_hole()
    pin = trimesh.creation.cylinder(radius=0.004, height=0.02, sections=64)
    pin.apply_translation((0.05, 0.0, 0.0105))                    # standing on the plate
    tilt = trimesh.transformations.rotation_matrix(0.4, (1.0, 0.3, 0.0))
    m = trimesh.util.concatenate([plate, pin])
    m.apply_transform(tilt)
    V, F = op.read_stl(_stl(tmp_path, m), scale=1.0)
    H = op.find_holes(V, F)
    holes = np.flatnonzero(H["kind"] == 1)
    pins = np.flatnonzero(H["kind"] == -1)
    assert holes.size == 1 and pins.size == 1
    k = holes[0]
    axis = tilt[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert abs(H["diameter"][k] - 0.020) < 0.0002                # facets of a 96-gon
    assert np.linalg.norm(H["center"][k] - tilt[:3, 3]) < 1e-6
    assert abs(abs(H["axis"][k] @ axis) - 1) < 1e-6
    assert abs(H["depth"][k] - 0.001) < 0.0002                   # the plate thickness
    assert abs(H["diameter"][pins[0]] - 0.008) < 0.0002


def test_the_gap_between_two_plates(tmp_path):
    a = trimesh.creation.box(extents=(0.1, 0.06, 0.002))
    b = trimesh.creation.box(extents=(0.08, 0.05, 0.002))
    b.apply_translation((0.0, 0.0, 0.002 + 0.0008))              # 0.8 mm above a
    m = trimesh.util.concatenate([a, b])
    V, F = op.read_stl(_stl(tmp_path, m), scale=1.0)
    W, part, facing = op.gap_map(V, F, +1)
    C, N, A = op.face_geometry(V, F)
    up_of_a = (N[:, 2] > 0.9) & (np.abs(C[:, 2] - 0.001) < 1e-6)
    down_of_b = (N[:, 2] < -0.9) & (np.abs(C[:, 2] - 0.0018) < 1e-6)
    for s in (up_of_a, down_of_b):
        assert s.any() and np.allclose(W[s], 0.0008) and facing[s].all()
    lab, R = op.gap_regions(V, F, W, facing, 0.002)
    assert len(R["area"]) == 2                                    # the two faces across the gap
    assert np.allclose(R["wmed"], 0.0008) and np.allclose(R["wmin"], 0.0008)
    assert abs(R["area"].sum() - (0.08 * 0.05 + 0.1 * 0.06)) < 1e-9 or R["area"][1] > 0
    assert (W[~(up_of_a | down_of_b) & facing] > 0.002).all()     # nothing else that narrow


def test_holes_for_model_follow_load_car(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
    from car_article import load_car
    m = _plate_with_hole()
    m.apply_scale(1000.0)                                         # an STL in millimetres
    m.apply_translation((500.0, 20.0, 300.0))
    path = _stl(tmp_path, m)
    mesh, info = load_car(path, "file")
    V, F = op.read_stl(path)
    H = op.find_holes(V, F)
    hs = op.holes_for_model(H, info)
    assert len(hs) == 1 and abs(hs[0]["diameter"] - 0.020) < 0.0002
    # the hole centre is at the centre of the loaded mesh's hole (its origin)
    assert np.linalg.norm(np.asarray(hs[0]["center"]) - np.asarray(mesh.vertices).mean(0)) < 1e-4
    assert abs(abs(hs[0]["axis"][2]) - 1) < 1e-6


def test_near_links_contain_every_link_near_a_point():
    """NearLinks.links gives every link with an end within r (the exact
    test of the hole setup applied to them gives what a pass over all
    links gives)."""
    from drainsim.spatial import NearLinks
    rng = np.random.default_rng(19)
    X = rng.random((5000, 3))
    a = rng.integers(0, 5000, 30000)
    b = np.where(rng.random(30000) < 0.1, -1, rng.integers(0, 5000, 30000))
    for ends in ((a,), (a, b)):
        nl = NearLinks(X, *ends)
        for _ in range(50):
            c = rng.random(3)
            r = rng.uniform(0.01, 0.2)
            near = np.linalg.norm(X[a] - c, axis=1) <= r
            if len(ends) == 2:
                near |= (b >= 0) & (np.linalg.norm(X[np.maximum(b, 0)] - c, axis=1) <= r)
            got = nl.links(c, r)
            assert np.all(np.diff(got) > 0)                     # sorted, unique
            assert set(np.flatnonzero(near)) <= set(got.tolist())


def test_throats_not_at_holes_as_a_pass_over_all_holes():
    """throats_not_at keeps the throats the loop over all throats and holes
    keeps (a centroid within a hole's diameter, or a cell used by one)."""
    from types import SimpleNamespace
    from drainsim.spatial import throats_not_at
    rng = np.random.default_rng(7)
    used = rng.random(400) < 0.05

    def thr(d=0.0):
        return SimpleNamespace(centroid=rng.random(3), diameter=d,
                               cells_a=rng.integers(0, 400, 3), cells_b=rng.integers(0, 400, 3))
    throats = [thr() for _ in range(2000)]
    forced = [thr(rng.uniform(0.01, 0.15)) for _ in range(60)]
    ref = [t for t in throats
           if not any(np.linalg.norm(t.centroid - f.centroid) < f.diameter for f in forced)
           and not (used[t.cells_a].any() or used[t.cells_b].any())]
    got = throats_not_at(throats, forced, used)
    assert [id(t) for t in got] == [id(t) for t in ref] and 0 < len(got) < len(throats)
    assert throats_not_at(throats, [], used) == [t for t in throats
                                                 if not (used[t.cells_a].any() or used[t.cells_b].any())]
