"""Regression / verification tests.  Run:  python -m pytest -q tests"""
import numpy as np
import pytest

from drainsim import cases, shapes as sh
from drainsim.compartments import segment
from drainsim.fsm import fill_spill
from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import dip
from drainsim.physics import Fluid, ThroatModel, torricelli_level


def _fsm_setup(g, up):
    fl = g.fluid.ravel()
    return (g.centers() @ up, np.where(fl, 0, -1), g.neighbors(),
            g.boundary_mask().ravel() & fl, fl)


def test_cup_fills_to_rim_and_conserves():
    g = Grid.empty([0, 0], [1, 1], 0.01)
    sh.add(g, sh.shell_box(g, [0.2, 0.2], [0.6, 0.5], 0.01, open_faces=[(1, 1)]))
    h, reg, nbr, sink, fl = _fsm_setup(g, np.array([0.0, 1.0]))
    src = np.where(fl, g.cell_volume, 0.0)            # flood everything
    ret, dr, lost, _ = fill_spill(h, reg, nbr, sink, src, g.cell_volume)
    interior = sh.rect(g, [0.21, 0.21], [0.59, 0.5]).ravel() & fl
    assert np.isclose(ret.sum(), interior.sum() * g.cell_volume)
    assert np.isclose(ret.sum() + dr.sum() + lost.sum(), src.sum())


def test_spill_is_routed_to_lower_cup():
    g = cases.two_cups(dx=0.01)
    h, reg, nbr, sink, fl = _fsm_setup(g, np.array([0.0, 1.0]))
    X = g.centers()
    src = np.where(fl & (X[:, 0] > 0.15) & (X[:, 0] < 0.35) & (X[:, 1] > 0.75)
                   & (X[:, 1] < 0.95), g.cell_volume, 0.0)
    ret, _, _, _ = fill_spill(h, reg, nbr, sink, src, g.cell_volume)
    th = np.radians(35)
    h2 = X @ np.array([-np.sin(th), np.cos(th)])
    new, dr, _, _ = fill_spill(h2, reg, nbr, sink, ret, g.cell_volume)
    old, dr_old, _, _ = fill_spill(h2, reg, nbr, sink, ret, g.cell_volume,
                                   spill_routing=False)
    upper = X[:, 1] > 0.45
    spilled = ret.sum() - new[upper].sum()
    assert spilled > 0.005
    assert new[~upper].sum() == pytest.approx(spilled)       # all caught below
    assert np.isclose(new.sum() + dr.sum(), ret.sum())
    assert old[~upper].sum() == 0.0 and dr_old.sum() > 0


def test_segmentation_finds_hole_and_vent():
    g, _ = cases.torricelli_box(dx=0.005, hole=0.015, vent=0.02)
    c = segment(g)
    assert c.n == 2
    assert len(c.throats) == 2
    areas = sorted(t.area for t in c.throats)
    assert areas[0] == pytest.approx(0.015, abs=0.006)


def test_torricelli_2d_matches_analytic():
    g, interior = cases.torricelli_box(dx=0.005, hole=0.015)
    sim = Simulation(g, cases.static(t_end=6), initial_L=interior.astype(float),
                     dt_max=0.02)
    H = sim.run().arrays()
    th = min(sim.comp.throats, key=lambda t: t.centroid[1])
    W = 0.38
    level = 0.31 + H["liquid_comp"][:, 1] / W
    ana = torricelli_level(H["t"], level[0] - th.centroid[1], 0.6, th.area, W,
                           h_cap=ThroatModel().holdup_head(th.diameter, Fluid(), 2))
    err = np.abs(level - th.centroid[1] - ana).max()
    assert err < 0.005                                    # < 5 mm over 0.38 m


def test_capillary_holdup_stops_drainage():
    # 2D slot of 2 mm: h_cap = 2 sigma / (rho g d) ~ 7 mm of standing liquid
    g, interior = cases.torricelli_box(dx=0.002, hole=0.002, W=0.1, H=0.1)
    sim = Simulation(g, cases.static(t_end=40), initial_L=interior.astype(float),
                     dt_max=0.05)
    H = sim.run().arrays()
    left = H["liquid_comp"][-1, 1]
    assert left > 0
    assert np.isclose(H["liquid_comp"][-1, 1], H["liquid_comp"][-50, 1], rtol=1e-3)


def test_unvented_small_hole_does_not_drain():
    g, interior = cases.torricelli_box(dx=0.002, hole=0.004, vent=0.0, W=0.1, H=0.1)
    sim = Simulation(g, cases.static(t_end=2), initial_L=interior.astype(float),
                     dt_max=0.05)
    H = sim.run().arrays()
    assert np.isclose(H["liquid_comp"][-1, 1], H["liquid_comp"][0, 1])


def test_trapped_air_and_cup_liquid_on_dip():
    g = cases.inverted_cup(dx=0.005)
    m = dip(2, depth=0.75, start_height=0.15, t_down=5, t_hold=1, t_up=5, t_after=1)
    sim = Simulation(g, m, dt_max=0.05)
    H = sim.run().arrays()
    cavity = (0.25 - 0.02) * (0.15 - 0.01)
    assert H["air_trapped"].max() == pytest.approx(cavity, rel=0.05)
    assert H["liquid_retained"][-1] == pytest.approx(cavity, rel=0.05)


def test_closed_box_conserves_liquid_while_rotating():
    g = Grid.empty([-0.3, -0.3], [0.3, 0.3], 0.005)
    sh.add(g, sh.shell_box(g, [-0.2, -0.2], [0.2, 0.2], 0.01))
    interior = sh.rect(g, [-0.19, -0.19], [0.19, 0.0]).ravel() & g.fluid.ravel()
    from drainsim.motion import Keyframes
    m = Keyframes([0, 5], [0, 170], np.zeros((2, 2)), 2, bath_level=-10)
    sim = Simulation(g, m, initial_L=interior.astype(float), dt_max=0.05)
    H = sim.run().arrays()
    assert np.allclose(H["liquid_retained"], H["liquid_retained"][0])


def test_3d_box_drains():
    g, interior = cases.box3d(dx=0.02, hole=0.04, vent=0.04)
    sim = Simulation(g, cases.static(ndim=3, t_end=3),
                     initial_L=interior.astype(float), dt_max=0.1,
                     segment_kwargs=dict(d_free=0.1))
    H = sim.run().arrays()
    assert sim.comp.n == 2
    assert H["liquid_retained"][-1] < H["liquid_retained"][0]
    assert np.isclose(H["liquid_retained"][-1] + H["drained"][-1],
                      H["liquid_retained"][0])


def test_stl_voxelisation():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    g = Grid.from_mesh(m, dx=0.01)
    assert g.ndim == 3 and g.solid.any()
    # closed shell: interior must be a separate fluid region
    from scipy import ndimage
    _, n = ndimage.label(g.fluid)
    assert n == 2


def test_world_frame_export(tmp_path):
    """Object-frame results are mapped back so that the object moves."""
    from drainsim.motion import Keyframes
    from drainsim.worldviz import export_world, to_world, animate_2d
    import xml.etree.ElementTree as ET
    g = cases.inverted_cup(dx=0.01)
    m = Keyframes([0, 2, 4], [0, 0, 30], np.array([[0, 0.2], [0, -0.5], [0.3, 0.2]]),
                  2, pivot=np.array([0, 0.2]))
    sim = Simulation(g, m, dt_max=0.05)
    H = sim.run(snapshot_every=1.0)
    files = export_world(sim, H, str(tmp_path), prefix="cup", volume_every=2)
    for k in ("object", "liquid", "air", "volume"):
        ds = list(ET.parse(tmp_path / files[k]).getroot().iter("DataSet"))
        assert len(ds) == (5 if k != "volume" else 3)
    # a point on the object follows the prescribed pose
    p = np.array([[0.1, 0.3]])
    w = to_world(p, m, 4.0)[0]
    R = np.array([[np.cos(np.radians(30)), -np.sin(np.radians(30))],
                  [np.sin(np.radians(30)), np.cos(np.radians(30))]])
    exp = R @ (p[0] - [0, 0.2]) + [0, 0.2] + [0.3, 0.2]
    assert np.allclose(w[:2], exp)
    vtk = pytest.importorskip("vtk")
    r = vtk.vtkXMLStructuredGridReader()
    r.SetFileName(str(tmp_path / "cup_frames" / "volume_00004.vts"))
    r.Update()
    b = np.array(r.GetOutput().GetBounds())
    corners = np.array([[g.origin[0], g.origin[1]],
                        [g.origin[0] + g.shape[0] * g.dx, g.origin[1] + g.shape[1] * g.dx],
                        [g.origin[0], g.origin[1] + g.shape[1] * g.dx],
                        [g.origin[0] + g.shape[0] * g.dx, g.origin[1]]])
    W = to_world(corners, m, 4.0)
    assert np.allclose(b[[0, 2]], W.min(0)[:2], atol=1e-5)
    assert np.allclose(b[[1, 3]], W.max(0)[:2], atol=1e-5)
    animate_2d(sim, H, str(tmp_path / "cup.gif"), fps=2, dpi=40)
    assert (tmp_path / "cup.gif").stat().st_size > 0


# ------------------------------------------------------------------ film
def _plate2d(dx):
    g = Grid.empty([0, 0], [0.3, 0.6], dx)
    sh.add(g, sh.box(g, [0.1, 0.05], [0.15, 0.55]))
    return g


def test_film_vertical_plate_matches_jeffreys():
    g = _plate2d(0.003)
    sim = Simulation(g, cases.static(t_end=30), dt_max=0.25, film=True)
    c = sim.film.c
    sim.film.s.h[c.normal[:, 0] > 0.7] = 1e-4
    H = sim.run(snapshot_times=[30])
    face = (c.normal[:, 0] > 0.95) & (c.x[:, 1] > 0.1) & (c.x[:, 1] < 0.5)
    x_top = 0.55 - c.x[face, 1]
    ana = np.minimum(np.sqrt(1e-3 * x_top / (1000 * 9.81 * 30)), 1e-4)
    err = np.abs(H.snap_film[30][face] - ana) / ana
    assert np.median(err) < 0.04


def test_film_deposition_follows_landau_levich():
    from drainsim.motion import Keyframes
    g = Grid.empty([0, -0.05], [0.3, 0.7], 0.003)
    sh.add(g, sh.box(g, [0.1, 0.05], [0.16, 0.6]))
    U = 0.05
    T = 0.65 / U
    m = Keyframes([0, T], [0, 0], np.array([[0, -0.65], [0, 0.0]]), 2)
    sim = Simulation(g, m, dt_max=0.05, film=True)
    H = sim.run(snapshot_times=[T])
    c = sim.film.c
    low = (c.normal[:, 0] > 0.95) & (c.x[:, 1] > 0.1) & (c.x[:, 1] < 0.15)
    h0 = sim.film.deposit_thickness(np.array([U]))[0]
    assert np.mean(H.snap_film[T][low]) == pytest.approx(h0, rel=0.03)


def test_film_conserves_mass_in_closed_box():
    from drainsim.motion import Keyframes
    g = Grid.empty([-0.3, -0.3], [0.3, 0.3], 0.005)
    sh.add(g, sh.shell_box(g, [-0.2, -0.2], [0.2, 0.2], 0.01))
    interior = sh.rect(g, [-0.19, -0.19], [0.19, 0.0]).ravel() & g.fluid.ravel()
    m = Keyframes([0, 6, 12], [0, 90, 90], np.zeros((3, 2)), 2, bath_level=-10)
    sim = Simulation(g, m, initial_L=interior.astype(float), dt_max=0.05,
                     film=True)
    H = sim.run().arrays()
    tot = H["liquid_retained"] + H["film_volume"] + H["film_pending"]
    assert H["film_volume"].max() > 0
    assert np.allclose(tot, tot[0], rtol=1e-9)


def test_film_pool_emersion_speed_uses_the_node_extent(monkeypatch):
    """The film left by an emptying internal pool is deposited at the speed
    2 e / (time the node took to empty), with e the half height of that node
    (sim.en), not of the finest cell: doubling every node's extent doubles
    the first deposition speed."""
    from drainsim import film as filmmod
    from drainsim.motion import Keyframes
    orig_thickness = filmmod.FilmModel.deposit_thickness
    orig_update = filmmod.FilmModel.update
    seen = []

    def spy(self, U):
        seen.append(np.array(U, float))
        return orig_thickness(self, U)
    monkeypatch.setattr(filmmod.FilmModel, "deposit_thickness", spy)

    def first_speed(scale):
        def update(self, dt):                  # scale the extents for the film only
            en = self.sim.en
            self.sim.en = scale * en
            try:
                return orig_update(self, dt)
            finally:
                self.sim.en = en
        monkeypatch.setattr(filmmod.FilmModel, "update", update)
        seen.clear()
        g = Grid.empty([-0.3, -0.3], [0.3, 0.3], 0.005)
        sh.add(g, sh.shell_box(g, [-0.2, -0.2], [0.2, 0.2], 0.01))
        interior = sh.rect(g, [-0.19, -0.19], [0.19, 0.0]).ravel() & g.fluid.ravel()
        m = Keyframes([0, 6, 12], [0, 90, 90], np.zeros((3, 2)), 2, bath_level=-10)
        Simulation(g, m, initial_L=interior.astype(float), dt_max=0.05, film=True).run()
        return seen[0]
    u1, u2 = first_speed(1.0), first_speed(2.0)
    assert u1.size and u1.size == u2.size
    assert np.allclose(u2, 2.0 * u1)


def test_hole_in_open_cup_is_rate_limited():
    """A cup open at the top is part of the exterior; its drain hole must
    still be a (internal) throat and drain at the orifice rate."""
    g = Grid.empty([0, 0], [1, 1], 0.005)
    sh.add(g, sh.shell_box(g, [0.3, 0.3], [0.7, 0.7], 0.01, open_faces=[(1, 1)]))
    sh.cut(g, sh.rect(g, [0.4925, 0.29], [0.5075, 0.315]))
    interior = (sh.rect(g, [0.31, 0.31], [0.69, 0.69]) & g.fluid).ravel()
    sim = Simulation(g, cases.static(t_end=4), initial_L=interior.astype(float),
                     dt_max=0.02)
    inner = [t for t in sim.comp.throats if t.a == t.b]
    assert sim.comp.n == 1 and len(inner) == 1
    th = inner[0]
    H = sim.run().arrays()
    W = 0.38
    level = 0.31 + H["liquid_retained"] / W
    ana = torricelli_level(H["t"], level[0] - th.centroid[1], 0.6, th.area, W,
                           h_cap=ThroatModel().holdup_head(th.diameter, Fluid(), 2))
    assert np.abs(level - th.centroid[1] - ana).max() < 0.006


def test_shallow_pockets_hold_nothing():
    """Depressions shallower than min_depth (voxel roughness) retain nothing;
    a real, deeper well on the same surface still does."""
    g = Grid.empty([0, 0, 0], [0.3, 0.3, 0.2], 0.005)
    sh.add(g, sh.box(g, [0.02, 0.02, 0.02], [0.28, 0.28, 0.08]))       # slab
    sh.cut(g, sh.box(g, [0.05, 0.05, 0.0751], [0.10, 0.10, 0.08]))       # 1-cell dent
    sh.cut(g, sh.box(g, [0.18, 0.18, 0.0551], [0.23, 0.23, 0.08]))       # 5-cell well
    X = g.centers()
    fl = g.fluid.ravel()
    reg = np.where(fl, 0, -1)
    sink = g.boundary_mask().ravel() & fl
    src = np.where(fl & (X[:, 2] > 0.08) & (X[:, 2] < 0.1), g.cell_volume, 0.0)
    h = X[:, 2]
    raw, _, _, _ = fill_spill(h, reg, g.neighbors(), sink, src, g.cell_volume)
    filt, _, _, _ = fill_spill(h, reg, g.neighbors(), sink, src, g.cell_volume,
                               min_depth=1.5 * g.dx)
    dent = (X[:, 0] < 0.15)
    well_vol = 0.05 * 0.05 * 0.025
    assert raw[dent].sum() > 0
    assert filt[dent].sum() == 0.0
    assert filt[~dent].sum() == pytest.approx(well_vol, rel=0.05)


def test_explicit_hole_uses_true_size():
    g = Grid.empty([0, 0], [1, 1], 0.005)
    sh.add(g, sh.shell_box(g, [0.3, 0.3], [0.7, 0.7], 0.01, open_faces=[(1, 1)]))
    interior = (sh.rect(g, [0.31, 0.31], [0.69, 0.69]) & g.fluid).ravel()
    d = 0.012                                   # not a multiple of dx
    sim = Simulation(g, cases.static(t_end=3), initial_L=interior.astype(float),
                     dt_max=0.02, holes=[((0.5, 0.305), d)])
    th = [t for t in sim.comp.throats if abs(t.diameter - d) < 1e-12]
    assert len(th) == 1 and th[0].area == pytest.approx(d)
    H = sim.run().arrays()
    W = 0.38
    level = 0.31 + H["liquid_retained"] / W
    ana = torricelli_level(H["t"], level[0] - th[0].centroid[1], 0.6, d, W,
                           h_cap=ThroatModel().holdup_head(d, Fluid(), 2))
    assert np.abs(level - th[0].centroid[1] - ana).max() < 0.008


def _thin_box_mesh(extents, shift, open_top=False):
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(shift)
    if open_top:
        keep = m.face_normals[:, 2] < 0.5
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    return m


def test_volume_fractions_of_thin_closed_box():
    """Side-aware cut-cell volumes: a zero-thickness box off the grid gets
    its exact interior volume (binary cells lose ~1 cell layer per face)."""
    from scipy import ndimage
    from drainsim.volfrac import effective_volumes
    ext = np.array([0.137, 0.113, 0.091])
    m = _thin_box_mesh(ext, [0.0031, -0.0017, 0.0023])
    g = Grid.from_mesh(m, dx=0.007, pad=0.03)
    lab, _ = ndimage.label(g.fluid)
    b = lab.ravel()
    inside = (b > 0) & (b != b[0])
    v1 = effective_volumes(g, 1)
    v4, st = effective_volumes(g, 4, return_stats=True)
    assert v1[inside].sum() < 0.85 * ext.prod()
    assert v4[inside].sum() == pytest.approx(ext.prod(), rel=0.02)
    # nothing is created: fluid + solid sub-cells never exceed the domain
    assert v4.sum() <= np.prod(g.shape) * g.cell_volume * (1 + 1e-12)
    assert st["dropped"] < 1e-3 * ext.prod()
    assert np.all(v4[~g.fluid.ravel()] == 0)


def test_cup_retains_true_volume_with_subcells():
    """End to end: an open thin-walled cup (off grid) emptied from a full
    start. Binary cells lose the wall volume; with sub-cells (volumes and
    connectivity) the rim is resolved and the cup keeps its true volume."""
    ext = np.array([0.1, 0.08, 0.06])
    m = _thin_box_mesh(ext, [0.0013, 0.0021, 0.0], open_top=True)
    g = Grid.from_mesh(m, dx=0.008, pad=0.024)
    fl = g.fluid.ravel()
    X = g.centers()
    res = {}
    for k in (1, 4):
        sim = Simulation(g, cases.static(ndim=3, t_end=0.1), subcells=k,
                         initial_L=fl.astype(float), dt_max=0.05)
        sim.run()
        res[k] = sim.hist.liquid_retained[-1]
    # coarse spill level: bottom of the first cell layer above the rim
    ztop = ext[2] / 2
    zc = g.origin[2] + (np.floor((ztop - g.origin[2]) / g.dx) + 1) * g.dx
    quantised = ext[0] * ext[1] * (zc - (-ext[2] / 2))
    exact = ext[0] * ext[1] * ext[2]
    assert res[1] < 0.8 * quantised
    assert res[4] == pytest.approx(exact, rel=0.05)
    assert abs(res[4] - exact) < 0.5 * abs(quantised - exact)


def test_hole_wet_matches_circle_segment():
    """Wetted fraction and centroid of a partly submerged explicit hole."""
    from types import SimpleNamespace
    from drainsim.compartments import Throat
    ax = np.array([0.0, np.sin(0.6), np.cos(0.6)])        # tilted hole axis
    t = Throat(a=0, b=0, cells_a=np.zeros(0, int), cells_b=np.zeros(0, int),
               normals=np.zeros((0, 3)), area=np.pi * 0.019 ** 2 / 4, diameter=0.019,
               centroid=np.array([0.1, 0.2, 0.3]), axis=ax)
    fake = SimpleNamespace(up=np.array([0.0, 0.0, 1.0]), grid=SimpleNamespace(ndim=3))
    # sample the disc
    u = np.cross(ax, [1.0, 0, 0]); u /= np.linalg.norm(u); w = np.cross(ax, u)
    s = np.linspace(-0.0095, 0.0095, 801)
    S, T = np.meshgrid(s, s)
    ins = S ** 2 + T ** 2 <= 0.0095 ** 2
    P = t.centroid + S[ins, None] * u + T[ins, None] * w
    z = P[:, 2]
    for H in (z.min() - 1e-3, z.min() + 0.001, 0.3, z.max() - 0.002, z.max() + 1e-3):
        frac, zc, sill = Simulation._hole_wet(fake, t, H)
        wet = z < H
        assert sill == pytest.approx(z.min(), abs=2e-5)
        assert frac == pytest.approx(wet.mean(), abs=5e-3)
        if wet.any():
            assert zc == pytest.approx(z[wet].mean(), abs=5e-5)


def test_plugged_hole_opens_at_given_time():
    """A hole with open_at holds everything until then, then drains."""
    g = Grid.empty([0, 0], [1, 1], 0.005)
    sh.add(g, sh.shell_box(g, [0.3, 0.3], [0.7, 0.7], 0.01, open_faces=[(1, 1)]))
    interior = (sh.rect(g, [0.31, 0.31], [0.69, 0.69]) & g.fluid).ravel()
    sim = Simulation(g, cases.static(t_end=3), initial_L=interior.astype(float),
                     dt_max=0.02, holes=[dict(center=(0.5, 0.305), diameter=0.012,
                                              open_at=1.0)])
    H = sim.run().arrays()
    before = H["liquid_retained"][H["t"] <= 1.0 + 1e-9]
    assert np.ptp(before) < 1e-12
    assert H["liquid_retained"][-1] < 0.8 * before[0]


def _bell(Ri=0.1, Hc=0.3, t=0.004):
    trimesh = pytest.importorskip("trimesh")
    tube = trimesh.creation.annulus(r_min=Ri, r_max=Ri + t, height=Hc + t)
    tube.apply_translation([0, 0, (Hc + t) / 2])
    cap = trimesh.creation.cylinder(radius=Ri + t, height=t)
    cap.apply_translation([0, 0, Hc + t / 2])
    return trimesh.util.concatenate([tube, cap])


def test_diving_bell_follows_boyle():
    """Inverted cup lowered to 5 m: the trapped air compresses as
    V (p_atm + rho g d_interface) = const, the gas amount is conserved, and
    with compressible_air=False the pocket keeps its volume."""
    from drainsim.motion import Keyframes
    g = Grid.from_mesh(_bell(), dx=0.015, pad=0.045)
    D = 5.0
    mo = Keyframes([0, 1 + D], np.zeros((2, 3)), np.array([[0, 0, 0.05], [0, 0, -D]]),
                   ndim=3, pivot=np.zeros(3))
    res = {}
    for comp in (False, True):
        sim = Simulation(g, mo, dt_max=0.05, split=False, compressible_air=comp)
        H = sim.run().arrays()
        res[comp] = (H["air_trapped"], H["air_gas"])
    V_inc = res[False][0][-1]
    V, n = res[True][0][-1], res[True][1]
    tail = n[len(n) // 3:]                                      # cup fully under
    assert np.ptp(tail) < 1e-9 * n.max()                        # gas conserved
    A = np.pi * 0.1 ** 2
    Hc = n[-1] / A                                              # model's own charge
    fp = Fluid()
    a, b, c = fp.rho * fp.g, fp.p_atm + fp.rho * fp.g * (D - 0.3), -Hc * fp.p_atm
    ha = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert V == pytest.approx(A * ha, rel=0.01)
    assert V_inc == pytest.approx(n[-1], rel=0.01)              # old model: no change
    assert V < 0.75 * V_inc


def _slot_cup_mesh(ext, shift, slot_bottom, slot_h, slot_w):
    """Open-top zero-thickness box whose +x wall has a rectangular slot
    (y in +-slot_w/2, z from slot_bottom to slot_bottom + slot_h, box frame)."""
    trimesh = pytest.importorskip("trimesh")
    hx, hy, hz = np.asarray(ext) / 2
    z0, z1, w = slot_bottom, slot_bottom + slot_h, slot_w / 2
    quads = [
        [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz)],      # bottom
        [(-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (-hx, -hy, hz)],      # -x
        [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],      # -y
        [(-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz)],          # +y
        [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, z0), (hx, -hy, z0)],          # +x below
        [(hx, -hy, z1), (hx, hy, z1), (hx, hy, hz), (hx, -hy, hz)],            # +x above
        [(hx, -hy, z0), (hx, -w, z0), (hx, -w, z1), (hx, -hy, z1)],            # +x left
        [(hx, w, z0), (hx, hy, z0), (hx, hy, z1), (hx, w, z1)],                # +x right
    ]
    V, F = [], []
    for q in quads:
        b = len(V)
        V += list(q)
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    m = trimesh.Trimesh(np.array(V, float), np.array(F), process=False)
    m.apply_translation(shift)
    return m


def test_subcell_graph_opens_narrow_slot():
    """A slot narrower than a cell (7 mm at dx = 8 mm) is shut in the coarse
    voxel model, so the cup keeps water to the rim. With the sub-cell graph
    the slot is open and the cup drains to the slot's lower edge."""
    ext = np.array([0.1, 0.08, 0.06])
    hz = ext[2] / 2
    zs = hz - 0.02
    m = _slot_cup_mesh(ext, [0.0017, -0.0023, 0.0011], zs, 0.007, 0.03)
    g = Grid.from_mesh(m, dx=0.008, pad=0.024)
    fl = g.fluid.ravel()
    res = {}
    for key, kw in (("coarse", dict(subcell_connect=False)), ("graph", {})):
        sim = Simulation(g, cases.static(ndim=3, t_end=0.1), subcells=4,
                         initial_L=fl.astype(float), dt_max=0.05, **kw)
        sim.run()
        res[key] = sim.hist.liquid_retained[-1]
    to_slot = ext[0] * ext[1] * (zs + hz)
    assert res["coarse"] > 1.3 * to_slot            # slot shut: water to the rim
    assert res["graph"] == pytest.approx(to_slot, rel=0.06)


def test_plug_seals_opening():
    """``plugs`` cut every link through the plugged opening."""
    ext = np.array([0.1, 0.08, 0.06])
    hz = ext[2] / 2
    zs = hz - 0.02
    shift = np.array([0.0017, -0.0023, 0.0011])
    m = _slot_cup_mesh(ext, shift, zs, 0.007, 0.03)
    g = Grid.from_mesh(m, dx=0.008, pad=0.024)
    fl = g.fluid.ravel()
    plug = dict(center=shift + [ext[0] / 2, 0.0, zs + 0.0035], diameter=0.034,
                axis=(1.0, 0.0, 0.0))
    sim = Simulation(g, cases.static(ndim=3, t_end=0.1), subcells=4, plugs=[plug],
                     initial_L=fl.astype(float), dt_max=0.05)
    sim.run()
    assert sim.hist.liquid_retained[-1] == pytest.approx(ext.prod(), rel=0.05)


def test_subcell_graph_cup_is_leak_free_and_accurate():
    """Open cups at several offsets (one exactly on the grid faces): no leak
    through the walls, and the retained level within about one sub-cell
    (3 mm of 60 mm) of the rim."""
    ext = np.array([0.1, 0.08, 0.06])
    for off in ([0.0, 0.0, 0.0], [0.004, 0.001, 0.0027], [0.0099, 0.0049, 0.0066]):
        m = _thin_box_mesh(ext, off, open_top=True)
        g = Grid.from_mesh(m, dx=0.012, pad=0.036)
        fl = g.fluid.ravel()
        sim = Simulation(g, cases.static(ndim=3, t_end=0.1), subcells=4,
                         initial_L=fl.astype(float), dt_max=0.05)
        sim.run()
        assert sim.hist.liquid_retained[-1] == pytest.approx(ext.prod(), rel=0.06)


def _closed_slot_box(ext, shift, slot_h, vent=None):
    """The slot cup of ``_slot_cup_mesh`` (slot from mid-height up) with a
    lid; ``vent``: width of a 30 mm long slot in the lid, None = no vent."""
    trimesh = pytest.importorskip("trimesh")
    hx, hy, hz = np.asarray(ext) / 2
    if vent is None:
        quads = [[(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]]
    else:
        a, b = vent / 2, 0.015
        quads = [[(-hx, -hy, hz), (-a, -hy, hz), (-a, hy, hz), (-hx, hy, hz)],
                 [(a, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (a, hy, hz)],
                 [(-a, -hy, hz), (a, -hy, hz), (a, -b, hz), (-a, -b, hz)],
                 [(-a, b, hz), (a, b, hz), (a, hy, hz), (-a, hy, hz)]]
    V, F = [], []
    for q in quads:
        o = len(V)
        V += q
        F += [[o, o + 1, o + 2], [o, o + 2, o + 3]]
    lid = trimesh.Trimesh(np.array(V, float) + shift, np.array(F), process=False)
    return trimesh.util.concatenate([_slot_cup_mesh(ext, shift, 0.0, slot_h, 0.03), lid])


def _drain_closed_box(dx, slot_h, vent, t_end, **kw):
    ext = np.array([0.1, 0.08, 0.06])
    shift = np.array([0.0017, -0.0023, 0.0011])
    g = Grid.from_mesh(_closed_slot_box(ext, shift, slot_h, vent), dx=dx, pad=0.024)
    X = g.centers()
    inside = g.fluid.ravel() & np.all(np.abs(X - shift) < ext / 2, axis=1)
    sim = Simulation(g, cases.static(ndim=3, t_end=t_end), subcells=4,
                     initial_L=inside.astype(float), dt_max=0.1, **kw)
    sim.run()
    to_slot = ext[0] * ext[1] * ext[2] / 2
    return sim, sim.hist.liquid_retained[-1] / to_slot


def test_subcell_opening_between_compartments():
    """A closed box whose drain slot and lid vent (5 mm, dx = 8 mm) are open
    only in the sub-cells: the inside is its own compartment, and the two
    slots must still connect it to the outside (as throats, or merged with
    split=False). Without that it stays full (ratio 2)."""
    sim, r = _drain_closed_box(0.008, 0.005, 0.005, 30.0)
    assert len(sim.comp.throats) == 2
    assert r < 1.6                  # drains to the slot + capillary hold-up
    _, r_eq = _drain_closed_box(0.008, 0.005, None, 1.0, split=False)
    assert r_eq == pytest.approx(1.0, abs=0.06)


def test_single_opening_glugs_with_subcell_width():
    """One 12 mm slot, no vent: wider than the Rayleigh-Taylor limit
    (~10 mm), so it drains while air comes in through the same opening. On a
    4 mm grid the whole cells see only a ~6 mm gap; the width must come from
    the sub-cells, or the box holds its water like a pipette."""
    sim, r = _drain_closed_box(0.004, 0.012, None, 60.0)
    d = max(t.diameter for t in sim.comp.throats)
    assert d >= ThroatModel().d_crit(Fluid(), 3)
    assert r < 1.3
    # a 5 mm slot on its own is below the limit and holds
    _, r5 = _drain_closed_box(0.008, 0.005, None, 10.0)
    assert r5 > 1.9


# ------------------------------------------------------------ v5: parallel
def _random_case(rng, trial):
    from drainsim.fsm import to_csr
    shape = tuple(int(x) for x in rng.integers(3, 12, size=rng.integers(2, 4)))
    g = Grid(shape=shape, dx=1.0, origin=np.zeros(len(shape)),
             solid=rng.random(shape) < rng.uniform(0, 0.4))
    P, I = to_csr(g.neighbors())
    N = g.ncells
    h = np.round(rng.integers(0, rng.integers(2, 30), N) * 0.1
                 + (rng.random(N) * 0.05 if trial % 2 else 0), 9)
    nreg = 3 if trial % 3 == 0 else 1
    region = np.where(g.fluid.ravel(), rng.integers(0, nreg, N), -1).astype(np.int64)
    sink = (rng.random(N) < rng.uniform(0, 0.2)) & (region >= 0)
    src = np.where(region >= 0, rng.random(N) * rng.uniform(0, 3), 0.0)
    vc = rng.uniform(0.5, 1.0, N)
    ec = rng.uniform(0.0, 0.2, N) if trial % 4 else np.zeros(N)
    md = [0.0, 0.1, 0.3][trial % 3]
    return h, region, (P, I), sink, src, vc, trial % 5 != 0, nreg, md, ec


@pytest.mark.parametrize("threads", [1, 2])
def test_split_kernel_matches_serial_sweep(threads):
    """The parallel fill-spill (any thread count, with the redundant-saddle
    filter and sink pruning) gives the serial sweep's result."""
    import numba
    from drainsim import fsm
    old = numba.get_num_threads()
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    fsm.CAND_FILTER = True
    try:
        rng = np.random.default_rng(3)
        rng_e = np.random.default_rng(5)
        for trial in range(120):
            h, region, nbr, sink, src, vc, sr, nreg, md, ec = _random_case(rng, trial)
            es = None if trial % 2 == 0 else fsm.box_shape(rng_e.normal(size=3))
            a = fill_spill(h, region, nbr, sink, src, vc, sr, nreg, md, ec, method="serial",
                           eshape=es)
            for prune in (False, True):
                b = fill_spill(h, region, nbr, sink, src, vc, sr, nreg, md, ec, prune=prune,
                               eshape=es)
                sc = max(np.abs(a[0]).max(), 1e-300)
                assert np.abs(a[0] - b[0]).max() <= 1e-12 * sc
                assert np.allclose(a[1], b[1], rtol=1e-12, atol=1e-15 * sc)
                assert np.allclose(a[2], b[2], rtol=1e-12, atol=1e-15 * sc)
                if not prune:
                    assert np.array_equal(a[3], b[3])
    finally:
        fsm.CAND_FILTER = None
        numba.set_num_threads(old)


def test_parallel_helpers_match_numpy():
    from drainsim import fsm, par
    from drainsim.model import _label_bodies_csr
    rng = np.random.default_rng(5)
    k = rng.permutation(400_000).astype(np.int64) * 7 - 11
    assert np.array_equal(par.par_sort(k), np.sort(k))
    f = rng.random(10_001) < 0.4
    assert np.array_equal(par.compact(f), np.flatnonzero(f))
    h = np.round(rng.random(5000), 3)
    sink = rng.random(5000) < 0.1
    act = np.flatnonzero(rng.random(5000) < 0.8)
    ref = act[np.argsort(np.where(sink[act], -np.inf, h[act]), kind="stable")]
    assert np.array_equal(fsm.height_order(h, sink, act), ref)
    g = Grid(shape=(30, 20, 10), dx=1.0, origin=np.zeros(3),
             solid=rng.random((30, 20, 10)) < 0.3)
    P, I = fsm.to_csr(g.neighbors())
    comp = rng.integers(0, 2, g.ncells)
    mask = g.fluid.ravel() & (rng.random(g.ncells) < 0.7)
    b1, n1 = _label_bodies_csr(mask, comp, P, I)
    b2, n2 = par.label_bodies(mask, comp, P, I)
    assert n1 == n2 and np.array_equal(b1, b2)


def test_results_independent_of_threads_and_reuse():
    """Same state with 1 or 2 numba threads, and with or without reusing the
    hierarchy while the part hangs still."""
    import numba
    from drainsim.motion import Keyframes
    g, interior = cases.box3d(dx=0.02, hole=0.04, vent=0.04)
    m = Keyframes([0, 1.0, 3.0], np.array([[0, 0, 0], [0, 20, 0], [0, 20, 0]]),
                  np.zeros((3, 3)), 3, bath_level=-10)
    out = []
    old = numba.get_num_threads()
    for thr, reuse in ((1, True), (2, True), (1, False)):
        numba.set_num_threads(min(thr, numba.config.NUMBA_NUM_THREADS))
        sim = Simulation(g, m, initial_L=interior.astype(float), dt_max=0.1,
                         segment_kwargs=dict(d_free=0.1))
        sim.reuse_static = reuse
        sim.run()
        out.append(sim.L.copy())
        if reuse:
            assert sim._fc_l.hits > 0
    numba.set_num_threads(old)
    assert np.array_equal(out[0], out[1])
    assert np.abs(out[0] - out[2]).max() < 1e-12


def _fake_hole(ax, d=0.019, c=(0.1, 0.2, 0.3), ndim=3, up=(0.0, 0.0, 1.0)):
    from types import SimpleNamespace
    from drainsim.compartments import Throat
    t = Throat(a=0, b=0, cells_a=np.zeros(0, int), cells_b=np.zeros(0, int),
               normals=np.zeros((0, 3)), area=np.pi * d ** 2 / 4 if ndim == 3 else d,
               diameter=d, centroid=np.asarray(c, float), axis=np.asarray(ax, float))
    fake = SimpleNamespace(up=np.asarray(up, float), grid=SimpleNamespace(ndim=ndim))
    return t, fake


def test_hole_flux_matches_disc_sampling():
    """int sqrt(2 g (H - z)) dA over the wetted part of a tilted and of a
    vertical hole (6.3, from agent A's OrificeDisk): against a fine sampling
    of the disc; deep under water it is the Torricelli value A sqrt(2 g h);
    partly covered it is below the centroid-head value used before."""
    g = 9.81
    for ax in ([0.0, np.sin(0.6), np.cos(0.6)], [1.0, 0.0, 0.0]):
        ax = np.asarray(ax) / np.linalg.norm(ax)
        t, fake = _fake_hole(ax)
        u = np.cross(ax, [0.3, 1.0, 0.2]); u /= np.linalg.norm(u); w = np.cross(ax, u)
        s = np.linspace(-0.0095, 0.0095, 1201)
        S, T = np.meshgrid(s, s)
        ins = S ** 2 + T ** 2 <= 0.0095 ** 2
        z = (t.centroid + S[ins, None] * u + T[ins, None] * w)[:, 2]
        dA = t.area / ins.sum()
        for H in (z.min() + 0.001, 0.3, z.max() - 0.002, z.max() + 0.01):
            ref = (np.sqrt(2 * g * np.maximum(H - z, 0.0)) * dA).sum()
            got = Simulation._hole_flux(fake, t, H, g)
            assert got == pytest.approx(ref, rel=2e-3)
            frac, zc, _ = Simulation._hole_wet(fake, t, H)
            assert got <= t.area * frac * np.sqrt(2 * g * (H - zc)) * (1 + 1e-9)
        assert Simulation._hole_flux(fake, t, z.min() - 1e-4, g) == 0.0
        H = 0.3 + 0.5                                        # deep: Torricelli
        assert Simulation._hole_flux(fake, t, H, g) == pytest.approx(
            t.area * np.sqrt(2 * g * (H - 0.3)), rel=1e-4)
    # half covered vertical hole: 0.94 of the centroid-head value
    t, fake = _fake_hole([1.0, 0.0, 0.0])
    frac, zc, _ = Simulation._hole_wet(fake, t, 0.3)
    r = Simulation._hole_flux(fake, t, 0.3, g) / (t.area * frac * np.sqrt(2 * g * (0.3 - zc)))
    assert 0.9 < r < 0.97


def test_hole_flux_2d_slot_is_the_weir_law():
    """2D: a vertical slot of height d covered to h above its sill passes
    (2/3) sqrt(2 g) h^1.5 per unit depth (the rectangular weir)."""
    g = 9.81
    t, fake = _fake_hole([1.0, 0.0, 0.0], d=0.02, c=(0.0, 0.5, 0.0), ndim=2,
                         up=(0.0, 1.0, 0.0))
    for h in (0.002, 0.01, 0.02):
        H = 0.5 - 0.01 + h
        assert Simulation._hole_flux(fake, t, H, g) == pytest.approx(
            2.0 / 3.0 * np.sqrt(2 * g) * h ** 1.5, rel=1e-3)


def test_side_hole_drains_like_the_weir_ode():
    """A cup with a 12 mm hole in its side wall (axis horizontal) drains
    through it: the level follows A dH/dt = -Cd int sqrt(2 g (H - h_c - z)) dA
    (h_c = capillary hold-up) and stops at least h_c above the sill."""
    trimesh = pytest.importorskip("trimesh")
    from scipy.integrate import solve_ivp
    from drainsim.physics import Fluid
    ext = np.array([0.1, 0.08, 0.06])
    shift = np.array([0.0017, -0.0023, 0.0011])
    m = trimesh.creation.box(extents=ext)
    m.apply_translation(shift)
    m = trimesh.Trimesh(m.vertices, m.faces[m.face_normals[:, 2] < 0.5], process=False)
    d = 0.012
    zc = shift[2] - ext[2] / 2 + 0.015
    hole = dict(center=(shift[0] + ext[0] / 2, shift[1], zc), diameter=d, axis=(1.0, 0, 0))
    h = 0.004
    lo = np.array([-0.08, -0.07, -0.05])
    g = Grid.from_mesh(m, h, bounds=(lo, lo + np.array([40, 36, 32]) * h))
    tm = ThroatModel(Cd=0.62)
    sim = Simulation(g, cases.static(ndim=3, t_end=8.0), subcells=4, dt_max=0.02,
                     holes=[hole], throat_model=tm)
    inside = sim.fl & np.all(np.abs(sim.X - shift) < ext / 2, axis=1) & \
        (sim.X[:, 2] < zc + 0.02)
    sim.L = np.where(inside, 1.0, 0.0)
    Hh = sim.run().arrays()
    bd = sim._bodies()
    k = int(np.argmax(np.where(bd["bath"], -1, bd["vol"])))
    area = bd["area"][k]
    fl = Fluid()
    hc = tm.holdup_head(d, fl, 3)
    sill = zc - d / 2
    # final level: at least the hold-up above the sill; the carved hole
    # (cells within d/2 - h/2 of its axis) ends within a cell of its sill
    assert sill + hc - 1e-4 <= bd["level"][k] <= sill + hc + 1.5 * h
    # the drained volume against the level ODE with the same pool area
    t_hole = sim.comp.throats[[i for i, t in enumerate(sim.comp.throats)
                               if t.axis is not None][0]]
    V = np.array(Hh["liquid_retained"])
    tt = np.array(Hh["t"])
    H0 = bd["level"][k] + (V[1] - V[-1]) / area

    def rhs(_, y):
        return [-tm.Cd * Simulation._hole_flux(sim, t_hole, y[0] - hc, fl.g) / area]
    sol = solve_ivp(rhs, (0, tt[-1]), [H0], dense_output=True, max_step=0.01, rtol=1e-8)
    Vode = V[-1] + area * (sol.sol(tt[1:])[0] - bd["level"][k])
    half = np.searchsorted(-V[1:], -(V[1] + V[-1]) / 2)
    t_sim = tt[1:][half]
    t_ode = tt[1:][np.searchsorted(-Vode, -(V[1] + V[-1]) / 2)]
    assert t_sim == pytest.approx(t_ode, rel=0.1)


def test_throat_sizes_from_the_geometry():
    """7.2: a closed box of 2 mm plates with a 30 x 5 mm slot in a side wall
    (dx = 8 mm: open in the grid only through a few sub-cells);
    drainsim.throat_size widens its throat to the slot's neck (5 mm, at
    least 3 mm), never wider than the slot."""
    trimesh = pytest.importorskip("trimesh")

    def box(lo, hi):
        b = trimesh.creation.box(extents=np.subtract(hi, lo))
        b.apply_translation(0.5 * (np.asarray(lo, float) + np.asarray(hi, float)))
        return b
    T, W, D, Hh = 0.002, 0.1, 0.08, 0.06
    sx, sz = 0.03, 0.005                                 # the slot, in the wall x = W/2
    z0 = 0.0
    m = trimesh.util.concatenate([
        box((-W / 2, -D / 2, -Hh / 2), (W / 2, D / 2, -Hh / 2 + T)),          # floor
        box((-W / 2, -D / 2, Hh / 2 - T), (W / 2, D / 2, Hh / 2)),            # lid
        box((-W / 2, -D / 2, -Hh / 2), (-W / 2 + T, D / 2, Hh / 2)),
        box((-W / 2, -D / 2, -Hh / 2), (W / 2, -D / 2 + T, Hh / 2)),
        box((-W / 2, D / 2 - T, -Hh / 2), (W / 2, D / 2, Hh / 2)),
        # the wall with the slot: below, above, and either side of it
        box((W / 2 - T, -D / 2, -Hh / 2), (W / 2, D / 2, z0 - sz / 2)),
        box((W / 2 - T, -D / 2, z0 + sz / 2), (W / 2, D / 2, Hh / 2)),
        box((W / 2 - T, -D / 2, z0 - sz / 2), (W / 2, -sx / 2, z0 + sz / 2)),
        box((W / 2 - T, sx / 2, z0 - sz / 2), (W / 2, D / 2, z0 + sz / 2))])
    m.apply_translation((0.0017, -0.0023, 0.0011))
    g = Grid.from_mesh(m, dx=0.008, pad=0.024)
    sim = Simulation(g, cases.static(ndim=3, t_end=0.1), subcells=4)
    ext = [t for t in sim.comp.throats if t.a != t.b]
    assert ext
    t = max(ext, key=lambda t: t.area)
    a0, d0 = t.area, t.diameter
    assert a0 < 0.9 * sx * sz                          # the grid's faces: less than the slot
    # the neck rule: the slot is 5 mm wide (at least 3 mm): its throat gets
    # about that width (a lower bound: voxels of 0.5 mm), never more
    from drainsim.throat_size import neck_size_throats
    neck_size_throats(sim, res=0.0005, verbose=False)
    t3 = max((t for t in sim.comp.throats if t.a != t.b), key=lambda t: t.diameter)
    assert 0.7 * sz <= t3.diameter <= 1.05 * sz
    assert t3.area <= sx * sz


def test_pressure_heads_pass_through_full_compartments():
    """7.2, ThroatModel.pressurised: a chain bath (0) - 1 (full) - 2 (full)
    - 3 (a free surface): 1 and 2 get the bath's head through their
    submerged throats, 3 keeps its own; a full compartment fed through a
    submerged throat counts as fed."""
    from types import SimpleNamespace
    sim = SimpleNamespace()
    sim.comp = SimpleNamespace(n=4, volume=np.array([1e3, 1.0, 1.0, 1.0]))
    sim._ta = np.array([0, 1, 2])
    sim._tb = np.array([1, 2, 3])
    # one face per throat, at heights 0.40, 0.30, 0.20 (faces between
    # nodes 0-1, 2-3, 4-5)
    sim.h = np.array([0.40, 0.40, 0.30, 0.30, 0.20, 0.20])
    sim._tca = np.array([0, 2, 4])
    sim._tcb = np.array([1, 3, 5])
    sim._tptr = np.array([0, 1, 2, 3])
    liq = np.array([0.0, 1.0, 1.0, 0.5])                 # 1, 2 full; 3 half
    bath = 1.0
    # side levels: the bath on 0's side; the full ones' own tops (0.45,
    # 0.35); 3's free surface at 0.25 (above its throat face at 0.20)
    slev = np.array([bath, 0.45, 0.45, 0.35, 0.35, 0.25])
    full, Hp, sub_a, sub_b, fed = Simulation._pressure_heads(sim, liq, slev, np.ones(3, bool))
    assert full.tolist() == [False, True, True, False]
    assert Hp[1] == pytest.approx(bath) and Hp[2] == pytest.approx(bath)
    assert Hp[3] == -np.inf
    assert sub_a.tolist() == [True, True, True]          # every throat under liquid from a
    assert fed[1] >= 1 and fed[2] >= 1 and fed[3] == 0

