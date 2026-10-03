"""Grid throats sized from the true geometry (7.2).

The segmentation joins compartments by throats made of the grid faces
where they meet. In a narrow passage that the grid resolves only with a
few sub-cells, those faces are a fraction of the real opening: a 6.5 mm
passage at the end of a bumper beam became 4 sub-cell faces, 0.07 cm2,
d 1.1 mm, and the beam held its air through a whole dip. Here each grid
throat's cross-section is measured on the triangles: rays from its centre
in its plane (perpendicular to the mean direction of its faces), each to
the first wall. Their lengths give the opening's area (the polygon they
span) and its width (the narrowest pair of opposite rays). A throat whose
grid area or width is smaller than that is raised to it; a throat whose
centre lies inside metal, or that is not a bounded opening (a quarter of
its rays reach no wall within ``rmax``), is left as it is. The walls must
be thin solids (as the car meshes): a single sheet's slot has its edges in
the plane of the rays, which do not hit them.
"""
from __future__ import annotations

import numpy as np


def true_sections(centres, normals, V, F, n_dirs=72, rmax=0.05, rmi=None):
    """Area (m^2) and width (m) of the openings at ``centres`` across
    ``normals`` (unit), from rays on the mesh (V, F) capped at ``rmax``.
    NaN where the centre is in metal or the direction is undefined."""
    from .openings import face_geometry
    c = np.asarray(centres, float)
    nrm = np.asarray(normals, float)
    m = len(c)
    ok = np.linalg.norm(nrm, axis=1) > 0.5
    nrm = np.where(ok[:, None], nrm, [[0.0, 0.0, 1.0]])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    e = np.where(np.abs(nrm[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = np.cross(nrm, e)
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(nrm, t1)
    th = np.arange(n_dirs) * 2 * np.pi / n_dirs
    D = (np.cos(th)[None, :, None] * t1[:, None, :] + np.sin(th)[None, :, None] * t2[:, None, :])
    P = np.repeat(c, n_dirs, axis=0)
    D = D.reshape(-1, 3)
    if rmi is None:
        import trimesh
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False, validate=False))
    N = face_geometry(V, F)[1]
    # the first wall along each ray (from inside metal: its inner face)
    hit = np.full(len(P), -1, np.int64)
    step = 4_000_000
    for s0 in range(0, len(P), step):
        hit[s0:s0 + step] = rmi.intersects_first(P[s0:s0 + step] + 1e-7 * D[s0:s0 + step],
                                                 D[s0:s0 + step])
    r = np.full(len(P), rmax)
    h = np.flatnonzero(hit >= 0)
    if h.size:
        j = hit[h]
        Pj = V[F[j, 0]]
        den = np.einsum("ij,ij->i", N[j], D[h])
        tt = np.einsum("ij,ij->i", N[j], Pj - P[h]) / np.where(np.abs(den) > 1e-12, den, 1e-12)
        tt = np.where(np.abs(den) > 1e-12, tt, rmax)
        r[h] = np.clip(tt, 0.0, rmax)
        # a centre in metal: the first hit faces away (its normal along the ray)
        inside = np.zeros(len(P), bool)
        inside[h] = den > 0
    else:
        inside = np.zeros(len(P), bool)
    r = r.reshape(m, n_dirs)
    inside = inside.reshape(m, n_dirs)
    area = 0.5 * np.sin(2 * np.pi / n_dirs) * (r * np.roll(r, -1, axis=1)).sum(1)
    half = n_dirs // 2
    width = (r[:, :half] + r[:, half:]).min(1)
    # in metal, or not a bounded opening (rays that reach no wall)
    bad = ~ok | (inside.mean(1) > 0.5) | ((r >= rmax).mean(1) > 0.25)
    area[bad] = np.nan
    width[bad] = np.nan
    return area, width


def size_throats(sim, V=None, F=None, n_dirs=72, rmax=0.05, verbose=True):
    """Raise the grid throats of ``sim`` (not explicit holes, not channel
    mouths) to their true cross-section (``true_sections``) where the grid
    gives less; the walls: (V, F), or the octree's triangles. Updates the
    throat arrays the stepping uses. Returns the number raised."""
    th = sim.comp.throats
    ids = [i for i, t in enumerate(th) if t.axis is None and not getattr(t, "slot", False)
           and t.a != t.b]
    if not ids:
        return 0
    if V is None:
        T = np.asarray(sim.grid.triangles, float)
        V = T.reshape(-1, 3)
        F = np.arange(len(V)).reshape(-1, 3)
    cen = np.array([np.asarray(th[i].centroid, float) for i in ids])
    nrm = np.zeros((len(ids), 3))
    for q, i in enumerate(ids):
        t = th[i]
        if t.normals is not None and len(t.normals):
            v = np.asarray(t.normals, float).mean(0)
            if np.linalg.norm(v) > 0.5:
                nrm[q, :v.size] = v
    area, width = true_sections(cen, nrm, V, F, n_dirs, rmax)
    n = 0
    for q, i in enumerate(ids):
        t = th[i]
        if not np.isfinite(area[q]):
            continue
        if area[q] > t.area or width[q] > t.diameter:
            t.area = float(max(t.area, area[q]))
            t.diameter = float(max(t.diameter, width[q]))
            n += 1
    sim._throat_csr()
    if verbose:
        print(f"throat sizes from the geometry: {n} of {len(ids)} grid throats raised", flush=True)
    return n
