"""Gap channels (7.1): narrow gaps between plates that the grid closes,
added to the model as compartments of their own.

A gap a few millimetres wide between two overlapping panels (a flange, a
seam, a reinforcement on a panel) is closed in a grid of cells larger than
it, although liquid and air pass through it in reality. ``openings`` finds
the gaps on the triangles (``gap_samples``: the width at points spread over
the faces). Here the samples whose mid-gap point lies in no fluid node of
the model (the gap is closed in the grid there) become **channel nodes**:

* the samples of a closed gap are joined along each wall (samples on the
  same surface, ``openings._same_surface``) and across the gap (a sample and
  the sample of the other wall nearest to where its ray hits), so channels
  on the two sides of one plate stay apart; each connected set is a **channel**, a compartment of its own;
* a channel's samples are gathered into nodes about ``node_size`` apart:
  position the mean mid-gap point, volume area x width (the two walls of a
  gap counted once), linked where their samples are;
* where a channel opens into fluid of the model (the gap ends, widens, or
  is open in the grid: ``openings.gap_mouths``) a **mouth** joins its nodes
  to the fluid nodes there. The mouths of a channel into one compartment,
  split where they are apart, are slot throats (``Throat.slot``): the
  orifice law over the wetted part, with the capillary hold-up of a slot of
  the gap's width (2 sigma / rho g w).

Inside a channel, liquid moves by fill-spill like in any compartment (no
friction along the channel); between a channel and the rest it is rate
limited by its mouths. Lengths in metres, in the model's frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from . import openings as op
from .par import dual


@dataclass
class Channels:
    X: np.ndarray            # (m, 3) node positions
    v: np.ndarray            # (m,) node volumes
    size: np.ndarray         # (m,) node size (edge of the cube used for the
                             # vertical extent of its volume)
    width: np.ndarray        # (m,) mean gap width of each node
    links: np.ndarray        # (k, 2) linked node pairs
    channel: np.ndarray      # (m,) channel of each node (0..n-1)
    n: int                   # number of channels
    mouths: list = field(default_factory=list)   # dicts, see build_channels
    stats: dict = field(default_factory=dict)


def _containing(tree, X, hs, P, k=8):
    """The node whose cube contains each point (-1: none)."""
    if len(P) == 0:
        return np.zeros(0, np.int64)
    k = min(k, len(X))
    _, j = tree.query(P, k=k)
    j = j.reshape(len(P), k)
    inside = np.all(np.abs(X[j] - P[:, None, :]) <= hs[j][:, :, None] * (1 + 1e-9), axis=2)
    return np.where(inside.any(1), j[np.arange(len(P)), inside.argmax(1)], -1)


def _components(n, pairs):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    G = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(G, directed=False)[1]


def _edge_length(P, chunk, sp):
    """Length of the edge a band of points along it marks (the band a
    couple of samples wide): in chunks of ``chunk``, each chunk's extent
    along its main direction, plus one sample spacing."""
    key = np.floor(P / chunk).astype(np.int64)
    _, g = np.unique(key, axis=0, return_inverse=True)
    g = g.ravel()
    total = 0.0
    for k in range(g.max() + 1):
        Q = P[g == k]
        if len(Q) < 2:
            total += sp
            continue
        Q = Q - Q.mean(0)
        ax = np.linalg.svd(Q, full_matrices=False)[2][0]
        x = Q @ ax
        total += float(x.max() - x.min()) + sp
    return total


def _probe(tree, X, hs, fl, lab, mid, n, w, u, lu, s, walls):
    """For each edge of a channel (mid-gap point mid of the sample at the
    edge, its wall normal n and gap width w, the direction u to the sample
    beyond at distance lu): the fluid node of the model at the first of a
    set of points beyond the edge, nearest first, that lies in a fluid node
    and is in sight of mid (no wall between; walls: (V, F, normals, ray
    intersector) of the walls, None: not checked). The points: lu + {0.5, 1, 2, 3} s along u, and across the
    gap's plane from mid-gap to s beyond either wall (a gap ends where one
    wall ends: its mouth can lie beside the other wall). -1: none."""
    du = np.array([0.5, 1.0, 2.0, 3.0])
    dn = np.array([0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0])    # x s, from mid-gap
    cand = [(a, b) for a in du for b in dn]
    out = np.full(len(mid), -1, np.int64)
    if len(mid) == 0:
        return out
    # nearest first (for a typical edge, lu ~ s)
    cand.sort(key=lambda c: (1.0 + c[0]) ** 2 + c[1] ** 2)
    todo = np.arange(len(mid))
    for a_, b_ in cand:
        if todo.size == 0:
            break
        # across: from mid-gap by b_ s, but beyond a wall only by the part
        # that exceeds half the gap
        off = np.sign(b_) * (0.5 * w[todo] + abs(b_) * s) if b_ != 0 else np.zeros(todo.size)
        Q = mid[todo] + u[todo] * (lu[todo] + a_ * s)[:, None] + n[todo] * off[:, None]
        j = _containing(tree, X, hs, Q)
        ok = (j >= 0) & fl[np.maximum(j, 0)] & (lab[np.maximum(j, 0)] >= 0)
        if walls is not None and ok.any():
            k = np.flatnonzero(ok)
            d = Q[k] - mid[todo[k]]
            L = np.linalg.norm(d, axis=1)
            V, F, N, rmi = walls
            hitw = op.gap_rays(V, F, mid[todo[k]], d / L[:, None], float(L.max()) * 1.001,
                               N=N, rmi=rmi)[0]
            ok[k] = ~(hitw < L)                        # no wall before the point
        out[todo[ok]] = j[ok]
        todo = todo[~ok]
    return out


def build_channels(S, X, nsize, fl, lab, lo=0.003, hi=0.020, node_size=None,
                   triangles=None, min_area=1e-4, min_mouth=0.01, verbose=False, mesh=None):
    """Channels of the gaps lo <= width < hi of ``gap_samples`` S (in the
    model's frame, ``openings.samples_to_model``) that are closed among the
    model's nodes (positions X, cube sizes nsize, fluid flags fl,
    compartment labels lab). Channels of less than ``min_area`` (one wall)
    are left out, and mouth pieces shorter than ``min_mouth``. node_size:
    the spacing of the channel nodes (default: the smallest node size);
    triangles: (n, 3, 3) the walls in the model's frame, for the line of
    sight from a gap to its mouths (None: not checked); or mesh: (V, F)
    the same with shared vertices (less memory). Returns
    ``Channels``; its ``mouths``:
    dicts with ``channel``, ``outer`` (model nodes), ``inner`` (channel
    nodes), ``weights`` (mouth cross-section per face, m^2), ``pos`` (where
    the gap ends, per face), ``width`` (m), ``compartment`` (of the outer
    nodes)."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, float)
    nsize = np.asarray(nsize, float)
    hs = 0.5 * nsize
    s = float(node_size or nsize[fl].min())
    walls = None
    if mesh is None and triangles is not None and len(triangles):
        V = np.asarray(triangles, float).reshape(-1, 3)
        mesh = (V, np.arange(len(V)).reshape(-1, 3))
    if mesh is not None:
        import trimesh
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        V, F = np.asarray(mesh[0], float), np.asarray(mesh[1])
        rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False, validate=False))
        walls = (V, F, op.face_geometry(V, F)[1], rmi)
    sp = float(S["spacing"])
    tree = cKDTree(X)
    W = S["width"]
    band = np.flatnonzero(S["facing"] & (W >= lo) & (W < hi))
    if band.size == 0:
        z = np.zeros(0)
        return Channels(X=np.zeros((0, 3)), v=z, size=z, width=z, links=np.zeros((0, 2), np.int64),
                        channel=np.zeros(0, np.int64), n=0)
    P, Nn, A, w = S["point"][band], S["normal"][band], S["area"][band], W[band]
    mid = P + Nn * (0.5 * w)[:, None]
    j = _containing(tree, X, hs, mid)
    open_ = (j >= 0) & fl[np.maximum(j, 0)] & (lab[np.maximum(j, 0)] >= 0)
    closed = ~open_
    nb = band.size
    # sample graph: along each wall, and across the gap (to a sample of the
    # face the ray hits)
    pr = cKDTree(P).query_pairs(1.5 * sp, output_type="ndarray") if nb else np.zeros((0, 2), int)
    pr = pr[op._same_surface(P, Nn, pr[:, 0], pr[:, 1], sp)]
    # across: the sample of the other wall nearest to where the ray hits it
    # (a face of the other wall can be large: not any sample of that face)
    dd, jx = cKDTree(P).query(P + Nn * w[:, None], distance_upper_bound=1.5 * sp)
    hit = np.isfinite(dd)
    jx = np.where(hit, jx, 0)
    hit &= np.einsum("ij,ij->i", Nn, Nn[jx]) < -0.5          # facing back
    across = np.stack([np.flatnonzero(hit), jx[hit]], 1)
    E = np.concatenate([pr, across])
    E = E[closed[E[:, 0]] & closed[E[:, 1]]]
    comp = _components(nb, E)
    # channels: closed components with enough area (one wall: half the
    # sampled area of both)
    ncomp = int(comp.max()) + 1 if nb else 0
    comp = np.where(closed, comp, -1)
    carea = np.bincount(comp[closed], weights=A[closed], minlength=ncomp) * 0.5
    keep = carea >= min_area
    renum = np.full(ncomp, -1, np.int64)
    renum[keep] = np.arange(keep.sum())
    ch = np.where(comp >= 0, renum[np.maximum(comp, 0)], -1)
    nch = int(keep.sum())
    on = np.flatnonzero(ch >= 0)
    # nodes: the samples of a channel gathered on a grid of node_size
    vox = np.floor(mid[on] / s).astype(np.int64)
    key = np.c_[ch[on], vox]
    ukey, node_of = np.unique(key, axis=0, return_inverse=True)
    node_of = node_of.ravel()
    m = len(ukey)
    node = np.full(nb, -1, np.int64)
    node[on] = node_of
    # the part of the gap that is fluid in the model already (fluid nodes
    # across it, beside the closed mid-gap point) is not counted again
    across = np.zeros(on.size)
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        jj = _containing(tree, X, hs, P[on] + Nn[on] * (f * w[on])[:, None])
        across += ((jj >= 0) & fl[np.maximum(jj, 0)]) / 5.0
    aw = A[on] * w[on] * (1.0 - across)
    # both walls of a gap sample the same space: per node, the samples whose
    # normal points the other way from the node's first sample are the other
    # wall; with both walls present each wall's volume is halved
    first = np.full(m, -1, np.int64)
    first[node_of[::-1]] = on[::-1]
    side = np.einsum("ij,ij->i", Nn[on], Nn[first[node_of]]) >= 0
    vp = np.bincount(node_of[side], weights=aw[side], minlength=m)
    vm = np.bincount(node_of[~side], weights=aw[~side], minlength=m)
    vol = np.where((vp > 0) & (vm > 0), 0.5 * (vp + vm), vp + vm)
    asum = np.bincount(node_of, weights=A[on], minlength=m)
    Xn = np.stack([np.bincount(node_of, weights=A[on] * mid[on, d], minlength=m)
                   for d in range(3)], 1) / asum[:, None]
    wn = np.bincount(node_of, weights=A[on] * w[on], minlength=m) / asum
    lk = E[(node[E[:, 0]] >= 0) & (node[E[:, 1]] >= 0)]
    lk = np.sort(node[lk], axis=1)
    lk = np.unique(lk[lk[:, 0] != lk[:, 1]], axis=0) if len(lk) else np.zeros((0, 2), np.int64)
    # mouths: beyond the edge of the closed samples (where the gap ends,
    # widens, narrows or is open in the grid), the nearest fluid node seen
    # from the gap (``_probe``)
    lab_s = np.full(W.size, -1, np.int64)
    lab_s[band[on]] = ch[on]
    ea, eu, elu = op.mouth_edges(S, lab_s)
    loc = np.full(W.size, -1, np.int64)
    loc[band] = np.arange(nb)
    ms = loc[ea]
    jo = _probe(tree, X, hs, fl, lab, mid[ms], Nn[ms], w[ms], eu, elu, s, walls)
    ok = jo >= 0
    ms, jo = ms[ok], jo[ok]
    # one face per (mouth sample's place on the mid-gap sheet, outer node):
    # both walls and several neighbours of one sample give it once
    fk = np.c_[np.floor(mid[ms] / sp).astype(np.int64), jo]
    _, first_f = np.unique(fk, axis=0, return_index=True)
    ms, jo = ms[first_f], jo[first_f]
    inner = node[ms]
    outer_c = lab[jo]
    mouths = []
    dropped = 0
    if len(ms):
        # pieces: per (channel, compartment), faces within 2 node sizes
        g = np.c_[ch[ms], outer_c]
        ug, gi = np.unique(g, axis=0, return_inverse=True)
        gi = gi.ravel()
        o = np.argsort(gi, kind="stable")
        st = np.searchsorted(gi[o], np.arange(len(ug) + 1))
        for q in range(len(ug)):
            f = o[st[q]:st[q + 1]]
            if f.size > 1:
                pp = cKDTree(mid[ms[f]]).query_pairs(max(2.0 * s, 3.0 * sp), output_type="ndarray")
                piece = _components(f.size, pp)
            else:
                piece = np.zeros(1, np.int64)
            for pc in np.unique(piece):
                ff = f[piece == pc]
                length = _edge_length(mid[ms[ff]], 4.0 * s, sp)
                if length < min_mouth:
                    dropped += 1
                    continue
                ii, oo, ww = inner[ff], jo[ff], w[ms[ff]]
                pair, inv = np.unique(np.c_[ii, oo], axis=0, return_inverse=True)
                # the cross-section length x width, over the faces by their
                # share of the edge samples
                inv = inv.ravel()
                cnt = np.bincount(inv).astype(float)
                wt = cnt / cnt.sum() * length * float(np.median(ww))
                # where each face's part of the gap ends (mean mid-gap point)
                fpos = np.stack([np.bincount(inv, weights=mid[ms[ff], d]) for d in range(3)],
                                1) / cnt[:, None]
                mouths.append(dict(channel=int(ug[q, 0]), compartment=int(ug[q, 1]),
                                   inner=pair[:, 0], outer=pair[:, 1], weights=wt, pos=fpos,
                                   width=float(np.median(ww)),
                                   centroid=mid[ms[ff]].mean(0)))
    stats = dict(band_samples=int(nb), closed_share=float(A[closed].sum() / max(A.sum(), 1e-30)),
                 channels=nch, nodes=int(m), links=int(len(lk)), mouths=len(mouths),
                 mouths_dropped=dropped, volume=float(vol.sum()),
                 channel_area=float(carea[keep].sum()))
    if verbose:
        print(f"channels: {nch} ({stats['channel_area']:.3f} m2, {vol.sum()*1e6:.0f} ml) of the "
              f"{lo*1e3:g}-{hi*1e3:g} mm gaps ({stats['closed_share']:.0%} closed in the grid); "
              f"{m} nodes, {len(mouths)} mouths ({dropped} shorter than {min_mouth*1e3:g} mm left "
              f"out)", flush=True)
    return Channels(X=Xn, v=vol, size=np.full(m, s), width=wn, links=lk,
                    channel=ukey[:, 0].astype(np.int64),
                    n=nch, mouths=mouths, stats=stats)


# ------------------------------------------------------------- the step
@njit(cache=True, nogil=True)
def _mouth_q(H, E, hs, cw, cz, n_f, apw, ho, inn_ok, out_ok, Cd, g2):
    """Q into a channel at level H through a mouth (faces at heights hs,
    sorted, with cumulative weights cw and weight x height cz) from the
    level E outside: the orifice law over the faces below the higher of the
    two; free when the lower one is below their centroid (with the slot's
    hold-up ho on free outflow)."""
    up_, lo_ = (E, H) if E > H else (H, E)
    n = np.searchsorted(hs[:n_f], up_)
    if n == 0:
        return 0.0
    A = apw * cw[n - 1]
    zc = cz[n - 1] / cw[n - 1]
    free = lo_ < zc
    head = up_ - (zc if free else lo_)
    if E > H:
        if not inn_ok or head <= 0.0:
            return 0.0
        return Cd * A * np.sqrt(g2 * head)
    if not out_ok:
        return 0.0
    if free:
        head -= ho
    if head <= 0.0:
        return 0.0
    return -Cd * A * np.sqrt(g2 * head)


@njit(cache=True, nogil=True)
def _net(H, m0, m1, E, fptr, hs, cw, cz, apw, ho, inn_ok, out_ok, Cd, g2):
    s = 0.0
    for q in range(m0, m1):
        f0 = fptr[q]
        s += _mouth_q(H, E[q], hs[f0:], cw[f0:], cz[f0:], fptr[q + 1] - f0, apw[q], ho[q],
                      inn_ok[q], out_ok[q], Cd, g2)
    return s


@njit(cache=True, nogil=True)
def _stage(H, n0, n1, nodes, v, h, en):
    s = 0.0
    for j in range(n0, n1):
        c = nodes[j]
        x = (H - h[c] + en[c]) / (2.0 * en[c])
        s += v[c] * min(max(x, 0.0), 1.0)
    return s


@dual
def channel_flows(mptr, fptr, fh, fw, apw, ho, E, inn_ok, out_ok, vent, nptr, nodes,
                  L, v, h, en, dt, Cd, g2, Q, hs, cw, cz):
    """The flows of the gap channels' mouths for one step (see
    ``Simulation._channel_step``): per channel the level H that solves its
    storage balance (or the pressurised / unvented balance), and Q per
    mouth (m^3/s, + into the channel) at that level. Mouths by channel
    (mptr), faces by mouth (fptr: heights fh, weights fw); E the level
    outside each mouth (-inf: none); vent: air outside some mouth of the
    channel. hs, cw, cz: work arrays of the faces' size."""
    nch = mptr.shape[0] - 1
    for k in prange(nch):
        m0, m1 = mptr[k], mptr[k + 1]
        for q in range(m0, m1):
            Q[q] = 0.0
        if m1 == m0:
            continue
        n0, n1 = nptr[k], nptr[k + 1]
        V = 0.0
        cap = 0.0
        bot = np.inf
        top = -np.inf
        for j in range(n0, n1):
            c = nodes[j]
            V += L[c] * v[c]
            cap += v[c]
            bot = min(bot, h[c] - en[c])
            top = max(top, h[c] + en[c])
        wet_out = False
        lmin = np.inf
        lmax = -np.inf
        for q in range(m0, m1):
            if E[q] > -np.inf:
                wet_out = True
                lmin = min(lmin, E[q])
                lmax = max(lmax, E[q])
        if V <= 0.0 and not wet_out:
            continue
        # the faces of each mouth by height
        for q in range(m0, m1):
            f0, f1 = fptr[q], fptr[q + 1]
            o = np.argsort(fh[f0:f1], kind="mergesort")
            a = 0.0
            b = 0.0
            for r in range(f1 - f0):
                j = f0 + o[r]
                hs[f0 + r] = fh[j]
                a += fw[j]
                b += fw[j] * fh[j]
                cw[f0 + r] = a
                cz[f0 + r] = b
        mode = 0                                    # 0 storage, 1 full, 2 no venting
        need = 0.0
        if vent[k]:
            if V + dt * _net(top, m0, m1, E, fptr, hs, cw, cz, apw, ho, inn_ok, out_ok,
                             Cd, g2) - cap >= 0.0:
                mode = 1
                need = (cap - V) / dt
                lo = top
                hi = top + 1.0
                while _net(hi, m0, m1, E, fptr, hs, cw, cz, apw, ho, inn_ok, out_ok,
                           Cd, g2) > need and hi < top + 1e3:
                    hi = top + 2.0 * (hi - top)
            else:
                lo = bot
                hi = top
        else:
            mode = 2
            lo = min(lmin, bot) - 1e-9
            hi = max(lmax, top) + 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            f = _net(mid, m0, m1, E, fptr, hs, cw, cz, apw, ho, inn_ok, out_ok, Cd, g2)
            if mode == 0:
                f = V + dt * f - _stage(mid, n0, n1, nodes, v, h, en)
            elif mode == 1:
                f -= need
            if f > 0.0:
                lo = mid
            else:
                hi = mid
        H = 0.5 * (lo + hi)
        for q in range(m0, m1):
            f0 = fptr[q]
            Q[q] = _mouth_q(H, E[q], hs[f0:], cw[f0:], cz[f0:], fptr[q + 1] - f0, apw[q],
                            ho[q], inn_ok[q], out_ok[q], Cd, g2)


# ------------------------------------------------------------- review
def export_channels(prefix, ch, comp_n0, to_file=None):
    """Files to review the gap channels of a model (open the .vtp files in
    ParaView over the mesh): ``<prefix>_channels.csv`` (one row per
    channel: its nodes, volume, mean width, the length of its mouths and
    the compartments it joins; compartment ids as in the model, channels
    being comp_n0 + channel), ``<prefix>_channels.vtp`` (the channel nodes
    and their links; point data ``channel``, ``width_mm``) and
    ``<prefix>_mouths.vtp`` (where each mouth opens; ``channel``,
    ``compartment``, ``mouth``). to_file: maps model-frame points to the
    frame wanted (e.g. the STL's, ``openings.from_model``); None: the
    model's frame."""
    from .worldviz import write_vtp
    f = to_file or (lambda p: p)
    m = ch.v.size
    vol = np.bincount(ch.channel, weights=ch.v, minlength=ch.n)
    wmean = np.bincount(ch.channel, weights=ch.v * ch.width, minlength=ch.n) / np.maximum(vol, 1e-30)
    cen = np.stack([np.bincount(ch.channel, weights=ch.v * ch.X[:, d], minlength=ch.n)
                    for d in range(3)], 1) / np.maximum(vol, 1e-30)[:, None]
    cen = f(cen) if ch.n else cen
    nn = np.bincount(ch.channel, minlength=ch.n)
    joins = [set() for _ in range(ch.n)]
    mlen = np.zeros(ch.n)
    nm = np.zeros(ch.n, np.int64)
    for mo in ch.mouths:
        k = mo["channel"]
        joins[k].add(mo["compartment"])
        mlen[k] += float(np.sum(mo["weights"])) / max(mo["width"], 1e-30)
        nm[k] += 1
    with open(prefix + "_channels.csv", "w") as fh:
        fh.write("channel,compartment,nodes,volume_ml,width_mean_mm,mouths,mouth_length_mm,"
                 "joins,x,y,z\n")
        for k in range(ch.n):
            c = cen[k]
            fh.write(f"{k},{comp_n0 + k},{nn[k]},{vol[k]*1e6:.3f},{wmean[k]*1e3:.2f},{nm[k]},"
                     f"{mlen[k]*1e3:.1f},{' '.join(map(str, sorted(joins[k])))},"
                     f"{c[0]:.5f},{c[1]:.5f},{c[2]:.5f}\n")
    if m:
        lines = [np.asarray(e, np.int64) for e in ch.links]
        write_vtp(prefix + "_channels.vtp", f(ch.X), lines=lines or None,
                  point_data=dict(channel=ch.channel.astype(np.float32),
                                  width_mm=(ch.width * 1e3).astype(np.float32)))
    if ch.mouths:
        P = np.concatenate([mo["pos"] for mo in ch.mouths])
        rep = [len(mo["pos"]) for mo in ch.mouths]
        write_vtp(prefix + "_mouths.vtp", f(P),
                  point_data=dict(channel=np.repeat([mo["channel"] for mo in ch.mouths],
                                                    rep).astype(np.float32),
                                  compartment=np.repeat([mo["compartment"] for mo in ch.mouths],
                                                        rep).astype(np.float32),
                                  mouth=np.repeat(np.arange(len(ch.mouths)), rep)
                                  .astype(np.float32)))
