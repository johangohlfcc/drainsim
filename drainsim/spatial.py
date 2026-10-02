"""Links near a point, for setup steps that visit many points (explicit
holes, plugs): a k-d tree on the node positions and the links indexed by
their end nodes, so that each point looks only at the links around it
instead of at all links. The candidates come with a small margin; the
callers apply their own (exact) test to them, so the result is the one of
a test over all links."""
from __future__ import annotations

import numpy as np


class NearLinks:
    """Links (a[e], b[e]) between nodes at positions X."""

    def __init__(self, X, a, b=None):
        from scipy.spatial import cKDTree
        self.X = X
        self.tree = cKDTree(X, balanced_tree=False, compact_nodes=False)
        N = X.shape[0]
        self.ends = []
        for end in (a, b):
            if end is None:
                continue
            end = np.asarray(end, np.int64)
            ok = end >= 0
            e = np.flatnonzero(ok)
            o = np.argsort(end[e], kind="stable")
            e = e[o]
            ptr = np.zeros(N + 1, np.int64)
            np.cumsum(np.bincount(end[e], minlength=N), out=ptr[1:])
            self.ends.append((ptr, e))

    def nodes(self, c, r):
        """The nodes within r of c (and a little beyond)."""
        r = float(r) * (1.0 + 1e-9) + 1e-12
        return np.asarray(self.tree.query_ball_point(np.asarray(c, float), r), np.int64)

    def links(self, c, r):
        """Sorted ids of the links with an end (a, or b if given) among the
        nodes within r of c (and a little beyond)."""
        n = self.nodes(c, r)
        if n.size == 0:
            return np.zeros(0, np.int64)
        parts = []
        for ptr, e in self.ends:
            lo, hi = ptr[n], ptr[n + 1]
            m = hi - lo
            if m.sum() == 0:
                continue
            start = np.repeat(lo - np.cumsum(np.r_[0, m[:-1]]), m)
            parts.append(e[start + np.arange(m.sum())])
        if not parts:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(parts))
