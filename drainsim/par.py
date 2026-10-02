"""Parallel building blocks (numba prange) for the per-step passes.

All functions give exactly the result of their serial counterparts, for any
number of threads, so a run is reproducible whatever ``NUMBA_NUM_THREADS``
is. With one thread they cost about the same as the serial code.

- ``sort_keys`` / ``par_sort``: sample sort of int64 keys (value ranges
  split in parallel, each sorted by numpy in a thread pool);
- ``compact``: indices of the True entries, in order;
- ``label_bodies``: connected components of a masked graph, numbered by
  their lowest node index (as ``model._label_bodies_csr``);
- ``connected_to``: nodes whose component contains a seed node;
- ``fingerprint``: order-independent 64-bit hash of an array (cache keys).
"""

import threading
import types

import numpy as np
from numba import njit, prange, get_num_threads, uint64


def nthreads():
    return int(get_num_threads())


_PAR_LOCK = threading.Lock()
_LOCAL = threading.local()


def serial_thread():
    """From now on the ``dual`` kernels called in this thread run their plain
    loop: a helper thread next to the model (e.g. the field writer of a
    recording) then leaves the parallel kernels and the cores to the model.
    Same results (see ``dual``)."""
    _LOCAL.serial = True


def dual(f):
    """Compile ``f`` twice: with ``parallel=True`` (used when numba has more
    than one thread) and as a plain ``nogil`` loop (one thread; safe to run
    from several Python threads at once, whatever the threading layer).
    Only one parallel kernel runs at a time: a call made while another
    thread is inside one runs the plain loop instead (numba's default
    'workqueue' threading layer does not allow concurrent launches); so do
    the calls from a ``serial_thread``. Both release the GIL while they
    run, so other Python threads go on meanwhile."""
    par = njit(parallel=True, cache=True, nogil=True)(f)
    g = types.FunctionType(f.__code__, f.__globals__, f.__name__ + "_serial",
                           f.__defaults__, f.__closure__)
    g.__qualname__ = f.__qualname__ + "_serial"
    g.__module__ = f.__module__
    ser = njit(cache=True, nogil=True)(g)

    def call(*args):
        if get_num_threads() > 1 and not getattr(_LOCAL, "serial", False) \
                and _PAR_LOCK.acquire(blocking=False):
            try:
                return par(*args)
            finally:
                _PAR_LOCK.release()
        return ser(*args)
    call.par = par
    call.serial = ser
    call.__name__ = f.__name__
    call.__doc__ = f.__doc__
    return call


# ----------------------------------------------------------------- chunks
@njit(cache=True, nogil=True)
def _nchunks(n, nt):
    c = 4 * nt
    if c > n:
        c = max(n, 1)
    return c


# ------------------------------------------------------------------- sort
def bucket(a):
    """See ``_bucket``."""
    return _bucket(a, nthreads())


@dual
def _bucket(a, nt):
    """Split the int64 keys ``a`` into value ranges: returns (out, starts),
    where out[starts[b]:starts[b+1]] holds the keys of range b (unsorted)
    and every key of a range is below every key of the next. Sorting each
    range then sorts the whole (``par_sort``, ``fsm.sort_keys``)."""
    n = a.shape[0]
    B = 8 * nt                                   # ranges
    OV = 64                                      # samples per range
    ns = B * OV
    if n < 4 * ns:
        out = a.copy()
        bst = np.zeros(2, np.int64)
        bst[1] = n
        return out, bst
    step = n // ns
    s = np.empty(ns, np.int64)
    for j in prange(ns):
        s[j] = a[j * step + (j * 7919) % step]
    s.sort()
    spl = np.empty(B - 1, np.int64)
    for j in range(B - 1):
        spl[j] = s[(j + 1) * OV]
    C = 4 * nt                                   # chunks of the input
    cnt = np.zeros((C, B), np.int64)
    bk = np.empty(n, np.int32)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        for j in range(lo, hi):
            b = np.searchsorted(spl, a[j])
            bk[j] = b
            cnt[k, b] += 1
    off = np.empty((C, B), np.int64)
    bst = np.empty(B + 1, np.int64)
    t = 0
    for b in range(B):
        bst[b] = t
        for k in range(C):
            off[k, b] = t
            t += cnt[k, b]
    bst[B] = n
    out = np.empty(n, np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        o = off[k].copy()
        for j in range(lo, hi):
            b = bk[j]
            out[o[b]] = a[j]
            o[b] += 1
    return out, bst


_POOL = [None, 0]


def sort_keys(k):
    """Sort the int64 array ``k`` in place. With several numba threads, the
    keys are first split into value ranges (``par.bucket``) and the ranges
    are sorted by numpy in a thread pool (numpy's sort releases the GIL)."""
    nt = get_num_threads()
    if nt == 1 or k.shape[0] < (1 << 18):
        k.sort()
        return k
    out, bst = bucket(k)
    if _POOL[0] is None or _POOL[1] != nt:
        from concurrent.futures import ThreadPoolExecutor
        _POOL[0] = ThreadPoolExecutor(max_workers=nt)
        _POOL[1] = nt
    B = bst.shape[0] - 1
    list(_POOL[0].map(lambda b: out[bst[b]:bst[b + 1]].sort(), range(B)))
    k[:] = out
    return k


def par_sort(a):
    """Sorted copy of the int64 array ``a``."""
    return sort_keys(np.array(a, np.int64))


# ---------------------------------------------------------------- compact
def compact(flag):
    """``np.flatnonzero(flag)`` (int64), in parallel."""
    return _compact(flag, nthreads())


@dual
def _compact(flag, nt):
    n = flag.shape[0]
    C = _nchunks(n, nt)
    cnt = np.zeros(C + 1, np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        m = 0
        for j in range(lo, hi):
            if flag[j]:
                m += 1
        cnt[k + 1] = m
    for k in range(C):
        cnt[k + 1] += cnt[k]
    out = np.empty(cnt[C], np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        o = cnt[k]
        for j in range(lo, hi):
            if flag[j]:
                out[o] = j
                o += 1
    return out


# -------------------------------------------------------------- labelling
@dual
def _roots(mask, comp, ptr, idx, nt):
    """Root (lowest node index) of the component of every masked node;
    -1 elsewhere. Edges join masked neighbours with equal ``comp``."""
    N = mask.shape[0]
    C = _nchunks(N, nt)
    uf = np.empty(N, np.int64)
    for c in prange(N):
        uf[c] = c
    # 1. unions inside each chunk of node indices (each chunk only writes
    #    its own entries); nodes with an edge to a later chunk are flagged
    ncross = np.zeros(C + 1, np.int64)
    cross = np.zeros(N, np.bool_)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        m = 0
        for c in range(lo, hi):
            if not mask[c]:
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n < 0 or n < c or not mask[n] or comp[n] != comp[c]:
                    continue
                if n >= hi:
                    if not cross[c]:
                        cross[c] = True
                    m += 1
                    continue
                a = c
                while uf[a] != a:
                    uf[a] = uf[uf[a]]
                    a = uf[a]
                b = n
                while uf[b] != b:
                    uf[b] = uf[uf[b]]
                    b = uf[b]
                if a != b:
                    if a < b:
                        uf[b] = a
                    else:
                        uf[a] = b
        ncross[k + 1] = m
    for k in range(C):
        ncross[k + 1] += ncross[k]
    E = ncross[C]
    ea = np.empty(E, np.int64)
    eb = np.empty(E, np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        o = ncross[k]
        for c in range(lo, hi):
            if not cross[c]:
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n < 0 or n < hi or not mask[n] or comp[n] != comp[c]:
                    continue
                ea[o] = c
                eb[o] = n
                o += 1
    # 2. the edges between chunks, in sequence (with path halving)
    for e in range(E):
        a = ea[e]
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        b = eb[e]
        while uf[b] != b:
            uf[b] = uf[uf[b]]
            b = uf[b]
        if a != b:
            if a < b:
                uf[b] = a
            else:
                uf[a] = b
    # 3. roots (read only)
    root = np.empty(N, np.int64)
    for c in prange(N):
        r = np.int64(-1)
        if mask[c]:
            r = np.int64(c)
            while uf[r] != r:
                r = uf[r]
        root[c] = r
    return root


@dual
def _number(mask, root, nt):
    N = mask.shape[0]
    C = _nchunks(N, nt)
    cnt = np.zeros(C + 1, np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        m = 0
        for c in range(lo, hi):
            if mask[c] and root[c] == c:
                m += 1
        cnt[k + 1] = m
    for k in range(C):
        cnt[k + 1] += cnt[k]
    rid = np.empty(N, np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        o = cnt[k]
        for c in range(lo, hi):
            if mask[c] and root[c] == c:
                rid[c] = o
                o += 1
    body = np.empty(N, np.int64)
    for c in prange(N):
        b = -1
        if mask[c]:
            b = rid[root[c]]
        body[c] = b
    return body, cnt[C]


def label_bodies(mask, comp, ptr, idx):
    """Same as ``model._label_bodies_csr``: (body id per node or -1, count),
    bodies numbered by their lowest node index."""
    mask = np.ascontiguousarray(mask, np.bool_)
    root = _roots(mask, comp, ptr, idx, nthreads())
    return _number(mask, root, nthreads())


@dual
def _connected_to(mask, root, seed):
    N = mask.shape[0]
    hit = np.zeros(N, np.bool_)
    for c in prange(N):
        if mask[c] and seed[c]:
            hit[root[c]] = True                  # same value from every thread
    out = np.zeros(N, np.bool_)
    for c in prange(N):
        if mask[c]:
            out[c] = hit[root[c]]
    return out


def connected_to(mask, comp, ptr, idx, seed):
    """Masked nodes whose component (see ``label_bodies``) holds a masked
    node with ``seed`` set."""
    mask = np.ascontiguousarray(mask, np.bool_)
    root = _roots(mask, comp, ptr, idx, nthreads())
    return _connected_to(mask, root, np.ascontiguousarray(seed, np.bool_))



def connected_sides(lo, hi, comp, ptr, idx, seed):
    """``connected_to(lo, ...)`` and ``connected_to(hi, ...)`` (the same
    ``comp`` and seeds) for two disjoint masks, from one labelling: the
    components are keyed by ``comp`` and the side, so none spans both
    masks. Returns (the lo result, the hi result, the roots), the roots for
    ``connected_across``."""
    lo = np.ascontiguousarray(lo, np.bool_)
    hi = np.ascontiguousarray(hi, np.bool_)
    mask, key = _side_key(lo, hi, comp)
    root = _roots(mask, key, ptr, idx, nthreads())
    out = _connected_to(mask, root, np.ascontiguousarray(seed, np.bool_))
    a, b = _split(out, lo)
    return a, b, root


@dual
def _side_key(lo, hi, comp):
    N = lo.shape[0]
    mask = np.empty(N, np.bool_)
    key = np.empty(N, np.int64)
    for c in prange(N):
        mask[c] = lo[c] or hi[c]
        key[c] = 2 * np.int64(comp[c]) + (1 if hi[c] else 0)
    return mask, key


@dual
def _split(out, lo):
    N = out.shape[0]
    a = np.empty(N, np.bool_)
    b = np.empty(N, np.bool_)
    for c in prange(N):
        a[c] = out[c] and lo[c]
        b[c] = out[c] and not lo[c]
    return a, b


def connected_across(lo, hi, A, root, comp, ptr, idx, seed):
    """``connected_to(lo | (hi & ~A), comp, ptr, idx, seed)`` for ``seed``
    within ``lo``, from the labelling of ``connected_sides(lo, hi, ...)``
    (``root``) when ``A`` is one of its results on ``hi`` (a union of whole
    hi components): its components are the side components joined by the
    links across (between lo and hi & ~A nodes, the links of
    ``label_bodies``). Only those links and the roots are joined."""
    lo = np.ascontiguousarray(lo, np.bool_)
    hi = np.ascontiguousarray(hi, np.bool_)
    A = np.ascontiguousarray(A, np.bool_)
    ra, rb = _across_links(lo, hi, A, root, comp, ptr, idx, nthreads())
    uf, met = _join_roots(ra, rb, lo.shape[0])
    return _joined_seeded(lo, hi, A, root, uf, met, np.ascontiguousarray(seed, np.bool_))


@dual
def _across_links(lo, hi, A, root, comp, ptr, idx, nt):
    """The roots of the two ends of every link between a lo node and a
    hi & ~A node (n > c in the row of c, the same comp)."""
    N = lo.shape[0]
    C = _nchunks(N, nt)
    cnt = np.zeros(C + 1, np.int64)
    for k in prange(C):
        m = 0
        for c in range(k * N // C, (k + 1) * N // C):
            if not (lo[c] or (hi[c] and not A[c])):
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n < 0 or n < c or lo[n] == lo[c] or comp[n] != comp[c]:
                    continue
                if lo[n] or (hi[n] and not A[n]):
                    m += 1
        cnt[k + 1] = m
    for k in range(C):
        cnt[k + 1] += cnt[k]
    ra = np.empty(cnt[C], np.int64)
    rb = np.empty(cnt[C], np.int64)
    for k in prange(C):
        o = cnt[k]
        for c in range(k * N // C, (k + 1) * N // C):
            if not (lo[c] or (hi[c] and not A[c])):
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n < 0 or n < c or lo[n] == lo[c] or comp[n] != comp[c]:
                    continue
                if lo[n] or (hi[n] and not A[n]):
                    ra[o] = root[c]
                    rb[o] = root[n]
                    o += 1
    return ra, rb


@njit(cache=True, nogil=True)
def _join_roots(ra, rb, N):
    """Union-find over the roots joined by the links (ra, rb), towards the
    lowest root; only the roots met are set (``met``)."""
    uf = np.empty(N, np.int64)
    met = np.zeros(N, np.bool_)
    for e in range(ra.shape[0]):
        for r in (ra[e], rb[e]):
            if not met[r]:
                met[r] = True
                uf[r] = r
    for e in range(ra.shape[0]):
        a = ra[e]
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        b = rb[e]
        while uf[b] != b:
            uf[b] = uf[uf[b]]
            b = uf[b]
        if a != b:
            if a < b:
                uf[b] = a
            else:
                uf[a] = b
    return uf, met


@dual
def _joined_seeded(lo, hi, A, root, uf, met, seed):
    """The nodes of lo | (hi & ~A) whose joined component holds a seed
    (in lo)."""
    N = lo.shape[0]
    hit = np.zeros(N, np.bool_)
    for c in prange(N):
        if seed[c] and lo[c]:
            r = root[c]
            if met[r]:
                while uf[r] != r:
                    r = uf[r]
            hit[r] = True                     # the same value from every thread
    out = np.empty(N, np.bool_)
    for c in prange(N):
        x = False
        if lo[c] or (hi[c] and not A[c]):
            r = root[c]
            if met[r]:
                while uf[r] != r:
                    r = uf[r]
            x = hit[r]
        out[c] = x
    return out

# ------------------------------------------------------------ fingerprint
@njit(cache=True, nogil=True)
def _mix(x):
    x = (x ^ (x >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> uint64(27))) * uint64(0x94D049BB133111EB)
    return x ^ (x >> uint64(31))


@dual
def _fp(u):
    n = u.shape[0]
    s = uint64(0)
    for j in prange(n):
        s += _mix(uint64(u[j]) ^ _mix(uint64(j) + uint64(0x9E3779B97F4A7C15)))
    return s


def fingerprint(*arrays):
    """64-bit hash of the arrays' values and shapes, independent of the
    number of threads."""
    out = []
    for a in arrays:
        a = np.ascontiguousarray(a)
        if a.dtype == np.bool_:
            a = a.view(np.uint8)
        b = a.reshape(-1)
        if b.dtype.itemsize == 8:
            b = b.view(np.uint64)
        elif b.dtype.itemsize == 4:
            b = b.view(np.uint32)
        elif b.dtype.itemsize == 1:
            b = b.view(np.uint8)
        else:
            b = b.astype(np.int64).view(np.uint64)
        out.append((b.shape[0], str(a.dtype), int(_fp(b))))
    return tuple(out)
