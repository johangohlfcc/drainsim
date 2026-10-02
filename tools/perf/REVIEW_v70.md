# drainsim 7.0 performance review (measured on 6.4, 30 Sep 2026)

Machine: 16 cores / 32 threads, 128 GB. Car xc90, octree 3 levels, 2 sub-cells, narrow 4, film on
(the settings of the 5 mm and 3 mm runs). Profiles: py-spy sampling (20 Hz) and a memory sampler
with stack dumps at each new peak. Raw data in runs\perf\results (local, not in git:
scaling_8mm.*, p12, p8, r5dip, r5hang); the scripts are in this folder (README.md).

## 1. Parallel loops: races

All 78 prange loops (fsm, par, stepk, film, sidesplit, voxel) read: no race and no unsafe
accumulation. Sums go into fixed blocks (stepk.NBLK = 64, film NB = 64) added in a fixed order; max
and min are numba reductions; the union-find in par._roots only links nodes inside its chunk.
The one shared write is par._connected_to (`hit[root] = True` from several threads, same value,
documented): benign.

## 2. Determinism

thread_scaling.py on the 8 mm car (10.8 M nodes), 6 steps of dip and 6 of drip-off from the same
state: the state (L, B, A) is identical for 1, 2, 4, 8, 16 and 32 threads. The film sweep is
serial, so the film is too.

## 3. Thread scaling: the step is mostly serial

| 8 mm, s per step | 1 thread | 16 threads | speed-up |
|---|---|---|---|
| dip (t = 8 s) | 2.27 | 1.99 | 1.14x |
| drip-off | 4.02 | 3.54 | 1.13x |

32 threads: no gain. At 16 threads the dip step is fill-spill 1.07 s (the serial cascade `S fill`
0.33 s, the height sort 0.16 s, the rest parallel kernels that scale 2-3x), film 0.52 s (serial),
bath/atmosphere 0.18 s, throats 0.12 s. The drip-off step is film 2.74 s (77 %, serial sweep, 10
sub-steps per 0.5 s step).

## 4. Time-loop hotspots (py-spy, 8 mm, 20 dip steps at t = 12-14 s, 4.0 s per step)

| part | share |
|---|---|
| fill-spill (liquid 30 %, air 20 %) | 48 % |
|  of which the serial cascade `fsm.solve`/`_fill` | 20 % |
|  of which the hierarchy kernels | 15 % |
|  of which the height sort | 7 % |
| film | 26 % |
|  of which the serial sweep | 12 % |
|  of which `_sorted_space` + `_sweep_order` (re-sorting the film edges every step while turning) | 12 % |
| bath / atmosphere labelling | 9 % |
| throats (per-throat Python loop, bodies) | 6 % |

## 5. Setup: time, memory, scaling

| grid | nodes | octree build | Simulation setup | setup peak | memory while stepping |
|---|---|---|---|---|---|
| 12 mm | 5.0 M | 28 s | 306 s | 19 GB | - |
| 8 mm | 10.8 M | 80 s | 568 s | 39 GB | 5-6 GB |
| 5 mm (29 Sep) | 21 M | 305 s | 1516 s | 86 GB | about 8-10 GB |
| 3 mm (30 Sep) | 60 M | 1411 s | 8061 s (swapping) | 272 GB | 30-38 GB |

Setup time by phase (share of Simulation setup):

| phase | 12 mm | 8 mm |
|---|---|---|
| `model._node_wall_distance` (+ its `sample_triangles`) | 40 % | 57 % |
| `octree._subcell_graph` (sub-cells, narrow passages, volume sharing) | 36 % | about 25 % |
|  `_triangles_near` (up to 125 serial numpy passes over all 6 M triangles) | 16 % | 7 % |
|  `narrow_cells` | 10 % | about 6 % |
| octree build (`_dilate`/`_unique_chunks`: serial numpy sorts) | 8 % of the total | 12 % |

Memory: the setup peak is `_node_wall_distance` alone (19 of 19 GB at 12 mm, 39 of 39 GB at 8 mm;
everything else stays under 6 GB at 8 mm). It samples every triangle by its longest edge
(`sample_triangles(..., spacing=1.0)`), so long thin CAD triangles give about n^2/2 points where
about n would do, and shared vertices repeat: about 240 M points at 12 mm where a 6 mm spacing needs
about 15 M. Then `list()` + `concatenate` hold the cloud twice, and a cKDTree is built on it.
`narrow_cells` does the same sampling but thins it to one point per cube (`_thin`) first.

Resident state after setup: 434 B/node (12 mm), 382 B/node (8 mm); + about 12 % of caches after
running. Of this:
- `ograph.indices`, `indptr`, `key`, `level`: 62 B/node, a second copy of the graph; only
  `ograph.boundary` is used after setup.
- `grid.triangles` + `grid.faces`: 550 MB (the full mesh, not needed after setup).
- `sim.host`: identity map on the octree, 8 B/node. `G`, `P` (compressible air, off): 16 B/node.
- int64 everywhere: `nbr` indices 45 B/node, labels, film element indices; the node and link counts
  fit int32 up to about 250 M nodes.

## 6. Allocation in the time loop

About 30 node-sized arrays are made fresh per step (step inputs, masks, `inj`, `L`, the fill-spill
copies `h - ec` etc.) and 3 per film sub-step. Measured: a fresh 87 MB result costs 13 ms against 6
ms into a kept buffer, so about 5 % of a step. `_throat_step`: `can_gain`/`can_lose` build a
throat-sized mask per call (per flowing throat), and the per-throat loop is Python. Small at 3000
throats.

## 7. Movie rendering (one worker, 5 mm recording)

| | dip frames | drip-off frames |
|---|---|---|
| loading before the first frame | 2.7 min | 2.7 min |
| per frame | 3.5 s | 6.0-6.5 s |
| peak memory | 24 GB | 26 GB |

- Loading and the memory spike are `_build_film_shells` (door_dip_film_movie.py): the display mesh
  is subdivided to cell size (`trimesh.remesh.subdivide_to_size`) and mapped to the film carrier,
  in every worker, for every part. The result depends only on the recording.
- Per frame, the film colouring is 80 % (dip) / 60 % (drip-off): log10 + colour + deep copy to VTK
  + culling of all faces + a new cell array for both shells, every frame, although the film changes
  only once per model step (3 frames per step in the dip). The VTK render itself is about 1.2 s.

## Plan for 7.0 (bit-identical unless marked)

1. **Setup memory: `_node_wall_distance`** (identical). Slab by slab (points within 3 dx of the
   slab), exact duplicate points dropped (the nearest distance is unchanged), no list+concat copy.
   Expected: 8 mm setup peak 39 -> about 6 GB, 3 mm 272 -> about 40-60 GB (no swap), setup time
   about half. The same pattern: gseg.py's surface distance (distance="surface").
   Later, not identical: exact point-triangle distances from the cells' triangle lists (more
   accurate), to be validated against the 20 mm and door cases.
2. **Render: cache the film shells per recording, colour the film once per model step** (examples
   only, same pixels). Expected: loading 2.7 min -> seconds, worker peak 24 -> about 5-10 GB
   (5 mm), 3 mm: 3-4 workers instead of 2; frames about 2x faster in the dip.
3. **Film in the time loop** (identical): the sweep level by level in parallel (receivers pull
   inflow in sender order), `_sorted_space`/`_sweep_order` as one parallel kernel. Expected:
   drip-off step 3.5 -> about 1.5 s, dip step -10 %.
4. **Fill cascade** (identical): O(1) ancestor tests (Euler-tour intervals of the hierarchy) for
   the "inside the overflowing depression" and "child of P" walks. Expected: `S fill` a fraction of
   0.33 s.
5. **Memory trims** (identical): free the graph copy, triangles and faces after setup, drop `host`
   on the octree, allocate `G`/`P` only with compressible air, int32 node/link indices, reused step
   buffers. Expected: resident 380-430 -> about 250 B/node; int32 also cuts memory traffic.
6. **Setup kernels** (identical): `_triangles_near` as one numba kernel over the triangles; the
   octree build's sorts with `par.sort_keys`.

Each step: bit-identical check against 6.4 (tests; 20 mm car uniform + octree; 8 mm thread
fingerprint), then timing on 8 mm, then a 5 mm or 3 mm run.
