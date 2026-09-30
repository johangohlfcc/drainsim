# drainsim — transient geometric drainage and accessibility (prototype)

A from-scratch Python prototype that extends the geometric electro-dip
method (Göhl et al., *A Novel Geometric Simulation Method for Rapid
Prediction of Accessibility and Drainage in the Electro-dip Coating Process*)
in two ways, still without resolving the flow:

1. **Spill routing (fill-spill-merge).** Liquid leaving a cavity is routed
   downhill to the next cavity, or to the bath, instead of disappearing. The
   same happens in reverse for air under the bath.
2. **Transient drainage and filling.** The fluid space is split into
   *compartments* joined by *throats* (holes, gaps and slots). Liquid moves
   between compartments through an orifice law, using the pool levels on both
   sides plus capillary hold-up and venting. Drainage and filling therefore
   take time that depends on hole size, and the model gives a real transient
   instead of a sequence of equilibria.

It runs on 2D sections and on 3D voxel grids (from synthetic shapes or an STL
file). The algorithms are deliberately simple and grid based, so that they
can later be moved into IBOFlow on the octree grid.

---

## Quick start

```bash
pip install numpy scipy numba scikit-image matplotlib trimesh pytest
pip install vtk                 # optional: 3D MP4 rendering (also needs ffmpeg)
export PYTHONPATH=$PWD          # run from this folder
python -m pytest -q tests       # 99 verification tests, ~2 min
python examples/01_torricelli.py
python examples/02_spill_cascade.py
python examples/03_door_section.py   # ~6 min
python examples/04_box3d_stl.py      # ~1 min: world-frame VTK + MP4 (3D)
python examples/05_door_animation.py # ~2 min: world-frame MP4/GIF + VTK (2D)
python examples/06_film_validation.py # ~20 s: film vs Jeffreys and Landau-Levich
python examples/07_door_film.py      # ~1.5 min: door section with wall film

# the article's car door (Fig. 7 / Fig. 8), mesh not included:
python examples/door_article.py --stl door.stl --dx 0.006 0.005 0.004 --tilt 22.5 45
#   --subcells 0  binary cell volumes (v2 behaviour); default 4
python examples/door_figures.py --stl door.stl --runs examples/out/door --fig8-dx 0.004
```

Figures are written to `examples/out/`.

Minimal use:

```python
from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes
import numpy as np

grid = Grid.from_mesh("door.stl", dx=0.004)            # metres
motion = Keyframes(times=[0, 6, 8, 14, 40],
                   angles=np.zeros((5, 3)),             # Euler x,y,z [deg]
                   shifts=[[0,0,.2],[0,0,-1],[0,0,-1],[0,0,.2],[0,0,.2]],
                   ndim=3, pivot=np.zeros(3), bath_level=0.0)
sim = Simulation(grid, motion)                         # transient model
hist = sim.run(snapshot_times=[7, 14, 40]).arrays()
hist["liquid_above_bath"], hist["air_below_bath"], sim.pockets()
```

---

## World-frame visualisation (the object moves, not the bath)

The solver works in the object frame: the grid is fixed to the object, and
the bath surface and gravity rotate around it. `drainsim.worldviz` reverses
this for output. Every stored snapshot is mapped with the motion's pose,
`x_world = R(t)(x_obj − pivot) + pivot + T(t)`, so the bath becomes a fixed box
and the object moves through it with the liquid it carries, as in the plant.

```python
hist = sim.run(snapshot_every=0.2)            # store frames for animation
from drainsim.worldviz import export_world, animate_2d
export_world(sim, hist, "out/vtk", prefix="door",
             object_mesh="door.stl")          # optional: draw the real mesh
animate_2d(sim, hist, "out/door.mp4", fps=5)  # 2D runs: MP4 or .gif
from drainsim.render3d import render_animation   # 3D runs (needs vtk, ffmpeg)
render_animation("out/vtk", "door", "out/door.mp4", fps=5)
```

`export_world` writes compressed VTK XML files plus `.pvd` time collections
(real times, so ParaView's Play and time slider follow the process):

| File | Content |
|---|---|
| `<prefix>_object.pvd` | object surface per frame (voxel iso-surface, or the given STL) |
| `<prefix>_liquid.pvd` | surface of all liquid that is not bath: cavity contents, compartments still draining, carried liquid |
| `<prefix>_air.pvd` | surface of air below the bath surface: trapped pockets and compartments not yet filled (accessibility) |
| `<prefix>_volume.pvd` | full cell grid in world coordinates with `liquid`, `retained`, `bath`, `air_below`, `solid`, `compartment` (every `volume_every`-th frame; it is large) |
| `<prefix>_bath.vtp` | static bath box |
| `<prefix>_open_in_paraview.py` | loads and colours everything (ParaView: View > Python Shell > Run Script) |

Free surfaces are drawn as flat planes that hold the same volume as the
cells (`flat_surfaces=True`). Without this, a pool in a rotated grid shows
a cell staircase. This is display only; the stored `volume` data are the raw
cell values. 2D runs give contour lines and a quad grid in the x-y plane.
The ParaView helper script has not been run here (there is no ParaView in the
build environment), but all VTK files were read back and checked with the
VTK library.

---

## The model

### State and frame
The grid is a uniform voxel grid fixed to the object. Cells cut by the surface
are *solid* for connectivity, as in IBOFlow. Each fluid cell stores a liquid
fraction `L ∈ [0,1]` of its *effective volume* (see Cut-cell volumes below).
As in the article, the object does not move on the grid: each step only
the up-vector `u` and the bath level `z_b` (along `u`) are updated in the
object frame. Cell height is `h = u·x`.

### Compartments and throats (`compartments.py`, done once)
* `D` = distance from each fluid cell to the nearest solid surface.
* Watershed of `−D` from the local maxima of `D`. Region boundaries land on the
  narrowest cross-sections. This is the SNOW pore-network extraction used for
  porous media.
* Neighbouring regions are merged unless their shared boundary is a real
  constriction: they merge if `r_throat ≥ β·min(r_peak)` (default β = 0.6), or if
  the opening is at least `d_free` wide (default 30 mm, treated as
  instantaneous).
* Everything touching the domain boundary is merged into compartment 0, the
  *exterior*.
* Each throat stores its face list, its area (vector sum of the staircase face
  normals, which is exact for a planar cut), and its inscribed diameter
  (accurate to about ±dx/2).

`split=False` skips this step, so every opening is instantaneous. That is the
equilibrium model.

### Fill-spill-merge kernel (`fsm.py`)
This is a 3D, arbitrary-gravity version of the depression-hierarchy
algorithm used in hydrology (Barnes, Callaghan & Wickert, *Earth Surf. Dynam.*
2020–21):

1. Sweep the cells in increasing `h` and merge them with union-find. A cell
   with no lower neighbour starts a *leaf* depression. A cell that joins two
   or more components is a *spill point*: the components become children of a
   new node, or of the *ocean* (sinks) if one of them already drains.
2. Every source parcel follows steepest descent to its leaf.
3. A leaf fills its own cells. Any excess spills over its saddle into whichever
   sibling the downhill path from the saddle leads to. When all siblings are
   full, the parent's own cells fill, and so on up the tree. Whatever reaches
   the ocean is counted as drained. Volume is conserved exactly.

The same kernel handles air by using `−h`. `spill_routing=False` sends every
overflow straight to the sink, which reproduces the original behaviour.

**Cells of different size** (v6). Nodes differ in size (sub-cells, octree
leaves), so the kernel works with each node's vertical extent
`[h − e, h + e]` (`ecell`):
- Cells are swept by their **floor** `h − e`. Water enters a cell at its
  floor, so a pool spills at the floor of its saddle node, and a large cell
  whose floor lies below a spill level belongs to that pool.
- A cell that straddles spill levels is cut into **portions**. How its
  volume is spread over its extent is `eshape`: `"box"` (default) is the
  exact volume of an axis-aligned cube below a tilted plane (`cell_cdf`: the
  sum of three uniform segments along the up-vector), `"linear"` spreads it
  evenly. With `"box"` a cube's portions are the sums of its eight
  children's, so a coarse octree cell holds what its children would.
- A cell that straddles more nested spill levels than it has portion slots
  (4) gives the part up to the highest of them to that depression (the
  ancestor of the others), the rest to its parent. Before v6 the rest went
  to the parent of the lowest, and a pool on a gently tilted, rough floor
  (many nested micro-depressions) lost up to 3 % of its capacity.
- A partly filled depression fills to a **flat level**: its portions below
  one level H hold the volume (bisection over the cells near the surface),
  instead of whole layers of cells in sweep order. The pool surface is
  flat, and its level, taken back from any surface cell, is H.
For 120 k cells a call takes about 10 ms (numba). Since v5 the sweep runs in
parallel; see *Parallel runs* below.

### One time step (`model.py`)
1. **Pose:** get the new `u` and `z_b`.
2. **Throats**, evaluated on the start-of-step state. For each throat:
   * *Levels* on both sides: the free surface of the liquid body touching the
     throat, or `z_b` if the body is the bath.
   * *Head*, measured from the higher side: `Δh = H_s − max(H_d, z_c)`, where
     `z_c` is the centroid height of the wetted part of the throat.
   * *Capillary hold-up:* if the outflow is free (a meniscus at the exit),
     `Δh −= 2(n−1)σ/(ρ g d)`. That is 4σ/ρgd for a hole and 2σ/ρgd for a slot.
   * *Flow:* `Q = C_d · A_wet · √(2 g Δh)`.
   * *Venting:* the source compartment needs air to get in and the receiving
     compartment needs air to get out. Air can pass through another opening
     (air on the far side, or on the compartment's own side where it can bubble
     out), or through the same opening if it has air on both sides somewhere.
     Otherwise the flow is counter-current: it is multiplied by 0.3, and set to
     zero if `d < d_crit` (the Rayleigh–Taylor limit, ≈ 3.7 l_c for a hole and
     π l_c for a slot; l_c is the capillary length).
   * *Limiters:* the source cannot be drained below the throat or by more than
     its volume, the two levels cannot overshoot each other (half the
     equalisation volume per step), and a closed receiver cannot exceed its air
     volume.
   * Outflow is taken from the top of the source pool. Inflow is released at the
     receiving side of the throat and routed by the kernel.
3. **Equilibrate:**
   * Exterior: `B` = bath (below `z_b` and connected to the domain boundary
     below it), `A` = atmosphere (the same above).
     * Liquid FSM with sinks `B` + boundary: all non-bath liquid is routed
       cavity to cavity.
     * Air FSM with sinks `A` + boundary + throat faces that have air on the far
       side: air under the bath rises until it escapes or is trapped.
   * Internal compartments: liquid FSM with no sinks, so their volume only
     changes through throats.
4. **Time step:** the bath surface moves at most `cells_per_step`·dx relative to
   the object (as in the article), and the step is capped at `dt_max`. After the
   motion ends, stepping continues at `dt_max` to capture the drainage tail.

### Parallel runs (v5)
Every pass of a time step runs on all of numba's threads
(`NUMBA_NUM_THREADS`, default: all cores). The results do not depend on the
number of threads: the state after any step is bit-for-bit the same with 1 or
16 threads (checked by the tests and by `examples/thread_scaling.py`).
Compared with v4.3 the results differ only by rounding (door drainage at 8 mm:
liquid fractions within 1e-10).

**Fill-spill-merge in parallel** (`fsm.py`, `method="split"`, default). The
serial sweep builds the depression hierarchy by a union-find in height order.
The same hierarchy, node numbers included, is built from:
1. *Height order:* an integer sort of (height in nm, index) keys. The keys are
   split into value ranges in parallel, and each range is sorted by numpy in a
   thread pool (`par.sort_keys`).
2. *Steepest-descent basins* (parallel): every node points to its lowest lower
   neighbour. A node with none starts a leaf; the others belong to the basin of
   the node they point to. Components of the sweep are always unions of
   basins.
3. *Candidate saddles* (parallel): nodes with lower neighbours in two or more
   basins; only there can components join. With 4 or more threads, the
   candidates that cannot join anything new are dropped first: a candidate
   whose basins are already linked through lower neighbours that are
   candidates themselves (about 95 % of them on the car).
4. *Joins* (serial, but only over the remaining candidates, about 1 % of the
   nodes): union-find over basins in height order, which creates the
   meta-depressions exactly as the sweep does.
5. *Owner of every node* (parallel): the top of its basin's branch at its own
   height.
6. Portions, capacities and retained volumes per node are parallel; the fill
   cascade itself is serial but only touches depression nodes.

Sinks surrounded by sinks (the bath interior for the liquid, the atmosphere
for the air) are left out of the sweep (`prune=True`); nothing flows through a
sink, so the retained volumes are the same. `method="serial"` keeps the
original single sweep for reference.

**Hanging still.** While the pose does not change (drip-off after the dip,
the door hanging before and after the plugs are pulled), the bath, the
atmosphere and the depression hierarchies do not change either. The model
keeps them (`Simulation.reuse_static`, default on) and only routes the new
volumes; a hierarchy is kept once a sweep's inputs were the same in two steps
in a row (the air sweep's sinks at throats change with the liquid there).

**Other passes.** The labelling of the bath, the atmosphere and the liquid
bodies (`par.label_bodies`: chunks of node indices are joined in parallel, the
few links between chunks in sequence), and the per-node array work of a step
(`stepk.py`: sweep inputs, height bands, the new liquid fraction, the sums of
the history, liquid bodies and the pool levels at throats) are parallel
kernels. The film (`film.py`) builds its edge list once and sweeps its
elements renumbered from the highest down, so that the sweep walks memory in
order (2× faster); its coefficients are computed in parallel. The film sweep
itself stays serial (each element needs the inflow from above).

**Threads and Python threads.** With one numba thread (`NUMBA_NUM_THREADS=1`)
the kernels are compiled as plain loops that release the GIL, and
`Simulation(threads=2)` overlaps the liquid and air sweeps in two Python
threads as before. With more numba threads, the passes run one after the
other, each on all threads. Only one parallel kernel runs at a time; a call
from another Python thread meanwhile (e.g. the movie's writer thread) runs the
plain loop.

**Do not oversubscribe.** The parallel kernels expect their threads to have
a core each. Running a second parallel job (another simulation, or render
workers) at the same time slows both down a lot; give each job its own share
(`NUMBA_NUM_THREADS`) of the cores, or run them one after the other.

**Scaling benchmark.** `examples/thread_scaling.py` builds the car as the
movie does (film on), and times model steps during the motion and during the
hang with 1, 2, 4, 8, 16 threads, from the same state; it also checks that
the state is identical for every thread count.

Timings in the sandbox (car as in the movie: 20 mm, 2 sub-cells, 2.8 M
nodes, film on). The sandbox has only two slow cores, so the gain from more
threads cannot be measured there; these are one-thread numbers (v4.3 with its
two Python threads):

| per step | v4.3 | v5, 1 thread |
|---|---|---|
| dip (dt 0.1 s) | 1.31 s | 0.57 s |
| hanging (dt 0.5 s, 10 film sub-steps) | 2.43 s | 0.79 s |

Door drainage (8 mm, 4 sub-cells, 500 steps): v4.3 159 s, v5 93 s with one
thread, 73 s with two.

### Explicit holes (`Simulation(..., holes=[...])`)
Known openings, such as drain holes from CAD, can be listed as
`(center, diameter)` or `dict(center=, diameter=, axis=)`. Each one is carved
open in the voxel grid (so it exists even on coarse grids). It is cut along
its mid-plane and always treated as a throat with its **true** area and
diameter, so the orifice flow and the capillary hold-up do not depend on the
resolution. If `axis` is not given, it is estimated from the voxels around
the rim; fitting it to the mesh rim is better. Automatically detected
throats that overlap a listed hole are dropped. A warning is printed if the
cut does not close the opening locally.

The wetted part of an explicit hole also comes from the true circle (a slot
in 2D), not from the voxel faces. For a pool level H, the wetted area is the
circular segment below H. With free outflow the orifice law is integrated
over that segment (v6.3, see "Accuracy fixes" below); a drowned hole uses
the level difference. The sill is the circle's lowest point. The pool on each side is looked up
among the fluid cells within d/2 + 1.5 dx of the hole centre that connect
to that side, so the level can fall below the voxel faces. Before this, the
face rows sat up to a cell above the true sill, and the door's 45° pool
stopped about 1 dx too high. With binary cells this bias hid part of the
wall-volume error. With cut-cell volumes it made the result 20–30 % too high
at 5–6 mm.

### Compressible air pockets (`Simulation(..., compressible_air=True)`)
Trapped air in the exterior (pockets under the bath) can be compressed
isothermally, following Boyle's law. The fill-spill for air then routes the
**gas amount** G (m³ at ambient pressure) instead of a volume:

- A cell of volume v at pressure p holds v·p/p_atm of gas.
- A pocket's pressure is uniform and equal to the hydrostatic pressure at
  its water surface, p = p_atm + ρg·depth.
- The pocket's volume sets that water level, and the level sets the
  pressure, so each step makes a few fixed-point passes (usually 2–3).

Above the bath level, in the part of the exterior connected to the bath
without passing through the atmosphere (``suction``, v6.3), the pressure is
below ambient and a pocket expands.

Gas is conserved by the routing itself. Pockets that merge add their gas, a
pocket that splits divides it by where the gas is, and gas that reaches the
atmosphere is vented. No pocket bookkeeping across steps is needed. Air
inside closed compartments is still passive.

The effect is small for shallow dips: 2.4 % less volume at 0.25 m. It
grows with depth: about 9 % at 1 m, 16 % at 2 m, 23 % at 3 m and 49 % at
10 m. The default is off, so the validated door results stay unchanged.
`Fluid.p_atm` sets the ambient pressure.

Check: an inverted cup (r = 10 cm, 30 cm high) lowered to 0.5–10 m. The
model follows the exact Boyle solution within 0.1 %, relative to its own
trapped gas amount, and the gas is conserved to machine precision. The
previous model keeps the pocket at its surface volume
(`tests: test_diving_bell_follows_boyle`).

`examples/diving_bell_movie.py` is a presentation movie. Two glass cups,
one with each model, go down to 10 m and back up, with the pocket volume
plotted against depth next to the exact curve. The cups are generated in
the script, so no customer geometry is involved.

### Presentation case: the cascade rack (`examples/cascade_rack.py`)
This case shows spill routing on a part built for it. Three stepped cups
sit on one rigid backplate, and the geometry is generated in the script:

- **A:** the top cup, no hole, about 3.3 l.
- **B:** the middle cup, with a 20 mm hole that drains into C.
- **C:** the bottom tray, with a 40 mm hole that drains to the bath.

The sequence:
1. Dip upright and lift: all three cups are full.
2. Hold at 0° for 30 s: B and C drain, and A stays full.
3. Tilt to 40° in 4 s: A pours over its rim into B, B drains and overflows
   into C, and C drains into the bath.
4. Hold to 70 s.

The previous method (equilibrium, escaped liquid lost) is run with the same
motion. There B and C empty the instant the rack is lifted, and nothing A
loses ever reaches them. `cascade_rack.py` writes the curves, and
`cascade_rack_movie.py` renders both racks side by side: water, hole jets,
and spill streams worked out from the volume balance between frames, with
the curves beside them. At 10 mm, A ends with 1.3 l. During the tilt B
peaks at 1.35 l and C receives about 0.25 l.

### Cut-cell volumes (`volfrac.py`, `Simulation(..., subcells=4)`)
With binary cells every sheet costs about one cell of fluid volume on each
side, and the error only falls linearly with dx. In the door's narrow
V channel this is large: the 45° corner pocket below the hole rim is 31 %
too small at 6 mm and 14 % at 2 mm. Each fluid cell therefore gets an
effective volume:

    v_eff(c) = dx³ + fluid parts of the neighbouring cut cells on c's side of the wall

1. Every cut cell (a solid cell with a fluid face neighbour) is split into
   k³ sub-cells, and the surface triangles are sampled again at the fine
   spacing. Sub-cells the surface passes through are solid.
2. A breadth-first search starts from the sub-cells that touch a coarse fluid
   cell and spreads through fluid sub-cells only. Each fluid sub-cell goes
   to the nearest coarse fluid cell it can reach without crossing the
   surface, so a cell that straddles a sheet gives its two halves to the
   two sides.
3. The surface sub-cells themselves were split equally between the fluid
   sub-cells next to them (half to each side of a sheet); since v6.3 by
   line of sight, without the metal (see "Accuracy fixes" below).
4. Pockets sealed at the fine scale that no fluid cell can reach are
   dropped. The run header reports how much.

Connectivity does not change. Cut cells stay closed, so openings, throats
and the pool topology are the same as before. Only the volume–level
relation of every pool, air pocket and compartment changes. The fill kernel
takes per-cell volumes, and a cell's liquid fraction `L` refers to its own
`v_eff`. `subcells=0` (or 1) gives the binary model. This needs the surface
triangles, so it works for STL geometries in 3D only; synthetic voxel
shapes keep dx³.

Accuracy on off-grid thin boxes, with k = 4 and 10–20 cells across: the
interior volume is within 0.2–2.5 % (binary cells: 9–33 % low), for both
zero-thickness and 3 mm walls. Cost on the door at 6 mm: 7 s. It scales
with the number of cut cells × k³ and runs once, at setup.

This is the v3 model; it is still available with
`Simulation(..., subcell_connect=False)`. Since v4 the cut cells are
refined instead (next section), which also fixes connectivity and the
spill level.

### Sub-cell graph and spill level (v4, default)
In the voxel model a cut cell is closed, so any opening narrower than about
two cells is shut, and every sill (the lower edge of a slot, the free edge
of a sheet) is moved up to the next open cell. On the car door this decides
the result: the 22.5° filling is set by a 25 × 14 mm clip slot in the inner
panel, which some grids opened and others closed (5.5 → 6.5 → 5.2 l at
6 → 5 → 4 mm in v3).

**Two-level graph** (`volfrac.cut_cell_graph`). Every fluid sub-cell of a
cut cell becomes a node of its own. Coarse fluid cells keep their index,
and the fine nodes get indices `ncells, ncells+1, …`. The graph is a CSR
pair `sim.nbr = (indptr, indices)`.
- A fine node links to its fluid face neighbours: fine nodes in the same
  or a neighbouring cut cell, and coarse fluid cells across a coarse face.
- The surface is re-sampled at dx/k and sub-cells it touches are solid, so
  sheets stay leak-free at the fine level. A sample that falls on a coarse
  face is given to the same cell as in the coarse voxelisation.
- Fine voids sealed from every coarse fluid cell (between two sheets) are
  inactive.
- Volumes are the sub-cell volume plus a share of the surface sub-cells, as
  in v3. Heights and levels use each node's own extent (`sim.en`: e/k for
  fine nodes).

**Spill level at the saddle's floor** (`fill_spill(..., ecell=)`). The level
at which a pool spills is the floor of its saddle node (h − e). Any node
whose extent straddles that level, or the levels of the enclosing pools, is
split into portions, so a pool no longer holds whole coarse cells above its
spill level. With `ecell=None` the old whole-cell rule is kept.

**Plugs** (`Simulation(..., plugs=[...])`). A plugged hole (centre,
diameter, axis) has every link through it cut, coarse and fine, so it needs
no throat and works on any grid. The filling cases use it for the plugged
drain holes. Explicit holes (`holes=`) still get throats; fine links around
their rims that cross the sheet plane are cut, so the orifice law still
governs them.

**Throats on the node graph** (v4.1, `compartments.node_throats`). The
segmentation runs on whole cells, so an opening that only the sub-cells
resolve used to be closed between two compartments, and a throat's width
was that of its whole cells (6 mm for a 12 mm slot at dx = 4 mm, below the
glugging limit). Now:
- Fine nodes take their compartment by a watershed flood on the node graph
  (nodes far from the walls first, distance from the surface sampled at
  dx/k), so a compartment boundary through an opening lies in its narrowest
  section.
- Every link between two compartments belongs to a throat: to the
  whole-cell throat it touches, or else to a new throat (an opening that
  only the sub-cells resolve). With `split=False` such compartments are
  merged instead.
- Sub-cell links that cross the plane of an internal throat (`a == b`)
  next to it are added to it, so no liquid bypasses its rate limit.
- Each throat's area is the net (projected) area of all its faces, whole and
  sub-cell, with per-face weights. Its width is the largest inscribed disc
  of the faces across the opening, drawn on a sub-cell raster (about ±1
  sub-cell per edge). The wetted fraction, centroid and sill use the
  sub-cell faces too.

Checks (closed box, 12 mm slot in a side wall, 5 mm slots):
- One 12 mm slot, no vent, dx = 4 mm: width 11.5 mm, above the ~10 mm
  limit, so the box glugs empty to the slot (before: stayed full).
- 5 mm drain slot plus 5 mm lid vent, dx = 8 mm (both open only in the
  sub-cells): drains through two new throats (before: stayed full).
  `split=False` drains it to the slot within 3 %.
- One 5 mm slot on its own is below the limit and holds, like a pipette.

**What an opening needs to pass liquid** (thin-walled slot, dx = 8 mm,
k = 4, six grid offsets): 1 sub-cell across, never open; 1.5 sub-cells,
open at 3 of 6 offsets; 2 sub-cells (dx/2), always open. IBOFlow needs about
3 cells across, so a drainsim grid of dx opens what IBOFlow needs cells of
about dx/6 for.

**Accuracy and cost**
- Open thin-walled cups at 8 offsets, dx 12, 8 and 5 mm: the retained
  volume is within 1.6 % on average and 3.3 % at worst of the true volume
  (v3: 5.8 % and 16 %).
- A 7 mm slot at dx = 8 mm is open, and the cup drains to the slot's edge.
- The fine nodes add about 1–4× the coarse cell count, and a door run takes
  about 2–4× longer than in v3 (e.g. 8 mm filling: 100 s instead of 25 s).

### Octree grid (v6, `Octree`, `Simulation(Octree.from_mesh(...))`)
A uniform grid resolves the walls and the bulk alike, and the bulk holds
most of the cells: the car at 6 mm is 82 M cells, of which about 6 M touch
the surface. An octree keeps the wall cells and makes the bulk coarse.

    from drainsim.octree import Octree
    ot = Octree.from_mesh(mesh, h=0.003, levels=4, bounds=(lo, hi))
    sim = Simulation(ot, motion, subcells=2, film=True, ...)

- **Levels.** Level 0 has the cell size `h`; it holds the closed cells
  (cut by the surface, as on the uniform grid) and at least `margin`
  (default 1) fluid cells around them. Each coarser level doubles the size
  up to `h · 2**levels` (the base cells). The grading is 2:1: neighbours
  differ by at most one level. An octree with `levels=0` is the uniform
  grid, cell for cell.
- **Graph.** The fluid leaves are the nodes, followed by the sub-cells of
  the closed level-0 cells (as on the uniform grid). A link joins face
  neighbours of the same level, or a leaf and the coarser leaf across its
  face. Every algorithm (fill-spill-merge, labelling, compartments, throats)
  needs only the links, a height, a volume and an extent per node, so it
  runs unchanged. Node sizes are `sim.nsize`, extents `sim.en`.
- **Compartments** (`gseg.segment_graph_voxel`). The uniform grid's
  segmentation, step by step on the node graph: distance to the nearest
  closed cell (k-d tree, as the distance transform), markers = maxima over
  the 26-neighbourhood, a priority flood (equal priorities first-in
  first-out, as skimage's watershed), merging across leaf faces, throats
  grouped by their 26-connected side cells with the whole-cell width rule;
  then the fine nodes are flooded and completed into throats by
  `compartments.node_throats`, as on the uniform grid. With no coarse
  levels it reproduces the uniform segmentation up to flooding ties (car at
  25 mm: 231 vs 235 compartments). `segment_kwargs=dict(distance="surface")`
  selects the older variant (distance to the surface triangles, face links
  only).
- **Film.** The carrier is the fluid-facing surface of the closed level-0
  cells, made by marching cubes block by block (the dense level-0 space is
  never allocated). Each element maps to the leaf next to it.
- **Display.** Movies and pocket renders need a uniform grid. Node fields
  are resampled onto a display grid of spacing `h · 2**j` (`octview.DisplayGrid`,
  volume-weighted), with `j` chosen to stay below `--display-max-cells`
  (default 100 M).

Checks (`tests/test_octree.py` and the car):
- `levels=0`: the same closed cells, links, sub-cells and volumes as the
  uniform grid, and the same door results.
- Tilted open cup, flooded: the octree with 2 or 3 levels keeps the uniform
  grid's volume within 1e-6 (coarse cells in the pool and at its surface).
- Door (5° hang, holes closed, 8 mm): 10.895 l on the uniform grid, the same
  with 2 and 3 levels. The four article cases (filling 0° and 22.5°,
  drainage 22.5° and 45°, explicit drain holes) at 12 mm (2 levels) and 6 mm
  (3 levels): the uniform grid's values to 4 decimals; with no coarse
  levels the drainage history is the same to 1e-19. The door is thin, so
  the octree saves little there (6 mm: 2.0 M instead of 3.0 M nodes).
- Open cup draining through an explicit 12 mm hole: identical with no coarse
  levels, within 1 % with 2.
- Closed box with 5 mm drain slot and lid vent at 8 mm (open only in the
  sub-cells): the same compartments and throats (area, width) as the
  uniform grid with 0 and 2 levels, and the same drainage within 2 %.
- Car body at 20 mm, short dip, 5 s hang (2 cores):

  | | nodes | setup | run | peak | air 15 s / 20 s | liquid 35 s / 40 s |
  |---|---|---|---|---|---|---|
  | uniform | 2.77 M | 98 s | 5.5 min | 3.1 GB | 170.3 / 134.0 l | 108.1 / 99.0 l |
  | octree, 0 levels | 2.60 M | 113 s | | 2.4 GB | 173.3 / 138.3 l | 102.4 / 92.6 l |
  | octree, 2 levels | 0.93 M | 101 s | 1.9 min | 2.1 GB | 173.8 / 138.7 l | 100.4 / 90.6 l |

  The coarse levels change the trapped volumes by 0.3–2 %. The rest of the
  difference to the uniform grid (air +2–3 %, liquid −5–6 %) comes from the
  segmentation (326 vs 330 compartments, 459 vs 472 throats, from ties in
  the flooding), to which the transient drainage at 20 mm is sensitive.
- Size of the car octree (k = 2, 3 levels): 12 mm: 0.50 M closed cells,
  2.9 M nodes, 4.8 GB to build; 20 mm (2 levels): 0.16 M closed, 0.93 M
  nodes. The nodes grow about as h^-2.2 (the wall cells and their
  sub-cells dominate): about 13 M at 6 mm, 32 M at 4 mm, 60 M at 3 mm.

Command line: `--levels N` in `car_article.py`, `car_movie.py`
(`--record`/`--render` only), `thread_scaling.py`, `door_article.py` and
`door_drain.py`; `dx` is then the finest cell, at the walls.

### Narrow passages (v6.1–6.2, `Simulation(..., narrow=dict(k=8))`, `--narrow K`)
An opening is always open once it is two sub-cells wide. Seams between
panels, lap joints, slots and small holes narrower than that stay shut, or
open only at some grid offsets. The narrow refinement gives only the
closed cells at such passages more sub-cells:

- **Finding them** (`octree.narrow_cells`, once at setup). The surface is
  sampled at the base sub-cell size with the triangle normals. A cell is
  narrow if it holds a wall with another wall facing it across fluid
  closer than `width` base sub-cells (default 2: what the base resolution
  opens at most at some offsets) but wider than 1.5 fine sub-cells (what
  the refinement can open): two nearly parallel faces offset along the
  normal, or two free mesh edges facing each other (slots, holes, gaps
  between the edges of two sheets). Meshes exported as thin plates (the
  car: every panel has two faces 0.5–2 mm apart) are recognised by their
  back-to-back face pairs (`_plate_sign`); only faces that face each other
  across fluid then count, and the free-edge test is skipped.
- **Refining them.** A narrow closed cell gets `k` sub-cells per edge
  (default 2 × `subcells`), the others keep `subcells`. The surface is
  resampled finely only near them. Sub-cells of neighbouring closed cells
  with different resolution link like hanging faces (each finer sub-cell
  to the coarser one it faces). Volumes, extents, fill-spill, compartments
  and throat areas and widths use each node's own size.
- **Only passages that connect** (v6.2, `connecting=True`, default;
  `octree._connecting_cells`). The candidates are refined first, then the
  fine fluid sub-cells in them are grouped into passages (connected
  components). A passage's *mouths* are the nodes outside the candidates
  it touches. It is kept if its mouths belong to two or more regions of
  the outside graph restricted to the nodes within `margin` cells
  (default 1) of the mouths; otherwise it is a dead end (a seam or crevice
  with one mouth, whose contacts are joined by the fluid in front of it)
  and its cells go back to the base resolution. A passage with two mouths
  is kept even when a wider opening elsewhere joins the two spaces, since
  that opening may lie higher.
- Off (`narrow=None`, default): the graph is bit-for-bit the v6 one.

Checks (`tests/test_octree.py`):
- Open cup at h = 8 mm, 2 sub-cells (4 mm), with a 3 mm lap-joint channel
  in its side, or a 3 mm slot: shut without refinement (water to the rim);
  with `k = 8` (1 mm) only the cells at the channel/slot are refined
  (10–41 of about 600 closed cells) and the cup drains to the channel's /
  slot's lower edge.
- A cup without narrow passages: nothing refined, the same result.
- The lap joint closed at the bottom of its channel (a dead-end crevice):
  41 candidate cells, none refined, the same result as without
  refinement; with `connecting=False` all 41 are refined.

Door, article cases, octree with 2 levels, 4 → 8 sub-cells at narrow cells
(litres; nodes):

| | nodes | fill 0° | fill 22.5° | drain 22.5° | drain 45° |
|---|---|---|---|---|---|
| 12 mm, no refinement | 0.37–0.52 M | 10.83 | 5.22 | 0.041 | 0.257 |
| 12 mm, all candidates | 0.57–0.80 M | 10.85 | 5.23 | 0.045 | 0.230 |
| 12 mm, connecting only | 0.47–0.72 M | 10.84 | 5.23 | 0.045 | 0.231 |
| 8 mm, no refinement | 1.05–1.52 M (uniform) | 10.90 | 5.27 | 0.036 | 0.222 |
| 8 mm, all candidates | 1.25–1.69 M | 10.89 | 5.22 | 0.036 | 0.213 |
| 8 mm, connecting only | 1.04–1.41 M | 10.90 | 5.28 | 0.037 | 0.222 |
| 5 mm, no refinement | 3.1–4.6 M | 10.93 | 5.24 | 0.033 | 0.228 |
| IBOFlow 0.35 mm / experiment | | 10.81 / 10.75 | 5.15 / 5.16 | 0.039 / 0.056 | 0.226 / 0.246 |

At 12 mm the connecting passages give the whole effect (the 45° drainage
close to the finer grids). At 8 mm the connecting passages change nothing;
the lower 22.5° filling with all candidates refined (5.22 l) comes from
cells the filter classes as dead ends (probably the volume of seams along
the pool), not from a new connection.

Car body, 0.6 m slice through the middle, octree with 3 levels, 2 → 4
sub-cells:

| | candidates (share of closed cells) | refined, connecting only | nodes: none / all / connecting | build |
|---|---|---|---|---|
| 12 mm | 16 039 (19 %) | 6 052 | 0.47 / 1.01 / 0.68 M | 34 / 55 / 57 s |
| 8 mm | 31 284 (16 %) | 10 423 | 1.11 / 2.10 / 1.45 M | 75 / – / 103 s |
| 6 mm | 47 686 (13 %) | 12 166 | 2.07 / 3.53 / 2.47 M | 130 / – / 173 s |

So the filter keeps about a third to a quarter of the candidates, and the
node increase at 6 mm drops from +71 % to +19 %. At 20 mm most of the car
counts as narrow (the refinement is meant for fine grids).

### Accuracy fixes from the compare_agents review (v6.3)
Two coding agents wrote independent re-implementations of the article's
method (`compare_agents.zip`). Agent A's code (a uniform grid, exact
floods, hole disks, compressible air, film on the triangles) gave four
things drainsim now does as well. Each is on by default and can be turned
off for comparison.

- **Exact cut cells** (`voxel.py`; `Octree.from_mesh(..., cut="sample")`,
  `Grid.from_mesh(..., cut="sample")` for the old rule). Cut cells were
  found by sampling points on the triangles at 0.45 dx. That misses cells a
  sheet only grazes: such a cell stayed a whole fluid cell, its volume
  beyond the sheet counted as fluid, and next to the sub-cells of a cut cell
  it could join the fluid on both sides of the sheet. At 8 sub-cells per
  edge the article door drained through its inner panel (6.59 l instead of
  10.8 l at 12 mm). Now a cell is cut if the surface enters its interior
  (separating-axis test, as in agent A's voxeliser), plus the sampled cells
  (so a sheet lying exactly on a cell face stays one cell thick).
- **Corner cells get sub-cells** (`octree.REFINE_EDGE_NEIGHBOURS`). Where
  a sheet runs obliquely the closed cells form a wall two cells thick, and a
  closed cell that touches the fluid only along an edge or a corner holds
  fluid too. Only closed cells with a fluid *face* neighbour were refined;
  a tilted box 12 cells across lost 1–6 % of its volume in its corners.
  Now closed cells with a fluid leaf among their 26 neighbours are refined
  (car slice at 12 mm: +7 % nodes, +20 % graph time).
- **Surface sub-cells shared by line of sight** (`sidesplit.py`;
  `Simulation(..., surface_share="equal")` for the old rule). A closed
  sub-cell at the surface was shared equally among its fluid neighbours,
  half to each side of a sheet, so the metal of every plate counted as
  fluid (a 0.7 mm panel adds 0.35 mm over each wetted side). Now each
  closed sub-cell is sampled with 2 x 2 x 2 points; a point goes to the
  fluid neighbour it can see (the segment to it crosses no triangle), and a
  point with an odd number of crossings to every neighbour lies in the metal
  and is dropped (agent A's cut-cell sampling). Closed sub-cells without a
  fluid neighbour use the neighbours of their closed cell. A turned box with
  0.7 mm walls, 12 cells across, now holds its volume within 0.2 % (equal
  share: +0.1 to +1.2 %, and -1.6 to -6 % without the corner cells).
- **Holes pass the weir flow** (`ThroatModel(hole_profile=False)` for the
  old rule). An explicit hole with free outflow passes
  Q = Cd ∫ sqrt(2 g (H - h_c - z)) dA over its wetted part (h_c the
  capillary hold-up), from agent A's `OrificeDisk`. The centroid head used
  before over-predicts a partly covered hole (a half covered vertical hole
  by 7 %; the 2D slot now gives the rectangular weir (2/3) sqrt(2g) h^1.5).
- **Suction** (`Simulation(..., suction=False)` for the old rule). The
  bath is the exterior below the bath level connected to the domain
  boundary *plus* everything connected to it without passing through the
  atmosphere (agent A's submerged system). An inverted glass lifted partly
  out keeps its water; the liquid above the bath level is at
  p_atm + ρ g (z_b - z) < p_atm, and a compressible pocket there expands
  (exact Boyle). Before, that liquid drained as if air could get in. This
  changes the timing of dip-out (a car cavity whose openings are still
  under the bath stays full until they emerge), not the final volumes.

Door (octree with 2 coarse levels, which gives the uniform grid's values),
trapped water in litres:

| | fill 0° | fill 22.5° | drain 22.5° | drain 45° |
|---|---|---|---|---|
| 12 mm, v6.2 | 10.83 | 5.22 | 0.041 | 0.257 |
| 12 mm, v6.3 | 10.80 | 5.23 | 0.035 | 0.239 |
| 10 mm, v6.2 | 10.90 | 5.28 | 0.030 | 0.211 |
| 10 mm, v6.3 | 10.85 | 5.26 | 0.044 | 0.222 |
| 8 mm, v6.2 | 10.90 | 5.27 | 0.036 | 0.222 |
| 8 mm, v6.3 | 10.83 | 5.24 | 0.046 | 0.230 |
| agent A, 0.69 mm (uniform, exact floods) | 10.81 | 5.23 | 0.041 | 0.234 |
| IBOFlow, 0.35 mm | 10.81 | 5.15 | 0.039 | 0.226 |
| experiment | 10.75 ± 0.14 | 5.16 ± 0.13 | 0.056 ± 0.005 | 0.246 ± 0.016 |

The 0° pool's volume below drainsim's level (0.1690–0.1692 m above the
door's lowest point), measured with agent A's code and extrapolated in its
cell size, is 10.80–10.82 l; v6.3 gives 10.80 (12 mm) and 10.83 (8 mm).
The +1 % of v6.0–6.2 came from the metal share and the grazed cells.

Tests: `tests/test_voxel.py` (exact cells vs brute force, grazing cells, a
sheet on a cell face, a tilted plate that leaked with the sampled cells,
hollow boxes, a 2 mm plate's metal), `tests/test_fsm_scenes.py` (agent A's
voxel scenes with exact cell counts: bottle on its side, cascades,
overflow landing next to the saddle, mirrored scenes), `tests/test_suction.py`
(inverted glass, turned and lifted glass, under-pressured pocket vs Boyle),
and the hole flux against a sampled disc, the weir law and a side-hole
drainage ODE in `tests/test_drainsim.py`.

### Fixes from the code review (v6.4)
A code review of all of 6.3 (package, tests, tools, examples) found eight
defects and a number of smaller issues; all are fixed, one commit each.
The car at 20 mm (uniform and octree, with and without the film) gives the
same numbers as 6.3 to the last digit.

- **Holes near the domain boundary** (`Octree.carve_holes`,
  `compartments.carve_holes`) were skipped without a warning when the
  centre was closer than a full diameter to the boundary; the margin is now
  the radius.
- **Throat axis for openings whose net area cancels** (`gseg._throat`, a
  jagged opening on the octree) was the axis of the single largest face; it
  is now the axis with the largest total area, as on the uniform grid.
- **Film left by an emptying pool** (`film.py`) was deposited at a speed
  based on the height of the finest cell; it now uses the height of the node
  the element maps to (sub-cells and octree leaves differ).
- **`export_world` on an octree run** failed with an `AttributeError`; it
  now says that it needs a uniform grid (see `octview.DisplayGrid`).
- **`car_movie.py --render --workers`** did not pass `--look`, `--orbit`,
  `--sim-dt` and `--dt-hang` to its workers, so a parallel movie had no
  camera orbit. It also waited for ever for the steps of a recording that
  had died; it now gives up after `--wait-timeout` s (default 3600) without
  a new step.
- **`door_drain_movie_hd.py`** showed only the liquid held above the surface
  before the plugs were pulled, not all the water in the door.
- **`door_figures.py`** looked for `results.json` one folder below where
  `door_article.py` writes it and skipped Fig. 7.
- Smaller: `touches_boundary` is set for `split=False` on the octree; the
  drop count is a running sum; `render3d` removes its frame folder when
  ffmpeg fails; `shapes.cylinder` handles a zero-length axis and thin walls;
  the peak memory reported off Windows is the true peak; duplicated code
  (region merging, raster width, sheet normal, sub-cell steps, triangle
  sampling) is shared; dead code is removed.

Tests: `tests/test_small_fixes.py`, `tests/test_examples_fixes.py` and new
tests in `tests/test_octree.py` and `tests/test_drainsim.py` (99 in all).

### Shallow pockets and ties
* **`min_depth_cells`** (default 1.5). Depressions, and air domes, shallower
  than this many cell heights retain nothing: they merge into the next
  deeper pool, or drain. A voxelised curved surface is rough at the cell
  scale, and at a tilt it forms many tiny pockets that a smooth surface does
  not have. On the door they held ~10 % of the retained water as scattered
  traces. Real dents shallower than ~1.5 cells are lost too; at 2 mm that is
  about 3–4 mm, which is at the capillary length, where the geometric model
  cannot decide anyway.
* **Deterministic heights.** Cell heights are rounded to 1 nm, and the
  up-vector to 1e-12. At exact angles such as 45°, whole layers of cells
  have equal heights. Without rounding, platform-dependent last-bit
  differences decided which way these ties broke: at 5 mm this changed the
  45° door result by 4.5 % between Windows and Linux.

### Holes inside a compartment
A cavity can be connected to the outside by a wide opening and also drain
through small holes. The article's door is an example: the door interior is
open to the cabin side through the large access aperture, and it drains
through three ~18 mm holes in the bottom flange of the inner panel. Merging
by the wide path puts both sides in one compartment. The segmentation
therefore also keeps narrow openings between pre-merge watershed regions as
*internal throats* (`a == b`). An opening qualifies if it is narrower than
`beta` × the adjacent pore radii and narrower than `d_free`, and if the two
sides are connected only through it within a local box. The simulation
removes throat faces from the instantaneous connectivity, i.e. from
fill-spill, the bath and atmosphere labelling and the pool labelling. Flow
through them then follows the orifice law. Whether a hole is detected
depends on resolution: a hole that is as wide as the channel it sits in is
not treated as a constriction, and then it drains instantly.

### Surface film and dripping (`film.py`, `Simulation(..., film=True)`)
The film lives on a *carrier* surface: the fluid-facing boundary of the solid
voxels (marching squares in 2D, marching cubes in 3D, Taubin-smoothed). Each
side of a sheet-metal wall is its own surface, so the inside and outside of a
cavity carry separate films. The state is the film thickness `h` on each
element, and its volume is `h·A`.

* **Submerged elements.** If the fluid cell next to an element is at least
  half full, the element's film belongs to that pool or the bath.
* **Deposition.** When the liquid level leaves an element, it deposits
  `h₀ = l_c·min(0.94 Ca^(2/3), Ca^(1/2))` (Landau–Levich–Derjaguin), with
  `Ca = μU/σ`. `U` is the withdrawal speed along the surface. For the bath it
  comes exactly from the motion. For pools inside the object it comes from
  how fast the next cell emptied. Film left by the bath is supplied by the
  bath. Film left by an internal pool is taken out of that compartment's
  liquid, so total liquid is conserved.
* **Drainage.** Gravity drives a thin film: `q = ρ g_t h³/(3μ)`, where
  `g_t = g − (g·n)n` is the component along the surface. It is solved with
  upwind finite volumes over the element edges and backward Euler, swept from
  the highest element down. This is stable for any time step and exactly
  conservative. Film that runs onto a submerged element joins that pool.
* **Drops.** Film collects at low points on downward-facing surfaces. A
  connected patch of hanging film releases a drop once it holds
  `drop_volume`: 3.8 l_c³ ≈ 0.075 ml of water in 3D, 2 l_c² per metre of
  depth in 2D (rough). The drop is handed to the voxel model in the air
  cell below, and fill-spill-merge routes it into a cavity or back to the
  bath. Every drop is logged (time, position, volume).
* **Puddles.** Film thicker than `h_bulk` on an upward-facing surface is
  handed to the voxel model, which pools or routes it.

Verified against Jeffreys' vertical-plate solution in 2D and 3D, and against
the deposition law for withdrawal speeds of 5 mm/s to 0.5 m/s (example 06,
`tests/`).

### Outputs (`History`)
Time series of:

- `liquid_above_bath`: the carried liquid.
- `liquid_retained`: all non-bath liquid.
- `liquid_comp`: liquid per compartment.
- `liquid_exterior`: exterior pools.
- `air_below_bath`: accessibility. Air not yet displaced, including
  compartments that are still filling.
- `liquid_trapped`: trapped liquid, i.e. liquid (not bath) **above** the
  bath surface of the current pose.
- `air_trapped`: trapped air, i.e. air (not atmosphere) **below** the
  surface, in every compartment. (Before v4.3: exterior pockets only; the
  old quantity is `air_gas` without compressible air.)

`sim.pockets()`, `car_article.py` and all movies use the same definitions
(`Simulation.trapped_fields`, `worldviz.trapped_display`): a node that
straddles the surface counts with the part of its extent on that side. They
only decide what is reported and drawn. The state is not changed: a
compartment crossing the surface keeps its liquid or air, and is drawn as
liquid above the surface and as air below it.
- `drained`: cumulative volume returned to the bath.
- `throat_flow`: flow through each throat, a→b.
- `film_volume`: film on the walls.
- `drip_volume` and `n_drips`: cumulative volume dripped and number of drops.
  `sim.film.s.drips` lists every drop.
- `film_pending`: film volume passed to the voxel model at the next step.

Also `snapshots` of `L`, `sim.pockets()` (volume and centroid of every
cavity/pocket), and `viz.write_vtk` for ParaView.

2D volumes are per unit depth: m² = 1000 l/m.

---

## Verification

| Test | Result |
|---|---|
| 2D box draining through 10/15/30 mm openings vs analytic orifice solution | max deviation 3–4.5 mm over a 0.38 m drop (Ex. 01) |
| 3D box, 30 mm hole, vs analytic | within 0.2 mm during the drop; residual of about one cell layer at the end |
| Upright cup: FSM fill level | exact interior volume |
| Two cups tilted 35°: spill routing | all overflow caught by the lower cup (legacy loses 14.4 of 38.4 l/m) |
| Inverted cup dip-in, upright cup dip-out | trapped air and liquid within 5 % of cavity volume |
| Closed box rotated 170° | liquid volume conserved to machine precision |
| Unvented box with a 4 mm slot | does not drain (Rayleigh–Taylor stable) |
| 2 mm slot | drainage stops with a standing liquid column (capillary hold-up) |
| Film on a vertical wall, 2D and 3D, vs Jeffreys | profiles within 1–5 % at 10–60 s (Ex. 06) |
| Film deposited on withdrawal, 5 mm/s – 0.5 m/s, vs Landau–Levich–Derjaguin | within 3 % |
| Closed box with pool and film, rotated 90° | pool + film + pending conserved to 1e-9 |
| Diving bell lowered to 5 m, compressible air vs Boyle | within 1 % (0.1 % over 0.5–10 m); gas conserved to 1e-9 |
| Partly wetted explicit hole: wetted fraction, centroid, sill vs sampled disc | within 0.5 % / 0.05 mm |
| Partly wetted hole: orifice flow integrated over the wetted disc vs sampled disc (v6.3) | within 0.2 %; 2D slot = weir law within 0.1 % |
| Cup draining through a side hole vs the weir ODE (v6.3) | half-drained time within 10 % |
| Cut cells vs brute-force triangle-box test (v6.3) | identical |
| Turned hollow box, 0.7 mm walls, 12 cells across (v6.3) | interior volume within 0.3 % |
| 2 mm plate across a sealed box (v6.3) | fluid on both sides within 0.2 % of the box minus the plate |
| Agent A's voxel scenes: bottle, cascades, saddle landing, mirrored (v6.3) | exact cell counts |
| Inverted glass half out of the bath; pocket above the bath vs Boyle (v6.3) | stays full; exact |
| Thin closed box off the grid, 7 mm cells: interior volume | k = 4: within 2 % (binary: 21 % low) |
| Open thin-walled cup, full start, drains to its rim | k = 4: within 1–2.5 % of area × quantised level (binary: 15–23 % low) |

---

## Car door of the article (Fig. 7 / Fig. 8)

> **v6 note.** v6 changed the fill-spill capacities (cells swept by their
> floor, cube volume distribution, flat filling, and the portion rule for
> cells that straddle many nested spill levels, which lost up to 3 % of a
> pool on a gently tilted rough floor). The article table below is the v6
> rerun; the other door and car numbers in this section and the next were
> computed with v4–v5.3.

`examples/door_article.py` sets up the drainage experiment:

- **Placement:** the door's lowest point starts 0.15 m above the water.
- **Motion:** it moves 0.4 m down and back up in 8 s, followed by 30 s of
  draining. The door is tilted 22.5° or 45° about its own normal, with the
  front (+x) end lowered so that the water collects in the front bottom
  corner as in Fig. 8.
- **Holes:** the drain holes are open, as in the drainage case. By default the
  three 19 mm drain holes are given explicitly (`DOOR_HOLES`: centre and axis
  fitted to the mesh rim); `--no-holes` relies on automatic detection only.
- **Drain time:** by default (`--t-drain auto`) the door hangs for at least
  30 s, then in 10 s steps until the retained water changes by less than
  0.5 % (or 0.1 ml) in 10 s, at most 300 s.
- **Crop:** only the part of the door that can get wet (bath level + 0.1 m at
  the deepest point) is voxelised.

Water left after drainage, in litres (22.5° / 45°). These are the v3
results (no hanging lean); for v4, see the next section:

| dx | v3, k = 4 (default) | v3, binary (`--subcells 0`) | v2 | v1 |
|---|---|---|---|---|
| 6 mm | 0.0444 / 0.2353 | 0.0232 / 0.1642 | | 0.042 / 0.224 |
| 5 mm | 0.0496 / 0.2314 | 0.0235 / 0.1636 | 0.0326 / 0.2051 | 0.037 / 0.189 |
| 4 mm | | | 0.0313 / 0.2040 | 0.037 / 0.198 |
| 3 mm | | | 0.0303 / 0.2221 | 0.035 / 0.203 |
| article, 7 ref. (0.35 mm) | 0.039 / 0.226 | | | |
| experiment | 0.056 / 0.246 | | | |

What each version has:
- **v1:** automatic hole detection, 30 s drain, no depth filter.
- **v2:** explicit holes, auto drain, depth filter.
- **v3:** cut-cell volumes (k = 4), and hole wetting from the true circle.
  The v3 binary column differs from v2 only by the hole wetting. The fine
  grids are being run by the local agent (`RUN_FINE_GRIDS_v3.md`).

The 45° pool can at most hold the corner pocket below the lowest drain-hole
sill. Its volume, from `examples/door_pocket.py`:

| dx | binary | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|
| 6 mm | 0.140 | 0.201 | 0.204 | 0.204 |
| 5 mm | 0.149 | 0.213 | 0.213 | 0.212 |
| 4 mm | 0.161 | 0.215 | 0.213 | 0.212 |
| 3 mm | 0.175 | 0.219 | 0.217 | 0.216 |

The retained 45° value is this pocket plus the band the hole holds up. The
flow stops when the head over the wetted segment's centroid equals the
capillary hold-up, about 2.5 × 1.5 mm above the sill. At 6 mm that band is
3.6 mm over about 86 cm², or roughly 30 ml.

Notes:

* **Voxel walls cost volume (fixed in v3).** Binary cells lose about one
  cell per sheet face. In the door's narrow V channel that is 31 % of the
  pocket at 6 mm and still 14 % at 2 mm. That was the main reason earlier
  45° values sat below the article's.
* **At 45° the access aperture never goes under.** Its lowest front corner
  is about 0.32 m above the lowest point of the door, and the door is only
  dipped 0.25 m. The door cavity can therefore only fill through the drain
  holes. With the explicit holes, filling is rate-limited through the
  holes at every resolution.
* **Remaining level quantisation.** Pools that spill over a sheet edge
  rather than through a listed hole still fill to the bottom of the first
  cell layer above the edge. This is O(dx) and positive, and it matters most
  at 22.5°, where part of the water is held that way.

### Recreating Table 1 and Figs. 5–8 (`examples/article_comparison.py`)

```bash
python examples/article_comparison.py --stl door.stl --dx 0.012 0.010 0.008 0.006 0.005 --out runs_article
python examples/article_comparison.py --stl door.stl --figures --stills-dx-fill 0.005 \
    --stills-dx-drain 0.005 --out runs_article [--prev runs_article_v3]
```

- **Placement:** the *untilted* door's lowest point is 0.15 m above the
  bath, and the door is then tilted about its centre
  (`place_untilted=True`). With the tilted door's lowest point at 0.15 m,
  the aperture never goes under at 22.5° and the filling case holds no
  water.
- **Hanging lean (`HANG = 5`, `--hang`):** on its two hooks the door leans
  about 5° towards −y (cabin side up), as in the article's simulations.
  This lifts the aperture lip and the clip slots relative to the pool.
- **Plugs:** the filling cases plug the three drain holes with
  `plugs=DOOR_HOLES`, which works on every grid.
- **Post-processing:** as in the article, bodies below 0.01 l are not
  counted and not drawn.
- **Outputs:**
  - `table1_comparison.md/.png`;
  - `fig5_comparison.png` (filling) and `fig7_comparison.png` (drainage),
    both in the article's style;
  - `convergence_comparison.png`, all four cases against cell size, with an
    earlier version in grey via `--prev`;
  - `fig6_comparison.png` and `fig8_comparison.png` (renders). Each run
    renders its still (`fill_0_5mm.png` etc.) as soon as it finishes, and
    `--figures` reuses it; only a missing still makes `--figures` run
    that case again.
- Results from several machines or folders can be combined: `--figures`
  reads every `results*.jsonl` in `--out`.
- `--progress MIN` prints a line every MIN minutes of wall time while a case
  runs (model time, steps, liquid, wall time, ETA, peak memory), plus the
  setup stages of a filling case. `Simulation.run(progress=callable)` calls
  the function after every step.

Trapped water in litres (v6, uniform grid, 2 cores; the octree with 2–3
coarse levels gives the same values to 4 decimals at 12 and 6 mm):

| | filling 0° | filling 22.5° | drainage 22.5° | drainage 45° |
|---|---|---|---|---|
| drainsim 12 mm | 10.83 | 5.22 | 0.041 | 0.257 |
| drainsim 10 mm | 10.90 | 5.28 | 0.030 | 0.211 |
| drainsim 8 mm | 10.90 | 5.27 | 0.036 | 0.222 |
| drainsim 6 mm | 10.88 | 5.26 | 0.038 | 0.245 |
| drainsim 5 mm | 10.93 (3 min) | 5.24 (4 min) | 0.033 | 0.228 |
| v4–v5.3, 8 mm | 10.74 | 5.17 | 0.033 | 0.213 |
| v3.3, 5 mm (cut cells closed, no hang) | 10.56 | 6.46 | 0.048 | 0.225 |
| article ref. 4 (2.75 mm) | 10.32 | 4.60 | 0.022 | 0.201 |
| article ref. 7 (0.35 mm) | 10.81 | 5.15 | 0.039 | 0.226 |
| experiment | 10.75 ± 0.14 | 5.16 ± 0.13 | 0.056 ± 0.005 | 0.246 ± 0.016 |

**What sets the filling volume.** The figures were found by tracing the
minimax spill path from the pool to the domain boundary. The lowest exits
of the door cavity are:
- the lower edge of the aperture in the inner panel (z ≈ 1.589 m at
  x ≈ −0.23 m);
- two clip slots in the panel, each about 25 × 14 mm. The one at
  x ≈ −0.10 m has a bottom tab with ~5 mm notches beside it that reach
  z ≈ 1.589 m. The one at x ≈ +0.225 m is the lowest exit at 22.5°.

In v3 the voxels shut the slots and notches on some grids and not on
others, so the 22.5° filling jumped between 5.0 and 7.2 l. With the
sub-cell graph they are resolved on every grid (5.22–5.28 l from 12 mm
to 5 mm in v6).

The volume below the spill level agrees between grids: 10.90 l at 8 mm
and 10.93 l at 5 mm (v6), for 0° with the 5° lean. Without the lean the
door holds about 10.2 l.

**Memory:** 5 mm filling at 22.5° (4.4 M nodes) peaks at 3.3 GB. 4 mm needs about
6–7 GB, which did not fit in this 8 GB sandbox next to a second run.

### Car body of the article (`examples/car_article.py`)

**Orientation and rotation (v5.1).** `load_car` turns the body upright with
the nose along +X, whatever the file's axes (`--orient auto`, default: the
height axis is the shortest, the roof is the narrower end of it and sits
behind the bonnet; `--orient file` keeps the file's axes). The 360° turn over
the dip is `--rotation pitch` (transverse axis; `--sense 1` nose down first,
`-1` tail first) or `--rotation roll` (long axis; `1` left side first, `-1`
right side first); `--reverse` starts tail first. `examples/car_orientation.py`
draws the four variants side by side (roof red, nose green) to compare with
the article's Fig. 9. car_movie.py, car_article.py and thread_scaling.py take
the same options; recordings store them.

**Presentation look (v5.2).** `car_movie.py --look hd` (default) renders with
three-point lighting, glossy deep-blue liquid, golden air and a clear bath
surface sheet over the pale bath; `--orbit DEG` turns the camera slowly
around the car over the movie; `--look plain` gives the earlier look.
`door_dip_film_movie.py` has the same `--look hd` (default; also the door
blue-tinted under the surface with the waterline, glossy jets and drops, the
film colour bar at the top) and `--orbit DEG`.

The article's last case (Sect. 3.3, Figs. 9–13) is a Volvo car body in a
55 s rotating dip. The body pitches nose-down through half a turn on the
way in, travels upside down through the bath with a slight rocking, and
turns through the second half turn on the way out.

- **Motion:** the keyframes are read off the article's Fig. 9 (see the
  module docstring). World +x is the travel direction, and the car frame is
  the mesh frame (+X forward, +Z up) centred on its bounding box.
- **Units:** meshes in millimetres are scaled to metres automatically.
- **Motions:**
  - `--motion short` (default, 35 s) keeps the dip-in and the dip-out and
    replaces the 25 s transport with 5 s upside down under the surface.
  - `--motion full` is the article's 55 s.
- **Time steps:** fixed at `--dt-max` (0.1 s, i.e. 1.2° of rotation per
  step), unless `--cells-per-step` limits them by the surface speed. This
  gives 350 steps for the short motion, plus the hang.
- **Modes:**
  - `--mode transient` (default) is drainsim with throats. After dip-out it
    hangs for `--hang` s (default 20) so that slow drainage can finish.
  - `--mode equilibrium` (`split=False`) makes every opening
    instantaneous, which is the article's assumption.
- **Reported:** trapped air at 15, 20, 30 and 40 s, and trapped liquid at
  55 s and at the end, both in bodies above 0.01 l. Pocket centres are
  given in the mesh's own coordinates.
- **Renders:** `--render <run folder>` draws the article's Figs. 10–13:
  top and side views with each pocket in its own colour.

```bash
python examples/car_article.py --stl xc90.stl --preview-motion --out runs_car   # Fig. 9 check
python examples/car_article.py --stl xc90.stl --dx 0.04 --subcells 2 --out runs_car
python examples/car_article.py --stl xc90.stl --render runs_car/car_40mm_k2_transient
```

`examples/car_movie.py` renders the same dip as a movie with the film
model on. It follows the car in 3D and shows:
- trapped air under the bath (yellow);
- liquid held by the car (blue);
- the wall film, coloured by thickness, and the drops.

A curve panel sits next to the 3D view. The model advances in fixed 0.1 s
steps; frames in between only move the car. The car is drawn from a
decimated copy of the mesh.

**Fine grids: record, then render in parallel** (v4.2).

```bash
python examples/car_movie.py --stl xc90.stl --dx 0.007 --subcells 4 --record runs_car/rec7
python examples/car_movie.py --render runs_car/rec7 --workers 10 --out runs_car/car_7mm.mp4
```

- `--record` runs only the model. Per step it writes the liquid and
  trapped-air fields exactly as the movie draws them (flat surfaces, sparse,
  8 bit), the film thickness (float16) and the new drops. The fields are
  extracted in a writer thread while the model runs on.
- `--render` splits the frames over `--workers` processes. Each draws its
  part with the same scene and writes a movie part; the parts are joined
  without re-encoding. It can start while the recording runs: workers wait
  for the steps they need.
- Liquid and air surfaces are extracted block by block (64³ cells) where
  the fields are, not over the whole grid (same triangles as the full-grid
  marching cubes), smoothed once per model step and moved with the car.
- `--render DIR --preview 8 17 27` draws stills from a recording.
- Drip-off: the car hangs `--hang` 120 s after the dip-out. The model steps
  0.1 s through the motion and the first `--slow-hang` 10 s of hanging, then
  `--dt-hang` 0.5 s (the film sub-steps on its own). The movie runs real
  time to the end of the dip-out, then `--speedup` 2× and `--speedup-late`
  10×: about 51 s in all. The curves are drawn against movie time, labelled
  in model time.
- The part of the car below the bath surface is drawn blue-tinted (the
  same sheet clipped at the surface plane) and the waterline is a dark blue
  line, so crossing the surface is easy to follow.

Article results: after dip-in, 0.61 l of air in 12 pockets; after
dip-out, 0.41 l of liquid in 8 cavities. The article used a 1.2 mm octree
grid (294 M cells) and took 5 h 20 min on 32 cores.

### Early test cases movie

`examples/early_cases_movie.py` runs four of the first 2D test cases in
lockstep with the current code and renders them side by side (30 s,
1920 × 1080, 24 fps). Each panel's camera follows its object.
1. Drainage through a hole: orifice law with capillary hold-up, and the
   analytic level as a dashed line.
2. Spill routing: two cups are tilted, and liquid from the upper cup is
   routed to the one below.
3. Trapped air and carried liquid: an upside-down cup and an open cup are
   dipped and lifted.
4. Door-like section: dip, tilt out and drain through 8 mm holes.

```bash
python examples/early_cases_movie.py --preview 4 9 17 26 --out early.mp4   # stills
python examples/early_cases_movie.py --out early_cases.mp4                 # ~10 min
```

---

## Door drainage transient (horizontal door, plugs pulled)

Experiment: the door hangs horizontally (0° tilt) with its three drain
holes plugged. It is dipped and lifted, then the three plugs are pulled at
the same moment by strings, and the outflow is weighed. Four runs are
digitised in `examples/data/door_drain_exp_0deg.csv`: time since opening,
cumulative outflow, and water left in the door. The runs agree closely:
50 / 90 / 99 % out at 7.0 / 14.1 / 17.3 s (±0.1 s), initial rate 0.77 l/s,
8–20 ml left (one outlier: 249 ml).

`examples/door_drain.py` repeats the same sequence:

1. The holes are given with `open_at` (explicit holes can be plugged until
   a given time: no liquid or air passes before then).
2. 4 s down, 4 s up, 5 s pause.
3. All three holes open at t = 13 s.

Water left = volume in the door at opening minus the cumulative flow through
the three holes, which is what the scale measures. With the plugs in, the
door holds 11.1 l at 8 mm and 10.9 l at 6 mm (measured: 10.5–10.9 l). That
amount does not depend on the dip depth, because it is set by the lower edge
of the access opening.

| | water at opening (l) | 50 % out (s) | 90 % out (s) | 99 % out (s) |
|---|---|---|---|---|
| experiment (4 runs) | 10.5–10.9 | 7.0 | 14.1 | 17.3 |
| 8 mm, Cd = 0.6 | 11.11 | 7.4 | 15.4 | 18.5 |
| 6 mm, Cd = 0.6 | 10.89 | 7.3 | 15.2 | 18.4 |
| 5 mm, Cd = 0.6 | 10.56 | 7.2 | 14.9 | 18.0 |
| 6 mm, Cd = 0.65 | 10.89 | 6.8 | 14.0 | 17.0 |
| 6 mm, Cd = 0.6, binary cells | 9.42 | 6.3 | 12.9 | 15.4 |
| v4.3, 5° hang, 6 mm, Cd = 0.60 | 10.91 | 7.3 | 15.1 | 18.2 |
| v4.3, 5° hang, 6 mm, Cd = 0.63 | 10.91 | 6.9 | 14.3 | 17.4 |
| v4.3, 5° hang, 6 mm, Cd = 0.65 | 10.91 | 6.7 | 13.9 | 16.8 |

With the door hanging at its 5° lean (`--hang`, default 5 in
`door_drain.py`), Cd = 0.63 matches all three times of the experiment
within 0.1 s.

`examples/door_drain_movie_hd.py` is the full-HD movie of this case
(1920 × 1080): the door with the underwater tint and waterline during the
dip, glossy water, plugs that drop out, tapered jets that turn into drops
at a trickle, ripples on the bath, and a slow camera push while it drains.
The panel shows the water left and the outflow rate against the four runs
(range and mean), with the 50 / 90 / 99 % times filled in as they are
reached. `--workers N` renders in N processes and joins the parts.

What the comparison shows:

* The outflow rate plotted against the remaining water is the same at 8, 6
  and 5 mm. It is set by the level–volume relation of the door and the
  orifice law, and with the cut-cell volumes it has already converged at
  these resolutions. What remains is a fixed ratio: the measured rate is
  about 7 % above the model's at Cd = 0.6, so Cd ≈ 0.65 fits the whole
  curve.
* Binary cells lose 13 % of the water at 6 mm and drain correspondingly too
  fast.
* The results do not depend on the time step (0.025–0.1 s at 8 mm).
* Cost at 6 mm (0.63 M cells, k = 4): 205 s on one core for the full 43 s
  sequence (12 s setup), about 5× slower than real time.
* `examples/door_drain_movie.py` renders the whole sequence in real time:
  the 3D world-frame view on the left, and on the right the measured curves
  with the model line growing as it runs (24 fps, vtk + ffmpeg).
  `--preview t1 t2 ...` writes stills instead.
* `examples/door_dip_film_movie.py` makes a showcase movie of a full dip
  (default 45°, holes open, wall film on). It shows the pools (drawn on
  top so they stay visible), the jets, the wall film coloured by thickness
  (1–100 µm, log scale) on two thin shells on the STL, one for each side
  of the sheet, and the drops. Drops fall straight down to the door or the
  bath and are drawn 3× enlarged. The first 28 s play in real time, then
  the dripping tail up to 128 s plays 5× faster. The right panel shows the
  pools, the film volume and the dripped volume over time.
  `--full-dip` puts the whole door under water: the travel is set from the
  tilted door's height, 8 s down and 8 s up. The camera follows 80 % of the
  door's motion. Real time runs to 36 s, then the tail to 136 s plays 5×
  faster. At 10 mm the door holds about 4 l just after lift-out, and the
  film peaks at about 130 ml.
* After drainage the model keeps < 1 ml; the experiment 8–20 ml. The wall
  film (`film=True`) is not included in these runs.

---

## Parameters

| Where | Name | Default | Meaning |
|---|---|---|---|
| `Simulation` | `split` | True | False = all openings instantaneous (equilibrium) |
| | `spill_routing` | True | False = escaped liquid vanishes (original method) |
| | `cells_per_step` | 1.0 | bath motion per step, in cells |
| | `dt_max` | 0.05 s | max step (drainage tail) |
| `segment_kwargs` | `beta` | 0.6 | constriction ratio for a throat |
| | `d_free` | 0.03 m | openings wider than this are instantaneous |
| `ThroatModel` | `Cd` | 0.6 | discharge coefficient |
| | `capillary` | True | capillary hold-up head |
| | `counter_current_factor` | 0.3 | unvented flow reduction |
| | `rt_factor` | 3.68 | `d_crit = rt_factor·l_c` (3D) |
| | `film` | False | True or `FilmParams(...)` enables the wall film |
| | `holes` | None | explicit holes `[(center, diameter)]` or dicts with `axis`, `open_at` (plugged until then) |
| | `min_depth_cells` | 1.5 | pockets shallower than this many cell heights hold nothing |
| | `subcells` | 4 | side-aware cut-cell volumes with k³ sub-cells per cut cell; 0 = binary |
| | `compressible_air` | False | trapped exterior air follows Boyle's law (hydrostatic pressure at the pocket's water surface) |
| | `narrow` | None | octree: `dict(k=K, width=2, connecting=True, margin=1)` gives the closed cells at passages narrower than `width` base sub-cells K sub-cells per edge (True: k = 2·subcells); `connecting`: only passages with two or more mouths |
| | `portion` | "box" | volume of a node below a level: "box" = exact for an axis-aligned cube (additive over octree levels), "linear" = spread evenly over its extent |
| `Octree.from_mesh` | `h` | – | finest cell (at the walls) |
| | `levels` | 3 | coarser levels; base cells h·2^levels |
| | `margin` | 1 | cells of each level kept around the finer one |
| `segment_kwargs` (octree) | `distance` | "voxel" | "voxel": the uniform grid's segmentation on the graph; "surface": distance to the triangles, face links only |
| `FilmParams` | `drop_volume` | 3.8 l_c³ (3D) | volume of one drop |
| | `h_bulk` | min(1 mm, dx/2) | puddle limit on upward faces |
| | `hang_normal` | −0.2 | `n·up` below this means the film hangs (drips) |
| | `slope_floor` | 0.3 | floor on sin α when converting withdrawal speed to along-surface speed |
| | `dt_film` | 0.05 s | film sub-step |
| `Fluid` | `rho, mu, sigma, g` | water | ED paint: set its μ, σ and ρ |

---

## Assumptions and known limitations

* **Air is passive inside internal compartments.** Air pockets there are
  not trapped separately: a dome inside a closed compartment fills with
  liquid. Exterior pockets are trapped as in the article, and they can be
  compressed with `compressible_air=True`.
* **Nested trapping** (an air pocket inside a liquid cavity, or the reverse) is
  not resolved: liquid wins.
* **Throat law.** A plain orifice law is used for every opening, including
  weir-type spill over a partly wetted edge. There is no inertia, no pipe
  friction for long narrow channels, and air-side resistance is ignored. Cd
  and the counter-current factor need calibration.
* **Throat detection depends on resolution.** Whole-cell throats need about
  3 cells across, as in the article; with the sub-cell graph any opening of
  at least 2 sub-cells (dx/2) is open and gets a throat. Width and area are
  good to about one sub-cell per edge. Choosing `beta`/`d_free` sets what
  counts as "instantaneous".
* **Capillary hold-up uses the hole formula** (4σ/ρgd) for every opening in
  3D, also for slots, where 2σ/ρgd would apply.
* **Film model:**
  * Surface tension is ignored on the walls: no rivulets, no film breakup, and
    no contact-angle hold-up on flat areas beyond `h_bulk`.
  * Upward-facing surfaces at or near horizontal keep their deposited film.
  * The deposition law is for a plate pulled out at a steady speed. For
    nearly horizontal surfaces the speed conversion is capped by
    `slope_floor`, which is uncertain.
  * Drop size is fixed (Tate). In 2D it is a rough per-unit-depth value.
  * The carrier comes from voxels, so edges are rounded at about one cell and
    film thickness is not resolved at sharp edges.
* **Resolution.** Since v6 the octree removes the bulk cells, but the wall
  cells and their sub-cells still grow as h^-2: the car at 3 mm is about
  60 M nodes (100 GB class). The article's 1.2 mm is out of reach on a
  workstation; 3–4 mm with k = 2 (sub-cells of 1.5–2 mm) is the practical
  limit.
* **Segmentation on the octree** reproduces the uniform grid's up to ties
  in the flooding; the transient on a coarse car grid is sensitive to that
  (a few per cent, see *Octree grid*).

## Roadmap

1. The real test case: the car door from the article (water), including film
   and drips, compared with the filling and drainage measurements.
2. Sub-cell spill heights for saddles that are not listed holes: use the
   lowest fluid sub-cell of the saddle cell (available from `volfrac.py`)
   instead of its centre. This removes the remaining O(dx) level bias.
3. Weir law for rim overflow; channel resistance for long gaps. The hold-up
   criterion for a partly wetted hole (head at the segment centroid) also
   needs checking against experiment.
4. Air in internal compartments (pressure and compression, nested pockets).
5. Calibration experiments: door on a load cell (mass vs time after the plugs
   are pulled), and plates with graded holes for `Cd` and the hold-up
   threshold.
6. Port to IBOFlow (see `COPILOT_DRAINSIM_TO_IBOFLOW.md`). drainsim runs on
   an octree of the same kind since v6.
7. Parallel parts still serial in v5: the film sweep (it dominates the
   drip-off with film on; it could be swept level by level, each level in
   parallel, pulling the inflow from above in sweep order so the result stays
   the same), the fill cascade and the per-throat loop in Python.

## Code map

| File | Content |
|---|---|
| `drainsim/grid.py` | voxel grid, neighbours, STL voxelisation (full or cropped to a box; memory-light surface sampling) |
| `drainsim/shapes.py` | primitives for synthetic geometries |
| `drainsim/cases.py` | test geometries (boxes, cups, door section, 3D box) |
| `drainsim/fsm.py` | fill-spill-merge kernel (numba): parallel hierarchy (`prepare`/`solve`, reusable `FSMCache`), serial reference sweep |
| `drainsim/par.py` | parallel building blocks: sort, compaction, connected components, `dual` (parallel + serial nogil compilation) |
| `drainsim/stepk.py` | per-step array passes of `Simulation` as parallel kernels |
| `drainsim/compartments.py` | distance field, watershed, throat extraction, explicit holes, throats on the node graph (`node_throats`) |
| `drainsim/octree.py` | octree grid (2:1 graded, keys per level), node graph with sub-cells (per-cell resolution, narrow-passage refinement) |
| `drainsim/gseg.py` | compartments and throats on a node graph (octree runs) |
| `drainsim/octview.py` | uniform display grid for octree runs (movies, renders) |
| `drainsim/volfrac.py` | side-aware cut-cell volumes (sub-cell sampling + geodesic BFS, numba) |
| `drainsim/physics.py` | fluid, orifice law, hold-up, venting limits, analytic solution |
| `drainsim/motion.py` | keyframe motions, `dip()` helper |
| `drainsim/model.py` | `Simulation`: time loop, throats, equilibration, diagnostics |
| `drainsim/film.py` | film carrier mesh, deposition, drainage (numba sweep), drops |
| `drainsim/viz.py` | 2D plots, object-frame VTK writer (debugging) |
| `drainsim/worldviz.py` | world-frame export (moving object), VTK XML/PVD writers, 2D animation |
| `drainsim/render3d.py` | off-screen 3D rendering of the world-frame series to MP4/GIF |
| `tests/` | verification tests |
| `examples/` | the seven examples above, `door_article.py` (door case CLI), `door_figures.py` (Fig. 7/8 replicas), `article_comparison.py` (Table 1 and Figs. 5–8 recreated), `car_article.py` (car body, Figs. 9–13), `car_movie.py` (car dip movie: trapped air, liquid, wall film and drops), `door_pocket.py` (pocket volume check), `door_drain.py` (plugged-hole drainage transient vs experiment), `door_drain_movie.py` (real-time movie with the measurements), `door_drain_movie_hd.py` (full-HD version, 5° hang, jets, drops, ripples, flow panel), `door_dip_film_movie.py` (showcase: full dip with film and drops), `diving_bell_movie.py` (compressible air presentation case), `cascade_rack.py` / `cascade_rack_movie.py` (spill-routing presentation case), `early_cases_movie.py` (four early 2D test cases side by side: orifice drainage vs analytic, spill routing, trapped air and carried liquid, door-like section), `thread_scaling.py` (v5 thread-scaling benchmark on the car), `data/` (digitised drainage measurements) |
