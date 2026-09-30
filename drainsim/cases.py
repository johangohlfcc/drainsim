"""Synthetic test geometries (2D unless stated otherwise). Units: metres."""
from __future__ import annotations

import numpy as np

from . import shapes as sh
from .grid import Grid
from .motion import Keyframes


def static(ndim=2, t_end=10.0, bath_level=-10.0, angle=0.0) -> Keyframes:
    """No motion; bath far below (pure drainage in air)."""
    if ndim == 2:
        ang = np.array([angle, angle], float)
    else:
        a3 = np.zeros(3) if np.ndim(angle) == 0 else np.asarray(angle, float)
        if np.ndim(angle) == 0:
            a3[0] = angle
        ang = np.tile(a3, (2, 1))
    return Keyframes([0.0, t_end], ang, np.zeros((2, ndim)), ndim,
                     bath_level=bath_level)


def torricelli_box(dx=0.005, hole=0.015, vent=0.02, W=0.4, H=0.4, wall=0.01):
    """Closed box with a hole in the bottom centre and a vent in the lid."""
    g = Grid.empty([0, 0], [1, 1], dx)
    lo = np.array([0.5 - W / 2, 0.3])
    hi = lo + [W, H]
    sh.add(g, sh.shell_box(g, lo, hi, wall))
    sh.cut(g, sh.rect(g, [0.5 - hole / 2, lo[1] - dx], [0.5 + hole / 2, lo[1] + wall + dx]))
    if vent:
        sh.cut(g, sh.rect(g, [lo[0] + 0.05, hi[1] - wall - dx], [lo[0] + 0.05 + vent, hi[1] + dx]))
    interior = sh.rect(g, lo + wall, hi - wall) & g.fluid
    return g, interior.ravel()


def two_cups(dx=0.005):
    """Upper cup that spills into a lower cup when tilted."""
    g = Grid.empty([0, 0], [1, 1], dx)
    sh.add(g, sh.shell_box(g, [0.1, 0.5], [0.4, 0.7], 0.02, open_faces=[(1, 1)]))
    sh.add(g, sh.shell_box(g, [0.35, 0.1], [0.8, 0.3], 0.02, open_faces=[(1, 1)]))
    return g


def inverted_cup(dx=0.005):
    """Downward-open cup (traps air on dip-in) above an upward-open cup."""
    g = Grid.empty([-0.5, -0.1], [0.5, 0.6], dx)
    sh.add(g, sh.shell_box(g, [-0.3, 0.3], [-0.05, 0.45], 0.01, open_faces=[(1, -1)]))
    sh.add(g, sh.shell_box(g, [0.05, 0.05], [0.3, 0.2], 0.01, open_faces=[(1, 1)]))
    return g


def door_section(dx=0.002, hole=0.008, vent=0.02, with_shelf=True):
    """2D cross-section loosely resembling a door: a tall cavity between two
    panels, drain holes in the bottom, a narrow opening at the top, an
    internal shelf (stiffener) that holds liquid, and an outside step."""
    g = Grid.empty([-0.3, -0.05], [0.3, 0.75], dx)
    lo = np.array([-0.2, 0.05])
    hi = np.array([0.2, 0.65])
    t = 0.004
    sh.add(g, sh.shell_box(g, lo, hi, t))
    # drain holes in the bottom, near both corners
    for x in (-0.15, 0.15):
        sh.cut(g, sh.rect(g, [x - hole / 2, lo[1] - dx], [x + hole / 2, lo[1] + t + dx]))
    # top opening (belt-line slot)
    sh.cut(g, sh.rect(g, [0.0 - vent / 2, hi[1] - t - dx], [0.0 + vent / 2, hi[1] + dx]))
    if with_shelf:
        # shelf with an upturned lip, attached to the left panel
        sh.add(g, sh.polyline(g, [[-0.196, 0.35], [-0.05, 0.33], [-0.05, 0.37]], t))
        # small cup below the shelf, fed only by liquid spilling from above
        sh.add(g, sh.polyline(g, [[-0.02, 0.22], [-0.02, 0.18], [0.12, 0.18], [0.12, 0.22]], t))
    # outside ledge on the right panel (exterior retention)
    sh.add(g, sh.polyline(g, [[0.2, 0.3], [0.26, 0.3], [0.26, 0.33]], t))
    return g


def box3d(dx=0.01, hole=0.03, vent=0.04, size=0.3, wall=0.01):
    """3D closed box with a square hole in the floor and a vent in the lid."""
    g = Grid.empty([-0.25, -0.25, -0.05], [0.25, 0.25, 0.45], dx)
    lo = np.array([-size / 2, -size / 2, 0.05])
    hi = lo + size
    sh.add(g, sh.shell_box(g, lo, hi, wall))
    sh.cut(g, sh.box(g, [-hole / 2, -hole / 2, lo[2] - dx], [hole / 2, hole / 2, lo[2] + wall + dx]))
    if vent:
        sh.cut(g, sh.box(g, [lo[0] + 0.03, -vent / 2, hi[2] - wall - dx],
                         [lo[0] + 0.03 + vent, vent / 2, hi[2] + dx]))
    interior = sh.box(g, lo + wall, hi - wall) & g.fluid
    return g, interior.ravel()
