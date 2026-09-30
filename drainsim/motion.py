"""Rigid-body dipping motions.

A pose maps object coordinates to world coordinates:

    x_world = R(t) (x_obj - pivot) + pivot + T(t)

World "up" is +y in 2D and +z in 3D. The bath surface is the horizontal
plane at world height ``bath_level``. Because the grid is fixed to the object,
the simulation only needs, per time, the up-vector expressed in the object
frame and the bath height along it (``Motion.frame``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rot2(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def rot3(angles_deg) -> np.ndarray:
    """Rotation from x-y-z Euler angles (degrees), applied in order x, y, z."""
    ax, ay, az = np.radians(angles_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@dataclass
class Keyframes:
    """Piecewise-linear motion through keyframes.

    times  : (K,) seconds, increasing.
    angles : (K,) degrees (2D, about the out-of-plane axis) or (K, 3) Euler
             angles in degrees (3D).
    shifts : (K, ndim) world translation T(t) of the object.
    pivot  : rotation centre in object coordinates.
    bath_level : world height of the bath surface.
    """
    times: np.ndarray
    angles: np.ndarray
    shifts: np.ndarray
    ndim: int = 2
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(2))
    bath_level: float = 0.0

    def __post_init__(self):
        self.times = np.asarray(self.times, float)
        self.angles = np.asarray(self.angles, float)
        self.shifts = np.asarray(self.shifts, float)
        self.pivot = np.asarray(self.pivot, float)
        if self.pivot.size != self.ndim:
            self.pivot = np.zeros(self.ndim)

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def pose(self, t: float):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        if self.ndim == 2:
            ang = np.interp(t, self.times, self.angles)
            R = rot2(ang)
        else:
            ang = [np.interp(t, self.times, self.angles[:, k]) for k in range(3)]
            R = rot3(ang)
        T = np.array([np.interp(t, self.times, self.shifts[:, k])
                      for k in range(self.ndim)])
        return R, T

    def frame(self, t: float):
        """(up vector in object frame, bath level along it)."""
        R, T = self.pose(t)
        e_up = np.zeros(self.ndim)
        e_up[-1] = 1.0
        up = R.T @ e_up
        zb = self.bath_level - e_up @ (self.pivot + T) + up @ self.pivot
        return up, zb

    def speed_bound(self, t: float, dt: float, extent: float) -> float:
        """Max speed of the bath surface relative to object points (m/s)."""
        u0, z0 = self.frame(t)
        u1, z1 = self.frame(t + dt)
        dang = np.linalg.norm(u1 - u0)
        return (abs(z1 - z0) + dang * extent) / dt


def dip(ndim=2, depth=0.4, start_height=0.15, t_down=4.0, t_hold=0.0,
        t_up=4.0, t_after=0.0, tilt=0.0, pivot=None, bath_level=0.0,
        extra_keys=None) -> Keyframes:
    """Simple vertical dip: down by ``depth``, hold, back up, then wait.

    ``tilt`` is a constant tilt angle (deg, 2D) or Euler triple (3D).
    The object starts ``start_height`` above its initial position, i.e. the
    object's y=0 (z=0) plane is at world height ``start_height``.
    """
    up = np.zeros(ndim)
    up[-1] = 1.0
    ts = [0.0, t_down, t_down + t_hold, t_down + t_hold + t_up,
          t_down + t_hold + t_up + t_after]
    zs = [start_height, start_height - depth, start_height - depth,
          start_height, start_height]
    shifts = np.array([z * up for z in zs])
    if ndim == 2:
        angles = np.full(len(ts), float(tilt))
    else:
        angles = np.tile(np.asarray(tilt if np.ndim(tilt) else [0, 0, tilt],
                                    float), (len(ts), 1))
    # remove zero-length segments
    keep = [0] + [i for i in range(1, len(ts)) if ts[i] > ts[i - 1]]
    return Keyframes(np.array(ts)[keep], angles[keep], shifts[keep], ndim,
                     pivot=np.zeros(ndim) if pivot is None else pivot,
                     bath_level=bath_level)
