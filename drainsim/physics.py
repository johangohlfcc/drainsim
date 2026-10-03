"""Lumped physics used for rate-limited exchange through throats.

Nothing here resolves the flow field. The throat model is an orifice law with
two geometric corrections:

* **Capillary hold-up** - a meniscus pinned at the exit of an opening of width
  ``d`` withstands a hydrostatic head ``h_cap = 2(n-1) sigma / (rho g d)``
  (n = 3: circular hole, 4 sigma / rho g d; n = 2: slot, 2 sigma / rho g d).
  Liquid above an opening therefore stops draining at a finite depth.
* **Venting** - a compartment can only lose (gain) liquid if air can get in
  (out). If the only connection is the draining opening itself, the flow is
  counter-current ("glugging"): reduced by ``counter_current_factor``, and
  zero if the opening is narrower than the Rayleigh-Taylor stability limit
  ``d_crit = rt_factor * l_c`` (an inverted straw holds its water).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fluid:
    rho: float = 1000.0         # kg/m^3
    mu: float = 1.0e-3          # Pa s (not used by the throat model yet)
    sigma: float = 0.072        # N/m
    g: float = 9.81             # m/s^2
    p_atm: float = 101325.0     # Pa, ambient pressure (compressible air pockets)

    @property
    def capillary_length(self) -> float:
        return float(np.sqrt(self.sigma / (self.rho * self.g)))


@dataclass
class ThroatModel:
    Cd: float = 0.6                       # discharge coefficient
    capillary: bool = True                # apply capillary hold-up head
    counter_current_factor: float = 0.3   # flow factor for unvented openings
    rt_factor: float = 3.68               # d_crit = rt_factor * l_c (circular)
    instant: bool = False                 # True: throats equalise in one step
    # explicit holes with free outflow: integrate the orifice law over the
    # wetted part of the hole, Q = Cd * int sqrt(2 g (H - z)) dA, so a partly
    # covered hole behaves like a weir (6.3); False: Cd * A_wet * sqrt(2 g
    # (H - z_centroid)) as before, which over-predicts a partly covered hole
    hole_profile: bool = True
    # the Rayleigh-Taylor cut-off (no counter-current flow through an
    # opening narrower than d_crit) holds for an opening facing up or down,
    # with the heavy liquid over the light air across it. True: only for
    # openings within ``rt_angle`` deg of horizontal (their normal within
    # that of vertical); through an opening in a steep wall liquid and air
    # pass each other side by side (the counter-current factor applies).
    # False: for every opening (as before 7.1)
    rt_orientation: bool = False
    rt_angle: float = 45.0

    def holdup_head(self, d: float, fluid: Fluid, ndim: int) -> float:
        if not self.capillary or d <= 0:
            return 0.0
        return 2.0 * (ndim - 1) * fluid.sigma / (fluid.rho * fluid.g * d)

    def d_crit(self, fluid: Fluid, ndim: int) -> float:
        # 2D slot: Rayleigh-Taylor cut-off wavelength / 2 = pi * l_c
        f = self.rt_factor if ndim == 3 else np.pi
        return f * fluid.capillary_length

    def flow(self, area: float, head: float, fluid: Fluid) -> float:
        """Orifice (Torricelli) law. m^3/s in 3D, m^2/s per unit depth in 2D."""
        if head <= 0.0:
            return 0.0
        return self.Cd * area * np.sqrt(2.0 * fluid.g * head)


def torricelli_level(t, h0, Cd, a, A, g=9.81, h_cap=0.0):
    """Analytic level in a prismatic tank of plan area A draining through an
    orifice of area a at its bottom, with an optional hold-up head.

    Level above the orifice: sqrt(h - h_cap) decreases linearly in time.
    """
    s0 = np.sqrt(max(h0 - h_cap, 0.0))
    s = np.maximum(s0 - Cd * a / A * np.sqrt(g / 2.0) * np.asarray(t), 0.0)
    return s ** 2 + h_cap
