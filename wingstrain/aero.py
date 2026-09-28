"""Aerodynamic loading of the instrumented element.

The element you strain gauge (normally the main plane) carries
  * its own lift and drag, distributed over the span, and
  * point loads from any elements mounted onto it (upper flaps on brackets).

Everything scales with dynamic pressure, so the strain-to-downforce calibration
is speed independent; what is NOT speed independent is how the load is *shared*
and where it acts, which is why `load_fraction` and `x_cp` carry their own
uncertainties in the error budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .airfoil import transform

RHO_SEA_LEVEL = 1.225


def air_density(temp_C: float = 15.0, pressure_Pa: float = 101325.0) -> float:
    return pressure_Pa / (287.05 * (temp_C + 273.15))


def spanwise_shape(kind: str = "blend", tip_loading: float = 0.75,
                   span: float = 1.0, y0: float = 0.0, n: int = 401):
    """Normalised spanwise load shape, integral over the span = 1.

    'uniform'    : constant (2D-like, strong endplates)
    'elliptical' : sqrt(1-(2eta-1)^2), classic unswept lift distribution
    'blend'      : 1-(1-tip_loading)*(2eta-1)^2  -- `tip_loading` is the load
                   per unit span at the tip as a fraction of mid-span.  0.75 is
                   a reasonable stand-in for a well-sealed endplate; drop it to
                   ~0.4 for a weak tip.
    """
    y = np.linspace(y0, y0 + span, n)
    eta = (y - y0) / span
    s = 2 * eta - 1
    if kind == "uniform":
        f = np.ones_like(eta)
    elif kind == "elliptical":
        f = np.sqrt(np.clip(1 - s ** 2, 0, None))
    elif kind == "blend":
        f = 1.0 - (1.0 - tip_loading) * s ** 2
    else:
        raise ValueError(f"unknown distribution '{kind}'")
    f /= np.trapezoid(f, y)
    return lambda yq: np.interp(yq, y, f)


@dataclass
class CarriedElement:
    """An upper element whose load passes through the instrumented one."""
    name: str
    CL_S: float                     # C_L * area [m^2], of this element alone
    CD_S: float = 0.0
    bracket_y: tuple = (0.30, 0.70) # bracket stations as span fractions
    xc_app: float = 0.55            # where the bracket load lands, chord fraction
    share: tuple | None = None      # load split between brackets (default even)


@dataclass
class AeroSpec:
    """Aero definition for the instrumented element plus what it carries."""
    CL: float = 1.60                # of the instrumented element alone
    CD: float = 0.35
    xc_cp: float = 0.32             # centre of pressure, chord fraction
    distribution: str = "blend"
    tip_loading: float = 0.75
    rho: float = RHO_SEA_LEVEL
    carried: list = field(default_factory=list)

    def q(self, V: float) -> float:
        return 0.5 * self.rho * V * V


@dataclass
class AeroLoads:
    """Resolved loading at one speed."""
    V: float
    q: float
    D_element: float                # downforce from the instrumented element [N]
    D_carried: float
    D_total: float
    drag_total: float
    pz: object                       # callable y -> N/m (negative)
    px: object
    x_cp: object                     # callable y -> m, car axes
    z_cp: object
    point_loads: list

    @property
    def carried_fraction(self) -> float:
        return self.D_carried / self.D_total if self.D_total else 0.0


def build_loads(spec: AeroSpec, V: float, span: float, chord_fn, y0: float,
                area: float, angle_deg: float, inverted: bool,
                x_ref: float) -> AeroLoads:
    """Turn an AeroSpec into distributed and point loads in car axes."""
    from .beam import PointLoad

    q = spec.q(V)
    D_el = q * area * spec.CL
    Dr_el = q * area * spec.CD
    shape = spanwise_shape(spec.distribution, spec.tip_loading, span, y0)

    def cp_point(xc: float, y: float):
        c = chord_fn(y)
        p = transform(np.array([[xc, 0.0]]), c, angle_deg, inverted, x_ref)[0]
        return float(p[0]), float(p[1])

    pz = lambda y: -D_el * float(shape(y))
    px = lambda y: Dr_el * float(shape(y))
    x_cp = lambda y: cp_point(spec.xc_cp, y)[0]
    z_cp = lambda y: cp_point(spec.xc_cp, y)[1]

    pls, D_car = [], 0.0
    for el in spec.carried:
        Dz = q * el.CL_S
        Dx = q * el.CD_S
        D_car += Dz
        shares = el.share or [1.0 / len(el.bracket_y)] * len(el.bracket_y)
        shares = np.array(shares, float)
        shares /= shares.sum()
        for frac, sh in zip(el.bracket_y, shares):
            yb = y0 + frac * span
            xa, za = cp_point(el.xc_app, yb)
            pls.append(PointLoad(yb, -Dz * sh, Dx * sh, xa, za,
                                 f"{el.name}@{frac:.2f}"))
    drag = Dr_el + sum(q * el.CD_S for el in spec.carried)
    return AeroLoads(V, q, D_el, D_car, D_el + D_car, drag,
                     pz, px, x_cp, z_cp, pls)
