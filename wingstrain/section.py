"""Thin-walled composite section analysis.

A wing section is idealised as a single closed cell (the skin wrapping the nose
plus a shear web) built from discrete wall segments, each with its own laminate,
plus optional lumped spar caps.  All stiffnesses are E-weighted, so mixed
layups (fabric skin + UD caps) are handled directly.

What is computed
    EA, elastic centroid, EI_xx, EI_zz, EI_xz   (full unsymmetric bending)
    A_enc, GJ                                   (Bredt-Batho single cell)
    x_sc                                        (shear centre, for the torsion arm)
    q_hat(s)                                    (shear flow per unit dM_x/dy)

Sign conventions (car axes: x aft, y spanwise, z up)
    eps_yy(x,z) = N/EA + b*(x - x_ea) + c*(z - z_ea)
    M_x = integral(sigma * (z - z_ea) dA)   -> positive = tension on the upper skin
    M_z = integral(sigma * (x - x_ea) dA)   -> positive = tension at the trailing edge
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .airfoil import Airfoil, transform
from .laminate import Laminate, MATERIALS


@dataclass
class Segment:
    p0: np.ndarray       # (x, z) start, car axes [m]
    p1: np.ndarray       # (x, z) end
    lam: Laminate
    surface: str = "skin"   # 'upper' | 'lower' | 'web' | 'te'
    xc: float = 0.0         # chord fraction of the midpoint (aerofoil axes)
    structural_only: bool = False   # counts for EI but not for the torsion cell

    @property
    def ds(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    @property
    def tangent(self) -> np.ndarray:
        d = self.p1 - self.p0
        n = np.hypot(*d)
        return d / n if n > 0 else np.array([1.0, 0.0])

    @property
    def t(self) -> float:
        return self.lam.thickness

    @property
    def E(self) -> float:
        return self.lam.Ex

    @property
    def G(self) -> float:
        return self.lam.Gxy


@dataclass
class Cap:
    """Lumped spar cap (unidirectional tows) treated as a point area."""
    p: np.ndarray
    area: float
    E: float
    name: str = "cap"
    rho: float = 1600.0


@dataclass
class Section:
    loop: list[Segment]          # closed cell, ordered, counter-clockwise
    caps: list[Cap] = field(default_factory=list)
    extra: list[Segment] = field(default_factory=list)   # bending-only walls
    chord: float = 1.0
    name: str = "section"
    _cache: dict = field(default_factory=dict, repr=False)

    # -- basic properties --------------------------------------------------
    @property
    def all_walls(self) -> list[Segment]:
        return list(self.loop) + list(self.extra)

    def props(self) -> dict:
        if "props" in self._cache:
            return self._cache["props"]
        EA = 0.0
        Sx = 0.0   # first moment about x-axis (sum E t ds z)
        Sz = 0.0
        for s in self.all_walls:
            w = s.E * s.t * s.ds
            EA += w
            Sx += w * s.mid[1]
            Sz += w * s.mid[0]
        for c in self.caps:
            w = c.E * c.area
            EA += w
            Sx += w * c.p[1]
            Sz += w * c.p[0]
        x_ea, z_ea = Sz / EA, Sx / EA

        EIxx = EIzz = EIxz = 0.0
        for s in self.all_walls:
            w = s.E * s.t * s.ds
            dx, dz = s.mid[0] - x_ea, s.mid[1] - z_ea
            EIxx += w * dz ** 2
            EIzz += w * dx ** 2
            EIxz += w * dx * dz
            # the wall's own second moment about its midpoint (small but free)
            tx, tz = s.tangent
            EIxx += s.E * s.ds ** 3 * s.t / 12.0 * tz ** 2
            EIzz += s.E * s.ds ** 3 * s.t / 12.0 * tx ** 2
        for c in self.caps:
            w = c.E * c.area
            dx, dz = c.p[0] - x_ea, c.p[1] - z_ea
            EIxx += w * dz ** 2
            EIzz += w * dx ** 2
            EIxz += w * dx * dz

        A_enc = _polygon_area(np.array([s.p0 for s in self.loop]))
        circ = sum(s.ds / (s.G * s.t) for s in self.loop)
        GJ = 4.0 * A_enc ** 2 / circ if circ > 0 else 0.0

        mass = sum(s.lam.areal_mass * s.ds for s in self.all_walls)
        mass += sum(c.area * c.rho for c in self.caps)

        p = dict(EA=EA, x_ea=x_ea, z_ea=z_ea, EIxx=EIxx, EIzz=EIzz, EIxz=EIxz,
                 A_enc=A_enc, GJ=GJ, mass_per_span=mass,
                 perimeter=sum(s.ds for s in self.loop),
                 z_max=max(s.mid[1] for s in self.all_walls),
                 z_min=min(s.mid[1] for s in self.all_walls))
        p["x_sc"], p["q_hat"] = self._shear_centre(p)
        self._cache["props"] = p
        return p

    # -- shear flow and shear centre ---------------------------------------
    def _shear_centre(self, p: dict):
        """Single-cell shear-flow analysis for a vertical shear.

        Returns (x_sc, q_hat) where q_hat[i] is the shear flow in loop segment
        i per unit dM_x/dy, for a load applied through the shear centre.
        I_xz coupling is neglected here (second order for wing sections).
        """
        loop = self.loop
        n = len(loop)
        if n < 3 or p["A_enc"] <= 0:
            return p["x_ea"], np.zeros(n)
        # open-cell (cut at segment 0) shear flow, built up segment by segment
        qb = np.zeros(n)
        acc = 0.0
        for i, s in enumerate(loop):
            dz = s.mid[1] - p["z_ea"]
            acc += s.E * s.t * s.ds * dz
            qb[i] = -acc / p["EIxx"]      # per unit dM_x/dy
        # close the cell: no twist for a load through the shear centre
        num = sum(qb[i] * loop[i].ds / (loop[i].G * loop[i].t) for i in range(n))
        den = sum(loop[i].ds / (loop[i].G * loop[i].t) for i in range(n))
        q = qb - num / den
        # resultants of that shear flow
        Fz = sum(q[i] * (loop[i].p1[1] - loop[i].p0[1]) for i in range(n))
        My = 0.0
        for i, s in enumerate(loop):
            dx, dz = s.p1 - s.p0
            xm, zm = s.mid
            My += q[i] * ((xm - p["x_ea"]) * dz - (zm - p["z_ea"]) * dx)
        x_sc = p["x_ea"] + (My / Fz if abs(Fz) > 1e-12 else 0.0)
        return x_sc, q

    # -- strain recovery ---------------------------------------------------
    def bending_curvatures(self, Mx: float, Mz: float) -> tuple[float, float]:
        """Solve the coupled bending problem for the strain-plane gradients."""
        p = self.props()
        K = np.array([[p["EIzz"], p["EIxz"]], [p["EIxz"], p["EIxx"]]])
        b, c = np.linalg.solve(K, np.array([Mz, Mx]))
        return float(b), float(c)

    def eps_yy(self, x, z, Mx: float, Mz: float, N: float = 0.0):
        p = self.props()
        b, c = self.bending_curvatures(Mx, Mz)
        return N / p["EA"] + b * (np.asarray(x) - p["x_ea"]) + c * (np.asarray(z) - p["z_ea"])

    def shear_flow(self, seg_index: int, T: float, dMx_dy: float = 0.0) -> float:
        """Shear flow [N/m] in a loop segment from torque T and shear dM_x/dy."""
        p = self.props()
        q_t = T / (2.0 * p["A_enc"]) if p["A_enc"] > 0 else 0.0
        q_v = p["q_hat"][seg_index] * dMx_dy if seg_index < len(self.loop) else 0.0
        return float(q_t + q_v)

    # -- geometry lookup ---------------------------------------------------
    ALIASES = {"suction": "bottom", "pressure": "top",
               "upper": "top", "lower": "bottom", "web": "web_rear"}

    def locate(self, xc: float, surface: str = "top") -> int:
        """Index of the loop segment nearest chord fraction `xc` on `surface`.

        `surface` is physical, as seen on the car: 'top' | 'bottom' |
        'web_rear' | 'web_front'.  For a downforce wing 'bottom' is the suction
        side.  'suction'/'pressure'/'upper'/'lower' are accepted as aliases.
        """
        have = {s.surface for s in self.loop}
        if surface not in have:
            surface = self.ALIASES.get(surface, surface)
        cand = [i for i, s in enumerate(self.loop) if s.surface == surface]
        if not cand:
            raise ValueError(f"no '{surface}' segments; have "
                             f"{sorted({s.surface for s in self.loop})}")
        return min(cand, key=lambda i: abs(self.loop[i].xc - xc))

    def summary(self) -> str:
        p = self.props()
        return (f"{self.name}: chord {self.chord*1e3:.0f} mm  "
                f"depth {(p['z_max']-p['z_min'])*1e3:.1f} mm\n"
                f"  EI_xx = {p['EIxx']:.1f} N.m^2   EI_zz = {p['EIzz']:.1f} N.m^2   "
                f"EI_xz = {p['EIxz']:.2f} N.m^2\n"
                f"  GJ    = {p['GJ']:.1f} N.m^2   A_enc = {p['A_enc']*1e4:.1f} cm^2   "
                f"EA = {p['EA']/1e6:.1f} MN\n"
                f"  elastic centroid x = {p['x_ea']*1e3:.1f} mm, z = {p['z_ea']*1e3:.1f} mm  "
                f"| shear centre x = {p['x_sc']*1e3:.1f} mm\n"
                f"  mass = {p['mass_per_span']:.3f} kg/m of span")


def _polygon_area(pts: np.ndarray) -> float:
    x, z = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


# --------------------------------------------------------------------------
# Builder: aerofoil -> closed-cell section
# --------------------------------------------------------------------------
def box_section(airfoil: Airfoil, chord: float, skin: Laminate,
                web: Laminate | None = None, x_front: float = 0.0,
                x_rear: float = 0.65, angle_deg: float = 0.0,
                inverted: bool = True, n_skin: int = 48,
                lower_skin: Laminate | None = None,
                caps: list[dict] | tuple = (),
                include_trailing_skin: bool = False,
                x_ref: float = 0.25, name: str | None = None) -> Section:
    """Build a single-cell section: skin from `x_front` to `x_rear` plus webs.

    x_front = 0 wraps the leading edge (no front web needed); x_rear = 1 closes
    at the trailing edge.  `caps` entries look like
        {"xc": 0.30, "surface": "upper", "layers": 4,
         "material": "ud_hm", "width": 0.020}
    """
    lower_skin = lower_skin or skin
    runs: list[tuple[np.ndarray, Laminate, str, np.ndarray]] = []

    xc = _cosine(x_front, x_rear, n_skin)
    lo = np.column_stack([xc, airfoil.z_lower(xc)])
    up = np.column_stack([xc, airfoil.z_upper(xc)])

    # loop: lower surface aft->fwd, front web up, upper surface fwd->aft, rear web down
    runs.append((lo[::-1], lower_skin, "lower", xc[::-1]))
    if x_front > 1e-6:
        if web is None:
            raise ValueError("x_front > 0 needs a `web` laminate to close the cell")
        runs.append((np.array([[x_front, float(airfoil.z_lower(x_front))],
                               [x_front, float(airfoil.z_upper(x_front))]]),
                     web, "web_front", np.array([x_front])))
    runs.append((up, skin, "upper", xc))
    if x_rear < 1.0 - 1e-6:
        if web is None:
            raise ValueError("x_rear < 1 needs a `web` laminate to close the cell")
        runs.append((np.array([[x_rear, airfoil.z_upper(x_rear)],
                               [x_rear, airfoil.z_lower(x_rear)]]),
                     web, "web_rear", np.array([x_rear])))

    # Physical labels: for an inverted (downforce) wing the aerofoil's upper
    # surface faces the ground, so expose 'top'/'bottom' as seen on the car.
    phys = {"upper": "bottom", "lower": "top"} if inverted else \
           {"upper": "top", "lower": "bottom"}

    segs: list[Segment] = []
    for pts, lam, surf, xcs in runs:
        surf = phys.get(surf, surf)
        P = transform(pts, chord, angle_deg, inverted, x_ref)
        mid_xc = (xcs[:-1] + xcs[1:]) / 2.0 if len(xcs) > 1 else \
                 np.full(len(P) - 1, xcs[0])
        for k in range(len(P) - 1):
            if np.hypot(*(P[k + 1] - P[k])) < 1e-9:
                continue
            segs.append(Segment(P[k], P[k + 1], lam, surf, float(mid_xc[k])))
    # stitch: make the loop exactly closed and counter-clockwise
    segs = _close_loop(segs)
    if _polygon_area(np.array([s.p0 for s in segs])) < 0:
        segs = [Segment(s.p1, s.p0, s.lam, s.surface, s.xc) for s in segs[::-1]]

    extra: list[Segment] = []
    if include_trailing_skin and x_rear < 1.0 - 1e-6:
        xt = _cosine(x_rear, 1.0, max(6, n_skin // 4))
        for surf, zf, lam in ((phys["upper"] + "_te", airfoil.z_upper, skin),
                              (phys["lower"] + "_te", airfoil.z_lower, lower_skin)):
            P = transform(np.column_stack([xt, zf(xt)]), chord, angle_deg,
                          inverted, x_ref)
            mid = (xt[:-1] + xt[1:]) / 2.0
            for k in range(len(P) - 1):
                extra.append(Segment(P[k], P[k + 1], lam, surf, float(mid[k]),
                                     structural_only=True))

    cap_objs = []
    for c in caps:
        mat = c.get("material", "ud_hm")
        ply = MATERIALS[mat] if isinstance(mat, str) else mat
        t = c.get("layers", 1) * c.get("ply_thickness", ply.thickness)
        area = t * c["width"]
        zf = airfoil.z_upper if c.get("surface", "upper") == "upper" else airfoil.z_lower
        pt = transform(np.array([[c["xc"], float(zf(c["xc"]))]]), chord,
                       angle_deg, inverted, x_ref)[0]
        cap_objs.append(Cap(pt, area, c.get("E", ply.E1), mat, ply.rho))

    return Section(segs, cap_objs, extra, chord,
                   name or f"{airfoil.name} c={chord*1e3:.0f}mm")


def _cosine(a: float, b: float, n: int) -> np.ndarray:
    """Cosine-clustered points on [a,b]; denser at the ends."""
    return a + (b - a) * (1 - np.cos(np.linspace(0, np.pi, n))) / 2.0


def _close_loop(segs: list[Segment]) -> list[Segment]:
    out = list(segs)
    for i in range(len(out) - 1):
        if np.hypot(*(out[i + 1].p0 - out[i].p1)) > 1e-9:
            out[i + 1] = Segment(out[i].p1.copy(), out[i + 1].p1, out[i + 1].lam,
                                 out[i + 1].surface, out[i + 1].xc)
    if np.hypot(*(out[0].p0 - out[-1].p1)) > 1e-9:
        out[-1] = Segment(out[-1].p0, out[0].p0.copy(), out[-1].lam,
                          out[-1].surface, out[-1].xc)
    return out
