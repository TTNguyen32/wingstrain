"""Euler-Bernoulli beam solver for the wing as a spanwise beam.

Three uncoupled problems are solved on the same mesh: vertical bending (EI_xx),
chordwise bending (EI_zz) and St Venant torsion (GJ).  Section properties may
vary along the span (taper).  Supports can be anywhere - the usual Formula
Student case is two pylons with overhanging tips, which makes the wing
statically determinate, but three mounts or a clamped centre mount work too.

Internal loads are recovered by integrating statics with the solved reactions,
so M(y) is exact for determinate spans regardless of mesh density.  Residual
moment at a free tip is reported as an equilibrium check.

Sign conventions (car axes: x aft, y spanwise, z up)
    F_z < 0 is downforce.
    M_x(y) = -sum_{s<y} F_z,i * (y - s_i)   -> positive = tension on the top skin
    M_z(y) = -sum_{s<y} F_x,i * (y - s_i)   -> positive = tension at the trailing edge
    T(y)   = -sum_{s<y} tau_i               -> positive by the right-hand rule about +y
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Support:
    """Wing mount. 'pin' restrains z only; 'clamp' also restrains dw/dy."""
    y: float
    kind: str = "pin"           # 'pin' | 'clamp'
    restrain_twist: bool = True
    restrain_chordwise: bool = True
    label: str = "mount"


@dataclass
class PointLoad:
    """Discrete load, e.g. a flap bracket reaction or an endplate load."""
    y: float
    Fz: float = 0.0             # [N], negative = downforce
    Fx: float = 0.0             # [N], positive = drag (aft)
    x_app: float | None = None  # chordwise point of application [m, car axes]
    z_app: float | None = None
    label: str = "point"


@dataclass
class BeamResult:
    y: np.ndarray
    Mx: np.ndarray
    Mz: np.ndarray
    T: np.ndarray
    Sz: np.ndarray              # dMx/dy, drives the transverse shear flow
    w: np.ndarray               # vertical deflection [m]
    u: np.ndarray               # chordwise deflection [m]
    phi: np.ndarray             # twist [rad]
    reactions: list = field(default_factory=list)
    equilibrium_residual: float = 0.0

    def at(self, yq, arr_name: str) -> float:
        return float(np.interp(yq, self.y, getattr(self, arr_name)))


class Beam:
    """Spanwise beam. `prop_fn(y)` returns a dict with EIxx, EIzz, GJ, x_sc, z_sc."""

    def __init__(self, span: float, prop_fn, supports: list[Support],
                 n_elem: int = 80, y_start: float = 0.0):
        self.span = float(span)
        self.y0 = float(y_start)
        self.prop_fn = prop_fn
        self.supports = sorted(supports, key=lambda s: s.y)
        self._n_elem = int(n_elem)
        self._extra_nodes: list[float] = []

    def require_nodes(self, ys) -> None:
        """Force mesh nodes at these spanwise stations (supports, point loads)."""
        self._extra_nodes.extend(float(y) for y in ys)

    def mesh(self) -> np.ndarray:
        base = np.linspace(self.y0, self.y0 + self.span, self._n_elem + 1)
        req = [s.y for s in self.supports] + self._extra_nodes
        y = np.unique(np.round(np.concatenate([base, np.array(req, float)]), 9))
        return y[(y >= self.y0 - 1e-9) & (y <= self.y0 + self.span + 1e-9)]

    # -- solve -------------------------------------------------------------
    def solve(self, pz, px=None, x_cp=None, z_cp=None,
              point_loads: list[PointLoad] | None = None,
              tq_extra=None) -> BeamResult:
        """`pz`, `px`, `x_cp`, `z_cp` are callables of y (N/m, N/m, m, m).

        `tq_extra(y)` adds a distributed torque [N.m/m] that carries no net
        force - used for pure centre-of-pressure load cases.
        """
        point_loads = list(point_loads or [])
        self.require_nodes([p.y for p in point_loads])
        y = self.mesh()
        n = len(y)
        L = np.diff(y)
        ym = 0.5 * (y[:-1] + y[1:])
        props = [self.prop_fn(v) for v in ym]
        EIxx = np.array([p["EIxx"] for p in props])
        EIzz = np.array([p["EIzz"] for p in props])
        GJ = np.array([p["GJ"] for p in props])
        x_sc = np.array([p["x_sc"] for p in props])
        z_sc = np.array([p.get("z_sc", 0.0) for p in props])

        # two-point Gauss average over each element rather than the midpoint
        # value: keeps the integrated load (and so the mount reactions) faithful
        # to the applied downforce to ~1e-9 instead of ~1e-5
        off = L / (2.0 * np.sqrt(3.0))
        avg = lambda f: 0.5 * (np.array([f(v) for v in ym - off], float)
                               + np.array([f(v) for v in ym + off], float))
        wz = avg(pz)
        wx = avg(px) if px else np.zeros(n - 1)
        xcp = np.array([x_cp(v) for v in ym], float) if x_cp else x_sc.copy()
        zcp = np.array([z_cp(v) for v in ym], float) if z_cp else z_sc.copy()
        # distributed torque about the shear centre
        tq = -(xcp - x_sc) * wz + (zcp - z_sc) * wx
        if tq_extra is not None:
            tq = tq + np.array([tq_extra(v) for v in ym], float)

        # --- bending in the vertical plane
        fix_w = [(self._node(y, s.y), 0) for s in self.supports]
        fix_th = [(self._node(y, s.y), 1) for s in self.supports if s.kind == "clamp"]
        pl_w = [(p.y, p.Fz) for p in point_loads if p.Fz]
        w, Rz = _solve_bending(y, EIxx, wz, fix_w + fix_th, pl_w)

        # --- bending in the chordwise plane
        sup_x = [s for s in self.supports if s.restrain_chordwise]
        if len(sup_x) >= 2 or any(s.kind == "clamp" for s in sup_x):
            fx = [(self._node(y, s.y), 0) for s in sup_x]
            fx += [(self._node(y, s.y), 1) for s in sup_x if s.kind == "clamp"]
            pl_x = [(p.y, p.Fx) for p in point_loads if p.Fx]
            u, Rx = _solve_bending(y, EIzz, wx, fx, pl_x)
        else:
            u, Rx = np.zeros(n), {}

        # --- torsion
        tw = [self._node(y, s.y) for s in self.supports if s.restrain_twist]
        pl_t = []
        for p in point_loads:
            arm_x = (p.x_app if p.x_app is not None else
                     float(np.interp(p.y, ym, x_sc))) - float(np.interp(p.y, ym, x_sc))
            arm_z = (p.z_app if p.z_app is not None else
                     float(np.interp(p.y, ym, z_sc))) - float(np.interp(p.y, ym, z_sc))
            t = -arm_x * p.Fz + arm_z * p.Fx
            if t:
                pl_t.append((p.y, t))
        if tw:
            phi, Rt = _solve_torsion(y, GJ, tq, tw, pl_t)
        else:
            phi, Rt = np.zeros(n), {}

        # --- internal loads by statics, using the solved reactions
        Mx, Sz = _integrate_moment(y, wz, [(p.y, p.Fz) for p in point_loads], Rz)
        Mz, _ = _integrate_moment(y, wx, [(p.y, p.Fx) for p in point_loads], Rx)
        T = _integrate_torque(y, tq, pl_t, Rt)

        free_tip = not any(abs(s.y - y[-1]) < 1e-9 for s in self.supports)
        resid = abs(Mx[-1]) / (np.abs(Mx).max() + 1e-12) if free_tip else 0.0

        reactions = []
        for s in self.supports:
            i = self._node(y, s.y)
            reactions.append(dict(label=s.label, y=s.y, Fz=Rz.get((i, 0), 0.0),
                                  Fx=Rx.get((i, 0), 0.0), My=Rt.get(i, 0.0),
                                  Mx=Rz.get((i, 1), 0.0)))
        return BeamResult(y, Mx, Mz, T, Sz, w, u, phi, reactions, resid)

    @staticmethod
    def _node(y: np.ndarray, yq: float) -> int:
        return int(np.argmin(np.abs(y - yq)))


# --------------------------------------------------------------------------
# element-level machinery
# --------------------------------------------------------------------------
def _bending_k(EI: float, L: float) -> np.ndarray:
    return EI / L ** 3 * np.array([
        [12.0, 6 * L, -12.0, 6 * L],
        [6 * L, 4 * L * L, -6 * L, 2 * L * L],
        [-12.0, -6 * L, 12.0, -6 * L],
        [6 * L, 2 * L * L, -6 * L, 4 * L * L]])


def _assemble_bending(y, EI, p):
    n = len(y)
    K = np.zeros((2 * n, 2 * n))
    f = np.zeros(2 * n)
    for e in range(n - 1):
        L = y[e + 1] - y[e]
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += _bending_k(EI[e], L)
        w = p[e]
        f[idx] += w * np.array([L / 2, L * L / 12, L / 2, -L * L / 12])
    return K, f


def _solve_bending(y, EI, p, fixed, point_loads):
    n = len(y)
    K, f = _assemble_bending(y, EI, p)
    for yq, F in point_loads:
        i = int(np.argmin(np.abs(y - yq)))
        f[2 * i] += F
    dofs = sorted({2 * i + k for i, k in fixed})
    free = np.setdiff1d(np.arange(2 * n), np.array(dofs, int))
    d = np.zeros(2 * n)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    R = K @ d - f
    reac = {(i, k): R[2 * i + k] for i, k in fixed}
    return d[0::2], reac


def _solve_torsion(y, GJ, tq, fixed_nodes, point_torques):
    n = len(y)
    K = np.zeros((n, n))
    f = np.zeros(n)
    for e in range(n - 1):
        L = y[e + 1] - y[e]
        k = GJ[e] / L
        K[e:e + 2, e:e + 2] += np.array([[k, -k], [-k, k]])
        f[e:e + 2] += tq[e] * L / 2.0
    for yq, t in point_torques:
        f[int(np.argmin(np.abs(y - yq)))] += t
    free = np.setdiff1d(np.arange(n), np.array(sorted(set(fixed_nodes)), int))
    d = np.zeros(n)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    R = K @ d - f
    return d, {i: R[i] for i in set(fixed_nodes)}


def _integrate_moment(y, p, point_loads, reactions):
    """M(y) = -integral of the shear of external forces to the left, plus any
    reaction moment from a clamped mount.

    Point loads and reactions make the shear discontinuous at a node, so the
    shear is tracked as the value just after each node and the element integral
    uses the limits on either side.  At a clamped station the internal moment
    itself jumps; the value reported is the limit from the side on which the
    beam continues.
    """
    n = len(y)
    jump = np.zeros(n)                       # concentrated transverse forces
    for yq, F in point_loads:
        if F:
            jump[int(np.argmin(np.abs(y - yq)))] += F
    for (i, k), R in reactions.items():
        if k == 0:
            jump[i] += R

    S = np.zeros(n)                          # shear just to the right of node i
    M = np.zeros(n)
    S[0] = jump[0]
    for e in range(n - 1):
        h = y[e + 1] - y[e]
        S_before = S[e] + p[e] * h           # shear just left of node e+1
        M[e + 1] = M[e] - 0.5 * (S[e] + S_before) * h
        S[e + 1] = S_before + jump[e + 1]
    for (i, k), R in reactions.items():
        if k == 1:                           # clamp reaction moment: a step
            if i < n - 1:
                M[y >= y[i] - 1e-12] += R    # beam continues to the right
            else:
                M[y > y[i] + 1e-12] += R
    return M, S


def _integrate_torque(y, tq, point_torques, reac_torques):
    n = len(y)
    acc = np.zeros(n)
    for e in range(n - 1):
        acc[e + 1] = acc[e] + tq[e] * (y[e + 1] - y[e])
    for yq, t in point_torques:
        i = int(np.argmin(np.abs(y - yq)))
        acc[y >= y[i] - 1e-12 if i < n - 1 else y > y[i] + 1e-12] += t
    for i, R in reac_torques.items():
        acc[y >= y[i] - 1e-12 if i < n - 1 else y > y[i] + 1e-12] += R
    return -acc
