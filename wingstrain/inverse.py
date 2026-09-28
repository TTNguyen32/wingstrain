"""Inverse model: measured strains -> downforce.

The forward model is linear in the applied load, so for a set of load *cases*
(a basis) the gauge strains are

    eps = A a,      A[i,k] = microstrain at gauge i per unit of load case k

The basis is written as the NOMINAL load case plus zero-net-force
perturbations, so the first coefficient IS the downforce and the rest are
nuisance parameters with physically meaningful priors:

    D_total    nominal load case, scaled to 1 N of total downforce
               (its own distributed load AND the bracket loads from any
                elements it carries, in the nominal proportion)
    dS_shape   elliptical minus nominal, distributed        net 0 N
    dA_asym    antisymmetric                                net 0 N  (yaw)
    dF_split   1 N moved between its own surface and the brackets   net 0 N
    dD_drag    extra drag at constant downforce             net 0 N  (L/D change)
    dP_cop     pure couple                                  net 0 N  (CoP shift)

A handful of gauges cannot separate all of those - the columns are close to
collinear - so the estimator is a MAP (prior-regularised) solution rather than
plain least squares.  That turns an ill-posed inverse into a well-posed one
whose reported sigma_D honestly includes the part of the error no amount of
signal conditioning will remove.

The product is one row of weights:

    D_hat [N] = sum_i w_i * eps_i [microstrain]

ready to program into the DAQ, plus the uncertainty that comes with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aero import build_loads, spanwise_shape
from .airfoil import transform
from .beam import PointLoad


@dataclass
class LoadBasis:
    """One unit load case."""
    name: str
    shape: object                 # callable y -> N/m of downforce per unit coeff
    net: float                    # net downforce [N] per unit coefficient
    shape_x: object = None        # callable y -> N/m of drag per unit coeff
    xc_cp: float = 0.32           # chord fraction where the distributed part acts
    couple: float = 0.0           # distributed torque [N.m/m] per unit coeff
    prior: float = 1.0e6          # 1-sigma prior on the coefficient
    point_loads: list = field(default_factory=list)   # per unit coefficient
    note: str = ""


def default_bases(wing, D_ref: float | None = None, shape_prior: float = 0.25,
                  asym_prior: float = 0.10, split_prior: float = 0.15,
                  cp_prior_xc: float = 0.05,
                  drag_prior: float = 0.20) -> list[LoadBasis]:
    """Build the nominal-plus-perturbations basis for this wing.

    Priors are 1-sigma, as a fraction of `D_ref` (except `cp_prior_xc`, which is
    a centre-of-pressure uncertainty in chord fractions).  They encode what you
    genuinely do not know run to run: how tip-loaded the wing is, how much yaw
    asymmetry there is, how the flaps share load into the brackets, and where
    the chordwise centre of pressure sits.
    """
    sp = wing.spec
    span, chord = sp.span, sp.chord
    if D_ref is None:
        D_ref = wing.solve(20.0, gauges=[]).D_total

    # --- nominal load case, scaled to exactly 1 N of total downforce
    L = build_loads(sp.aero, 20.0, span, sp.chord_at, 0.0, sp.area,
                    sp.angle_deg, sp.inverted, sp.x_ref)
    k = 1.0 / L.D_total
    nom_pls = [PointLoad(p.y, p.Fz * k, p.Fx * k, p.x_app, p.z_app, p.label)
               for p in L.point_loads]
    nom_dist = lambda y, k=k: -L.pz(y) * k          # N/m of downforce, >= 0
    f_dist = L.D_element / L.D_total                 # share carried by the skin

    y = np.linspace(0.0, span, 401)
    s = 2 * y / span - 1.0
    shape_nom = spanwise_shape(sp.aero.distribution, sp.aero.tip_loading, span)
    nomf = np.array([shape_nom(v) for v in y])
    ell = np.sqrt(np.clip(1 - s ** 2, 0.0, None))
    ell /= np.trapezoid(ell, y)
    asym = s / np.trapezoid(np.abs(s), y)

    def interp(f):
        f = np.asarray(f, float)
        return lambda yq: float(np.interp(yq, y, f))

    drag_ref = L.drag_total
    bases = [
        # flat prior on the quantity being measured: never shrink the answer
        LoadBasis("D_total", nom_dist, 1.0, shape_x=lambda y, k=k: L.px(y) * k,
                  xc_cp=sp.aero.xc_cp, prior=np.inf, point_loads=nom_pls,
                  note="nominal load path, 1 N total"),
        LoadBasis("dS_shape", interp((ell - nomf) * f_dist), 0.0,
                  xc_cp=sp.aero.xc_cp, prior=D_ref * shape_prior,
                  note="tip-loaded vs nominal"),
        LoadBasis("dA_asym", interp(asym), 0.0, xc_cp=sp.aero.xc_cp,
                  prior=D_ref * asym_prior, note="asymmetric (yaw)"),
        LoadBasis("dD_drag", lambda y: 0.0, 0.0,
                  shape_x=interp(nomf), xc_cp=sp.aero.xc_cp,
                  prior=drag_ref * drag_prior, note="drag / L-over-D change"),
    ]
    if nom_pls:
        # 1 N taken OFF the brackets (hence the sign flip) and put onto the
        # element's own surface, so the net force of this case is zero
        br_pls = [PointLoad(p.y, -p.Fz / (1.0 - f_dist) if f_dist < 1 else 0.0,
                            0.0, p.x_app, p.z_app, p.label + "_split")
                  for p in nom_pls]
        bases.append(LoadBasis("dF_split", interp(nomf), 0.0,
                               xc_cp=sp.aero.xc_cp, prior=D_ref * split_prior,
                               point_loads=br_pls,
                               note="load moved from brackets to skin"))
    bases.append(LoadBasis("dP_cop", lambda yq: 0.0, 0.0, xc_cp=sp.aero.xc_cp,
                           couple=1.0 / span,
                           prior=D_ref * cp_prior_xc * chord,
                           note="centre-of-pressure shift"))
    return bases


def influence_matrix(wing, gauges=None, bases: list[LoadBasis] | None = None):
    """A [n_gauge x n_basis], MICROSTRAIN per unit basis coefficient."""
    sp = wing.spec
    gauges = gauges if gauges is not None else sp.gauges
    bases = bases or default_bases(wing)
    A = np.zeros((len(gauges), len(bases)))
    for j, b in enumerate(bases):
        xcp = _cp_fn(sp, b.xc_cp, 0)
        zcp = _cp_fn(sp, b.xc_cp, 1)
        pz = lambda y, b=b: -b.shape(y)
        px = (lambda y, b=b: b.shape_x(y)) if b.shape_x else None
        tq = (lambda y, b=b: b.couple) if b.couple else None
        _, readings = wing.solve_raw(pz, px, xcp, zcp, b.point_loads or None,
                                     gauges, tq)
        A[:, j] = [r.microstrain for r in readings]
    return A, list(bases), list(gauges)


def _cp_fn(sp, xc: float, comp: int):
    def f(y):
        return float(transform(np.array([[xc, 0.0]]), sp.chord_at(y),
                               sp.angle_deg, sp.inverted, sp.x_ref)[0][comp])
    return f


@dataclass
class Calibration:
    """Downforce estimator plus its error budget.

    Weights come from a MAP (prior-regularised) inverse - that is what keeps a
    near-collinear problem well posed - and are then rescaled so the estimator
    reads exactly right for the nominal load case, removing the small shrinkage
    bias that ridge estimators otherwise carry.

    The budget is then explicit: each unresolved load case k contributes
    |w . A[:,k]| * prior_k newtons of possible error, and gauge noise
    contributes sqrt(sum (w_i sigma_i)^2).
    """
    A: np.ndarray
    bases: list
    gauges: list
    weights: np.ndarray            # N per microstrain
    cond: float
    sigma_noise: float             # [N]
    sigma_model: float             # [N]
    sigma_total: float             # [N]
    budget: list = field(default_factory=list)      # (name, newtons)
    warnings: list = field(default_factory=list)
    resolution: np.ndarray = field(default=None, repr=False)
    M: np.ndarray = field(default=None, repr=False)
    D_ref: float = 0.0

    def estimate(self, microstrain) -> float:
        """Downforce [N] from gauge readings [microstrain]."""
        return float(np.dot(self.weights, np.asarray(microstrain, float)))

    def fit(self, microstrain):
        """Full MAP fit -> (coefficients, D_hat, residual microstrain)."""
        eps = np.asarray(microstrain, float)
        a = self.M @ eps
        return a, self.estimate(eps), eps - self.A @ a

    @property
    def relative_error(self) -> float:
        return self.sigma_total / self.D_ref if self.D_ref else float("nan")

    def describe(self) -> str:
        terms = "  ".join(f"{w:+.4f}*{g.name}"
                          for w, g in zip(self.weights, self.gauges))
        lines = [
            f"Downforce estimator: {len(self.gauges)} channels, "
            f"{len(self.bases)} load cases, cond(A) = {self.cond:.0f}",
            f"  D [N] = {terms}     (strains in microstrain)",
            f"  1-sigma error {self.sigma_total:6.2f} N "
            f"= {100*self.relative_error:.1f}% of {self.D_ref:.0f} N",
            "  error budget:",
            f"    {'gauge noise':<26s} {self.sigma_noise:7.2f} N",
        ]
        for name, val in self.budget:
            lines.append(f"    {name:<26s} {val:7.2f} N")
        lines.append("  observability (1 = measured, 0 = taken on trust): "
                     + ", ".join(f"{b.name} {r:.2f}"
                                 for b, r in zip(self.bases, self.resolution)))
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def build_calibration(wing, gauges=None, bases=None, noise_ue=None,
                      D_ref: float | None = None, reject_thermal: bool = True,
                      temp_prior_K: float = 12.0, **basis_kw) -> Calibration:
    """Build the downforce estimator and its error budget for this setup.

    With `reject_thermal` (default) the common-mode thermal apparent strain is
    carried as an extra unknown, so the weights come out orthogonal to it as far
    as the channels allow - the multi-gauge equivalent of wiring a compensating
    gauge into the bridge.  It costs some noise performance and buys a large
    reduction in drift sensitivity; set it False to see the difference.
    """
    if D_ref is None:
        D_ref = wing.solve(20.0, gauges=[]).D_total
    bases = bases or default_bases(wing, D_ref, **basis_kw)
    A, bases, gauges = influence_matrix(wing, gauges, bases)
    if reject_thermal:
        tcol = np.array([getattr(g, "temp_apparent_ue_per_K", 0.0)
                         for g in gauges], float)
        if np.any(tcol != 0.0):
            A = np.hstack([A, tcol[:, None]])
            bases = list(bases) + [LoadBasis(
                "dT_thermal", lambda y: 0.0, 0.0, prior=temp_prior_K,
                note="thermal apparent strain, per K")]
    net = np.array([b.net for b in bases])
    sig = (np.array([g.noise_ue for g in gauges], float) if noise_ue is None
           else np.full(len(gauges), float(noise_ue)))
    sig = np.where(sig > 0, sig, 1e-6)
    pri = np.array([b.prior for b in bases], float)

    Ci = np.diag(1.0 / sig ** 2)
    Pi = np.diag(np.where(np.isfinite(pri), 1.0 / pri ** 2, 0.0))
    post = np.linalg.inv(A.T @ Ci @ A + Pi)
    M = post @ A.T @ Ci
    w = net @ M
    # rescale so the estimator is exact for the nominal load case
    gain = float(w @ A[:, 0])
    if abs(gain) < 1e-12:
        raise ValueError("these channels cannot see the nominal load case at all")
    w = w / gain
    M = M / gain

    sigma_noise = float(np.linalg.norm(w * sig))
    budget = []
    for k, b in enumerate(bases):
        if b.net != 0.0 or not np.isfinite(pri[k]):
            continue
        budget.append((f"{b.name} ({b.note})", abs(float(w @ A[:, k])) * pri[k]))
    sigma_model = float(np.sqrt(sum(v ** 2 for _, v in budget)))
    warn = []
    for name, val in budget:
        if val > 0.5 * D_ref:
            short = name.split(" (")[0]
            warn.append(f"{short} alone can shift the answer by {val:.0f} N "
                        f"({100*val/D_ref:.0f}% of the load): these channels "
                        f"cannot separate it from downforce.")
            if short == "dT_thermal":
                warn.append("  add a channel whose thermal-to-load ratio "
                            "differs - a Poisson or dummy-compensated bridge, "
                            "or gauges on surfaces of opposite strain sign.")
    sigma_total = float(np.hypot(sigma_noise, sigma_model))
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else np.inf
    return Calibration(A, bases, gauges, w, cond, sigma_noise, sigma_model,
                       sigma_total, budget, warn,
                       np.clip(np.diag(M @ A), 0.0, 1.0), M, D_ref)


def single_gauge_calibration(wing, V: float = 20.0):
    """The simple approach: one scalar N-per-microstrain per gauge."""
    res = wing.solve(V)
    return {r.gauge.name: dict(
        ue_per_N=r.microstrain / res.D_total,
        N_per_ue=res.D_total / r.microstrain if r.microstrain else np.inf,
        microstrain=r.microstrain, mV_per_V=r.mV_per_V) for r in res.readings}


def recommend_stations(wing, V: float = 20.0, xc: float | None = None,
                       surface: str = "bottom", clearance_depths: float = 2.5,
                       n: int = 121):
    """Score spanwise stations for a bending gauge.

    A good station has large strain per newton (signal), a small spanwise strain
    gradient (so a few mm of bonding error does not matter) and enough distance
    from a mount for the local load introduction to have died away - St Venant,
    a few section depths.
    """
    from .gauges import Gauge
    sp = wing.spec
    xc = sp.airfoil.x_max_thickness if xc is None else xc
    ys = np.linspace(0.005, sp.span - 0.005, n)
    res = wing.solve(V, gauges=[Gauge(f"p{i}", y, xc, surface)
                                for i, y in enumerate(ys)])
    sens = np.array([r.microstrain for r in res.readings]) / res.D_total
    grad = np.gradient(sens, ys)
    depth = res.props_mid["z_max"] - res.props_mid["z_min"]
    dist = np.min([np.abs(ys - s.y) for s in sp.supports], axis=0)
    ok = dist > clearance_depths * depth
    score = np.where(ok, np.abs(sens) / (1.0 + 0.01 * np.abs(grad)), 0.0)
    best = int(np.argmax(score))
    return dict(y=ys, sensitivity=sens, gradient=grad, allowed=ok, score=score,
                best_y=float(ys[best]), best_sensitivity=float(sens[best]),
                min_clearance=float(clearance_depths * depth), xc=xc,
                surface=surface, result=res)


def best_chordwise(wing, y: float, V: float = 20.0, surface: str = "bottom",
                   n: int = 60):
    """Strain per newton along the chord at one station -> where to bond."""
    from .gauges import Gauge
    sp = wing.spec
    xcs = np.linspace(max(sp.x_front + 0.02, 0.02), sp.x_rear - 0.02, n)
    res = wing.solve(V, gauges=[Gauge(f"c{i}", y, xc, surface)
                                for i, xc in enumerate(xcs)])
    sens = np.array([r.microstrain for r in res.readings]) / res.D_total
    i = int(np.argmax(np.abs(sens)))
    return dict(xc=xcs, sensitivity=sens, best_xc=float(xcs[i]),
                best_sensitivity=float(sens[i]), surface=surface, y=y)
