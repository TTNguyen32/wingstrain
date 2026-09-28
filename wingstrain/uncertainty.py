"""Monte Carlo error budget for the strain-to-downforce measurement.

The analytic budget in `inverse.Calibration` covers what the load itself can do.
This module adds everything else that makes the real number differ from the
predicted one: laminate thickness and modulus scatter, incidence error, where
the gauge actually ended up, gauge factor and amplifier gain, thermal apparent
strain, and electrical noise.

Two questions get separate answers, because they have different fixes:

  * SCALE error   - the calibration constant itself is wrong (wrong EI, gauge a
    few mm off, gauge factor tolerance).  This is systematic and it is removed
    by a static calibration: hang known weights on the wing and fit the gain.
  * RUN-TO-RUN error - noise, temperature, and the aerodynamic load moving
    around (distribution, load split, centre of pressure).  A static
    calibration does NOT remove these, and they set the floor on what the
    measurement can tell you.

Note that a uniform error in C_L or in air density cancels exactly: it changes
the true downforce and the strain in the same proportion.  Only *relative*
changes - shape, split, centre of pressure - actually hurt.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .beam import PointLoad
from .gauges import PylonGauge
from .laminate import Laminate, Ply
from .model import Wing


@dataclass
class Spread:
    """1-sigma spread of one error source (Gaussian unless `uniform`)."""
    value: float
    uniform: bool = False

    def draw(self, rng) -> float:
        if self.uniform:
            return rng.uniform(-self.value, self.value)
        return rng.normal(0.0, self.value)


DEFAULT_SOURCES: dict[str, Spread] = {
    # --- structure and model (scale errors, removable by static calibration)
    "skin_thickness_rel": Spread(0.05),      # cured ply thickness scatter
    "skin_modulus_rel": Spread(0.05),        # fibre volume fraction scatter
    "incidence_deg": Spread(0.30),           # flap/incidence setting repeatability
    # --- installation (also scale errors)
    "gauge_y_m": Spread(0.003),              # spanwise bonding position
    "gauge_xc": Spread(0.010),               # chordwise bonding position
    "gauge_angle_deg": Spread(2.0),          # grid alignment
    "gauge_factor_rel": Spread(0.010),       # gauge factor tolerance
    "gain_rel": Spread(0.005),               # amplifier / excitation gain
    # --- run to run (NOT removable by static calibration)
    "noise": Spread(1.0),                    # multiplier on each gauge noise_ue
    "delta_T_K": Spread(12.0),               # temperature swing vs zeroing
    "tip_loading": Spread(0.12),             # spanwise distribution shape
    "carried_fraction_rel": Spread(0.15),    # flap/main load split
    "xc_cp": Spread(0.04),                   # chordwise centre of pressure
}

SCALE_SOURCES = ("skin_thickness_rel", "skin_modulus_rel", "incidence_deg",
                 "gauge_y_m", "gauge_xc", "gauge_angle_deg",
                 "gauge_factor_rel", "gain_rel")


def scale_laminate(lam: Laminate, E_factor: float = 1.0,
                   t_factor: float = 1.0) -> Laminate:
    plies = [Ply(p.name, p.E1 * E_factor, p.E2 * E_factor, p.G12 * E_factor,
                 p.nu12, p.thickness * t_factor, p.rho,
                 p.strain_allow_t, p.strain_allow_c) for p in lam.plies]
    return Laminate(list(lam.angles), plies, lam.name)


def perturbed_wing(wing: Wing, draws: dict, gauges: list) -> tuple[Wing, list]:
    """Apply one Monte Carlo draw to the wing and its gauges."""
    spec = copy.deepcopy(wing.spec)
    Ef = 1.0 + draws.get("skin_modulus_rel", 0.0)
    tf = 1.0 + draws.get("skin_thickness_rel", 0.0)
    spec.skin = scale_laminate(spec.skin, Ef, tf)
    if spec.lower_skin is not None:
        spec.lower_skin = scale_laminate(spec.lower_skin, Ef, tf)
    if spec.web is not None:
        spec.web = scale_laminate(spec.web, Ef, tf)
    spec.angle_deg += draws.get("incidence_deg", 0.0)
    spec.aero.tip_loading = float(np.clip(
        spec.aero.tip_loading + draws.get("tip_loading", 0.0), 0.2, 1.0))
    spec.aero.xc_cp += draws.get("xc_cp", 0.0)
    cf = 1.0 + draws.get("carried_fraction_rel", 0.0)
    for el in spec.aero.carried:
        el.CL_S *= cf
        el.CD_S *= cf

    gs = []
    for g in gauges:
        g2 = copy.deepcopy(g)
        if not isinstance(g2, PylonGauge):
            g2.y = float(np.clip(g2.y + draws.get("gauge_y_m", 0.0),
                                 0.002, spec.span - 0.002))
            g2.xc = float(np.clip(g2.xc + draws.get("gauge_xc", 0.0),
                                  spec.x_front + 0.02, spec.x_rear - 0.02))
            g2.misalignment_deg += draws.get("gauge_angle_deg", 0.0)
        gs.append(g2)
    spec.gauges = gs
    return Wing(spec), gs


def static_load_case(span: float, total_N: float = 150.0,
                     stations=(0.12, 0.38, 0.62, 0.88), xc: float = 0.32):
    """Known weights hung on the wing for a bench calibration."""
    return [dict(y=f * span, N=total_N / len(stations), xc=xc)
            for f in stations], total_N


def _static_reading(wing: Wing, weights, gauges, calib,
                    instrument_gain: float = 1.0) -> float:
    """Estimator output when the known weights are hung on THIS wing."""
    from .airfoil import transform
    sp = wing.spec
    pls = []
    for w in weights:
        p = transform(np.array([[w["xc"], 0.0]]), sp.chord_at(w["y"]),
                      sp.angle_deg, sp.inverted, sp.x_ref)[0]
        pls.append(PointLoad(w["y"], -w["N"], 0.0, float(p[0]), float(p[1]),
                             "cal_weight"))
    _, readings = wing.solve_raw(lambda y: 0.0, None, lambda y: 0.0,
                                 lambda y: 0.0, pls, gauges)
    eps = np.array([r.microstrain for r in readings]) * instrument_gain
    return float(calib.weights @ eps)


def static_gain(nominal_wing: Wing, real_wing: Wing, weights, gauges_nominal,
                gauges_real, calib, instrument_gain: float = 1.0) -> float:
    """Gain correction a static hang test yields for a wing that is not nominal.

    The test compares the strain the nominal model PREDICTS for a known static
    load with the strain the real wing actually produces under that same load.
    The ratio corrects stiffness, gauge-position and instrumentation errors
    without assuming anything about the aerodynamic load path - which matters,
    because hanging weights on the main plane does not reproduce how the flaps
    feed load in, so the raw hang-test reading is NOT a downforce calibration
    on its own.
    """
    pred = _static_reading(nominal_wing, weights, gauges_nominal, calib, 1.0)
    meas = _static_reading(real_wing, weights, gauges_real, calib,
                           instrument_gain)
    return pred / meas if abs(meas) > 1e-9 else np.nan


@dataclass
class MCResult:
    V: float
    n: int
    D_true_nominal: float
    errors_raw: np.ndarray          # relative error, as-modelled calibration
    errors_static: np.ndarray       # relative error after a static hang test
    tornado: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def _stats(self, e):
        return dict(bias=float(np.mean(e)), sigma=float(np.std(e)),
                    p95=float(np.percentile(np.abs(e), 95)))

    def report(self) -> str:
        r, s = self._stats(self.errors_raw), self._stats(self.errors_static)
        D = self.D_true_nominal
        lines = [
            f"Monte Carlo, {self.n} samples at V = {self.V:.1f} m/s "
            f"(nominal downforce {D:.1f} N)",
            f"  as-modelled calibration : bias {100*r['bias']:+6.2f} %   "
            f"1-sigma {100*r['sigma']:6.2f} % ({r['sigma']*D:6.1f} N)   "
            f"95th pct |err| {100*r['p95']:6.2f} %",
            f"  after static hang test  : bias {100*s['bias']:+6.2f} %   "
            f"1-sigma {100*s['sigma']:6.2f} % ({s['sigma']*D:6.1f} N)   "
            f"95th pct |err| {100*s['p95']:6.2f} %",
        ]
        if self.tornado:
            lines.append("  contribution of each source in isolation "
                         "(1-sigma of relative error):")
            for k, v in sorted(self.tornado.items(), key=lambda kv: -kv[1]):
                bar = "#" * max(1, int(round(v * 400)))
                tag = " [static-calibratable]" if k in SCALE_SOURCES else ""
                lines.append(f"    {k:<22s} {100*v:6.2f} %  {bar}{tag}")
        return "\n".join(lines)


def monte_carlo(wing: Wing, calib, V: float = 20.0, n: int = 300,
                gauges=None, sources: dict | None = None, seed: int = 0,
                tornado: bool = True, static_weights=None) -> MCResult:
    """Propagate every error source through the full forward + inverse chain."""
    gauges = gauges if gauges is not None else calib.gauges
    src = dict(DEFAULT_SOURCES if sources is None else sources)
    rng = np.random.default_rng(seed)
    D_nom = wing.solve(V, gauges=[]).D_total
    weights, total_N = (static_load_case(wing.spec.span)
                        if static_weights is None else static_weights)

    def run(active: tuple, m: int, rng) -> tuple[np.ndarray, np.ndarray]:
        e_raw, e_stat = [], []
        for _ in range(m):
            draws = {k: src[k].draw(rng) for k in active if k in src}
            pw, pg = perturbed_wing(wing, draws, gauges)
            res = pw.solve(V, gauges=pg)
            eps = np.array([r.microstrain for r in res.readings])
            ig = ((1.0 + draws.get("gauge_factor_rel", 0.0))
                  * (1.0 + draws.get("gain_rel", 0.0)))
            # instrumentation effects act on the reading, not the structure
            noisy = eps * ig
            if "noise" in draws:
                noisy = noisy + np.array([rng.normal(0.0, g.noise_ue)
                                          for g in pg])
            dT = draws.get("delta_T_K", 0.0)
            if dT:
                noisy = noisy + np.array([g.temp_apparent_ue_per_K * dT
                                          for g in pg])
            D_hat = calib.estimate(noisy)
            D_true = res.D_total
            e_raw.append(D_hat / D_true - 1.0)
            k = static_gain(wing, pw, weights, gauges, pg, calib, ig)
            e_stat.append(D_hat * k / D_true - 1.0)
        return np.array(e_raw), np.array(e_stat)

    raw, stat = run(tuple(src), n, rng)
    tor = {}
    if tornado:
        m = max(30, n // 4)
        for k in src:
            r, _ = run((k,), m, np.random.default_rng(seed + hash(k) % 10000))
            tor[k] = float(np.std(r))
    return MCResult(V, n, D_nom, raw, stat, tor, src)
