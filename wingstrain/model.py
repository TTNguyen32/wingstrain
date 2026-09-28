"""The wing model: geometry + layup + mounts + aero + gauges -> strains.

    spec = WingSpec(...)
    wing = Wing(spec)
    res  = wing.solve(V=15.0)
    print(res.report())

Everything downstream (calibration, error budget, sweeps) uses `Wing.solve`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import aero as aeromod
from .airfoil import Airfoil
from .beam import Beam, BeamResult, PointLoad, Support
from .gauges import Gauge, PylonGauge, Reading, bridge_output, resolve
from .laminate import Laminate
from .section import Section, box_section


@dataclass
class WingSpec:
    """Complete definition of one instrumented wing element."""
    # -- planform
    span: float = 1.0                  # tip to tip [m]
    chord: float = 0.300               # chord at mid-span [m]
    taper_ratio: float = 1.0           # tip chord / mid-span chord
    angle_deg: float = 8.0             # incidence, +ve = trailing edge down
    inverted: bool = True              # downforce wing
    airfoil: Airfoil = field(default_factory=lambda: Airfoil.naca4("4412"))
    x_ref: float = 0.25                # datum for section x, chord fraction

    # -- structure
    skin: Laminate = field(default_factory=lambda: Laminate.symmetric(
        [0, 45, -45], "twill_200", "skin"))
    lower_skin: Laminate | None = None
    web: Laminate | None = field(default_factory=lambda: Laminate.build(
        [45, -45, -45, 45], "twill_200", "web"))
    x_front: float = 0.0               # structural box starts (0 = wraps the LE)
    x_rear: float = 0.65               # structural box ends, web here
    caps: list = field(default_factory=list)
    include_trailing_skin: bool = False
    n_skin: int = 72

    # -- mounts
    supports: list = field(default_factory=lambda: [Support(0.25, "pin", label="pylon_L"),
                                                    Support(0.75, "pin", label="pylon_R")])

    # -- aero
    aero: aeromod.AeroSpec = field(default_factory=aeromod.AeroSpec)

    # -- instrumentation
    gauges: list = field(default_factory=list)

    n_elem: int = 80

    def chord_at(self, y) -> float:
        """Linear taper, symmetric about mid-span."""
        ym = self.span / 2.0
        f = np.abs(np.asarray(y, float) - ym) / ym
        return float(self.chord * (1.0 + (self.taper_ratio - 1.0) * f))

    @property
    def area(self) -> float:
        y = np.linspace(0.0, self.span, 201)
        return float(np.trapezoid([self.chord_at(v) for v in y], y))

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.area


@dataclass
class Result:
    V: float
    loads: aeromod.AeroLoads
    beam: BeamResult
    readings: list
    spec: WingSpec
    props_mid: dict
    section_mid: Section
    max_strain: float
    max_strain_y: float
    reserve_factor: float
    tip_deflection: float
    tip_twist_deg: float
    aoa_change_deg: float

    @property
    def D_total(self) -> float:
        return self.loads.D_total

    def reading(self, name: str) -> Reading:
        for r in self.readings:
            if r.gauge.name == name:
                return r
        raise KeyError(name)

    def sensitivity(self, name: str) -> float:
        """Microstrain per newton of TOTAL wing downforce."""
        return self.reading(name).microstrain / self.D_total

    def report(self) -> str:
        L = self.loads
        s = self.spec
        out = [
            f"=== Wing strain model @ V = {self.V:.1f} m/s "
            f"({self.V*3.6:.0f} km/h), q = {L.q:.1f} Pa ===",
            f"instrumented element: span {s.span*1e3:.0f} mm, chord "
            f"{s.chord*1e3:.0f} mm, AR {s.aspect_ratio:.2f}, "
            f"incidence {s.angle_deg:.1f} deg, area {s.area:.3f} m2",
            f"downforce: element {L.D_element:6.1f} N + carried "
            f"{L.D_carried:6.1f} N = {L.D_total:6.1f} N   (drag {L.drag_total:.1f} N)",
            "",
            self.section_mid.summary(),
            f"  skin: {s.skin.describe()}",
            "",
            f"mounts: " + ", ".join(
                f"{r['label']} @ {r['y']*1e3:.0f} mm  Fz {r['Fz']:+7.1f} N"
                + (f", My {r['My']:+6.2f} N.m" if abs(r['My']) > 1e-9 else "")
                for r in self.beam.reactions),
            f"peak bending moment  {self.beam.Mx[np.argmax(np.abs(self.beam.Mx))]:+7.2f} N.m at y = "
            f"{self.beam.y[np.argmax(np.abs(self.beam.Mx))]*1e3:.0f} mm "
            f"(+ve = tension on the top skin)",
            f"peak torque          {np.abs(self.beam.T).max():7.2f} N.m",
            f"tip deflection       {self.tip_deflection*1e3:+7.2f} mm   "
            f"tip twist {self.tip_twist_deg:+.3f} deg "
            f"(aero incidence change {self.aoa_change_deg:+.3f} deg)",
            f"peak skin strain     {self.max_strain*1e6:7.0f} ue at y = "
            f"{self.max_strain_y*1e3:.0f} mm -> reserve factor "
            f"{self.reserve_factor:.1f} on first-ply strain",
            f"equilibrium residual {self.beam.equilibrium_residual:.2e}",
            "",
            "gauges:  " + f"{'':>12s}  {'station':>12s}  {'':>9s} {'':>10s}"
            f"{'strain':>12s} {'bridge':>12s}   sensitivity",
        ]
        for r in self.readings:
            out.append(f"  {r}   {r.microstrain/self.D_total:+.4f} ue/N"
                       f"  ({self.D_total/max(abs(r.microstrain),1e-9):6.1f} N/ue)")
        return "\n".join(out)


class Wing:
    def __init__(self, spec: WingSpec):
        self.spec = spec
        self._sec_cache: dict = {}

    # -- section handling --------------------------------------------------
    def section_at(self, y: float) -> Section:
        s = self.spec
        c = s.chord_at(y)
        key = round(c, 6)
        if key not in self._sec_cache:
            self._sec_cache[key] = box_section(
                s.airfoil, c, s.skin, s.web, s.x_front, s.x_rear, s.angle_deg,
                s.inverted, s.n_skin, s.lower_skin, s.caps,
                s.include_trailing_skin, s.x_ref)
        return self._sec_cache[key]

    def prop_fn(self, y: float) -> dict:
        p = dict(self.section_at(y).props())
        p["z_sc"] = p["z_ea"]        # vertical shear-centre offset ~ centroid
        return p

    # -- solve -------------------------------------------------------------
    def _read_pylon(self, pg: PylonGauge, br) -> Reading:
        """Strain in a mount strut from its reaction."""
        rc = next((r for r in br.reactions if r["label"] == pg.support_label),
                  br.reactions[0])
        e = pg.strain(rc["Fz"], rc["Fx"])
        dR, ratio, V_out = bridge_output(pg, e, 0.0)
        return Reading(pg, e, 0.0, 0.0, e, e * 1e6, dR, ratio * 1e3,
                       V_out * 1e3, 0.0, 0.0)

    def solve_raw(self, pz, px=None, x_cp=None, z_cp=None, point_loads=None,
                  gauges: list | None = None, tq_extra=None):
        """Solve one arbitrary load case. Returns (BeamResult, [Reading]).

        Used directly by the inverse model, which needs unit load cases rather
        than an aero specification.
        """
        s = self.spec
        gauges = gauges if gauges is not None else s.gauges
        beam = Beam(s.span, self.prop_fn, s.supports, s.n_elem)
        beam.require_nodes([g.y for g in gauges
                            if not isinstance(g, PylonGauge)])
        br = beam.solve(pz, px, x_cp, z_cp, point_loads, tq_extra)
        readings = [self._read_pylon(g, br) if isinstance(g, PylonGauge)
                    else self._read(g, br) for g in gauges]
        return br, readings

    def solve(self, V: float, aero: aeromod.AeroSpec | None = None,
              gauges: list | None = None) -> Result:
        s = self.spec
        spec_aero = aero or s.aero
        gauges = gauges if gauges is not None else s.gauges
        loads = aeromod.build_loads(spec_aero, V, s.span, s.chord_at, 0.0,
                                    s.area, s.angle_deg, s.inverted, s.x_ref)
        br, readings = self.solve_raw(loads.pz, loads.px, loads.x_cp,
                                      loads.z_cp, loads.point_loads, gauges)

        # worst-case skin strain over the span, for a structural sanity check
        i_worst = int(np.argmax(np.abs(br.Mx)))
        y_w = float(br.y[i_worst])
        sec_w = self.section_at(y_w)
        eps_all = np.array([sec_w.eps_yy(*sg.mid, br.Mx[i_worst], br.Mz[i_worst])
                            for sg in sec_w.all_walls])
        eps_max = float(np.abs(eps_all).max())
        at, ac = s.skin.min_strain_allowable()
        rf = min(at / max(eps_all.max(), 1e-12), ac / max(-eps_all.min(), 1e-12))

        mid = s.span / 2.0
        sec_mid = self.section_at(mid)
        tip_defl = float(br.w[0] if abs(br.w[0]) > abs(br.w[-1]) else br.w[-1])
        twist = float(np.degrees(br.phi[np.argmax(np.abs(br.phi))]))
        return Result(V, loads, br, readings, s, sec_mid.props(), sec_mid,
                      eps_max, y_w, rf, tip_defl, twist, twist)

    # -- gauge strain ------------------------------------------------------
    def _read(self, g: Gauge, br: BeamResult) -> Reading:
        s = self.spec
        # grid-length averaging along the span (moment gradients are steep
        # near the mounts; a 3 mm grid samples a finite slice of the span)
        ang = np.radians(g.total_angle)
        half = 0.5 * g.grid_length * abs(np.cos(ang))
        ys = np.clip(g.y + np.linspace(-half, half, 5 if half > 1e-6 else 1),
                     0.0, s.span)
        acc = np.zeros(3)
        for yq in ys:
            acc += self._strain_state(g, yq, br)
        e_span, e_chord, gam = acc / len(ys)

        sec = self.section_at(g.y)
        i = sec.locate(g.xc, g.surface)
        x, z = sec.loop[i].mid
        e_grid = resolve(e_span, e_chord, gam, g.total_angle)
        e_trans = resolve(e_span, e_chord, gam, g.total_angle + 90.0)
        dR, ratio, V_out = bridge_output(g, e_grid, e_trans)
        return Reading(g, e_span, e_chord, gam, e_grid, e_grid * 1e6,
                       dR, ratio * 1e3, V_out * 1e3, float(z), float(x))

    def _strain_state(self, g: Gauge, y: float, br: BeamResult):
        """(eps_span, eps_chord, gamma) on the skin surface at station y."""
        sec = self.section_at(y)
        i = sec.locate(g.xc, g.surface)
        seg = sec.loop[i]
        Mx = float(np.interp(y, br.y, br.Mx))
        Mz = float(np.interp(y, br.y, br.Mz))
        T = float(np.interp(y, br.y, br.T))
        Sz = float(np.interp(y, br.y, br.Sz))
        e_span = float(sec.eps_yy(seg.mid[0], seg.mid[1], Mx, Mz))
        # transverse strain on a beam surface: Poisson contraction of the skin
        e_chord = -seg.lam.nu_xy * e_span
        q = sec.shear_flow(i, T, -Sz)          # dMx/dy = -Sz
        gam = q / (seg.G * seg.t)
        return np.array([e_span, e_chord, gam])

    # -- maps --------------------------------------------------------------
    def strain_map(self, V: float, surface: str = "top", n_y: int = 60,
                   n_xc: int = 40, aero=None):
        """Spanwise x chordwise map of spanwise surface strain [microstrain]."""
        s = self.spec
        res = self.solve(V, aero, gauges=[])
        br = res.beam
        ys = np.linspace(0.0, s.span, n_y)
        xcs = np.linspace(max(s.x_front, 0.01), s.x_rear - 0.01, n_xc)
        E = np.zeros((n_xc, n_y))
        for j, y in enumerate(ys):
            sec = self.section_at(y)
            Mx = float(np.interp(y, br.y, br.Mx))
            Mz = float(np.interp(y, br.y, br.Mz))
            for k, xc in enumerate(xcs):
                seg = sec.loop[sec.locate(xc, surface)]
                E[k, j] = 1e6 * float(sec.eps_yy(seg.mid[0], seg.mid[1], Mx, Mz))
        return ys, xcs, E, res
