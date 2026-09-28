"""Verification against closed-form results.

Run standalone:   python tests/test_validation.py
or under pytest:  pytest tests/
Every number below has a textbook answer; the model has to reproduce it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wingstrain.airfoil import Airfoil
from wingstrain.beam import Beam, PointLoad, Support
from wingstrain.gauges import Gauge, resolve
from wingstrain.inverse import build_calibration
from wingstrain.laminate import Laminate, Ply
from wingstrain.presets import baseline
from wingstrain.section import Section, Segment

TOL = 1e-6


def _close(a, b, tol=TOL, what=""):
    err = abs(a - b) / max(abs(b), 1e-12)
    assert err < tol, f"{what}: got {a!r}, expected {b!r} (rel err {err:.3e})"


# ---------------------------------------------------------------- laminate
def test_laminate_unidirectional():
    """A 0-degree UD stack must return its own fibre-direction modulus."""
    lam = Laminate.build([0, 0, 0, 0], "ud_t700")
    _close(lam.Ex, 135e9, 1e-9, "UD Ex")
    _close(lam.nu_xy, 0.28, 1e-9, "UD nu")
    _close(lam.thickness, 4 * 0.15e-3, 1e-12, "thickness")


def test_laminate_pm45_is_shear_stiff():
    """+-45 fabric: low axial stiffness, high shear stiffness."""
    pm = Laminate.build([45, -45, -45, 45], "twill_200")
    ud = Laminate.build([0, 0, 0, 0], "twill_200")
    assert pm.Gxy > 4 * ud.Gxy, "pm45 should be far stiffer in shear"
    assert pm.Ex < 0.4 * ud.Ex, "pm45 should be far softer axially"


def test_laminate_quasi_isotropic():
    """A quasi-isotropic stack has Ex = Ey and G = E/(2(1+nu))."""
    q = Laminate.symmetric([0, 45, -45, 90], "ud_t700")
    m = q.moduli()
    _close(m["Ex"], m["Ey"], 1e-6, "QI isotropy")
    _close(m["Gxy"], m["Ex"] / (2 * (1 + m["nu_xy"])), 1e-6, "QI shear modulus")


# ----------------------------------------------------------------- section
def _rect_box(w=0.20, h=0.05, t=1.0e-3, E=70e9, G=26.9e9, n=200):
    lam = Laminate([0.0], [Ply("iso", E, E, G, 0.3, t)], "iso")
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    segs = []
    for i in range(4):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % 4])
        for k in range(n):
            segs.append(Segment(a + (b - a) * k / n, a + (b - a) * (k + 1) / n,
                                lam, "wall", 0.0))
    return Section(segs, [], [], 1.0, "box"), (w, h, t, E, G)


def test_box_section_properties():
    """Thin-walled rectangular box against the closed-form I and J."""
    sec, (w, h, t, E, G) = _rect_box()
    p = sec.props()
    _close(p["EIxx"], E * (t * h ** 3 / 6 + w * t * h ** 2 / 2), 1e-6, "EIxx")
    _close(p["EIzz"], E * (t * w ** 3 / 6 + h * t * w ** 2 / 2), 1e-6, "EIzz")
    _close(p["GJ"], G * 2 * t * w ** 2 * h ** 2 / (w + h), 1e-6, "GJ (Bredt)")
    _close(p["A_enc"], w * h, 1e-9, "enclosed area")
    assert abs(p["EIxz"]) < 1e-6, "symmetric box must have no product of inertia"
    assert abs(p["x_sc"]) < 1e-9, "shear centre of a symmetric box is central"


def test_box_bending_strain():
    """eps = M z / EI on a symmetric section."""
    sec, (w, h, t, E, G) = _rect_box()
    M = 50.0
    _close(sec.eps_yy(0.0, h / 2, M, 0.0), M * (h / 2) / sec.props()["EIxx"],
           1e-6, "surface strain")
    _close(sec.eps_yy(0.0, -h / 2, M, 0.0), -M * (h / 2) / sec.props()["EIxx"],
           1e-6, "opposite surface")


def test_shear_centre_moves_with_web_stiffness():
    """Stiffening one web must pull the shear centre towards it."""
    af = Airfoil.naca4("0012")
    from wingstrain.section import box_section
    thin = Laminate.build([45, -45], "twill_200")
    thick = Laminate.build([45, -45] * 4, "twill_200")
    skin = Laminate.symmetric([45, 0], "twill_200")
    a = box_section(af, 0.3, skin, thin, x_rear=0.6, inverted=False)
    b = box_section(af, 0.3, skin, thick, x_rear=0.6, inverted=False)
    assert b.props()["x_sc"] > a.props()["x_sc"], "stiffer rear web -> x_sc aft"


# -------------------------------------------------------------------- beam
def _beam(EI=2000.0, GJ=1000.0, L=1.0, supports=None, n=80):
    pf = lambda y: dict(EIxx=EI, EIzz=EI, GJ=GJ, x_sc=0.0, z_sc=0.0)
    return Beam(L, pf, supports, n)


def test_cantilever_tip_load():
    EI, L, P = 2000.0, 1.0, 100.0
    r = _beam(EI, supports=[Support(0.0, "clamp")]).solve(
        lambda y: 0.0, point_loads=[PointLoad(L, Fz=-P)])
    _close(r.w[-1], -P * L ** 3 / (3 * EI), 1e-9, "tip deflection")
    _close(r.Mx[0], P * L, 1e-9, "root moment")
    assert abs(r.Mx[-1]) < 1e-9 * P * L, "free end must carry no moment"


def test_cantilever_udl():
    EI, L, w0 = 2000.0, 1.0, -200.0
    r = _beam(EI, supports=[Support(L, "clamp")]).solve(lambda y: w0)
    _close(r.w[0], w0 * L ** 4 / (8 * EI), 1e-9, "tip deflection")
    _close(r.Mx[-1], abs(w0) * L ** 2 / 2, 1e-9, "root moment")


def test_simply_supported_udl():
    EI, L, w0 = 2000.0, 1.0, -200.0
    r = _beam(EI, supports=[Support(0.0), Support(L)]).solve(lambda y: w0)
    _close(float(np.interp(0.5, r.y, r.w)), 5 * w0 * L ** 4 / (384 * EI), 1e-9,
           "mid-span deflection")
    _close(float(np.interp(0.5, r.y, r.Mx)), -abs(w0) * L ** 2 / 8, 1e-9,
           "mid-span moment")
    _close(r.reactions[0]["Fz"], abs(w0) * L / 2, 1e-9, "reaction")


def test_overhanging_beam():
    """Two mounts with overhangs: the classic wing-on-pylons case."""
    L, a, w0 = 1.0, 0.2, -200.0
    r = _beam(supports=[Support(a), Support(L - a)], n=120).solve(lambda y: w0)
    _close(float(np.interp(a, r.y, r.Mx)), abs(w0) * a ** 2 / 2, 1e-9,
           "moment over the mount")
    _close(float(np.interp(0.5, r.y, r.Mx)),
           -(abs(w0) * (L - 2 * a) ** 2 / 8 - abs(w0) * a ** 2 / 2), 1e-9,
           "mid-span moment")


def test_two_span_indeterminate():
    """Two equal spans on three pins: R_centre = 1.25 w l, M = w l^2 / 8."""
    L, w0 = 1.0, -200.0
    r = _beam(supports=[Support(0.0), Support(0.5), Support(1.0)], n=100).solve(
        lambda y: w0)
    _close(r.reactions[1]["Fz"], 1.25 * abs(w0) * 0.5, 1e-6, "centre reaction")
    _close(float(np.interp(0.5, r.y, r.Mx)), abs(w0) * 0.25 / 8, 1e-6,
           "moment at the centre support")


def test_torsion_cantilever():
    GJ, L, t0 = 1000.0, 1.0, 50.0
    r = _beam(GJ=GJ, supports=[Support(0.0, "clamp")]).solve(
        lambda y: -t0, x_cp=lambda y: 1.0)
    _close(r.T[0], t0 * L, 1e-9, "root torque")
    _close(r.phi[-1], t0 * L * L / (2 * GJ), 1e-9, "tip twist")
    assert abs(r.T[-1]) < 1e-9 * t0 * L, "free end must carry no torque"


def test_global_equilibrium():
    """Mount reactions must sum to the total applied downforce."""
    w = baseline()
    res = w.solve(20.0)
    _close(sum(r["Fz"] for r in res.beam.reactions), res.D_total, 1e-5,
           "sum of reactions")
    assert res.beam.equilibrium_residual < 1e-6, "free tips carry no moment"


# ------------------------------------------------------------------ gauges
def test_strain_transformation():
    """Mohr's circle: a 45-degree grid reads half the shear."""
    _close(resolve(100.0, -30.0, 0.0, 0.0), 100.0, 1e-12, "0 deg")
    _close(resolve(100.0, -30.0, 0.0, 90.0), -30.0, 1e-12, "90 deg")
    _close(resolve(0.0, 0.0, 80.0, 45.0), 40.0, 1e-12, "45 deg on pure shear")
    _close(resolve(0.0, 0.0, 80.0, -45.0), -40.0, 1e-12, "-45 deg on pure shear")


def test_bridge_output():
    """Quarter bridge: V/V = GF eps / 4."""
    from wingstrain.gauges import bridge_output
    g = Gauge("g", 0.5, gauge_factor=2.0, bridge="quarter", excitation_V=5.0)
    dR, ratio, V = bridge_output(g, 1000e-6)
    _close(dR, 2.0 * 1000e-6, 1e-12, "dR/R")
    _close(ratio, 2.0 * 1000e-6 / 4, 1e-12, "V/V")
    _close(V, 5.0 * 2.0 * 1000e-6 / 4, 1e-12, "output volts")


def test_strain_scales_with_dynamic_pressure():
    """Everything is linear in q, so strain must go as V^2."""
    w = baseline()
    a, b = w.solve(10.0), w.solve(20.0)
    _close(b.reading("B1_top").microstrain / a.reading("B1_top").microstrain,
           4.0, 1e-9, "V^2 scaling")


# ------------------------------------------------------- custom aerofoils
def _write_formats(tmp):
    """The same section written the four ways coordinate files turn up."""
    ref = Airfoil.naca4("6412", 120)
    xc = (1 - np.cos(np.linspace(0, np.pi, 60))) / 2
    f = {}
    f["selig"] = "my section\n" + "\n".join(
        f"{x:.6f} {ref.z_upper(x):.6f}" for x in xc[::-1]) + "\n" + "\n".join(
        f"{x:.6f} {ref.z_lower(x):.6f}" for x in xc[1:])
    f["lednicer"] = "my section\n   60.   60.\n\n" + "\n".join(
        f"{x:.6f} {ref.z_upper(x):.6f}" for x in xc) + "\n\n" + "\n".join(
        f"{x:.6f} {ref.z_lower(x):.6f}" for x in xc)
    f["mm_comma"] = "\n".join(
        f"{40+250*x:.4f},{18+250*ref.z_upper(x):.4f}" for x in xc[::-1]) + "\n" + \
        "\n".join(f"{40+250*x:.4f},{18+250*ref.z_lower(x):.4f}" for x in xc[1:])
    f["lower_first"] = "\n".join(
        f"{x:.6f} {ref.z_lower(x):.6f}" for x in xc) + "\n" + "\n".join(
        f"{x:.6f} {ref.z_upper(x):.6f}" for x in xc)
    return ref, f


def test_custom_profile_formats_agree():
    """Selig, Lednicer, millimetre/comma and reversed order must all agree."""
    ref, files = _write_formats(None)
    xc = np.linspace(0.02, 0.98, 40)
    cam = lambda a: max((a.z_upper(x) + a.z_lower(x)) / 2 for x in xc)
    for key, text in files.items():
        af = Airfoil.from_text(text)
        _close(af.max_thickness, ref.max_thickness, 3e-3, f"{key} thickness")
        _close(cam(af), cam(ref), 5e-3, f"{key} camber")


def test_custom_profile_errors():
    cases = ["1 2\n3 4", "hello\nworld",
             "\n".join(f"0.5 {i*0.01}" for i in range(20)),
             "\n".join(f"{x:.4f} {0.06*np.sin(np.pi*x):.4f}"
                        for x in np.linspace(0, 1, 40))]
    for text in cases:
        try:
            Airfoil.from_text(text)
        except ValueError:
            continue
        raise AssertionError(f"should have been rejected: {text[:30]!r}")


def test_custom_profile_solves_and_round_trips():
    """A custom profile must drive the whole model and survive a config save."""
    import json
    import tempfile
    from wingstrain.config import spec_from_dict, spec_to_dict
    ref, files = _write_formats(None)
    w = baseline()
    w.spec.airfoil = Airfoil.from_text(files["mm_comma"], "my wing")
    w._sec_cache = {}
    r = w.solve(20.0)
    assert abs(r.reading("B1_bot").microstrain) > 1.0, "custom profile gave no strain"
    d = spec_to_dict(w.spec)
    assert "points" in d["airfoil"], "custom profile must serialise its points"
    back = spec_from_dict(json.loads(json.dumps(d)))
    _close(back.airfoil.max_thickness, w.spec.airfoil.max_thickness, 1e-3,
           "thickness through a config round trip")


# ----------------------------------------------------------------- inverse
def test_calibration_is_exact_for_the_nominal_load():
    """The estimator must return the truth for the load case it was built on."""
    w = baseline()
    cal = build_calibration(w)
    for V in (8.0, 15.0, 25.0):
        r = w.solve(V)
        _close(cal.estimate([x.microstrain for x in r.readings]), r.D_total,
               1e-9, f"estimate at {V} m/s")


def test_pylon_channels_reject_load_shape():
    """Mount struts carry the whole load, so nuisance cases must not move them."""
    from wingstrain.gauges import tube_pylon
    w = baseline()
    pyl = [tube_pylon("pL", support_label="pylon_L"),
           tube_pylon("pR", support_label="pylon_R")]
    cal = build_calibration(w, gauges=pyl)
    signal = abs(cal.A[:, 0]).max()          # the nominal load case
    for k, b in enumerate(cal.bases):
        if b.name in ("dS_shape", "dF_split", "dP_cop"):
            assert abs(cal.A[:, k]).max() < 1e-3 * signal, \
                f"{b.name} should be invisible to a pair of strut gauges"
    # thermal drift is a separate matter (see the collinearity test below):
    # here only the aerodynamic load cases should vanish
    aero_err = np.hypot.reduce([v for n, v in cal.budget
                                if not n.startswith("dT")])
    assert aero_err < 0.1, \
        f"strut pair should have no load-shape error, got {aero_err:.3f} N"


def test_identical_struts_cannot_separate_temperature_from_load():
    """Two identical strut channels see drift and load in the same direction."""
    from wingstrain.gauges import tube_pylon
    w = baseline()
    bad = [tube_pylon("pL", support_label="pylon_L", temp_apparent_ue_per_K=1.5),
           tube_pylon("pR", support_label="pylon_R", temp_apparent_ue_per_K=1.5)]
    cal = build_calibration(w, gauges=bad)
    assert any("dT_thermal" in wmsg for wmsg in cal.warnings), \
        "a collinear thermal direction must raise a warning"


def test_thermal_rejection_reduces_drift_sensitivity():
    """Fitting out common-mode drift must shrink the summed weights."""
    w = baseline()
    hot = build_calibration(w, reject_thermal=False)
    cold = build_calibration(w, reject_thermal=True)
    drift_hot = abs(sum(hot.weights[i] * g.temp_apparent_ue_per_K
                        for i, g in enumerate(hot.gauges)))
    drift_cold = abs(sum(cold.weights[i] * g.temp_apparent_ue_per_K
                         for i, g in enumerate(cold.gauges)))
    assert drift_cold < 0.02 * drift_hot, \
        f"thermal rejection barely helped: {drift_cold:.3f} vs {drift_hot:.3f}"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}\n        {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
