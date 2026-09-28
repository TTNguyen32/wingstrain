"""JSON configuration: describe a wing in a file, load it, solve it.

    python -m wingstrain --dump-config my_wing.json   # write a starting point
    python -m wingstrain --config my_wing.json --speed 20 --plots out/
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .aero import AeroSpec, CarriedElement
from .airfoil import Airfoil
from .beam import Support
from .gauges import Gauge, PylonGauge, tube_pylon
from .laminate import Laminate
from .model import Wing, WingSpec


def _airfoil(d) -> Airfoil:
    """Accepts {"naca": "4412"}, {"file": "s1223.dat"}, {"points": [[x, z], ...]}
    or {"flat_plate": 0.03} - or a bare string, read as a NACA code or a path."""
    if isinstance(d, str):
        return Airfoil.naca4(d) if d.isdigit() or d.upper().startswith("NACA") \
            else Airfoil.from_file(d)
    if "naca" in d:
        return Airfoil.naca4(str(d["naca"]))
    if "file" in d:
        return Airfoil.from_file(d["file"], d.get("name"))
    if "points" in d:
        return Airfoil.from_points(d["points"], d.get("name", "custom"))
    if "flat_plate" in d:
        return Airfoil.flat_plate(float(d["flat_plate"]))
    raise ValueError(f"cannot build an aerofoil from {d!r}")


def _laminate(d, name="layup") -> Laminate | None:
    if d is None:
        return None
    if d.get("symmetric"):
        return Laminate.symmetric(d["angles"], d.get("material", "twill_200"),
                                  d.get("name", name))
    return Laminate.build(d["angles"], d.get("material", "twill_200"),
                          d.get("name", name), d.get("thicknesses"))


def spec_from_dict(d: dict) -> WingSpec:
    g = [Gauge(**x) for x in d.get("gauges", [])]
    for x in d.get("pylon_gauges", []):
        x = dict(x)
        g.append(tube_pylon(**x) if "od" in x else PylonGauge(**x))
    return WingSpec(
        span=d.get("span", 1.0), chord=d.get("chord", 0.30),
        taper_ratio=d.get("taper_ratio", 1.0),
        angle_deg=d.get("angle_deg", 8.0), inverted=d.get("inverted", True),
        airfoil=_airfoil(d.get("airfoil", {"naca": "4412"})),
        x_ref=d.get("x_ref", 0.25),
        skin=_laminate(d.get("skin", {"angles": [45, 0, 0, 45]}), "skin"),
        lower_skin=_laminate(d.get("lower_skin"), "lower skin"),
        web=_laminate(d.get("web", {"angles": [45, -45]}), "web"),
        x_front=d.get("x_front", 0.0), x_rear=d.get("x_rear", 0.65),
        caps=d.get("caps", []),
        include_trailing_skin=d.get("include_trailing_skin", False),
        n_skin=d.get("n_skin", 72),
        supports=[Support(**s) for s in d.get(
            "supports", [{"y": 0.32, "label": "pylon_L"},
                         {"y": 0.68, "label": "pylon_R"}])],
        aero=AeroSpec(**{**{k: v for k, v in d.get("aero", {}).items()
                           if k != "carried"},
                         "carried": [CarriedElement(**c) for c in
                                     d.get("aero", {}).get("carried", [])]}),
        gauges=g, n_elem=d.get("n_elem", 80))


def _contour_points(af: Airfoil) -> list:
    """Selig-ordered contour, so a custom profile survives a config round trip."""
    import numpy as np
    xc = (1 - np.cos(np.linspace(0, np.pi, 90))) / 2
    up = [[float(x), float(af.z_upper(x))] for x in xc[::-1]]
    lo = [[float(x), float(af.z_lower(x))] for x in xc[1:]]
    return [[round(a, 6), round(b, 6)] for a, b in up + lo]


def load(path: str) -> Wing:
    with open(path) as fh:
        return Wing(spec_from_dict(json.load(fh)))


def spec_to_dict(spec: WingSpec) -> dict:
    lam = lambda l: None if l is None else {
        "angles": list(l.angles), "material": l.plies[0].name, "name": l.name}
    gauges, pylons = [], []
    for g in spec.gauges:
        if isinstance(g, PylonGauge):
            pylons.append({k: v for k, v in asdict(g).items()
                           if k not in ("y", "xc", "surface", "angle_deg")})
        else:
            gauges.append(asdict(g))
    return {
        "span": spec.span, "chord": spec.chord, "taper_ratio": spec.taper_ratio,
        "angle_deg": spec.angle_deg, "inverted": spec.inverted,
        "airfoil": ({"naca": spec.airfoil.name.replace("NACA", "")}
                    if spec.airfoil.name.startswith("NACA")
                    else {"name": spec.airfoil.name,
                          "points": _contour_points(spec.airfoil)}),
        "x_ref": spec.x_ref, "skin": lam(spec.skin),
        "lower_skin": lam(spec.lower_skin), "web": lam(spec.web),
        "x_front": spec.x_front, "x_rear": spec.x_rear, "caps": list(spec.caps),
        "include_trailing_skin": spec.include_trailing_skin,
        "n_skin": spec.n_skin, "n_elem": spec.n_elem,
        "supports": [{"y": s.y, "kind": s.kind, "label": s.label,
                      "restrain_twist": s.restrain_twist} for s in spec.supports],
        "aero": {"CL": spec.aero.CL, "CD": spec.aero.CD,
                 "xc_cp": spec.aero.xc_cp,
                 "distribution": spec.aero.distribution,
                 "tip_loading": spec.aero.tip_loading, "rho": spec.aero.rho,
                 "carried": [asdict(c) for c in spec.aero.carried]},
        "gauges": gauges, "pylon_gauges": pylons}


def save(spec: WingSpec, path: str) -> str:
    with open(path, "w") as fh:
        json.dump(spec_to_dict(spec), fh, indent=2)
    return path
