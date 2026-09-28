"""Vary the wing itself and watch the measurement quality change.

Sweeps skin thickness, mount spacing, incidence, chord and span, reporting both
the structural answer (stiffness, strain, margin) and the measurement answer
(sensitivity, sigma_D).

    python examples/03_design_sweeps.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from wingstrain.beam import Support
from wingstrain.inverse import build_calibration
from wingstrain.laminate import Laminate
from wingstrain.model import Wing
from wingstrain.presets import baseline

V = 20.0
HEAD = (f"  {'case':<22s} {'EI [N.m2]':>10s} {'D [N]':>7s} {'peak µε':>8s} "
        f"{'µε/N @B1':>9s} {'sigma_D':>8s} {'RF':>6s} {'tip mm':>7s}")


def row(label, wing):
    cal = build_calibration(wing)
    r = wing.solve(V)
    s = abs(r.reading("B1_bot").microstrain) / r.D_total
    print(f"  {label:<22s} {r.props_mid['EIxx']:10.0f} {r.D_total:7.0f} "
          f"{r.max_strain*1e6:8.0f} {s:9.4f} {cal.sigma_total:8.1f} "
          f"{r.reserve_factor:6.0f} {r.tip_deflection*1e3:7.2f}")


def variant(**kw):
    w = Wing(copy.deepcopy(baseline().spec))
    for k, v in kw.items():
        setattr(w.spec, k, v)
    return w


print(f"=== skin layup (thinner skin = more strain per newton) ===\n{HEAD}")
for plies, name in [([45, 0, 0, 45], "4 ply  1.00 mm"),
                    ([45, -45], "2 ply  0.50 mm"),
                    ([45, 0, -45], "3 ply  0.75 mm"),
                    ([45, 0, 0, 0, 0, 45], "6 ply  1.50 mm"),
                    ([0, 45, -45, 0], "4 ply, 0 outside")]:
    row(name, variant(skin=Laminate.build(plies, "twill_200", name)))

print(f"\n=== mount spacing (span 1000 mm) ===\n{HEAD}")
for a in (0.10, 0.21, 0.25, 0.32, 0.40, 0.45):
    row(f"mounts at ±{a*1e3:.0f} mm",
        variant(supports=[Support(a, "pin", label="pylon_L"),
                          Support(1.0 - a, "pin", label="pylon_R")]))
print("  note: mounts at 0.207*span make the hogging and sagging moments equal")
print("  and nearly cancel mid-span - structurally elegant, and the worst")
print("  possible layout for a strain measurement.")

print(f"\n=== incidence ===\n{HEAD}")
for ang in (0.0, 4.0, 8.0, 12.0, 16.0):
    row(f"{ang:.0f} deg", variant(angle_deg=ang))
print("  more incidence deepens the section in car axes, so EI rises and the")
print("  strain per newton falls even though the load rises.")

print(f"\n=== chord and span ===\n{HEAD}")
for c in (0.20, 0.25, 0.30, 0.35):
    row(f"chord {c*1e3:.0f} mm", variant(chord=c))
for b in (0.8, 1.0, 1.2):
    w = variant(span=b, supports=[Support(0.32 * b, "pin", label="pylon_L"),
                                  Support(0.68 * b, "pin", label="pylon_R")])
    for g in w.spec.gauges:
        g.y *= b
    row(f"span {b*1e3:.0f} mm", w)

print(f"\n=== structural box: where the shear web sits ===\n{HEAD}")
for xr in (0.45, 0.55, 0.65, 0.75, 1.0):
    row(f"web at x/c {xr:.2f}", variant(x_rear=xr))

print(f"\n=== aerofoil section ===\n{HEAD}")
from wingstrain.airfoil import Airfoil
for code in ("0012", "2412", "4412", "6412", "4415", "4409"):
    row(f"NACA {code}", variant(airfoil=Airfoil.naca4(code)))
