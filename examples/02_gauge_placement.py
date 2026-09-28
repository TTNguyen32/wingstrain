"""Where should the gauges go?

Scans spanwise and chordwise position, on both skins, and reports the
sensitivity you would get - then checks what each candidate does to the
downforce error budget.

    python examples/02_gauge_placement.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from wingstrain.gauges import bending_pair, shear_pair, tube_pylon
from wingstrain.inverse import (best_chordwise, build_calibration,
                                recommend_stations)
from wingstrain.presets import baseline

V = 20.0
wing = baseline()
D = wing.solve(V, gauges=[]).D_total
print(f"baseline wing, {D:.0f} N of downforce at {V:.0f} m/s\n")

print("spanwise scan (strain per newton of TOTAL downforce):")
print(f"  {'y [mm]':>8s} {'top µε/N':>10s} {'bottom µε/N':>13s}  {'':>8s}")
top = recommend_stations(wing, V, surface="top")
bot = recommend_stations(wing, V, surface="bottom")
for i in range(0, len(top["y"]), 10):
    flag = "" if top["allowed"][i] else "too close to a mount"
    print(f"  {top['y'][i]*1e3:8.0f} {top['sensitivity'][i]:10.4f} "
          f"{bot['sensitivity'][i]:13.4f}  {flag}")
print(f"\n  best usable station: y = {bot['best_y']*1e3:.0f} mm "
      f"({abs(bot['best_sensitivity']):.4f} µε/N on the bottom skin)")
print(f"  keep at least {bot['min_clearance']*1e3:.0f} mm "
      f"(2.5 section depths) clear of a mount")

print("\nchordwise scan at that station:")
for surf in ("top", "bottom"):
    c = best_chordwise(wing, bot["best_y"], V, surf)
    print(f"  {surf:>6s} skin: peak at x/c = {c['best_xc']:.2f}  "
          f"({abs(c['best_sensitivity']):.4f} µε/N);  "
          f"max thickness is at x/c = {wing.spec.airfoil.x_max_thickness:.2f}")
print("  note the peak is NOT at max thickness: the section sits at incidence,")
print("  so the deepest material relative to the neutral axis is further aft.")

print("\nwhat different instrumentation plans buy you:")
y0 = bot["best_y"]
plans = {
    "1 gauge, bottom skin": [bending_pair("G", y0, 0.34)[1]],
    "bending pair (top+bottom)": bending_pair("G", y0, 0.34),
    "2 stations, both skins": bending_pair("G1", y0, 0.34) + bending_pair("G2", 0.16, 0.34),
    "+ 45 deg shear pair": (bending_pair("G1", y0, 0.34) + bending_pair("G2", 0.16, 0.34)
                            + shear_pair("S1", y0, 0.34)),
    "+ strut gauges": (bending_pair("G1", y0, 0.34) + bending_pair("G2", 0.16, 0.34)
                       + shear_pair("S1", y0, 0.34)
                       + [tube_pylon("pL", support_label="pylon_L"),
                          tube_pylon("pR", support_label="pylon_R")]),
    "strut gauges only": [tube_pylon("pL", support_label="pylon_L"),
                          tube_pylon("pR", support_label="pylon_R")],
}
print(f"  {'plan':<28s} {'channels':>8s} {'sigma_D [N]':>12s} {'of which noise':>15s} {'error':>7s}")
for name, gs in plans.items():
    for g in gs:
        g.temp_apparent_ue_per_K = 1.5
    cal = build_calibration(wing, gauges=gs)
    print(f"  {name:<28s} {len(gs):8d} {cal.sigma_total:12.1f} "
          f"{cal.sigma_noise:15.1f} {100*cal.relative_error:6.1f}%")
