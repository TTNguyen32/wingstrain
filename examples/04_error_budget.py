"""How accurately can this actually measure downforce?

Runs the analytic budget and then a Monte Carlo over laminate scatter, bonding
tolerance, gauge factor, thermal drift, noise and aerodynamic load movement -
and separates the errors a static hang test removes from those it cannot.

    python examples/04_error_budget.py [n_samples]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from wingstrain.gauges import tube_pylon
from wingstrain.inverse import build_calibration
from wingstrain.plots import plot_error_budget
from wingstrain.presets import baseline
from wingstrain.uncertainty import monte_carlo

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
V = 20.0
wing = baseline()
# strut channels are Poisson full bridges - the standard way to turn a strut
# into a load cell, with thermal drift cancelled in hardware
struts = [tube_pylon("pylonL", support_label="pylon_L", noise_ue=1.0),
          tube_pylon("pylonR", support_label="pylon_R", noise_ue=1.0)]

plans = {"skin gauges only": list(wing.spec.gauges),
         "skin + strut gauges": list(wing.spec.gauges) + struts,
         "strut gauges only": struts}

print(f"{'plan':<22s} {'sigma_D [N]':>11s} {'noise':>7s} {'load shape':>11s} "
      f"{'% of D':>7s}")
cals = {}
for name, gs in plans.items():
    cals[name] = c = build_calibration(wing, gauges=gs)
    print(f"{name:<22s} {c.sigma_total:11.1f} {c.sigma_noise:7.1f} "
          f"{c.sigma_model:11.1f} {100*c.relative_error:6.1f}%")

print("\nthermal rejection (fitting common-mode drift out as an extra unknown):")
for name, gs in plans.items():
    off = build_calibration(wing, gauges=gs, reject_thermal=False)
    on = cals[name]
    drift = lambda c: abs(sum(w * g.temp_apparent_ue_per_K
                              for w, g in zip(c.weights, c.gauges)))
    print(f"  {name:<22s} {drift(off):7.2f} N/K  ->  {drift(on):6.3f} N/K "
          f"(noise cost {100*(on.sigma_noise/off.sigma_noise-1):+.1f}%)")

print(f"\nMonte Carlo, {N} samples:")
for name, gs in plans.items():
    mc = monte_carlo(wing, cals[name], V, N, gauges=gs,
                     tornado=(name == "skin gauges only"))
    print(f"\n--- {name}")
    print(mc.report())
    if name == "skin gauges only":
        import matplotlib.pyplot as plt
        os.makedirs("out", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        plot_error_budget(cals[name], mc, ax)
        fig.savefig("out/error_budget.png", dpi=140, bbox_inches="tight",
                    facecolor="#fcfcfb")
        print("\nwrote out/error_budget.png")
