"""Solve the baseline wing, print the report, write the dashboard figure.

    python examples/01_baseline_report.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wingstrain.inverse import build_calibration, single_gauge_calibration
from wingstrain.plots import dashboard
from wingstrain.presets import baseline

V = 20.0
wing = baseline()
res = wing.solve(V)
print(res.report())

print("\n" + "=" * 78)
cal = build_calibration(wing)
print(cal.describe())

print("\nIf you would rather use one gauge and one constant:")
for name, d in single_gauge_calibration(wing, V).items():
    print(f"  {name:>8s}:  D [N] = {d['N_per_ue']:+8.2f} x strain [µε]"
          f"      ({d['microstrain']:+7.1f} µε at {res.D_total:.0f} N)")
print("  - but that assumes the load distribution never changes; the "
      "multi-gauge\n    estimator above is what protects you when it does.")

os.makedirs("out", exist_ok=True)
print("\nwrote", dashboard(wing, V, cal, None, "out/dashboard.png"))
