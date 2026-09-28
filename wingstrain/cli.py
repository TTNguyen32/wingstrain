"""Command line entry point: `python -m wingstrain ...`"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="wingstrain",
        description="Strain-gauge downforce measurement model for a composite "
                    "wing: vary geometry, layup, mounts and gauge positions, "
                    "and see the strain, the calibration and the error budget.")
    ap.add_argument("--config", help="wing definition JSON (default: baseline)")
    ap.add_argument("--dump-config", metavar="PATH",
                    help="write the baseline configuration and exit")
    ap.add_argument("--profile", metavar="FILE",
                    help="aerofoil coordinate file (.dat/.csv, Selig or "
                         "Lednicer order, any units) to use instead of the "
                         "configured section")
    ap.add_argument("--speed", type=float, default=20.0, help="m/s (default 20)")
    ap.add_argument("--sweep", type=str, default="",
                    help="speed sweep, e.g. 8:28:5 (min:max:count)")
    ap.add_argument("--pylons", action="store_true",
                    help="add an axial gauge on each mount strut")
    ap.add_argument("--mc", type=int, default=0,
                    help="Monte Carlo samples for the full error budget")
    ap.add_argument("--no-thermal-rejection", action="store_true",
                    help="do not fit out common-mode thermal drift")
    ap.add_argument("--plots", metavar="DIR", help="write figures to DIR")
    ap.add_argument("--recommend", action="store_true",
                    help="print the best gauge stations for this wing")
    args = ap.parse_args(argv)

    from .config import load, save
    from .gauges import tube_pylon
    from .inverse import (best_chordwise, build_calibration,
                          recommend_stations, single_gauge_calibration)
    from .presets import baseline

    if args.dump_config:
        from .presets import baseline as bl
        print("wrote", save(bl().spec, args.dump_config))
        return 0

    wing = load(args.config) if args.config else baseline()
    if args.profile:
        from .airfoil import Airfoil
        wing.spec.airfoil = Airfoil.from_file(args.profile)
        wing._sec_cache = {}
        af = wing.spec.airfoil
        print(f"profile: {af.name} - {af.max_thickness*100:.1f}% thick at "
              f"x/c {af.x_max_thickness:.2f}\n")
    gauges = list(wing.spec.gauges)
    if args.pylons:
        gauges += [tube_pylon(f"pylon_{s.label[-1]}", support_label=s.label,
                              noise_ue=1.0)
                   for s in wing.spec.supports]

    res = wing.solve(args.speed, gauges=gauges)
    print(res.report())

    print()
    cal = build_calibration(wing, gauges=gauges,
                            reject_thermal=not args.no_thermal_rejection)
    print(cal.describe())

    print("\nsingle-gauge calibration constants (ignores load-shape error):")
    for name, d in single_gauge_calibration(wing, args.speed).items():
        print(f"  {name:>10s}  {d['ue_per_N']:+8.4f} µε/N   "
              f"{d['N_per_ue']:+9.2f} N/µε   {d['mV_per_V']:+7.3f} mV/V")

    if args.sweep:
        lo, hi, n = args.sweep.split(":")
        print("\nspeed sweep:")
        print(f"  {'V [m/s]':>8s} {'q [Pa]':>8s} {'D [N]':>9s} "
              f"{'peak M [N.m]':>13s} {'peak µε':>9s} {'tip δ [mm]':>11s}")
        for V in np.linspace(float(lo), float(hi), int(n)):
            r = wing.solve(V, gauges=gauges)
            print(f"  {V:8.1f} {r.loads.q:8.1f} {r.D_total:9.1f} "
                  f"{np.abs(r.beam.Mx).max():13.2f} {r.max_strain*1e6:9.0f} "
                  f"{r.tip_deflection*1e3:11.2f}")

    if args.recommend:
        print("\ngauge siting:")
        for surf in ("top", "bottom"):
            rc = recommend_stations(wing, args.speed, surface=surf)
            bc = best_chordwise(wing, rc["best_y"], args.speed, surf)
            print(f"  {surf:>6s} skin: y = {rc['best_y']*1e3:5.0f} mm, "
                  f"x/c = {bc['best_xc']:.2f}  ->  "
                  f"{abs(bc['best_sensitivity']):.4f} µε/N  "
                  f"(keep > {rc['min_clearance']*1e3:.0f} mm from a mount)")

    mc = None
    if args.mc:
        from .uncertainty import monte_carlo
        print()
        mc = monte_carlo(wing, cal, args.speed, args.mc, gauges=gauges)
        print(mc.report())

    if args.plots:
        from .plots import dashboard
        os.makedirs(args.plots, exist_ok=True)
        path = dashboard(wing, args.speed, cal, mc,
                         os.path.join(args.plots, "dashboard.png"),
                         gauges=gauges)
        print(f"\nwrote {path}")
    return 0
