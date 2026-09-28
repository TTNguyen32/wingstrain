"""Figures: where the strain is, where to bond, and how good the answer is.

Colours follow a validated palette: categorical blue / orange / aqua for
series, and a blue-grey-red diverging ramp for strain, whose sign (tension vs
compression) is exactly what a diverging scale is for.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
STRAIN_CMAP = LinearSegmentedColormap.from_list(
    "strain", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
               "#f3a09f", "#e34948", "#8f1f1f"])


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    return ax


def _group(pairs) -> dict:
    """Collapse co-located gauges so their labels do not overprint."""
    out = {}
    for r, key in pairs:
        out.setdefault(key, []).append(r.gauge.name)
    return out


def _mounts(ax, wing, label=True):
    for i, s in enumerate(wing.spec.supports):
        ax.axvline(s.y * 1e3, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        if label:
            ax.annotate(s.label, (s.y * 1e3, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -11), textcoords="offset points",
                        fontsize=7.5, color=MUTED)


# --------------------------------------------------------------------------
def plot_spanwise(res, wing, ax=None):
    """Internal loads along the span, with the gauge stations marked."""
    ax = ax or plt.subplots(figsize=(7, 3.4))[1]
    br = res.beam
    y = br.y * 1e3
    ax.plot(y, br.Mx, color=C1, lw=2.0, zorder=4, label="bending $M_x$")
    ax.plot(y, br.T, color=C2, lw=2.0, zorder=3, label="torque $T$")
    ax.plot(y, br.Mz, color=C3, lw=2.0, zorder=2, label="chordwise $M_z$")
    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
    _mounts(ax, wing)
    for g in {r.gauge.y for r in res.readings if r.gauge.surface != "pylon"}:
        ax.plot([g * 1e3], [np.interp(g, br.y, br.Mx)], "o", ms=6,
                mfc="white", mec=INK, mew=1.4, zorder=6)
    i = int(np.argmax(np.abs(br.Mx)))
    ax.annotate(f"peak {br.Mx[i]:+.2f} N·m", (y[i], br.Mx[i]),
                xytext=(6, 6), textcoords="offset points", fontsize=8, color=C1)
    _style(ax, "spanwise station [mm]", "internal load [N·m]",
           f"Internal loads at {res.V:.0f} m/s  –  "
           f"{res.D_total:.0f} N total downforce")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3,
              loc="lower center")
    return ax


def plot_gauge_strain_span(res, wing, xc=None, ax=None):
    """Surface strain along the span on both skins - the signal you can bond to."""
    from .gauges import Gauge
    ax = ax or plt.subplots(figsize=(7, 3.4))[1]
    sp = wing.spec
    xc = sp.airfoil.x_max_thickness if xc is None else xc
    ys = np.linspace(0.004, sp.span - 0.004, 121)
    for surf, col, lab in (("top", C1, "top skin"), ("bottom", C2, "bottom skin")):
        r = wing.solve(res.V, gauges=[Gauge(f"{surf}{i}", v, xc, surf)
                                      for i, v in enumerate(ys)])
        e = np.array([x.microstrain for x in r.readings])
        ax.plot(ys * 1e3, e, color=col, lw=2.0, label=lab)
        j = int(np.argmax(np.abs(e)))
        ax.annotate(f"{lab}: {e[j]:+.0f} µε", (ys[j] * 1e3, e[j]),
                    xytext=(6, -4 if e[j] < 0 else 4), textcoords="offset points",
                    fontsize=8, color=col)
    ax.axhline(0, color=INK2, lw=0.9)
    _mounts(ax, wing)
    # only grids aligned with the span read this curve; +-45 grids do not
    for (yy, e), names in _group(
            [(r, (round(r.gauge.y, 4), round(r.microstrain, 2)))
             for r in res.readings
             if r.gauge.surface in ("top", "bottom")
             and abs(r.gauge.total_angle) < 1e-9]).items():
        ax.plot([yy * 1e3], [e], "o", ms=6, mfc="white", mec=INK, mew=1.4,
                zorder=6)
        ax.annotate("/".join(names), (yy * 1e3, e), xytext=(0, 9),
                    textcoords="offset points", fontsize=7.5, color=INK,
                    ha="center")
    _style(ax, "spanwise station [mm]", "spanwise strain [µε]",
           f"Skin strain at x/c = {xc:.2f}, {res.V:.0f} m/s")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="best")
    return ax


def plot_strain_map(wing, V=20.0, ax=None, surface="bottom", n_y=70, n_xc=50):
    """Chord x span map of surface strain: the gauge siting chart."""
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    ys, xcs, E, res = wing.strain_map(V, surface, n_y, n_xc)
    lim = max(abs(E.min()), abs(E.max()), 1e-9)
    m = ax.pcolormesh(ys * 1e3, xcs, E, cmap=STRAIN_CMAP, shading="auto",
                      norm=TwoSlopeNorm(0.0, -lim, lim), rasterized=True)
    if E.min() < -1e-9 < 1e-9 < E.max():
        ax.contour(ys * 1e3, xcs, E, levels=[0.0], colors=[INK2],
                   linewidths=0.9, linestyles="dashed")
    cb = plt.gcf().colorbar(m, ax=ax, pad=0.015)
    cb.set_label("spanwise strain [µε]   (+ tension)", color=INK2,
                 fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    for s in wing.spec.supports:
        ax.axvline(s.y * 1e3, color=INK, lw=1.0, ls=(0, (4, 3)))
    for g in wing.spec.gauges:
        if g.surface == surface:
            ax.plot([g.y * 1e3], [g.xc], "o", ms=7, mfc="none", mec=INK, mew=1.6)
            ax.annotate(g.name, (g.y * 1e3, g.xc), xytext=(7, 0),
                        textcoords="offset points", fontsize=7.5, color=INK,
                        va="center")
    k = np.unravel_index(np.argmax(np.abs(E)), E.shape)
    ax.plot([ys[k[1]] * 1e3], [xcs[k[0]]], "*", ms=12, mfc=GOOD, mec="white",
            mew=0.8)
    ax.annotate(f"max |ε| {E[k]:+.0f} µε", (ys[k[1]] * 1e3, xcs[k[0]]),
                xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK)
    _style(ax, "spanwise station [mm]", "chord fraction x/c",
           f"{surface} skin strain at {V:.0f} m/s  ({res.D_total:.0f} N)"
           + (" – dashed line = zero strain" if E.min() < 0 < E.max() else ""))
    return ax


def plot_position_study(wing, V=20.0, ax=None, surface="bottom"):
    """Sensitivity vs spanwise station, with the no-go zones near the mounts."""
    from .inverse import recommend_stations
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    for surf, col in ((surface, C1), ("top" if surface == "bottom" else "bottom", C2)):
        r = recommend_stations(wing, V, surface=surf)
        ax.plot(r["y"] * 1e3, np.abs(r["sensitivity"]), color=col, lw=2.0,
                label=f"{surf} skin, x/c={r['xc']:.2f}")
        if surf == surface:
            ax.plot([r["best_y"] * 1e3], [abs(r["best_sensitivity"])], "*",
                    ms=14, mfc=GOOD, mec="white", mew=0.8, zorder=6)
            ax.annotate(f"best {r['best_y']*1e3:.0f} mm\n"
                        f"{abs(r['best_sensitivity']):.3f} µε/N",
                        (r["best_y"] * 1e3, abs(r["best_sensitivity"])),
                        xytext=(8, -18), textcoords="offset points", fontsize=8,
                        color=INK)
            cl = r["min_clearance"]
    for s in wing.spec.supports:
        ax.axvspan((s.y - cl) * 1e3, (s.y + cl) * 1e3, color=CRITICAL,
                   alpha=0.08, lw=0, zorder=0)
        ax.axvline(s.y * 1e3, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f"shaded: within {cl*1e3:.0f} mm of a mount – local load\n"
                "introduction, strain not beam-like",
                (0.02, 0.96), xycoords="axes fraction", va="top", fontsize=7.5,
                color=CRITICAL)
    _style(ax, "gauge spanwise station [mm]",
           "|sensitivity| [µε per N of downforce]",
           "Where to put a bending gauge")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    return ax


def plot_chordwise_study(wing, y=None, V=20.0, ax=None):
    """Sensitivity along the chord at one station, both skins."""
    from .inverse import best_chordwise
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    y = wing.spec.span * 0.46 if y is None else y
    for surf, col in (("top", C1), ("bottom", C2)):
        r = best_chordwise(wing, y, V, surf)
        ax.plot(r["xc"], np.abs(r["sensitivity"]), color=col, lw=2.0,
                label=f"{surf} skin")
        ax.plot([r["best_xc"]], [abs(r["best_sensitivity"])], "o", ms=7,
                mfc="white", mec=col, mew=1.6)
        ax.annotate(f"x/c {r['best_xc']:.2f}\n{abs(r['best_sensitivity']):.3f} µε/N",
                    (r["best_xc"], abs(r["best_sensitivity"])), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=col)
    ax.axvline(wing.spec.airfoil.x_max_thickness, color=MUTED, lw=1.0,
               ls=(0, (4, 3)))
    ax.annotate("max thickness", (wing.spec.airfoil.x_max_thickness, 0.02),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=MUTED)
    _style(ax, "chord fraction x/c", "|sensitivity| [µε per N]",
           f"Chordwise siting at y = {y*1e3:.0f} mm")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    return ax


def plot_section(wing, y=None, V=20.0, ax=None):
    """Section in car axes, coloured by strain per newton, with gauge spots."""
    from matplotlib.collections import LineCollection
    ax = ax or plt.subplots(figsize=(6, 3.4))[1]
    sp = wing.spec
    y = sp.span * 0.46 if y is None else y
    res = wing.solve(V, gauges=[])
    sec = wing.section_at(y)
    Mx = float(np.interp(y, res.beam.y, res.beam.Mx))
    Mz = float(np.interp(y, res.beam.y, res.beam.Mz))
    segs, vals = [], []
    for sg in sec.all_walls:
        segs.append([(sg.p0[0] * 1e3, sg.p0[1] * 1e3),
                     (sg.p1[0] * 1e3, sg.p1[1] * 1e3)])
        vals.append(1e6 * float(sec.eps_yy(*sg.mid, Mx, Mz)) / res.D_total)
    vals = np.array(vals)
    lim = max(np.abs(vals).max(), 1e-12)
    lc = LineCollection(segs, cmap=STRAIN_CMAP, norm=TwoSlopeNorm(0, -lim, lim),
                        linewidths=3.2)
    lc.set_array(vals)
    ax.add_collection(lc)
    cb = plt.gcf().colorbar(lc, ax=ax, pad=0.015)
    cb.set_label("µε per N of downforce", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    p = sec.props()
    ax.plot([p["x_ea"] * 1e3], [p["z_ea"] * 1e3], "+", ms=11, color=INK, mew=1.6)
    ax.annotate("elastic centroid", (p["x_ea"] * 1e3, p["z_ea"] * 1e3),
                xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.plot([p["x_sc"] * 1e3], [p["z_ea"] * 1e3], "x", ms=8, color=C2, mew=1.8)
    ax.annotate("shear centre", (p["x_sc"] * 1e3, p["z_ea"] * 1e3),
                xytext=(-6, 9), textcoords="offset points", fontsize=7.5,
                color=C2, ha="right")
    spots: dict = {}
    for g in sp.gauges:
        if g.surface in ("top", "bottom"):
            sg = sec.loop[sec.locate(g.xc, g.surface)]
            spots.setdefault((round(float(sg.mid[0]), 5),
                              round(float(sg.mid[1]), 5)), []).append(g.name)
    for (gx, gz), names in spots.items():
        ax.plot([gx * 1e3], [gz * 1e3], "o", ms=8, mfc="none", mec=INK, mew=1.6)
        ax.annotate("/".join(names), (gx * 1e3, gz * 1e3), xytext=(0, 10),
                    textcoords="offset points", fontsize=7.5, color=INK,
                    ha="center")
    ax.autoscale_view()
    ax.set_aspect("equal")
    _style(ax, "x [mm, aft of quarter chord]", "z [mm, up]",
           f"Section at y = {y*1e3:.0f} mm, {sp.angle_deg:.0f}° incidence "
           f"– all gauge chord positions projected onto one section")
    return ax


def plot_calibration(wing, cal, speeds=(8, 12, 16, 20, 24, 28), ax=None,
                     gauges=None):
    """Estimator output vs true downforce over a speed sweep, with its 1-sigma band."""
    ax = ax or plt.subplots(figsize=(6, 3.4))[1]
    gs = gauges if gauges is not None else cal.gauges
    D, Dh = [], []
    for V in speeds:
        r = wing.solve(V, gauges=gs)
        D.append(r.D_total)
        Dh.append(cal.estimate([x.microstrain for x in r.readings]))
    D, Dh = np.array(D), np.array(Dh)
    ax.fill_between(D, Dh - cal.sigma_total, Dh + cal.sigma_total, color=C1,
                    alpha=0.16, lw=0, label=f"±1σ = {cal.sigma_total:.0f} N")
    ax.plot(D, D, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="ideal 1:1")
    ax.plot(D, Dh, "o-", color=C1, lw=2.0, ms=6, label="estimator")
    for V, a, b in zip(speeds, D, Dh):
        ax.annotate(f"{V:.0f} m/s", (a, b), xytext=(6, -10),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    _style(ax, "true downforce [N]", "estimated downforce [N]",
           f"Calibration check  –  1σ {cal.sigma_total:.1f} N "
           f"({100*cal.relative_error:.1f}% at {cal.D_ref:.0f} N)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    return ax


def plot_error_budget(cal, mc=None, ax=None):
    """Horizontal bars: what limits the measurement."""
    ax = ax or plt.subplots(figsize=(7, 3.4))[1]
    if mc is not None:
        from .uncertainty import SCALE_SOURCES
        items = sorted(mc.tornado.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals = np.array([100 * v for _, v in items])
        cols = [C2 if n in SCALE_SOURCES else C1 for n in names]
        unit, xlabel = "%", "1σ contribution [% of downforce]"
        title = (f"Monte Carlo error budget, {mc.n} samples  –  total "
                 f"{100*np.std(mc.errors_raw):.1f}% as modelled, "
                 f"{100*np.std(mc.errors_static):.1f}% after a hang test")
    else:
        items = sorted([("gauge noise", cal.sigma_noise)]
                       + [(n.split(" (")[0], v) for n, v in cal.budget],
                       key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals = np.array([v for _, v in items])
        cols = [C1] * len(names)
        unit, xlabel = "N", "1σ contribution [N]"
        title = (f"Analytic error budget  –  total {cal.sigma_total:.1f} N "
                 f"({100*cal.relative_error:.1f}%)")
    pos = np.arange(len(names))
    ax.barh(pos, vals, color=cols, height=0.62, zorder=3)
    for p, v in zip(pos, vals):
        ax.annotate(f"{v:.2f} {unit}", (v, p), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=8.5, color=INK2)
    ax.set_xlim(0, vals.max() * 1.22 if vals.size else 1)
    _style(ax, xlabel, "", title)
    ax.grid(False, axis="y")
    if mc is not None:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=C2, label="removed by a static hang test"),
                           Patch(color=C1, label="run-to-run, not calibratable")],
                  frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    return ax


def dashboard(wing, V=20.0, cal=None, mc=None, path="out/dashboard.png",
              gauges=None):
    """One page: loads, siting, section, calibration, budget."""
    from .inverse import build_calibration
    cal = cal or build_calibration(wing, gauges=gauges)
    res = wing.solve(V, gauges=gauges if gauges is not None else wing.spec.gauges)
    fig = plt.figure(figsize=(15.5, 13.0), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.20,
                          left=0.07, right=0.97, top=0.945, bottom=0.05)
    plot_spanwise(res, wing, fig.add_subplot(gs[0, 0]))
    plot_gauge_strain_span(res, wing, ax=fig.add_subplot(gs[0, 1]))
    plot_strain_map(wing, V, fig.add_subplot(gs[1, 0]), "bottom")
    plot_section(wing, V=V, ax=fig.add_subplot(gs[1, 1]))
    plot_position_study(wing, V, fig.add_subplot(gs[2, 0]))
    plot_chordwise_study(wing, V=V, ax=fig.add_subplot(gs[2, 1]))
    plot_calibration(wing, cal, ax=fig.add_subplot(gs[3, 0]), gauges=gauges)
    plot_error_budget(cal, mc, fig.add_subplot(gs[3, 1]))
    sp = wing.spec
    fig.suptitle(f"Wing strain → downforce  |  {sp.airfoil.name}, "
                 f"{sp.span*1e3:.0f} × {sp.chord*1e3:.0f} mm, "
                 f"{sp.angle_deg:.0f}°, skin {sp.skin.thickness*1e3:.2f} mm, "
                 f"mounts at {', '.join(f'{s.y*1e3:.0f}' for s in sp.supports)} mm"
                 f"   |   {V:.0f} m/s, {res.D_total:.0f} N",
                 fontsize=12.5, color=INK, x=0.07, ha="left")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path
