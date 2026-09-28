"""Aerofoil geometry for the structural model.

Sections are generated in aerofoil axes (xi along the chord from the leading
edge, zeta up) and then transformed into car axes by `inverted` and
`angle_deg`.  For a downforce wing set inverted=True: the high-camber suction
side then faces the ground.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Airfoil:
    """Unit-chord aerofoil: upper and lower surfaces, leading edge at xi=0."""

    name: str
    xu: np.ndarray   # upper surface xi/c, increasing 0 -> 1
    zu: np.ndarray
    xl: np.ndarray   # lower surface xi/c, increasing 0 -> 1
    zl: np.ndarray

    # -- constructors ------------------------------------------------------
    @classmethod
    def naca4(cls, code: str = "4412", n: int = 160) -> "Airfoil":
        code = str(code).zfill(4)
        m = int(code[0]) / 100.0
        p = int(code[1]) / 10.0
        t = int(code[2:]) / 100.0
        # cosine spacing: fine at the leading edge where curvature is high
        beta = np.linspace(0.0, np.pi, n)
        x = (1.0 - np.cos(beta)) / 2.0
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1036 * x ** 4)   # closed trailing edge
        if m == 0.0 or p == 0.0:
            yc = np.zeros_like(x)
            dyc = np.zeros_like(x)
        else:
            yc = np.where(x < p,
                          m / p ** 2 * (2 * p * x - x ** 2),
                          m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
            dyc = np.where(x < p,
                           2 * m / p ** 2 * (p - x),
                           2 * m / (1 - p) ** 2 * (p - x))
        th = np.arctan(dyc)
        xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th)
        xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
        return cls(f"NACA{code}", *_monotonic(xu, zu), *_monotonic(xl, zl))

    @classmethod
    def from_points(cls, points, name: str = "custom") -> "Airfoil":
        """Build from an ordered contour, in either of the two usual orderings.

        Accepts the two formats coordinate files actually come in:

        * **Selig** - one loop starting at the trailing edge, forward along one
          surface to the leading edge, then back to the trailing edge.
        * **Lednicer** - two runs, each starting at the leading edge and going
          aft, upper surface first (often with a point-count header line).

        Coordinates may be in any units and need not start at x = 0; the
        contour is scaled to unit chord about its own leading edge. Which run
        is the upper surface is decided by where the material actually is, not
        by the file order.
        """
        P = np.asarray(points, float)
        if P.ndim != 2 or P.shape[1] < 2 or len(P) < 10:
            raise ValueError(f"need at least 10 (x, z) pairs, got {P.shape}")
        P = P[:, :2]
        x0, c = P[:, 0].min(), P[:, 0].max() - P[:, 0].min()
        if c <= 0:
            raise ValueError("all points share one x position - not an aerofoil")
        # datum on the leading edge, so a profile exported in millimetres from
        # somewhere else in a CAD sketch does not pick up phantom camber
        z0 = float(P[int(np.argmin(P[:, 0])), 1])
        P = np.column_stack([(P[:, 0] - x0) / c, (P[:, 1] - z0) / c])

        if P[0, 0] > 0.5:                       # Selig: starts at the trailing edge
            i = int(np.argmin(P[:, 0]))
            runs = [P[:i + 1][::-1], P[i:]]
        else:                                   # two runs, each from the nose aft
            d = np.diff(P[:, 0])
            i = int(np.argmin(d)) + 1           # the jump back to the nose
            if i < 4 or len(P) - i < 4:         # no jump: one run from the nose
                i = int(np.argmax(P[:, 0])) + 1
            runs = [P[:i], P[i:]]
        runs = [r for r in runs if len(r) >= 4]
        if len(runs) != 2:
            raise ValueError("could not split the contour into two surfaces")
        # both surfaces have to run most of the chord and overlap each other;
        # a single unclosed curve would otherwise split into two nonsense halves
        lo_x = max(r[:, 0].min() for r in runs)
        hi_x = min(r[:, 0].max() for r in runs)
        if hi_x - lo_x < 0.5:
            raise ValueError(
                "the two surfaces only share %.0f%% of the chord - this looks "
                "like a single open curve rather than a closed aerofoil "
                "contour" % (100 * max(hi_x - lo_x, 0)))
        if runs[0][:, 1].mean() < runs[1][:, 1].mean():
            runs = runs[::-1]                   # put the upper surface first
        up, lo = (_monotonic(r[:, 0], r[:, 1]) for r in runs)
        af = cls(name, *up, *lo)
        if af.max_thickness <= 0:
            raise ValueError("the two surfaces cross - check the point order")
        return af

    @classmethod
    def from_text(cls, text: str, name: str | None = None) -> "Airfoil":
        """Parse the contents of a .dat / .csv coordinate file."""
        pts, header = [], None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "!", ";")):
                continue
            try:
                a, b = (float(v) for v in line.replace(",", " ").split()[:2])
            except ValueError:
                header = header or line          # keep the name as written
                continue
            pts.append((a, b))
        if not pts:
            raise ValueError("no coordinate pairs found in the file")
        # a Lednicer point-count header looks like two whole numbers whose sum
        # is the number of points that follow
        h = pts[0]
        if (len(pts) > 1 and all(abs(v - round(v)) < 1e-6 and v >= 3 for v in h)
                and abs(h[0] + h[1] - (len(pts) - 1)) <= 2):
            pts = pts[1:]
        return cls.from_points(pts, name or header or "custom")

    @classmethod
    def from_file(cls, path: str, name: str | None = None) -> "Airfoil":
        """Read a coordinate file. The name comes from its header line if it has
        one, otherwise from the file name."""
        with open(path) as fh:
            af = cls.from_text(fh.read(), name)
        if af.name == "custom":
            af.name = path.split("/")[-1].rsplit(".", 1)[0]
        return af

    # kept for callers written against the older name
    from_selig = from_file

    @classmethod
    def flat_plate(cls, thickness: float = 0.03, n: int = 80) -> "Airfoil":
        """Constant-thickness plate; useful for closed-form cross checks."""
        x = np.linspace(0.0, 1.0, n)
        h = np.full_like(x, thickness / 2.0)
        h[0] = h[-1] = thickness / 2.0
        return cls(f"plate{thickness:g}", x, h, x.copy(), -h)

    # -- queries -----------------------------------------------------------
    def z_upper(self, xc) -> np.ndarray:
        return np.interp(np.asarray(xc, float), self.xu, self.zu)

    def z_lower(self, xc) -> np.ndarray:
        return np.interp(np.asarray(xc, float), self.xl, self.zl)

    def thickness_at(self, xc) -> np.ndarray:
        return self.z_upper(xc) - self.z_lower(xc)

    @property
    def max_thickness(self) -> float:
        xc = np.linspace(0.0, 1.0, 400)
        return float(self.thickness_at(xc).max())

    @property
    def x_max_thickness(self) -> float:
        """Chord fraction of maximum thickness - the deepest, best-signal spot."""
        xc = np.linspace(0.0, 1.0, 400)
        return float(xc[int(np.argmax(self.thickness_at(xc)))])

    def contour(self, n: int = 200):
        """Closed loop, counter-clockwise in (xi, zeta): lower TE->LE, upper LE->TE."""
        xc = (1 - np.cos(np.linspace(0, np.pi, n))) / 2.0
        lo = np.column_stack([xc[::-1], self.z_lower(xc[::-1])])
        up = np.column_stack([xc[1:], self.z_upper(xc[1:])])
        return np.vstack([lo, up])


def _monotonic(x, z):
    """Sort a surface by x and drop duplicate abscissae so np.interp is valid."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    o = np.argsort(x, kind="stable")
    x, z = x[o], z[o]
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    return x[keep], z[keep]


def transform(pts: np.ndarray, chord: float, angle_deg: float,
              inverted: bool, x_ref: float = 0.0) -> np.ndarray:
    """Aerofoil axes -> car axes (x aft, z up), scaled to `chord`.

    `angle_deg` > 0 rotates the trailing edge DOWN, i.e. adds incidence to an
    inverted (downforce) wing.  Rotation is about (x_ref*chord, 0).
    """
    p = np.array(pts, float) * chord
    if inverted:
        p[:, 1] *= -1.0
    b = np.radians(angle_deg)
    cb, sb = np.cos(b), np.sin(b)
    x = p[:, 0] - x_ref * chord
    z = p[:, 1]
    return np.column_stack([x * cb + z * sb, -x * sb + z * cb])
