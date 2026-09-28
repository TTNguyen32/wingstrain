"""Classical laminate theory: ply materials -> effective membrane moduli.

Only the membrane (A-matrix) behaviour is used downstream: the wing skin is
treated as a thin wall carrying in-plane load, so what the section model needs
from a layup is (E_span, E_chord, G, nu, thickness).  Wall bending stiffness
(D-matrix) is not used for the beam solve, but is reported so you can sanity
check local panel buckling separately.

Ply angles are measured from the SPANWISE axis (y).  A 0 deg ply therefore runs
along the span and feeds bending stiffness; +-45 plies feed torsion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Ply:
    """Orthotropic ply (cured, in-plane properties). SI units: Pa, m."""

    name: str
    E1: float          # fibre direction modulus [Pa]
    E2: float          # transverse modulus [Pa]
    G12: float         # in-plane shear modulus [Pa]
    nu12: float        # major Poisson ratio
    thickness: float   # cured ply thickness [m]
    rho: float = 1550.0            # [kg/m^3]
    strain_allow_t: float = 0.011  # tensile strain allowable [-]
    strain_allow_c: float = 0.008  # compressive strain allowable [-]

    def Q(self) -> np.ndarray:
        """Reduced stiffness matrix in ply material axes."""
        nu21 = self.nu12 * self.E2 / self.E1
        d = 1.0 - self.nu12 * nu21
        return np.array([
            [self.E1 / d, self.nu12 * self.E2 / d, 0.0],
            [self.nu12 * self.E2 / d, self.E2 / d, 0.0],
            [0.0, 0.0, self.G12],
        ])


# --- Library of materials typical of a Formula Student wing shop -------------
MATERIALS: dict[str, Ply] = {
    # 200 gsm 2x2 twill T300-class fabric / epoxy, ~50% Vf, wet or prepreg.
    "twill_200": Ply("twill_200", 61e9, 61e9, 4.2e9, 0.05, 0.25e-3, 1500.0),
    # 300 gsm 2x2 twill.
    "twill_300": Ply("twill_300", 60e9, 60e9, 4.2e9, 0.05, 0.36e-3, 1500.0),
    # Unidirectional T700-class prepreg, 150 gsm.
    "ud_t700": Ply("ud_t700", 135e9, 9.0e9, 4.8e9, 0.28, 0.15e-3, 1570.0),
    # Unidirectional high-modulus (M46J-class) for spar caps.
    "ud_hm": Ply("ud_hm", 230e9, 8.0e9, 4.5e9, 0.28, 0.15e-3, 1600.0),
    # Glass fabric, 200 gsm, sometimes used for a tough outer ply.
    "glass_200": Ply("glass_200", 23e9, 23e9, 3.5e9, 0.13, 0.22e-3, 1900.0),
    # Isotropic-ish core plies (Rohacell 71, Nomex): thickness set per use.
    "rohacell_71": Ply("rohacell_71", 92e6, 92e6, 42e6, 0.30, 3.0e-3, 75.0,
                       strain_allow_t=0.02, strain_allow_c=0.02),
    "nomex_3mm": Ply("nomex_3mm", 1e6, 1e6, 40e6, 0.30, 3.0e-3, 48.0,
                     strain_allow_t=0.02, strain_allow_c=0.02),
}


@dataclass
class Laminate:
    """A stack, bottom to top. `angles` in degrees from spanwise.

    `plies` may be a single material name (used for every angle) or one name
    per angle.  A `core` entry is just another ply with a low modulus.
    """

    angles: list[float]
    plies: list[Ply]
    name: str = "layup"
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(cls, angles, material="twill_200", name="layup",
              thicknesses=None) -> "Laminate":
        angles = [float(a) for a in angles]
        if isinstance(material, str):
            mats = [MATERIALS[material]] * len(angles)
        else:
            mats = [MATERIALS[m] if isinstance(m, str) else m for m in material]
        if thicknesses is not None:
            mats = [Ply(m.name, m.E1, m.E2, m.G12, m.nu12, float(t), m.rho,
                        m.strain_allow_t, m.strain_allow_c)
                    for m, t in zip(mats, thicknesses)]
        if len(mats) != len(angles):
            raise ValueError("need one material per ply angle")
        return cls(angles, mats, name)

    @classmethod
    def symmetric(cls, half_angles, material="twill_200", name="layup"):
        """[a,b,c] -> [a,b,c,c,b,a] so the laminate is mirror symmetric."""
        ang = list(half_angles) + list(half_angles)[::-1]
        if isinstance(material, str):
            return cls.build(ang, material, name)
        mat = list(material) + list(material)[::-1]
        return cls.build(ang, mat, name)

    # -- geometry ----------------------------------------------------------
    @property
    def thickness(self) -> float:
        return float(sum(p.thickness for p in self.plies))

    @property
    def areal_mass(self) -> float:
        """Mass per unit area of the wall [kg/m^2]."""
        return float(sum(p.thickness * p.rho for p in self.plies))

    def _z_bounds(self):
        h = self.thickness
        z = -h / 2.0
        out = []
        for p in self.plies:
            out.append((z, z + p.thickness))
            z += p.thickness
        return out

    # -- CLT ---------------------------------------------------------------
    def ABD(self):
        if "ABD" in self._cache:
            return self._cache["ABD"]
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        for ang, p, (z0, z1) in zip(self.angles, self.plies, self._z_bounds()):
            Qb = _q_bar(p.Q(), ang)
            A += Qb * (z1 - z0)
            B += Qb * (z1 ** 2 - z0 ** 2) / 2.0
            D += Qb * (z1 ** 3 - z0 ** 3) / 3.0
        self._cache["ABD"] = (A, B, D)
        return A, B, D

    def moduli(self) -> dict:
        """Effective membrane engineering constants.

        x = spanwise (ply 0 deg direction), y = wall-tangential (chordwise).
        """
        if "mod" in self._cache:
            return self._cache["mod"]
        A, B, D = self.ABD()
        h = self.thickness
        a = np.linalg.inv(A)
        out = {
            "Ex": 1.0 / (a[0, 0] * h),
            "Ey": 1.0 / (a[1, 1] * h),
            "Gxy": 1.0 / (a[2, 2] * h),
            "nu_xy": -a[0, 1] / a[0, 0],
            "thickness": h,
            "areal_mass": self.areal_mass,
            # how far from mirror-symmetric this stack is (0 = symmetric)
            "bend_coupling": float(np.abs(B).max() / (np.abs(A).max() * h + 1e-30)),
            "D11": float(D[0, 0]),
        }
        self._cache["mod"] = out
        return out

    # convenience accessors used by the section model
    @property
    def Ex(self) -> float:
        return self.moduli()["Ex"]

    @property
    def Gxy(self) -> float:
        return self.moduli()["Gxy"]

    @property
    def nu_xy(self) -> float:
        return self.moduli()["nu_xy"]

    def describe(self) -> str:
        m = self.moduli()
        ang = "/".join(f"{a:g}" for a in self.angles)
        return (f"[{ang}] {self.plies[0].name} t={m['thickness']*1e3:.2f} mm  "
                f"Ex={m['Ex']/1e9:.1f} GPa  G={m['Gxy']/1e9:.1f} GPa  "
                f"nu={m['nu_xy']:.3f}  {m['areal_mass']:.2f} kg/m2")

    def min_strain_allowable(self) -> tuple[float, float]:
        """Governing (tension, compression) strain allowables of the stack."""
        struct = [p for p in self.plies if p.E1 > 5e9]
        if not struct:
            struct = list(self.plies)
        return (min(p.strain_allow_t for p in struct),
                min(p.strain_allow_c for p in struct))


def _q_bar(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate a reduced stiffness matrix to the laminate axes."""
    t = np.radians(theta_deg)
    m, n = np.cos(t), np.sin(t)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    Qb = np.zeros((3, 3))
    Qb[0, 0] = Q11 * m4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * n4
    Qb[0, 1] = (Q11 + Q22 - 4 * Q66) * m2 * n2 + Q12 * (m4 + n4)
    Qb[1, 1] = Q11 * n4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * m4
    Qb[0, 2] = (Q11 - Q12 - 2 * Q66) * m2 * m * n + (Q12 - Q22 + 2 * Q66) * m * n2 * n
    Qb[1, 2] = (Q11 - Q12 - 2 * Q66) * m * n2 * n + (Q12 - Q22 + 2 * Q66) * m2 * m * n
    Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m2 * n2 + Q66 * (m4 + n4)
    Qb[1, 0], Qb[2, 0], Qb[2, 1] = Qb[0, 1], Qb[0, 2], Qb[1, 2]
    return Qb
