"""Strain gauge model: from the section strain state to what the DAQ sees.

A gauge is placed by (spanwise station, chord fraction, surface, grid angle) -
the same four things you tell the person bonding it.  The model then gives
    eps along the grid       [microstrain]
    dR/R and bridge output   [mV/V at the stated excitation]
including the effects that actually bite in practice: grid length averaging
across a steep moment gradient, transverse sensitivity, misalignment, thermal
apparent strain and amplifier noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BRIDGE_FACTOR = {          # V_out/V_ex = factor * GF * eps_effective
    "quarter": 0.25,
    "half": 0.5,           # two active gauges (e.g. top/bottom, opposite sign)
    "full": 1.0,           # four active gauges at +-eps
    # Poisson bridges: the transverse arms see -nu*eps, so they add (1+nu)/2
    # rather than doubling.  This is how a strut is turned into a load cell -
    # the axial and transverse gauges share a temperature, so drift cancels in
    # hardware and only the mechanical strain survives.
    "half_poisson": 0.25 * 1.30,
    "full_poisson": 0.50 * 1.30,
}


@dataclass
class Gauge:
    """One strain gauge (or one grid of a rosette).

    `angle_deg` is the grid direction measured from SPANWISE, rotating towards
    the chordwise (wall-tangent) direction.  0 deg = along the span = maximum
    bending sensitivity; +-45 deg picks up torsional shear.
    """
    name: str
    y: float                       # spanwise station [m]
    xc: float = 0.30               # chord fraction
    surface: str = "top"           # 'top' | 'bottom' | 'web_rear' | 'web_front'
    angle_deg: float = 0.0
    gauge_factor: float = 2.05
    grid_length: float = 3.0e-3    # active grid length [m], along its own axis
    bridge: str = "quarter"
    excitation_V: float = 5.0
    noise_ue: float = 1.0          # RMS noise referred to strain [microstrain]
    transverse_sensitivity: float = 0.0    # K_t, typically 0.00-0.02
    temp_apparent_ue_per_K: float = 0.0    # residual mismatch, CFRP ~1-3 ue/K
    misalignment_deg: float = 0.0          # bonded-in error, added to angle

    @property
    def total_angle(self) -> float:
        return self.angle_deg + self.misalignment_deg


@dataclass
class Reading:
    gauge: Gauge
    eps_span: float       # eps_yy at the gauge centre
    eps_chord: float      # eps in the wall-tangent direction
    gamma: float          # in-plane shear strain (tangent-span)
    eps_grid: float       # strain resolved along the grid, grid-length averaged
    microstrain: float
    dR_over_R: float
    mV_per_V: float      # bridge output per volt of excitation [mV/V]
    mV: float
    z: float              # gauge height in car axes [m]
    x: float

    def __str__(self) -> str:
        g = self.gauge
        return (f"{g.name:>12s}  y={g.y*1e3:6.0f} mm  x/c={g.xc:.2f}  "
                f"{g.surface:<9s} {g.total_angle:+5.1f} deg  "
                f"{self.microstrain:+9.1f} ue  {self.mV_per_V:+8.3f} mV/V")


def resolve(eps_span: float, eps_chord: float, gamma: float,
            angle_deg: float) -> float:
    """Strain along a grid at `angle_deg` from spanwise (Mohr transformation)."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return eps_span * c * c + eps_chord * s * s + gamma * s * c


def bridge_output(gauge: Gauge, eps_grid: float, eps_transverse: float = 0.0,
                  nu0: float = 0.285) -> tuple[float, float, float]:
    """Return (dR/R, V_out/V_ex, V_out [V]) for a gauge reading eps_grid."""
    Kt = gauge.transverse_sensitivity
    eps_ind = eps_grid + Kt * eps_transverse
    if Kt:
        eps_ind /= (1.0 - nu0 * Kt)
    dR = gauge.gauge_factor * eps_ind
    f = BRIDGE_FACTOR.get(gauge.bridge, 0.25)
    ratio = f * dR
    return dR, ratio, ratio * gauge.excitation_V


def rosette(name: str, y: float, xc: float, surface: str = "top",
            **kw) -> list[Gauge]:
    """0/45/90 rectangular rosette at one point: separates bending and torsion."""
    return [Gauge(f"{name}_0", y, xc, surface, 0.0, **kw),
            Gauge(f"{name}_45", y, xc, surface, 45.0, **kw),
            Gauge(f"{name}_90", y, xc, surface, 90.0, **kw)]


def shear_pair(name: str, y: float, xc: float, surface: str = "top",
               **kw) -> list[Gauge]:
    """+-45 pair: their difference is the torsional shear, sum rejects it."""
    return [Gauge(f"{name}_p45", y, xc, surface, 45.0, **kw),
            Gauge(f"{name}_m45", y, xc, surface, -45.0, **kw)]


def bending_pair(name: str, y: float, xc: float, **kw) -> list[Gauge]:
    """Top/bottom pair at the same station: sum rejects axial and thermal drift."""
    return [Gauge(f"{name}_top", y, xc, "top", 0.0, **kw),
            Gauge(f"{name}_bot", y, xc, "bottom", 0.0, **kw)]


@dataclass
class PylonGauge:
    """A gauge on a wing mount strut, not on the wing skin.

    The struts carry the ENTIRE wing load into the chassis, so a strut gauge is
    insensitive to how the load is shared between elements or distributed along
    the span - which is exactly the ambiguity that limits a skin gauge.  The
    penalty is that it tells you nothing about where the load acts.

    Modelled as a vertical strut of axial stiffness EA and bending stiffness EI:
        eps = Fz/EA  +/-  M * c / EI,    M = Fz*ecc_x + Fx*height
    With `axial_only=True` (a gauge pair on opposite faces wired to cancel
    bending) only the axial term is kept.
    """
    name: str
    support_label: str = "pylon_L"
    EA: float = 7.0e6              # [N]
    EI: float = 400.0              # [N.m^2]
    height: float = 0.150          # gauge station below the wing joint [m]
    ecc_x: float = 0.010           # load-path eccentricity at the gauge [m]
    c: float = 0.0125              # centroid-to-surface distance [m]
    axial_only: bool = True
    gauge_factor: float = 2.05
    bridge: str = "full_poisson"
    excitation_V: float = 5.0
    noise_ue: float = 1.0
    transverse_sensitivity: float = 0.0
    # a Poisson full bridge cancels drift in hardware, so the residual is small
    temp_apparent_ue_per_K: float = 0.2
    misalignment_deg: float = 0.0
    grid_length: float = 3.0e-3
    y: float = 0.0                 # for reporting only
    xc: float = 0.0
    surface: str = "pylon"
    angle_deg: float = 0.0

    @property
    def total_angle(self) -> float:
        return 0.0

    def strain(self, Fz: float, Fx: float = 0.0) -> float:
        """Strain along the strut axis. Fz is the reaction on the wing [N]."""
        axial = -Fz / self.EA          # wing pushed down -> strut in compression
        if self.axial_only:
            return axial
        M = Fz * self.ecc_x + Fx * self.height
        return axial + M * self.c / self.EI


def tube_pylon(name: str, od: float = 0.025, wall: float = 0.0015,
               E: float = 70e9, **kw) -> PylonGauge:
    """Round tube strut: convenience constructor from OD and wall thickness."""
    import math
    ri = od / 2.0 - wall
    ro = od / 2.0
    A = math.pi * (ro ** 2 - ri ** 2)
    I = math.pi / 4.0 * (ro ** 4 - ri ** 4)
    return PylonGauge(name, EA=E * A, EI=E * I, c=ro, **kw)
