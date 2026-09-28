"""Ready-made configurations to start from.

`baseline()` is a plausible Formula Student three-element rear wing whose main
plane is instrumented: 1 m span, 300 mm chord, 1 mm twill skin with a shear web
at 65% chord, two pylons at the quarter-span points, and the two upper flaps
delivering their load through brackets.  Change what you need and re-solve.
"""
from __future__ import annotations

from .aero import AeroSpec, CarriedElement
from .airfoil import Airfoil
from .beam import Support
from .gauges import Gauge, bending_pair, shear_pair
from .laminate import Laminate
from .model import Wing, WingSpec


def baseline_gauges() -> list:
    """A starting instrumentation plan: two bending stations + one shear pair."""
    g = []
    # main bending station, inboard of the left pylon, at max section depth
    g += bending_pair("B1", y=0.46, xc=0.30, bridge="half", noise_ue=1.0,
                      transverse_sensitivity=0.01,
                      temp_apparent_ue_per_K=1.5)
    # second station on the overhang, opposite moment sign - gives a second
    # independent equation for reconstructing the load distribution
    g += bending_pair("B2", y=0.16, xc=0.30, bridge="half", noise_ue=1.0,
                      transverse_sensitivity=0.01,
                      temp_apparent_ue_per_K=1.5)
    # +-45 pair for torsion -> centre of pressure
    g += shear_pair("S1", y=0.46, xc=0.30, surface="top", bridge="half",
                    noise_ue=1.5, temp_apparent_ue_per_K=1.5)
    return g


def baseline() -> Wing:
    spec = WingSpec(
        span=1.000,
        chord=0.300,
        taper_ratio=1.0,
        angle_deg=8.0,
        inverted=True,
        airfoil=Airfoil.naca4("4412"),
        skin=Laminate.symmetric([45, 0], "twill_200", "skin 4-ply twill"),
        web=Laminate.build([45, -45], "twill_200", "web 2-ply twill"),
        x_front=0.0,
        x_rear=0.65,
        # two pylons off the chassis, 360 mm apart: typical FS layout, and far
        # from the moment-cancelling spacing of span/4 that kills gauge signal
        supports=[Support(0.32, "pin", label="pylon_L"),
                  Support(0.68, "pin", label="pylon_R")],
        aero=AeroSpec(
            CL=1.70, CD=0.36, xc_cp=0.32,
            distribution="blend", tip_loading=0.75,
            carried=[
                CarriedElement("flap_1", CL_S=0.36, CD_S=0.10,
                               bracket_y=(0.30, 0.70), xc_app=0.55),
                CarriedElement("flap_2", CL_S=0.27, CD_S=0.09,
                               bracket_y=(0.30, 0.70), xc_app=0.60),
            ]),
        gauges=baseline_gauges(),
    )
    return Wing(spec)


def endplate_mounted_flaps(wing: Wing) -> Wing:
    """Variant: flaps hang off the endplates, so their load enters at the tips."""
    import copy
    w = Wing(copy.deepcopy(wing.spec))
    for el in w.spec.aero.carried:
        el.bracket_y = (0.02, 0.98)
    return w


def thin_skin(wing: Wing, plies=(45, 0)) -> Wing:
    """Variant: lighter skin -> more strain per newton, less stiffness."""
    import copy
    w = Wing(copy.deepcopy(wing.spec))
    w.spec.skin = Laminate.build(list(plies), "twill_200", "thin skin")
    return w
