"""Strain-gauge-based downforce measurement model for a composite wing.

Forward path : geometry + layup + mounts + aero  ->  strain at each gauge
Inverse path : measured strains  ->  total downforce (+ load distribution, CoP)
"""
from .aero import AeroSpec, CarriedElement, air_density, spanwise_shape
from .airfoil import Airfoil
from .beam import Beam, PointLoad, Support
from .gauges import Gauge, bending_pair, rosette, shear_pair
from .laminate import MATERIALS, Laminate, Ply
from .model import Result, Wing, WingSpec
from .section import box_section

__all__ = ["AeroSpec", "CarriedElement", "air_density", "spanwise_shape",
           "Airfoil", "Beam", "PointLoad", "Support", "Gauge", "bending_pair",
           "rosette", "shear_pair", "MATERIALS", "Laminate", "Ply", "Result",
           "Wing", "WingSpec", "box_section"]
__version__ = "1.0"
