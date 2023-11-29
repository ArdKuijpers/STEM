from dataclasses import dataclass
from abc import ABC


@dataclass
class WaterProcessParametersABC(ABC):
    """
    Abstract class which contains the parameters for a water boundary.
    """


@dataclass
class UniformWaterPressure(WaterProcessParametersABC):
    """
    Class which contains the parameters for a uniform water boundary.

    Attributes:
        - water_pressure (float): The water pressure.
        - is_fixed (bool): Whether the water pressure is fixed or not (default: True).

    """
    water_pressure: float
    is_fixed: bool = True

