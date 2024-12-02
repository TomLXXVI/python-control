"""
Defines the components of a thermal network: resistor and capacitor.
"""
from abc import ABC, abstractmethod
import sympy as sp
from python_control import Quantity
from python_control.core import TransferFunction


ThermalImpedance = TransferFunction


class ThermalComponent(ABC):
    """
    Abstract base class that models any thermal component.

    Attributes
    ----------
    value: Quantity | str
        The value of the thermal component. E.g., in the case of a resistor,
        this would represent its resistance value. When the value is not known,
        it can be represented by a symbol (i.e., a string).
    Z: ThermalImpedance
        The impedance of the thermal component.
    Y: ThermalImpedance
        The admittance of the thermal component, i.e., the inverse of the
        impedance.
    """

    def __init__(self, value: Quantity | str):
        if isinstance(value, Quantity):
            self.value = value.to_base_units()
            self.Z = self._create_impedance(self.value.m)
        else:
            self.value = sp.Symbol(value)
            self.Z = self._create_impedance(self.value.m)
        self.Y = ThermalImpedance(1 / self.Z.expr)

    @abstractmethod
    def _create_impedance(self, value) -> ThermalImpedance:
        pass


class Resistor(ThermalComponent):
    """
    Thermal component that models a resistor. Its value represents the
    resistance of the resistor in units K/W.
    """
    def _create_impedance(self, value) -> ThermalImpedance:
        return ThermalImpedance(f"{value}")


class Capacitor(ThermalComponent):
    """
    Thermal component that models a capacitor. Its value represents the
    capacitance of the capacitor in units J/K.
    """
    def _create_impedance(self, value) -> ThermalImpedance:
        return ThermalImpedance(f"1 / ({value} * s)")
