"""
Defines the passive linear components of an electrical circuit:
- resistor,
- capacitor, and
- inductor.
"""
from abc import ABC, abstractmethod
import sympy as sp

from python_control import Quantity
from python_control.core.symbols import s
from python_control.core import TransferFunction


ElectricalImpedance = TransferFunction


class ElectricalComponent(ABC):
    """
    Abstract base class that models any electrical component.

    Attributes
    ----------
    value: Quantity | str
        The value of the electrical component. E.g., in the case of a resistor,
        this would be the magnitude of its resistance. When the value is not
        known, it can be represented symbolically by a string.
    Z: ElectricalImpedance
        The impedance of the electrical component.
    Y: ElectricalImpedance
        The admittance of the electrical component, i.e. the inverted impedance.
    """

    def __init__(self, value: Quantity | str):
        if isinstance(value, Quantity):
            self.value = value.to_base_units()
            self.Z = self._create_impedance(self.value.m)  # impedance
        else:
            # self.value = sp.Symbol(value, positive=True, real=True)
            self.value = sp.parse_expr(value)
            self.Z = self._create_impedance(self.value)
        self.Y = ElectricalImpedance(1 / self.Z.expr)  # admittance

    @abstractmethod
    def _create_impedance(self, value) -> ElectricalImpedance:
        pass


class Resistor(ElectricalComponent):
    """
    Electrical component that models a resistor. Its value represents the
    resistance of the resistor in units Ohm.
    """
    def _create_impedance(self, value) -> ElectricalImpedance:
        return ElectricalImpedance(value)


class Capacitor(ElectricalComponent):
    """
    Electrical component that models a capacitor. Its value represents the
    capacitance of the capacitor in units Farad (F).
    """
    def _create_impedance(self, value) -> ElectricalImpedance:
        return ElectricalImpedance(1 / (value * s))


class Inductor(ElectricalComponent):
    """
    Electrical component that models an inductor. Its value represents the
    inductance of the inductor in units Henry (H).
    """
    def _create_impedance(self, value) -> ElectricalImpedance:
        return ElectricalImpedance(value * s)
