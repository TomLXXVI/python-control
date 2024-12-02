"""
Definitions of components in a translational mechanical system: mass, spring,
and damper.
"""
from abc import ABC, abstractmethod
import sympy as sp
from python_control import Quantity
from python_control.core.symbols import s
from python_control.core import TransferFunction


TranslationalImpedance = TransferFunction


class TranslationalComponent(ABC):
    """
    Abstract base class that models a translational mechanical component.

    Attributes
    ----------
    value: Quantity | sympy.Symbol
        The value of the translational component.
    Z: TransferFunction
        The (displacement) impedance of the translational component, defined as
        the ratio of force F(s) to displacement X(s).
    Y: TransferFunction
        The (displacement) admittance of the translational component, i.e. the
        inverse of the impedance.
    Z_v: TransferFunction
        Velocity impedance, defined as the ratio of force F(s) to speed V(s)
        (see notes).
    Y_v: TransferFunction
        Velocity admittance, i.e. the inverse of velocity impedance.

    Notes
    -----
    If impedance Z = F(s) / X(s) is divided by (Sympy symbol) s, we get the
    transfer function F(s) / V(s), with V(s) the Laplace transform of the
    velocity-time function v(t):
        F(s) = Z * X(s)
        F(s) = (Z / s) * s * X(s)
        F(s) = (Z / s) * V(s)
    The transfer function F(s) / V(s) is called the "velocity impedance" and can
    be retrieved through attribute `Z_v` of this class. The inverse of the
    velocity impedance is the "velocity admittance", which is retrievable
    through attribute `Y_v`.
    """
    def __init__(self, value: Quantity | str):
        if isinstance(value, Quantity):
            self.value = value.to_base_units()
            # displacement impedance F(s) / X(s)
            self.Z = self._create_impedance(self.value.m)
        else:
            self.value = sp.Symbol(value)
            self.Z = self._create_impedance(self.value)
        # displacement admittance X(s) / F(s)
        self.Y = TranslationalImpedance(1 / self.Z.expr)
        # velocity impedance F(s) / V(s) and admittance V(s) / F(s)
        self.Z_v = self.Z / s
        self.Y_v = self.Y / s

    @abstractmethod
    def _create_impedance(self, value) -> TranslationalImpedance:
        pass


class Spring(TranslationalComponent):
    """
    Translational component that models a spring. Its value represents the
    spring constant in units N / m:
    F(t) = k * x(t) [Laplace transform: F(s) = k * X(s)].
    """
    def _create_impedance(self, value) -> TranslationalImpedance:
        return TranslationalImpedance(value)


class Damper(TranslationalComponent):
    """
    Translational component that models a viscous damper. Its value represents
    the coefficient of viscous friction in units N / (m / s):
    F(t) = f_v * dx(t)/dt [Laplace transform: F(s) = f_v * s * X(s)].
    """
    def _create_impedance(self, value) -> TranslationalImpedance:
        return TranslationalImpedance(value * s)


class Mass(TranslationalComponent):
    """
    Translational component that models the mass of a body. Its value represents
    the mass in units N / (m / s²) = kg:
    F(t) = m * dx²(t)/dt² [Laplace transform: F(s) = k * s² * X(s)].
    """
    def _create_impedance(self, value) -> TranslationalImpedance:
        return TranslationalImpedance(value * s ** 2)
