"""
Defines the components of a translational mechanical system: mass, spring,
and damper.
"""
from abc import ABC, abstractmethod
import sympy as sp
from python_control import Quantity
from python_control.core.symbols import s
from python_control.core import TransferFunction


RotationalImpedance = TransferFunction


class RotationalComponent(ABC):
    """
    Abstract base class that models any rotational mechanical component.

    Attributes
    ----------
    value: Quantity | str
        The value of the rotational component.
    Z: TransferFunction
        The (displacement) impedance of the rotational component, defined as the
        ratio of torque T(s) to angular displacement Θ(s).
    Y: TransferFunction
        The (displacement) admittance of the rotational component, i.e. the
        inverse of the impedance.
    Z_v: TransferFunction
        Velocity impedance, defined as the ratio of torque T(s) to angular
        speed Ω(s).
    Y_v: TransferFunction
        Velocity admittance, i.e. the inverse of velocity impedance.
    """
    def __init__(self, value: Quantity | str):
        if isinstance(value, Quantity):
            self.value = value.to_base_units()
            self.Z = self._create_impedance(self.value.m)  # impedance
        else:
            self.value = sp.Symbol(value)
            self.Z = self._create_impedance(self.value)
        self.Y = RotationalImpedance(1 / self.Z.expr)  # admittance
        self.Z_v = self.Z / s
        self.Y_v = self.Y / s

    @abstractmethod
    def _create_impedance(self, value) -> RotationalImpedance:
        pass


class TorsionSpring(RotationalComponent):
    """
    Rotational component that models a torsion spring. Its value represents the
    spring constant in units N.m / rad:
    GH(t) = k * theta(t) [Laplace transform: GH(s) = k * theta(s)].
    """
    def _create_impedance(self, value) -> RotationalImpedance:
        return RotationalImpedance(value)


class TorsionDamper(RotationalComponent):
    """
    Rotational component that models a viscous damper. Its value represents
    the coefficient of viscous friction in units N.m / (rad / s):
    GH(t) = D * theta(t)/dt [Laplace transform: GH(s) = D * s * theta(s)].
    """
    def _create_impedance(self, value) -> RotationalImpedance:
        return RotationalImpedance(value * s)


class Inertia(RotationalComponent):
    """
    Rotational component that models the inertia of a body. Its value represents
    the inertia in units N.m / (rad / s²) = kg.m²:
    GH(t) = J * dx²(t)/dt² [Laplace transform: GH(s) = J * s² * theta(s)].
    """
    def _create_impedance(self, value) -> RotationalImpedance:
        return RotationalImpedance(value * s ** 2)


class GearRatio:

    def __init__(
        self,
        N_in: int | str | sp.Symbol,
        N_out: int | str | sp.Symbol
    ) -> None:
        """
        Creates a `GearRatio` object.

        Parameters
        ----------
        N_in:
            Number of teeth of the input gear.
        N_out:
            Number of teeth of the output gear.
        """
        if isinstance(N_in, int):
            self.N_in = N_in
        elif isinstance(N_in, str):
            self.N_in = sp.Symbol(N_in)
        else:
            self.N_in = N_in

        if isinstance(N_out, int):
            self.N_out = N_out
        elif isinstance(N_out, str):
            self.N_out = sp.Symbol(N_out)
        else:
            self.N_out = N_out

        self.torque_ratio = self.N_out / self.N_in     # T_out / T_in
        self.angle_ratio = self.N_in / self.N_out      # theta_out / theta_in
        self.impedance_ratio = self.torque_ratio ** 2  # Z_out / Z_in

    def reflect_to_output(
        self, *,
        Z_in: RotationalImpedance | None = None,
        T_in: Quantity | sp.Expr | None = None,
        theta_in: Quantity | sp.Expr | None = None
    ) -> RotationalImpedance | Quantity | sp.Expr:
        """
        Reflects the impedance `Z_in` or torque `T_in` on the input side to the
        output side of the gears.
        """
        if Z_in is not None:
            Z_in_to_out = self.impedance_ratio * Z_in.expr
            Z_in_to_out = RotationalImpedance(Z_in_to_out)
            return Z_in_to_out
        if T_in is not None:
            T_in_to_out = self.torque_ratio * T_in
            return T_in_to_out
        if theta_in is not None:
            theta_in_to_out = self.angle_ratio * theta_in
            return theta_in_to_out

    def reflect_to_input(
        self, *,
        Z_out: RotationalImpedance | None = None,
        T_out: Quantity | sp.Expr | None = None,
        theta_out: Quantity | sp.Expr | None = None
    ) -> RotationalImpedance | Quantity | sp.Expr:
        """
        Reflects the impedance `Z_out` or torque `T_out` on the output side to
        the input side of the gears.
        """
        if Z_out is not None:
            Z_out_to_in = (self.impedance_ratio ** -1) * Z_out.expr
            Z_out_to_in = RotationalImpedance(Z_out_to_in)
            return Z_out_to_in
        if T_out is not None:
            T_out_to_in = (self.torque_ratio ** -1) * T_out
            return T_out_to_in
        if theta_out is not None:
            theta_out_to_in = (self.angle_ratio ** -1) * theta_out
            return theta_out_to_in

    def __add__(self, other: 'GearRatio') -> 'GearRatio':
        """
        Combines two cascading gear ratios into one overall gear ratio.
        """
        N_in = self.N_in * other.N_in
        N_out = self.N_out * other.N_out
        return GearRatio(N_in, N_out)
