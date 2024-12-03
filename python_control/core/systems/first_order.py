import sympy as sp

from python_control import Quantity

from ..transfer_function import TransferFunction
from ..symbols import s, t
from .. import equation

Q_ = Quantity


class FirstOrderSystem(TransferFunction):
    """
    Implements a first-order system of the general form: K / (s + a)
    """
    def __init__(self, K: float, a: float, name: str = '') -> None:
        self.name = name
        super().__init__(K / (s + a))
        # unit step response in s-domain and time-domain:
        self.U = self.response(1 / s)
        self.u = self.U.inverse()

    @property
    def tau(self) -> Quantity:
        """Returns the time constant."""
        p = self.poles_control[0]
        tau = abs(1 / p)
        return Q_(tau, 's')

    @property
    def steady_state_output(self) -> float:
        """Returns the steady-state value of the unit step response."""
        output = self.u.evaluate(t=float('inf'))
        return output

    @property
    def rise_time(self) -> Quantity:
        """Returns the rise time of the unit step response."""
        sso = self.steady_state_output
        output_10per = 0.1 * sso
        output_90per = 0.9 * sso
        eq_10per = sp.Eq(self.u.expr, output_10per)
        eq_90per = sp.Eq(self.u.expr, output_90per)
        t_10per = equation.solve(eq_10per, t, domain=sp.S.Reals)
        t_90per = equation.solve(eq_90per, t, domain=sp.S.Reals)
        t_rise = t_90per - t_10per
        return Q_(t_rise, 's')

    @property
    def settling_time(self) -> Quantity:
        """Returns the settling time of the unit step response."""
        output_98per = 0.98 * self.steady_state_output
        eq_98per = sp.Eq(self.u.expr, output_98per)
        t_98per = equation.solve(eq_98per, t, domain=sp.S.Reals)
        return Q_(float(t_98per), 's')
