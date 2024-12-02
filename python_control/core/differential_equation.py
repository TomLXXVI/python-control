from typing import Union
from collections.abc import Sequence

import sympy as sp

from .symbols import s, t
from .laplace_transform import LaplaceTransform, InverseLaplaceTransform


class DifferentialEquation:
    """
    Implements a linear time invariant (LTI) differential equation.
    """
    def __init__(
        self,
        f: str | sp.Function,
        coeffs: Sequence[float],
        init_vals: Sequence[float | str] | None = None
    ) -> None:
        """
        Creates a `DifferentialEquation` object.

        Parameters
        ----------
        f:
            Either a string representing the symbolic name of the function for
            which the differential equation needs to be solved (this could be
            the output function of a system), or a known Sympy function, being
            the input function to the system (in that case, the differential
            equation to be solved is actually the response of the system).
        coeffs:
            The coefficients of the terms in the differential equation, ordered
            from highest to lowest order.
        init_vals: optional
            The initial values of the function f and its derivatives (except
            the highest order derivative of f), ordered from highest to lowest
            order. If `None`, then all initial values are assumed zero.
        """
        if isinstance(f, str):
            self.f = sp.Function(f)(t)
            self.F = sp.Function(self.f.func.name.upper())(s)
        else:
            self.f = f
            self.F = sp.laplace_transform(self.f, t, s, noconds=True)

        self.coeffs = coeffs
        self.order = len(coeffs) - 1

        if init_vals is None:
            self.init_vals = [0.0 for _ in range(self.order)]
        else:
            self.init_vals = init_vals

        # Expression of the differential equation in the time domain:
        self.expr = self._build_diff_eq()

        # Expression of the Laplace transform of the differential equation:
        self.laplace = self._transform_diff_eq()

    def _build_diff_eq(self):
        """
        Builds the symbolic expression of the differential equation in the
        time domain.
        """
        order = [t] * self.order
        terms = []
        for coeff in self.coeffs[:-1]:
            deriv = sp.Derivative(self.f, *order)
            terms.append(coeff * deriv)
            order.pop()
        terms.append(self.coeffs[-1] * self.f)
        diff_eq = sp.Add(*terms)
        return diff_eq

    def _init_vals_dict(self):
        """
        Creates a dictionary that maps the initial values to the initial
        condition expressions present in the differential equation, e.g., y(0),
        y'(0), y"(0) in case of a 3rd order differential equation.
        """
        ini_dict = {self.f.subs(t, 0): self.init_vals[-1]}
        ini_dict.update({
            self.f.diff(t, i).subs(t, 0): v
            for v, i in zip(
                self.init_vals[:-1],
                (i for i in range(self.order - 1, 0, -1))
            )
        })
        return ini_dict

    def _transform_diff_eq(self):
        """
        Returns the Laplace transform of the differential equation.
        """
        lap_eq = sp.laplace_transform(self.expr, t, s, noconds=True)
        # Replace the symbolic notation of the Laplace transform of self.f in
        # the expression by the notation referred to by self.F:
        lap_eq = lap_eq.subs({sp.LaplaceTransform(self.f, t, s): self.F})
        if self.init_vals:
            # Substitute the initial values into the expression:
            lap_eq = lap_eq.subs(self._init_vals_dict())
        return lap_eq

    def solve(
        self,
        rhs: Union['DifferentialEquation', sp.Function, sp.Expr]
    ) -> tuple[LaplaceTransform, InverseLaplaceTransform]:
        """
        Algebraically solves the differential equation for the function f in
        the lhs of the equation.

        Parameters
        ----------
        rhs:
            The right-hand side of the differential equation, which can be
            another `DifferentialEquation` object, a single Sympy function, or
            a Sympy expr.

        Returns
        -------
        F:
            The Laplace transform of the solution (`LaplaceTransform` object).
        f:
            The inverse Laplace transform of the solution
            (`InverseLaplaceTransform` object), which is actually the solution
            for the function in the time domain.
        """
        if isinstance(rhs, DifferentialEquation):
            rhs = rhs.laplace
        else:
            rhs = sp.laplace_transform(rhs, t, s, noconds=True)
        eq = self.laplace - rhs
        F = sp.solve(eq, self.F)[0]
        f = sp.inverse_laplace_transform(F, s, t, simplify=True)
        return LaplaceTransform(F), InverseLaplaceTransform(f)
