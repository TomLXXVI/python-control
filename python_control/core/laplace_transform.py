import warnings
import sympy as sp
import numpy as np
from numpy.exceptions import ComplexWarning

from .symbols import s, t

warnings.filterwarnings('ignore', category=ComplexWarning)


class InverseLaplaceTransform:
    """
    Wrapper around the expression of a function *f(t)* in the time domain,
    which can be considered as the inverse Laplace transform of a function
    *F(s)* in the s-domain.
    """
    def __init__(self, f: sp.Expr | str) -> None:
        """
        Creates an `InverseLaplaceTransform` object.

        Parameters
        ----------
        f:
            Function in the time domain *f(t)* which is the inverse Laplace
            transform of the function *F(s)* in the s-domain.
        """
        if isinstance(f, str):
            f = sp.parse_expr(f)
            f = f.subs('t', t)
        # self.f = f.expand(complex=True)
        self.f = f
        self._fun_mpmath = sp.lambdify(t, self.f, 'mpmath')
        self._fun_numpy = sp.lambdify(t, self.f, 'numpy')

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return str(self.expr)

    @property
    def expr(self) -> sp.Expr:
        """
        Returns the Sympy expression of the function *f(t)*.
        """
        f = self.f
        f = sp.expand(f)
        f = sp.collect(f, sp.exp(sp.Wild('w')))
        return f

    def transform(self) -> 'LaplaceTransform':
        """
        Returns the Laplace transform *F(s)* of the time-function *f(t)*
        """
        F = sp.laplace_transform(self.f, t, s, noconds=True)
        F = LaplaceTransform(F)
        return F

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Returns the value(s) of *f(t)* at time moment(s) *t*.
        """
        y = None
        if isinstance(t, (int, float)):
            y = self._fun_mpmath(t)
            try:
                y = float(y)
            except TypeError:
                y = complex(y).real
        elif isinstance(t, np.ndarray):
            y = self._fun_numpy(t)
            if not isinstance(y, np.ndarray) and len(t) > 1:
                # y is a constant (step function)
                y = np.array([y] * len(t), dtype=float)
        return y


class LaplaceTransform:
    """
    Wrapper around the expression of the Laplace transform *F(s)* of a function
    *f(t)* in the time domain.
    """
    def __init__(self, F: sp.Expr | str) -> None:
        """
        Creates a `LaplaceTransform` object.

        Parameters
        ----------
        F:
            Expression of the function in the s-domain *F(s)* which is the
            Laplace transform of the t-domain function *f(t)*.
        """
        if isinstance(F, str):
            F = sp.parse_expr(F)
        F = F.subs('s', s)
        F = F.normal()
        F = sp.cancel(F, s)
        self.F = F

    def __repr__(self):
        return str(self.expr)

    def __str__(self):
        return str(self.expr)

    @property
    def expr(self) -> sp.Expr:
        """Returns the Sympy expression of the function F(s)."""
        F = self.F
        F = F.normal()
        F = sp.expand(F)
        F = sp.collect(F, s)
        F = sp.simplify(F)
        return F

    def inverse(self) -> InverseLaplaceTransform:
        """
        Returns the inverse Laplace transform *f(t)* of the s-domain function
        *F(s)*
        """
        F = self.expanded(evaluate=True)
        f = sp.inverse_laplace_transform(F, s, t)
        f = InverseLaplaceTransform(f)
        return f

    def expanded(self, evaluate: bool = False) -> sp.Expr:
        """
        Returns the partial fraction expansion of *F(s)*.
        """
        F = sp.nsimplify(self.F)  # convert floating-point numbers to rationals
        F = sp.apart(F, s, full=True)  # full partial fraction decomposition
        F = F.doit()  # evaluate any unevaluated expressions
        if evaluate: F = F.evalf()  # numerically evaluate the expression
        return F

    @property
    def numerator(self) -> np.ndarray:
        """
        Returns the numerator of *F(s)* as a Numpy-array of the polynomial
        lhs in the numerator expression.
        """
        num, _ = sp.fraction(self.F)
        num = sp.Poly(num, s).all_coeffs()
        num = np.array([x for x in map(lambda x: float(x), num)])
        return num

    @property
    def denominator(self) -> np.ndarray:
        """
        Returns the denominator of *F(s)* as a Numpy-array of the polynomial
        lhs in the denominator expression.
        """
        _, den = sp.fraction(self.F)
        den = sp.Poly(den, s).all_coeffs()
        den = np.array([x for x in map(lambda x: float(x), den)])
        return den
