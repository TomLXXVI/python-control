"""
Implementation of the class `TransferFunction` which wraps both the Sympy and
the "Python Control Systems Library" (PCSL) implementations of a single-block
transfer function.
"""
from __future__ import annotations
from collections.abc import Sequence
from collections import namedtuple
import numpy as np
import sympy as sp
import control as ct
import sympy.physics.control.lti as sp_lti
import sympy.physics.control.control_plots as sp_plot
from sympy.printing.latex import latex
import matplotlib.pyplot as plt
from .state_space import StateSpace
from .laplace_transform import LaplaceTransform
from .symbols import s


Denominator = namedtuple('Denominator', ('as_poly', 'as_array'))
Numerator = namedtuple('Numerator', ('as_poly', 'as_array'))


class TransferFunction:
    """
    Implements properties and methods that can be applied to a transfer function
    of a single-block system:
                            G(s) = C(s) / R(s)
    where:
    C(s) is the Laplace transform of the output signal leaving the system, and
    R(s) is the Laplace transform of the input signal entering the system.
    """
    def __init__(self, expr: sp.Expr | str | float | int) -> None:
        """
        Creates a `TransferFunction` object from a Sympy or string expression
        of the transfer function.
        """
        self._create_sympy_transfer_function(expr)
        self._create_pcsl_transfer_function()

    def _create_sympy_transfer_function(self, expr):
        if isinstance(expr, str):
            expr = sp.parse_expr(expr)
        elif isinstance(expr, (float, int)):
            expr = sp.parse_expr(str(expr))
        expr = expr.subs('s', s)
        expr = sp.nsimplify(expr)
        expr = expr.normal()
        expr = sp.cancel(expr, s)
        self._sp_expr = expr
        self._sp_tf = sp_lti.TransferFunction.from_rational_expression(self._sp_expr, var=s)

    def _create_pcsl_transfer_function(self):
        num_coeffs = self._get_num_coeffs()
        den_coeffs = self._get_den_coeffs()
        nonzero_indices = np.nonzero(den_coeffs)[0]
        i = nonzero_indices[0] if nonzero_indices.size > 0 else None
        if i is not None:
            self.num_coeffs = num_coeffs / den_coeffs[i]
            self.den_coeffs = den_coeffs / den_coeffs[i]
        else:
            self.num_coeffs = np.array([])
            self.den_coeffs = np.array([])
        try:
            self._ct_tf = ct.TransferFunction(self.num_coeffs, self.den_coeffs)
        except (sp.PolynomialError, TypeError):
            self._ct_tf = None

    @classmethod
    def from_coefficients(
        cls,
        num: Sequence[float],
        den: Sequence[float]
    ) -> TransferFunction:
        """
        Creates a `TransferFunction` object when the coefficients of the
        numerator and the denominator of the transfer function (or differential
        equation) are given.
        
        Parameters
        ----------
        num:
            Sequence of the polynomial coefficients of the numerator (i.e. the
            right-hand side or input side of the differential equation) ordered
            from highest to lowest degree.
        den:
            Sequence of the polynomial coefficients of the denominator (i.e. the
            left-hand side or output side of the differential equation) ordered
            from highest to lowest degree.
        """
        num = cls._create_poly_expr(num)
        den = cls._create_poly_expr(den)
        tf = cls(f"({num}) / ({den})")
        return tf

    @classmethod
    def from_state_space(cls, ss: StateSpace):
        tf = ct.ss2tf(ss.as_ct)
        num = np.round(tf.num[0][0], 6)
        den = np.round(tf.den[0][0], 6)
        tf = cls.from_coefficients(num, den)
        return tf

    @staticmethod
    def _create_poly_expr(coeff: Sequence[float]) -> str:
        """
        Transforms the sequence of polynomial coefficients into a string
        expression.
        """
        poly_str = []
        n = len(coeff) - 1
        for i, c in enumerate(coeff):
            c = float(c)
            e = n - i
            if e > 1:
                poly_str.append(f"{c} * s ** {e}")
            elif e == 1:
                poly_str.append(f"{c} * s")
            else:
                poly_str.append(f"{c}")
        poly_str = "+".join(poly_str)
        return poly_str

    @property
    def __old_expr(self) -> sp.Expr:
        """
        Returns the Sympy expression of the `TransferFunction` object.
        """
        expr = self._sp_expr
        num, den = expr.as_numer_denom()
        if num.has(s):
            num = sp.expand(num)
            num = sp.collect(num, s)
        if den.has(s):
            den = sp.expand(den)
            den = sp.collect(den, s)
        expr = num / den
        expr = expr.normal()
        expr = sp.cancel(expr, s)
        return expr

    @property
    def expr(self) -> sp.Expr:
        num = sp.Poly(self.num_coeffs, s).as_expr()
        den = sp.Poly(self.den_coeffs, s).as_expr()
        return num / den

    @property
    def as_sympy(self) -> sp_lti.TransferFunction:
        """
        Returns the Sympy implementation of the transfer function.
        """
        return self._sp_tf

    @property
    def as_ct(self) -> ct.TransferFunction:
        """
        Returns the PCSL implementation of the transfer function.
        """
        return self._ct_tf

    def _get_num_coeffs(self) -> np.ndarray:
        """
        Returns the coefficients of the numerator of the transfer function as a
        Numpy array.
        """
        num = self._sp_tf.num.as_poly(s) or np.array([float(self._sp_tf.num)])
        if isinstance(num, sp.Poly):
            try:
                num = np.array([
                    x for x in map(lambda x: float(x), num.all_coeffs())
                ])
            except TypeError:
                num = np.array(num.all_coeffs())
        return num
    
    def _get_den_coeffs(self) -> np.ndarray:
        """
        Returns the coefficients of the denominator of the transfer function as
        a Numpy array.
        """
        den = self._sp_tf.den.as_poly(s) or np.array([float(self._sp_tf.den)])
        if isinstance(den, sp.Poly):
            try:
                den = np.array([
                    x for x in map(lambda x: float(x), den.all_coeffs())
                ])
            except TypeError:
                den = np.array(den.all_coeffs())
        return den

    @property
    def numerator_poly(self) -> sp.Poly:
        """
        Returns the numerator of the transfer function as a Sympy `Poly` object.
        """
        return self._sp_tf.num.as_poly(s)

    @property
    def denominator_poly(self) -> sp.Poly:
        """
        Returns the denominator of the transfer function as a Sympy `Poly`
        object.
        """
        return self._sp_tf.den.as_poly(s)

    @property
    def numerator(self) -> Numerator:
        """
        Returns a named tuple of type `Numerator` with two members:
        -   `as_poly`:
                Returns the numerator of the transfer function as a Sympy
                `Poly` object.
        -   `as_array`:
                Returns the coefficients of the transfer function's numerator.
        """
        num_arr = self.num_coeffs
        num_poly = self.numerator_poly
        return Numerator(num_poly, num_arr)

    @property
    def denominator(self) -> Denominator:
        """
        Returns a named tuple of type `Denominator` with two members:
        -   `as_poly`:
                Returns the denominator of the transfer function as a
                Sympy `Poly` object.
        -   `as_array`:
                Returns the coefficients of the transfer function's denominator.
        """
        den_arr = self.den_coeffs
        den_poly = self.denominator_poly
        return Denominator(den_poly, den_arr)

    @property
    def poles_pcsl(self) -> list[complex]:
        """
        Returns the poles of the transfer function using PCSL.
        """
        return self._ct_tf.poles().tolist()

    @property
    def poles_sympy(self) -> list[complex]:
        """
        Returns the poles of the transfer function using Sympy.
        """
        poles = self._sp_tf.poles()
        poles = [pole for pole in poles]
        return poles

    @property
    def poles(self) -> list[complex]:
        """
        Returns the poles of the transfer function.
        """
        poles_sympy = self.poles_sympy
        try:
            poles_control = self.poles_pcsl
        except AttributeError:
            poles_control = None
        if poles_control is not None:
            return poles_control
        else:
            return poles_sympy

    @property
    def zeros_pcsl(self) -> list[complex]:
        """
        Returns the zeros of the transfer function using PCSL.
        """
        return self._ct_tf.zeros()

    @property
    def zeros_sympy(self) -> list[complex]:
        """
        Returns the zeros of the transfer function using Sympy.
        """
        zeros = self._sp_tf.zeros()
        zeros = [zero for zero in zeros]
        return zeros

    @property
    def zeros(self) -> list[complex]:
        """
        Returns the zeros of the transfer function.
        """
        zeros_sympy = self.zeros_sympy
        try:
            zeros_control = self.zeros_pcsl
        except AttributeError:
            zeros_control = None
        if zeros_control is not None:
            return zeros_control
        else:
            return zeros_sympy

    @property
    def dc_gain(self) -> sp.Expr | float:
        """
        Returns the dc gain of the transfer function, i.e. the total gain
        when s = 0.
        """
        try:
            return float(self.as_sympy.dc_gain())
        except TypeError:
            return self.as_sympy.dc_gain()

    @property
    def gain(self) -> float:
        """
        Returns the gain factor K of the transfer function.
        """
        dc_gain = self.dc_gain
        T = self
        while dc_gain == float('inf'):
            T = self * s
            dc_gain = T.dc_gain
        prod_zeros = np.prod([-z for z in T.zeros])
        prod_poles = np.prod([-p for p in T.poles])
        K = (dc_gain / (prod_zeros / prod_poles))
        if isinstance(K, complex):
            K = K.real
        return K

    @property
    def is_stable(self) -> bool:
        """
        Indicates whether the system is stable or not.
        """
        if self._ct_tf is not None:
            if all(p.real < 0 for p in self.poles_pcsl):
                return True
            return False
        return True
        # if the transfer function contains symbolic coefficients, we assume
        # these coefficients will attain values so that the system will be
        # stable

    def steady_state_error(self, R: sp.Expr) -> sp.Number:
        """
        Returns the steady-state error between the input `R` and the output `C`
        of the system.

        Parameters
        ----------
        R:
            Sympy expression of the input signal entering the single-block
            system.

        Notes
        -----
        The error signal E is defined as E = R - C with C = T * R where T is the
        transfer function of the single-block system. So, the error signal E can
        also be written as E = R - T * R = (1 - T) * R.
        The steady-state error is then, using Sympy syntax, defined as:
        e_oo = limit(s * E, s, 0).
        """
        E = (1 - self._sp_expr) * R
        e_oo = sp.limit(s * E, s, 0)
        return e_oo

    def response(
        self,
        input_: LaplaceTransform | sp.Expr | str
    ) -> LaplaceTransform:
        """
        Returns the Laplace transform of the output signal when the Laplace
        transform of the input signal is given.
        """
        if isinstance(input_, str):
            input_ = sp.parse_expr(input_)
        elif isinstance(input_, LaplaceTransform):
            input_ = input_.expr
        output_expr = sp.Mul(self._sp_expr, input_)
        return LaplaceTransform(output_expr.evalf())

    def __str__(self):
        return str(self.expr)

    def __mul__(self, other: TransferFunction | int | float) -> TransferFunction:
        if isinstance(other, TransferFunction):
            series = sp_lti.Series(self._sp_tf, other._sp_tf)
            tf = series.doit()
            return TransferFunction(tf.to_expr())
        else:
            expr = self._sp_expr * other
            return TransferFunction(expr)

    def __rmul__(self, other: TransferFunction | int | float) -> TransferFunction:
        return self.__mul__(other)

    def __truediv__(self, other: TransferFunction | int | float) -> TransferFunction:
        if isinstance(other, TransferFunction):
            series = sp_lti.Series(self._sp_tf, other._sp_tf ** -1)
            tf = series.doit()
            return TransferFunction(tf.to_expr())
        else:
            expr = self._sp_expr / other
            return TransferFunction(expr)

    def __rtruediv__(self, other: TransferFunction | int | float) -> TransferFunction:
        if isinstance(other, (int, float)) and other == 1:
            tf = self._sp_tf ** -1
            return TransferFunction(tf.to_expr())
        elif isinstance(other, TransferFunction):
            series = sp_lti.Series(other._sp_tf, self._sp_tf ** -1)
            tf = series.doit()
            return TransferFunction(tf.to_expr())
        else:
            expr = other / self._sp_expr
            return TransferFunction(expr)

    def __add__(self, other: TransferFunction) -> TransferFunction:
        if isinstance(other, TransferFunction):
            parallel = sp_lti.Parallel(self._sp_tf, other._sp_tf)
            tf = parallel.doit()
            return TransferFunction(tf.to_expr())
        else:
            expr = self._sp_expr + other
            return TransferFunction(expr)

    def __radd__(self, other: TransferFunction | int) -> TransferFunction:
        if isinstance(other, TransferFunction):
            return other + self
        elif isinstance(other, int) and other == 0:
            return self
        else:
            expr = other + self._sp_expr
            return TransferFunction(expr)

    def __sub__(self, other: TransferFunction | int) -> TransferFunction:
        if isinstance(other, TransferFunction):
            parallel = sp_lti.Parallel(self._sp_tf, -other._sp_tf)
            tf = parallel.doit()
            return TransferFunction(tf.to_expr())
        else:
            expr = self._sp_expr - other
            return TransferFunction(expr)

    def __rsub__(self, other: TransferFunction | int) -> TransferFunction:
        if isinstance(other, TransferFunction):
            return other - self
        else:
            expr = other - self._sp_expr
            return TransferFunction(expr)

    def __neg__(self) -> TransferFunction:
        tf_expr = -self._sp_expr
        return TransferFunction(tf_expr)

    def feedback(self, other: TransferFunction, sign: int = -1) -> TransferFunction:
        feedback = sp_lti.Feedback(self._sp_tf, other._sp_tf, sign)
        tf = feedback.doit(cancel=True)
        return TransferFunction(tf.to_expr())

    def unit_impulse_response(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the numerical values of the points in the unit impulse response
        plot of the system.

        kwargs
        ------
        prec: 8
            Decimal point precision for the coordinates.
        lower_limit: 0
            Lower limit of the time axis.
        upper_limit: 10
            Upper limit of the time axis.

        Returns
        -------
        time_values:
            Numpy array with the time values of the points (abscissa)
        output_values:
            Numpy array with the system's output values (ordinate).
        """
        time, output = sp_plot.impulse_response_numerical_data(
            self._sp_tf,
            prec=kwargs.pop('prec', 8),
            lower_limit=kwargs.pop('lower_limit', 0),
            upper_limit=kwargs.pop('upper_limit', 10),
            **kwargs
        )
        return np.array(time), np.array(output)

    def plot_unit_impulse_response(self, **kwargs):
        """
        Plots the unit impulse response of the transfer function.

        kwargs
        ------
        prec: 8
            Decimal point precision for the coordinates.
        lower_limit: 0
            Lower limit of the time axis.
        upper_limit: 10
            Upper limit of the time axis.
        color: str, 'b' (blue)
            The color of the plot line.
        show_axes: bool, False
            Whether to accentuate the axes with a black line, or not.
        grid: bool, True
            Whether to show a grid, or not.
        show: bool, True
            Whether to already show the figure, or not. If `False` the
            Matplotlib interface `pyplot` is returned.
        title_on: bool, True
            If `True`, the transfer function equation is displayed in the title
            of the figure.
        """
        title_on = kwargs.pop('title_on', True)
        t_ax, y_ax = self.unit_impulse_response(**kwargs)
        if title_on:
            title = f'Impulse Response of ${latex(self._sp_tf)}$'
        else:
            title = None
        return self._plot_response(
            t_ax, y_ax, title,
            **kwargs
        )

    def unit_step_response(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the numerical values of the points in the unit step response
        plot of the system.

        kwargs
        ------
        prec: 8
            Decimal point precision for the coordinates.
        lower_limit: 0
            Lower time limit of the plot range in seconds.
        upper_limit: 10
            Upper time limit of the plot range in seconds.

        Returns
        -------
        time_values:
            Numpy array with the time values of the points (abscissa)
        output_values:
            Numpy array with the system's output values (ordinate).
        """
        time, output = sp_plot.step_response_numerical_data(
            self._sp_tf,
            prec=kwargs.pop('prec', 8),
            lower_limit=kwargs.pop('lower_limit', 0),
            upper_limit=kwargs.pop('upper_limit', 10),
            **kwargs
        )
        return np.array(time), np.array(output)

    def plot_unit_step_response(self, **kwargs):
        """
        Plots the unit step response of the transfer function.

        kwargs
        ------
        prec: 8
            Decimal point precision for the coordinates.
        lower_limit: 0
            Lower limit of the time axis in seconds.
        upper_limit: 10
            Upper limit of the time axis in seconds.
        color: str, 'b' (blue)
            The color of the plot line.
        show_axes: bool, False
            Whether to accentuate the axes with a black line, or not.
        grid: bool, True
            Whether to show a grid, or not.
        show: bool, True
            Whether to already show the figure, or not. If `False` the
            Matplotlib interface `pyplot` is returned.
        title_on: bool, True
            If `True`, the transfer function equation is displayed in the title
            of the figure.
        """
        title_on = kwargs.pop('title_on', True)
        t_ax, y_ax = self.unit_step_response(**kwargs)
        if title_on:
            title = f'Step Response of ${latex(self._sp_tf)}$'
        else:
            title = None
        return self._plot_response(
            t_ax, y_ax, title,
            **kwargs
        )

    def ramp_response(
        self,
        slope: float = 1.0,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the coordinates of points on the ramp response of the transfer
        function. The ramp input signal is $k * t * u(t)$ or $k/s**2$ with $k$
        the slope.

        The ramp function is a straight line through the origin.

        Parameters
        ----------
        slope:
            Positive slope of the ramp input signal.
        kwargs:
            prec: 8
                Decimal point precision for the coordinates.
            lower_limit: 0
                Lower time limit of the plot range.
            upper_limit: 10
                Upper time limit of the plot range.

        Returns
        -------
        time_values:
            Numpy array with the time values of the points (abscissa)
        output_values:
            Numpy array with the system's output values (ordinate).
        """
        inv_L = self.response(slope / s ** 2).inverse()
        t_min = int(kwargs.pop('lower_limit', 0))
        t_max = int(kwargs.pop('upper_limit', 10))
        t_num = (t_max - t_min) * 50
        prec = kwargs.get('prec', 8)
        time = np.round(np.linspace(t_min, t_max, t_num, endpoint=True), prec)
        output = inv_L.evaluate(time)
        return time, output

    def plot_ramp_response(self, **kwargs):
        """
        Plots the ramp response of the transfer function.

        kwargs
        ------
        slope: 1.0
            Positive slope of the ramp input signal.
        prec: 8
            Decimal point precision for the coordinates.
        lower_limit: 0
            Lower time limit of the time axis.
        upper_limit: 10
            Upper time limit of the time axis.
        color: str, 'b' (blue)
            The color of the plot line.
        show_axes: bool, False
            Whether to accentuate the axes with a black line, or not.
        grid: bool, True
            Whether to show a grid, or not.
        show: bool, True
            Whether to already show the figure, or not. If `False` the
            Matplotlib interface `pyplot` is returned.
        title_on: bool, True
            If `True`, the transfer function equation is displayed in the title
            of the figure.
        """
        title_on = kwargs.pop('title_on', True)
        t_ax, y_ax = self.ramp_response(**kwargs)
        if title_on:
            title = f'Ramp Response of ${latex(self._sp_tf)}$'
        else:
            title = None
        return self._plot_response(
            t_ax, y_ax, title,
            **kwargs
        )

    def parabola_response(
        self,
        slope: float = 1.0,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the coordinates of points on the parabola response plot of the
        system. The parabola input signal is $0.5 * k * t**2 * u(t)$ or $k/s**3$
        with $k$ the slope.

        Parameters
        ----------
        slope:
            Positive slope of the ramp input signal.
        kwargs:
            prec: 8
                Decimal point precision for the coordinates.
            lower_limit: 0
                Lower time limit of the plot range.
            upper_limit: 10
                Upper time limit of the plot range.

        Returns
        -------
        time_values:
            Numpy array with the time values of the points (abscissa)
        output_values:
            Numpy array with the system's output values (ordinate).
        """
        inv_L = self.response(slope / s ** 3).inverse()
        t_min = kwargs.pop('lower_limit', 0)
        t_max = kwargs.pop('upper_limit', 10)
        t_num = (t_max - t_min) * 50
        prec = kwargs.get('prec', 8)
        time = np.round(np.linspace(t_min, t_max, t_num, endpoint=True), prec)
        output = inv_L.evaluate(time)
        return time, output

    def plot_parabola_response(self, **kwargs):
        """Plots the parabola response of the transfer function.

        kwargs
        ------
        slope: 1.0
            Positive slope of the parabola input signal.
        prec: 8
            Decimal point precision for the coordinates.
        lower_limit: 0
            Lower limit of the time axis in seconds.
        upper_limit: 10
            Upper limit of the time axis in seconds.
        color: str, 'b' (blue)
            The color of the plot line.
        show_axes: bool, False
            Whether to accentuate the axes with a black line, or not.
        grid: bool, True
            Whether to show a grid, or not.
        show: bool, True
            Whether to already show the figure, or not. If `False` the
            Matplotlib interface `pyplot` is returned.
        title_on: bool, True
            If `True`, the transfer function equation is displayed in the title
            of the figure.
        """
        title_on = kwargs.get('title_on', True)
        t_ax, y_ax = self.parabola_response(**kwargs)
        if title_on:
            title = f'Parabola Response of ${latex(self._sp_tf)}$'
        else:
            title = None
        return self._plot_response(
            t_ax, y_ax, title,
            **kwargs
        )

    @staticmethod
    def _plot_response(
        t_ax: np.ndarray,
        output_ax: np.ndarray,
        title: str | None,
        **kwargs
    ):
        color = kwargs.get('color', 'b')
        show_axes = kwargs.get('show_axes', False)
        grid = kwargs.get('grid', True)
        show = kwargs.get('show', True)

        plt.plot(t_ax, output_ax, color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if title is not None:
            plt.title(title, pad=20)

        if grid:
            plt.grid()
        if show_axes:
            plt.axhline(0, color='black')
            plt.axvline(0, color='black')
        if show:
            plt.show()
            return
        return plt


def create_time_delay(
    T: float,
    n: int = 3,
    numdeg: int | None = -1
) -> TransferFunction:
    """Creates a `TransferFunction` object which adds a time delay `T` when
    added in series to another `TransferFunction` object.

    Parameters
    ----------
    T:
        Amount of time delay in seconds.
    n:
        The time delay is implemented using a Padé approximation. Parameter
        `n` is the degree of the denominator of the approximation. See function
        `control.pade` in the docs of Python Control Systems Library (PCSL)
        for some more information about this function.
    numdeg:
        If `numdeg` is `None`, the degree of the numerator will equal the degree
        of the denominator. If `numdeg` >= 0, it specifies the degree of the
        numerator. If `numdeg` < 0, it specifies the numerator degree as
        `n + numdeg`.
    """
    num, den = ct.pade(T, n, numdeg)
    tf = TransferFunction.from_coefficients(num, den)
    return tf


def normalize(F: TransferFunction) -> TransferFunction:
    zfs = [(s - zero) / -zero if zero != 0 else s for zero in F.zeros]
    pfs = [(s - pole) / -pole if pole != 0 else s for pole in F.poles]
    numerator = sp.Mul(*zfs) if zfs else sp.Float(1)
    denominator = sp.Mul(*pfs) if pfs else sp.Float(1)
    F_n = TransferFunction(numerator / denominator)
    return F_n
