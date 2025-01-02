from collections import namedtuple

import numpy as np
import sympy as sp
from scipy.optimize import root_scalar
from ..symbols import s
from ..transfer_function import TransferFunction
from ..laplace_transform import LaplaceTransform, InverseLaplaceTransform

LAP = LaplaceTransform
ILAP = InverseLaplaceTransform

StaticErrorConstants = namedtuple('StaticErrorConstants', ('Kp', 'Kv', 'Ka'))
SteadyStateError = namedtuple('SteadyStateError', ('e_oo', 'eR_oo', 'eD_oo'))


class FeedbackSystem:

    def __init__(
        self,
        G_c: TransferFunction | None = None,
        G_p: TransferFunction | None = None,
        H: TransferFunction | None = None,
        name: str = ''
    ) -> None:
        self.G_c = G_c if G_c is not None else TransferFunction(1)
        self.G_p = G_p if G_p is not None else TransferFunction(1)
        self.H = H if H is not None else TransferFunction(1)
        self.name = name
        self.G = self._transform_to_unity_feedback_system()
        self.E_to_R = self._get_error_to_reference()
        self.E_to_D = self._get_error_to_disturbance()
        self.open_loop = self._get_open_loop()
        self.closed_loop = self._get_closed_loop()
        self.is_stable = self.closed_loop.is_stable
        self.unit_step_response, self.y_oo = self._get_unit_step_response()

    def _transform_to_unity_feedback_system(self) -> TransferFunction:
        """Transforms the feedback system to an equivalent unity-feedback
        system with only a forward-path transfer function G(s).
        """
        num = self.G_c * self.G_p
        den = 1 + self.G_c * self.G_p * (self.H - 1)
        G = num / den
        return G

    def _get_error_to_reference(self) -> TransferFunction:
        """Returns the transfer function between the reference input signal R(s)
        and the error signal E(s) = R(s) - Y(s).
        """
        E_to_R = 1 / (1 + self.G)
        return E_to_R

    def _get_error_to_disturbance(self) -> TransferFunction:
        """Returns the transfer function between the disturbance input signal
        D(s) and the error signal E(s) = R(s) - Y(s).
        """
        E_to_D = self.G / (self.G_c * (1 + self.G))
        return E_to_D

    def _get_open_loop(self) -> TransferFunction:
        """Returns the open-loop transfer function of the feedback system."""
        return self.G_c * self.G_p * self.H

    def _get_closed_loop(self) -> TransferFunction:
        """Returns the closed-loop transfer function of the feedback system."""
        return self.G.feedback(TransferFunction(1))

    def _get_unit_step_response(self) -> tuple[ILAP | None, float | None]:
        """
        Calculates the unit-step response of the feedback system in the time
        domain.

        Returns
        -------
        time_response:
            `InverseLaplaceTransform` object holding the unit-step response
            of the feedback system in the time domain.
        steady_state_value:
            The value of the unit-step response when time goes to infinity.
            If the system is unstable, `None` is returned.
        """
        if self.closed_loop._ct_tf is not None:
            laplace_resp = self.closed_loop.response(1 / s)
            time_resp = laplace_resp.inverse()
            steady_state_err = self.closed_loop.steady_state_error(1 / s)
            steady_state_val = 1 - steady_state_err
            try:
                steady_state_val = float(steady_state_val)
            except TypeError:
                # unstable system
                steady_state_val = None
            return time_resp, steady_state_val
        return None, None

    @property
    def Kp(self) -> float | sp.Expr:
        """Position error constant."""
        Kp = sp.limit(self.G.expr, s, 0)
        try:
            return float(Kp)
        except TypeError:
            return Kp

    @property
    def Kv(self) -> float | sp.Expr:
        """Velocity error constant."""
        Kv = sp.limit(s * self.G.expr, s, 0)
        try:
            return float(Kv)
        except TypeError:
            return Kv

    @property
    def Ka(self) -> float | sp.Expr:
        """Acceleration error constant."""
        Ka = sp.limit(s ** 2 * self.G.expr, s, 0)
        try:
            return float(Ka)
        except TypeError:
            return Ka

    @property
    def static_error_constants(self) -> StaticErrorConstants:
        """Returns the static error constants in a named tuple
        `StaticErrorConstants`.
        """
        return StaticErrorConstants(
            Kp=self.Kp,
            Kv=self.Kv,
            Ka=self.Ka
        )

    @property
    def system_type(self) -> str:
        """Returns the system type, which corresponds with the number of pure
        integrations in the forward path. If there are more than two
        integrations in the forward path, 'None' is returned, which means that
        the system type has not been determined.
        """
        if self.Kv == 0 and self.Ka == 0:
            return 'type_0'
            # no integrations in the forward path, constant position error
        elif self.Kp == sp.oo and self.Ka == 0:
            return 'type_1'
            # one integration in the forward path, constant velocity error
        elif self.Kp == sp.oo and self.Kv == sp.oo:
            return 'type_2'
            # two integrations in the forward path, constant acceleration error
        else:
            return 'None'
            # system type undetermined

    def steady_state_error(
        self,
        R: sp.Expr | LaplaceTransform | None,
        D: sp.Expr | LaplaceTransform | None = None
    ) -> SteadyStateError:
        """Returns the steady-state error of the feedback system for the given
        reference input signal R(s) and/or disturbance input signal D(s).

        Notes
        -----
        Test-input signals could be e.g. a step, ramp, or parabola function.
        -   Laplace transform of step input (position input)
            u(t): 1 / s
        -   Laplace transform of ramp input (velocity input)
            t * u(t): 1 / s**2
        -   Laplace transform of parabola input (acceleration input)
            0.5 * t**2 * u(t): 1 / s**3
        """
        if isinstance(R, LaplaceTransform): R = R.expr
        if isinstance(D, LaplaceTransform): D = D.expr
        E_R = self.E_to_R.expr * R if R is not None else None
        E_D = self.E_to_D.expr * D if D is not None else None
        E = None
        if E_R is not None:
            E = E_R
            eR_oo = sp.limit(s * E_R, s, 0)
        else:
            eR_oo = None
        if E_D is not None:
            if E is not None:
                E -= E_D
            else:
                E = -E_D
            eD_oo = -sp.limit(s * E_D, s, 0)
        else:
            eD_oo = None
        e_oo = sp.limit(s * E, s, 0)
        e_oo = e_oo.together()
        return SteadyStateError(e_oo, eR_oo, eD_oo)

    @property
    def dominant_pole_pair(self) -> tuple[complex, complex] | None:
        """Returns the complex, dominant pole pair of the closed-loop transfer
        function. If no dominant poles can be determined, `None` is returned.
        If multiple complex pole pairs are present, the pole pair closest to the
        imaginary axis (i.e. the "slowest" pole pair) is considered to be the
        dominant pole pair.
        """
        complex_poles = [
            pole
            for pole in self.closed_loop.poles
            if pole.imag > 0
        ]
        if complex_poles:
            real_parts = [p.real for p in complex_poles]
            i_max = real_parts.index(max(rp for rp in real_parts if rp < 0.0))
            dominant_pole = complex_poles[i_max]
            dominant_pole_pair = (dominant_pole, dominant_pole.conjugate())
            return dominant_pole_pair
        else:
            return None

    def settling_time(self, bracket: tuple[float, float] | None = None) -> float | None:
        """Returns the settling time of the unit-step response.

        Parameters
        ----------
        bracket: optional
            The start and final value of the time interval in which the response
            is settling. If the closed-loop transfer function has a dominant
            pole pair, `bracket` can be `None`. In that case, the start time is
            set to the peak time and the final time is determined as 100
            times the natural period of the dominant poles.

        Raises
        ------
        ValueError:
            - If the closed-loop transfer function has no dominant pole pair and
            `bracket` is not set (`None`).
            - If the unit-step response at the final value of the bracket is
            smaller than 0.98 * the steady-state value or greater than 1.02 *
            the steady-state value.

        Returns
        -------
        settling_time:
            If the system is stable, has a dominant pole pair or `bracket` is
            set.
        None:
            If the system is unstable.
        """
        if self.is_stable:
            if bracket is None:
                if self.dominant_pole_pair is not None:
                    omega_n = abs(self.dominant_pole_pair[0])
                    T_n = 2 * np.pi / omega_n
                    t_ini = self.peak_time()
                    t_fin = 100 * T_n
                else:
                    raise ValueError(
                        "The settling time cannot be determined. Use `bracket` to"
                        "set the time interval in which the unit-step response"
                        "settles."
                    )
            else:
                t_ini, t_fin = bracket
            t_values, y_values = self.closed_loop.unit_step_response(
                lower_limit=t_ini,
                upper_limit=t_fin
            )
            mask = (y_values >= 0.98 * self.y_oo) & (y_values <= 1.02 * self.y_oo)
            if mask[-1]:
                i = len(y_values) - 1
                while mask[i] and i >= 0: i -= 1
                t_settling = t_values[i + 1]
                return t_settling
            else:
                raise ValueError(
                    f"The unit-step response at the final time {t_fin} has not"
                    "been settled already. Move the bracket further along the"
                    "time axis."
                )
        return None

    def peak_time(self, bracket: tuple[float, float] | None = None) -> float | None:
        """Returns the peak time of the unit-step response.

        Parameters
        ----------
        bracket: optional
            The start and final value of the time interval in which the response
            reaches the peak value. If the closed-loop transfer function has a
            dominant pole pair, `bracket` can be `None`. In that case, the
            start time is set to zero and the final time is determined as 10
            times the natural period of the dominant poles.

        Raises
        ------
        ValueError:
            If the closed-loop transfer function has no dominant pole pair and
            `bracket` is not set (`None`).

        Returns
        -------
        peak_time:
            If the system is stable, has a dominant pole pair or `bracket` is
            set.
        None:
            If the system is unstable.
        """
        if self.is_stable:
            if bracket is None:
                if self.dominant_pole_pair is not None:
                    omega_n = abs(self.dominant_pole_pair[0])
                    T_n = 2 * np.pi / omega_n
                    t_ini = 0.0
                    t_fin = 10 * T_n
                else:
                    raise ValueError(
                        "The peak time cannot be determined. Use `bracket` to"
                        "set the time interval in which the unit-step response"
                        "reaches the peak value."
                    )
            else:
                t_ini, t_fin = bracket
            t_values, y_values = self.closed_loop.unit_step_response(
                lower_limit=t_ini,
                upper_limit=t_fin
            )
            t_peak = t_values[np.argmax(y_values)]
            return t_peak
        return None

    def rise_time(self, bracket: tuple[float, float] | None = None) -> float | None:
        """Returns the rise time of the unit-step response.

        Parameters
        ----------
        bracket:
            The start and final value of the time interval in which the response
            reaches the peak value. If the closed-loop transfer function has a
            dominant pole pair, `bracket` can be `None`. In that case, the
            start time is set to zero and the final time is determined as 10
            times the natural period of the dominant poles.

        Raises
        ------
        ValueError:
            If the closed-loop transfer function has no dominant pole pair and
            `bracket` is not set (`None`).

        Returns
        -------
        rise_time:
            If the system is stable, has a dominant pole pair or `bracket` is
            set.
        None:
            If the system is unstable.
        """
        if self.is_stable:
            u_10p = 0.1 * self.y_oo
            u_90p = 0.9 * self.y_oo
            t_peak = self.peak_time(bracket)

            def _objective1(t: float) -> float:
                u_value = self.unit_step_response.evaluate(t)
                return u_value - u_10p

            def _objective2(t: float) -> float:
                u_value = self.unit_step_response.evaluate(t)
                return u_value - u_90p

            t_10p = root_scalar(_objective1, bracket=(0.0, t_peak)).root
            t_90p = root_scalar(_objective2, bracket=(0.0, t_peak)).root
            t_rise = t_90p - t_10p
            return t_rise
        return None

    def percent_overshoot(self, bracket: tuple[float, float] | None = None) -> float | None:
        """Returns the percent overshoot of the unit-step response.

        Parameters
        ----------
        bracket: optional
            The start and final value of the time interval in which the response
            reaches the peak value. If the closed-loop transfer function has a
            dominant pole pair, `bracket` can be `None`. In that case, the
            start time is set to zero and the final time is determined as 10
            times the natural period of the dominant poles.

        Raises
        ------
        ValueError:
            If the closed-loop transfer function has no dominant pole pair and
            `bracket` is not set (`None`).

        Returns
        -------
        percent_overshoot:
            If the system is stable, has a dominant pole pair or `bracket` is
            set.
        None:
            If the system is unstable.
        """
        if self.is_stable:
            t_peak = self.peak_time(bracket)
            y_peak = self.unit_step_response.evaluate(t_peak)
            percent_overshoot = (y_peak - self.y_oo) / self.y_oo * 100.0
            return percent_overshoot
        return None


def sensitivity(f: sp.Expr, p: sp.Symbol) -> sp.Expr:
    """Sensitivity is defined as the ratio of the fractional change in the
    function `f` to the fractional change in its parameter `p`, when the
    fractional change of parameter `p` approaches zero.
    """
    der_f = sp.diff(f, p)
    S = (p / f) * der_f
    return S.together()


def is_second_order_approx(feedback_system: FeedbackSystem):
    """Checks whether the feedback system can be approximated as a second-order
    system.
    """
    poles = set(feedback_system.closed_loop.poles_pcsl)
    zeros = feedback_system.closed_loop.zeros_pcsl
    dominant_poles = feedback_system.dominant_pole_pair
    other_poles = list(poles.difference(dominant_poles))

    def _is_far_pole(pole: complex) -> bool:
        if abs(pole.real) > abs(5 * dominant_poles[0].real):
            return True
        return False

    def _pole_zero_cancellation(pole: complex) -> bool:
        if any([0.9 < abs(pole / zero) < 1.1 for zero in zeros]):
            return True
        return False

    def _validity_check(pole: complex) -> bool:
        cond1 = _is_far_pole(pole)
        cond2 = _pole_zero_cancellation(pole)
        return cond1 or cond2

    check = all([_validity_check(pole) for pole in other_poles])
    return check
