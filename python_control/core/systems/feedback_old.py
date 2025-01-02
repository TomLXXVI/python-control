from collections import namedtuple
import sympy as sp
import numpy as np
from scipy.optimize import root_scalar
from ..symbols import s
from ..laplace_transform import LaplaceTransform, InverseLaplaceTransform
from ..transfer_function import TransferFunction


StaticErrorConstants = namedtuple('StaticErrorConstants', ('Kp', 'Kv', 'Ka'))
SteadyStateError = namedtuple('SteadyStateError', ('e_oo', 'eR_oo', 'eD_oo'))


class FeedbackSystem:

    def __init__(
        self,
        G_c: TransferFunction,
        G_p: TransferFunction | None = None,
        H: TransferFunction | None = None,
        name: str = ''
    ) -> None:
        """
        Creates a (negative) `FeedbackSystem` instance.

        Parameters
        ----------
        G_c:
            Transfer function of the controller if parameter G_p is not None,
            else the forward transfer function of the closed-loop feedback
            system.
        G_p: optional
            Transfer function of the plant.
        H: optional
            Feedback transfer function. If None, unity-feedback is assumed.
        name: optional
            A name to identify the system, e.g. in the legend of a plot.
        """
        self.name = name
        self.G_c = G_c
        self.G_p = G_p
        self.H = H
        if self.G_p is None: self.G_p = TransferFunction(1)
        if self.H is None: self.H = TransferFunction(1)
        # Forward transfer function of the equivalent unity-feedback system.
        self.G = self.G_c * self.G_p / (1 + self.G_c * self.G_p * (self.H - 1))
        # Transfer function E(s)/R(s) with error E(s) defined as R(s) - C(s):
        T_ER = 1 / (1 + self.G)
        self.T_ER = T_ER.expr
        # Transfer function E(s)/D(s) with D(s) a disturbance input signal to
        # the plant of the feedback system:
        T_ED = self.G / (G_c * (1 + self.G))
        self.T_ED = T_ED.expr
        # Unit-step response of the feedback system in the time domain:
        self.u, self.u_oo = self._unit_step_time_response()

    @property
    def open_loop(self) -> TransferFunction:
        """
        Returns the open-loop transfer function (loop gain) of the feedback
        control system.
        """
        return self.G_c * self.G_p * self.H

    @property
    def closed_loop(self) -> TransferFunction:
        """
        Returns the closed-loop transfer function C(s)/R(s) of the feedback
        control system.
        """
        return self.G.feedback(TransferFunction(1))

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
        """
        Returns the static error constants in a named tuple `StaticErrorConstants`.
        """
        return StaticErrorConstants(
            Kp=self.Kp,
            Kv=self.Kv,
            Ka=self.Ka
        )

    @property
    def system_type(self) -> str:
        """
        Returns the system type, which corresponds with the number of pure
        integrations in the forward path. If there are more than two integrations
        in the forward path, 'None' is returned, which means that the system
        type has not been determined.
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
        """
        Returns the steady-state error of the feedback system for the given
        reference input signal R(s) and/or disturbance input signal D(s).

        Notes
        -----
        Test-input signals could be e.g. a step, ramp, or parabola function.
        - Laplace transform of step input (position input) u(t):
            1 / s
        - Laplace transform of ramp input (velocity input) t * u(t):
            1 / s**2
        - Laplace transform of parabola input (acceleration input) 0.5 * t**2 * u(t):
            1 / s**3
        """
        if isinstance(R, LaplaceTransform): R = R.expr
        if isinstance(D, LaplaceTransform): D = D.expr
        if R is not None:
            E_R = self.T_ER * R
        else:
            E_R = None
        if self.T_ED is not None and D is not None:
            E_D = self.T_ED * D
        else:
            E_D = None
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
    def dominant_poles(self) -> tuple[complex, complex] | None:
        """
        Returns the complex, dominant poles of the closed-loop transfer function.
        If no dominant poles could be determined, `None` is returned.

        Notes
        -----
        If multiple complex pole pairs are present, the pole pair closest to the
        imaginary axis is considered to be the dominant pole pair.
        """
        complex_poles = [
            pole
            for pole in self.closed_loop.poles
            if pole.imag > 0
        ]
        if complex_poles:
            negative_real_parts = [p.real for p in complex_poles if p.real < 0]
            i_min = negative_real_parts.index(max(negative_real_parts))
            dominant_pole = complex_poles[i_min]
            dominant_pole_pair = (dominant_pole, dominant_pole.conjugate())
            return dominant_pole_pair
        else:
            return None

    @property
    def natural_frequency(self) -> float:
        """Returns the natural frequency of the dominant pole pair.
        If no dominant poles could be determined, `float('nan')` is returned.
        """
        if (dominant_poles := self.dominant_poles) is not None:
            omega_n = abs(dominant_poles[0])
            return omega_n
        return float('nan')

    @property
    def damping_ratio(self) -> float:
        """Returns the damping ratio of the dominant pole pair.
        If no dominant poles could be determined, `float('nan')` is returned.
        """
        if (dominant_poles := self.dominant_poles) is not None:
            omega_d = abs(dominant_poles[0].imag)
            sigma_d = abs(dominant_poles[0].real)
            phi = np.arctan(omega_d / sigma_d)
            damping_ratio = np.cos(phi)
            return damping_ratio
        return float('nan')

    def _unit_step_time_response(self) -> tuple[InverseLaplaceTransform | None, float | None]:
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
            laplace_response = self.closed_loop.response(1 / s)
            time_response = laplace_response.inverse()
            steady_state_error = self.closed_loop.steady_state_error(1 / s)
            steady_state_value = 1 - steady_state_error
            try:
                steady_state_value = float(steady_state_value)
            except TypeError:
                # happens if the system is not stable
                steady_state_value = None
            return time_response, steady_state_value
        return None, None

    def _get_peak_time(self) -> float:
        """
        Returns the peak time of the second-order system approximation, which is
        based on the dominant pole pair. If the system is unstable,
        `float('nan')` is returned.

        Notes
        -----
        The peak time is defined as the time required to reach the first or
        maximum peak.
        """
        if self.u is not None and self.dominant_poles is not None:
            t_p_values = np.pi / np.array([p.imag for p in self.closed_loop.poles if p.imag > 0.0])
            u_p_values = [self.u.evaluate(t_p) for t_p in t_p_values]
            i_max = u_p_values.index(max(u_p_values))
            t_p = float(t_p_values[i_max])
            return t_p
        return float('nan')

    @property
    def rise_time(self) -> float:
        """
        Returns the rise time of the closed-loop unit-step response.
        If the system is unstable, `float('nan')` is returned.

        Notes
        -----
        The rise time is defined as the time required for the waveform to go
        from 0.1 of the final value to 0.9 of the final value.
        """
        if self.u_oo is not None and self.dominant_poles is not None:
            u_10 = 0.1 * self.u_oo
            u_90 = 0.9 * self.u_oo
            tfun_10 = lambda t: self.u.evaluate(t) - u_10
            tfun_90 = lambda t: self.u.evaluate(t) - u_90
            t_peak = self._get_peak_time()
            t_10 = root_scalar(tfun_10, bracket=[0.0, t_peak]).root
            t_90 = root_scalar(tfun_90, bracket=[0.0, t_peak]).root
            t_rise = t_90 - t_10
            return t_rise
        return float('nan')

    @property
    def settling_time(self) -> float:
        """
        Returns the estimated settling time of the closed-loop unit-step
        response. If the system is unstable, `float('nan')` is returned.

        Notes
        -----
        The settling time is defined as the time required for the transient's
        damped oscillations to reach and stay within +/- 2% of the steady-state
        value.
        """
        if (self.u_oo is not None) and (self.u_oo < float('inf')) and (self.dominant_poles is not None):
            t_peak = self._get_peak_time()
            if not np.isnan(t_peak):
                t = t_peak
                dt = 0.1
                times = []
                while True:
                    u = self.u.evaluate(t)
                    if 0.98 * self.u_oo < u < 1.02 * self.u_oo:
                        times.append(t)
                        size = len(times)
                        if size > 100:
                            chunck = times[size - 100:]
                            if all(chunck):
                                return chunck[0]
                    else:
                        times.append(False)
                    t += dt
        return float('nan')

    @property
    def percent_overshoot(self) -> float:
        """
        Returns the percent overshoot of the closed-loop unit-step response if
        the system is stable, else returns `float('nan')`.

        Notes
        -----
        Percent overshoot is defined as the height the waveform overshoots the
        steady-state or final value at the peak time, expressed as a percentage
        of the steady-state value.
        """
        if self.u_oo is not None:
            t_peak = self._get_peak_time()
            u_peak = self.u.evaluate(t_peak)
            percent_overshoot = max(0, (u_peak - self.u_oo) / self.u_oo * 100)
            return percent_overshoot
        return float('nan')

    @property
    def peak_time(self) -> float:
        """Returns the estimated peak time of the closed-loop unit-step
        response. If the system is unstable or if the peak time cannot be
        determined, `float('nan')` is returned.
        """
        t_peak = self._get_peak_time()
        if not np.isnan(t_peak):
            u_peak = self.u.evaluate(t_peak)
            percent_overshoot = max(0.0, (u_peak - self.u_oo) / self.u_oo * 100)
            if percent_overshoot == 0.0:
                return float('nan')
            else:
                return t_peak
        return float('nan')

    @property
    def characteristics(self) -> str:
        """Returns the transient response characteristics of the feedback
        system.
        """
        str_list = [''] * 10
        str_list[0] = f"system type: {self.system_type}\n"
        str_list[1] = f"dominant pole pair: {self.dominant_poles}\n"
        str_list[2] = f"natural frequency of dominant pole pair = {self.natural_frequency:.3f} rad/s\n"
        str_list[3] = f"damping ratio of dominant pole pair = {self.damping_ratio:.3f}\n"
        str_list[4] = f"percent overshoot = {self.percent_overshoot:.3f} %\n"
        str_list[5] = f"rise time = {self.rise_time:.3f} s\n"
        str_list[6] = f"peak time = {self.peak_time:.3f} s\n"
        str_list[7] = f"settling time = {self.settling_time:.3f} s\n"
        if self.system_type == 'type_0':
            e_oo, *_ = self.steady_state_error(1 / s)
            str_list[8] = f"steady-state error = {e_oo:.3f}\n"
            str_list[9] = f"position error constant = {self.Kp:.3f}"
        elif self.system_type == 'type_1':
            e_oo, *_ = self.steady_state_error(1 / s ** 2)
            str_list[8] = f"steady-state error = {e_oo:.3f}\n"
            str_list[9] = f"velocity error constant = {self.Kv:.3f}"
        elif self.system_type == 'type_2':
            e_oo, *_ = self.steady_state_error(1 / s ** 3)
            str_list[8] = f"steady-state error = {e_oo:.3f}\n"
            str_list[9] = f"velocity error constant = {self.Ka:.3f}"
        else:
            str_list[8] = f"steady-state error = {float('nan')}\n"
            str_list[9] = f"velocity error constant = {float('nan')}"
        string = ''.join(str_list)
        return string


def sensitivity(f: sp.Expr, p: sp.Symbol) -> sp.Expr:
    """
    Sensitivity is defined as the ratio of the fractional change in the function
    `f` to the fractional change in its parameter `p` when the fractional change
    of the parameter approaches zero.
    """
    der_f = sp.diff(f, p)
    S = (p / f) * der_f
    return S.together()


def is_second_order_approx(feedback_system: FeedbackSystem):
    """
    Checks whether the feedback system can be approximated as a second-order
    system.
    """
    poles = set(feedback_system.closed_loop.poles_pcsl)
    zeros = feedback_system.closed_loop.zeros_pcsl
    dominant_poles = feedback_system.dominant_poles
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
