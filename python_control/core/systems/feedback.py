from collections import namedtuple
import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from ..symbols import s, t
from ..laplace_transform import LaplaceTransform, InverseLaplaceTransform
from ..transfer_function import TransferFunction


StaticErrorConstants = namedtuple('StaticErrorConstants', ('Kp', 'Kv', 'Ka'))
SteadyStateError = namedtuple('SteadyStateError', ('e_oo', 'eR_oo', 'eD_oo'))


class FeedbackSystem:

    def __init__(
        self,
        G1: TransferFunction,
        G2: TransferFunction | None = None,
        H: TransferFunction | None = None,
        name: str = ''
    ) -> None:
        """
        Creates a `FeedbackControlSystem`.

        Parameters
        ----------
        G1:
            Transfer function of the controller (if parameter G2 is not None),
            or the forward transfer function of the closed-loop feedback system.
        G2: optional
            Transfer function of the plant.
        H: optional
            Feedback transfer function. If None, unity feedback is assumed.
        name: optional
            A name to identify the system, e.g. in the legend of a plot.
        """
        self.name = name
        self.G1 = G1
        self.G2 = G2
        self.H = H
        if self.G2 is None: self.G2 = TransferFunction(1)
        if self.H is None: self.H = TransferFunction(1)

        # Forward transfer function of equivalent unity-feedback system.
        self.G = self.G1 * self.G2 / (1 + self.G1 * self.G2 * (self.H - 1))

        # Transfer function E(s) / R(s) with error E(s) defined as R(s) - C(s):
        T_ER = 1 / (1 + self.G)
        self.T_ER = T_ER.expr

        # Transfer function E(s) / D(s) with D(s) a disturbance signal:
        T_ED = self.G / (G1 * (1 + self.G))
        self.T_ED = T_ED.expr

        # Unit step response of feedback system in time domain:
        self.u, self.u_oo = self._unit_step_time_response()

    @property
    def open_loop(self) -> TransferFunction:
        """
        Returns the open-loop transfer function (loop gain) of the feedback
        control system.
        """
        return self.G1 * self.G2 * self.H

    @property
    def closed_loop(self) -> TransferFunction:
        """
        Returns the closed-loop transfer function C(s)/R(s) of the feedback
        control system.
        """
        return self.G.feedback(TransferFunction(1))

    @property
    def Kp(self) -> float:
        """Position static error constant."""
        Kp = sp.limit(self.G.expr, s, 0)
        return float(Kp)

    @property
    def Kv(self) -> float:
        """Velocity static error constant."""
        Kv = sp.limit(s * self.G.expr, s, 0)
        return float(Kv)

    @property
    def Ka(self) -> float:
        """Acceleration static error constant."""
        Ka = sp.limit(s ** 2 * self.G.expr, s, 0)
        return float(Ka)

    @property
    def static_error_constants(self) -> StaticErrorConstants:
        return StaticErrorConstants(
            Kp=self.Kp,
            Kv=self.Kv,
            Ka=self.Ka
        )

    @property
    def system_type(self) -> str:
        """
        Returns the system type, which is equivalent to the number of pure
        integrations in the forward path. If more than two integrations are
        in the forward path, 'None' is returned, which means that the system
        type has not been determined.
        """
        if self.Kv == 0 and self.Ka == 0:
            return 'type_0'
            # no integrations in the forward path, i.e., constant position
        elif self.Kp == sp.oo and self.Ka == 0:
            return 'type_1'
            # one integration in the forward path, i.e., constant velocity
        elif self.Kp == sp.oo and self.Kv == sp.oo:
            return 'type_2'
            # two integrations in the forward path, i.e., constant acceleration
        else:
            return 'None'
            # undetermined

    def steady_state_error(
        self,
        R: sp.Expr | LaplaceTransform | None,
        D: sp.Expr | LaplaceTransform | None = None
    ) -> SteadyStateError:
        """
        Returns the steady-state error of the closed-loop system for the given
        input signal R(s) and/or disturbance signal D(s).

        Notes
        -----
        Test input signals could be e.g. a step, ramp, or parabola.
        - standard step: u(t) [1 / s]
        - standard ramp: t * u(t) [1 / s**2]
        - standard parabola: 0.5 * t**2 * u(t) [1 / s**3]
        """
        if isinstance(R, LaplaceTransform):
            R = R.expr
        if isinstance(D, LaplaceTransform):
            D = D.expr

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
        """
        complex_poles = [
            pole
            for pole in self.closed_loop.poles
            if pole.imag != 0
        ]
        if len(complex_poles) == 2:
            if complex_poles[1] == complex_poles[0].conjugate():
                # noinspection PyTypeChecker
                return tuple(complex_poles)
        else:
            return None

    def _unit_step_time_response(self) -> tuple[InverseLaplaceTransform, float | None]:
        """
        Calculates the unit step response of the feedback system in the time
        domain.

        Returns
        -------
        time_response:
            `InverseLaplaceTransform` object holding the unit step response
            of the feedback system in the time domain.
        steady_state_value:
            The value of the unit step response when time goes to infinity.
            If the system is unstable, `None` is returned.
        """
        laplace_response = self.closed_loop.response(1 / s)
        time_response = laplace_response.inverse()
        steady_state_value = sp.limit(time_response.expr, t, sp.oo)
        try:
            steady_state_value = float(steady_state_value)
        except TypeError:
            # happens if the system is not stable
            steady_state_value = None
        return time_response, steady_state_value

    @property
    def peak_time(self):
        """
        Returns the peak time of the closed-loop unit step response.

        The peak time is defined as the time required to reach the first or
        maximum peak.
        """
        def _objective(t: float) -> float:
            uval = self.u.evaluate(t)
            return -uval

        t_min = 0.0
        t_max = np.max(1 / np.abs(self.closed_loop.poles))
        res = minimize_scalar(_objective, bounds=(t_min, t_max))
        t_peak = res.x
        return t_peak

    @property
    def rise_time(self) -> float | None:
        """
        Returns the rise time of the closed-loop unit step response.
        If the system is unstable, `None` is returned.

        The rise time is defined as the time required for the waveform to go
        from 0.1 of the final value to 0.9 of the final value.
        """
        if self.u_oo is not None:
            u_10 = 0.1 * self.u_oo
            u_90 = 0.9 * self.u_oo
            tfun_10 = lambda t: self.u.evaluate(t) - u_10
            tfun_90 = lambda t: self.u.evaluate(t) - u_90
            t_peak = self.peak_time
            t_10 = root_scalar(tfun_10, bracket=[0.0, t_peak]).root
            t_90 = root_scalar(tfun_90, bracket=[0.0, t_peak]).root
            t_rise = t_90 - t_10
            return t_rise

    @property
    def settling_time(self) -> float | None:
        """
        Returns the estimated settling time of the closed-loop unit step
        response. If the system is unstable, `None` is returned.

        The settling time is defined as the time required for the transient's
        damped oscillations to reach and stay within +/- 2% of the steady-state
        value.
        """
        if self.u_oo is not None:
            dt = self.peak_time / 100
            t = 0.0
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

    @property
    def percent_overshoot(self) -> float:
        """
        Returns the percent overshoot of the closed-loop unit step response if
        the system is stable else returns `None`.

        Percent overshoot is defined as the height the waveform overshoots the
        steady-state or final value at the peak time, expressed as a percentage
        of the steady-state value.
        """
        if self.u_oo is not None:
            t_peak = self.peak_time
            u_peak = self.u.evaluate(t_peak)
            percent_overshoot = (u_peak - self.u_oo) / self.u_oo * 100
            return percent_overshoot


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
    poles = set(feedback_system.closed_loop.poles_control)
    zeros = feedback_system.closed_loop.zeros_control
    dominant_poles = [pole for pole in poles if pole.imag != 0]
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
