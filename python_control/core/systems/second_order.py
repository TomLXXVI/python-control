import math
from enum import Enum
from scipy.optimize import root_scalar
from ..transfer_function import TransferFunction
from ..symbols import s


class NaturalResponseType(Enum):
    UNDAMPED = 'undamped'
    UNDERDAMPED = 'underdamped'
    CRITICALLY_DAMPED = 'critically_damped'
    OVERDAMPED = 'overdamped'


class SecondOrderSystem(TransferFunction):
    """
    Implements a second-order system of the general form:
                    G(s) = b / (s² + a * s + b)
    which has 2 finite poles and no zeros, and where:
    - b : square of the system's natural frequency (omega_nat ** 2)
    - a : 2 * damping ratio (zeta) * natural frequency (omega_nat)
    """
    def __init__(self, a: float, b: float, name: str = ''):
        """Creates a `SecondOrderSystem` object.

        Parameters
        ----------
        a:
            First-order coefficient of G(s) = b / (s² + a * s + b)
        b:
            Zero-order coefficient of G(s) = b / (s² + a * s + b)
        name:
            Name to identify the second-order system (e.g. in the legend of a
            diagram).
        """
        self.name = name
        self.a = a
        self.b = b
        tf = b / (s ** 2 + a * s + b)
        super().__init__(tf)

        # unit step response in s-domain and time-domain:
        self.U = self.response(1 / s)
        self.u = self.U.inverse()

        self.t_inf = self.T_nat * 1.e2  # we assume steady state at t_inf = 100 x T_nat

    @classmethod
    def from_poles(
        cls,
        p1: float | complex,
        p2: float | complex,
        name: str = ''
    ) -> 'SecondOrderSystem':
        """
        Creates a second order system given its two poles:
        - p1 = -sigma_d + j * omega_d
        - p2 = -sigma_d - j * omega_d
        """
        a = -(p1 + p2)
        b = p1 * p2
        return cls(a.real, b.real, name)

    @property
    def omega_nat(self) -> float:
        """
        Returns the natural frequency (rad/sec).
        """
        return self.b ** 0.5

    @property
    def T_nat(self) -> float:
        """
        Returns the natural period (seconds).
        """
        return 2 * math.pi / self.omega_nat

    @property
    def omega_damped(self) -> float:
        """
        Returns the damped frequency of oscillation (i.e., the magnitude of
        the imaginary part of the pole). Returns NaN when the system is
        overdamped (damping ratio > 1).
        """
        if self.zeta <= 1.0:
            return self.omega_nat * math.sqrt(1 - self.zeta ** 2)
        else:
            return float('nan')

    @property
    def sigma_damped(self) -> float:
        """
        Returns the exponential damping frequency (i.e., the magnitude of the
        real part of the pole).
        """
        return self.zeta * self.omega_nat

    @property
    def zeta(self) -> float:
        """
        Returns the damping ratio.
        """
        return (self.a / 2) / self.omega_nat

    def get_natural_response_type(self) -> NaturalResponseType:
        """
        Returns the type of natural response.
        """
        if self.zeta == 0.0:
            return NaturalResponseType.UNDAMPED
            # 2 opposite, imaginary poles
        elif 0.0 < self.zeta < 1.0:
            return NaturalResponseType.UNDERDAMPED
            # 2 opposite, complex poles
        elif self.zeta == 1.0:
            return NaturalResponseType.CRITICALLY_DAMPED
            # 2 coincident, real poles
        else:  # damping ratio > 1
            return NaturalResponseType.OVERDAMPED
            # 2 different, real poles

    @property
    def steady_state_output(self) -> float:
        """
        Returns the steady-state output of the system based on a unit step
        response. If, however, the system is undamped, NaN is returned.
        """
        if self.get_natural_response_type() != NaturalResponseType.UNDAMPED:
            sso = self.u.evaluate(self.t_inf)
            return sso
        return float('nan')

    @property
    def rise_time(self) -> float:
        """
        Returns the rise time of the system based on a unit step response. If,
        however, the system is undamped, NaN is returned.

        The rise time is defined as the time required for the waveform to go
        from 0.1 of the final value to 0.9 of the final value.
        """
        if self.get_natural_response_type() != NaturalResponseType.UNDAMPED:
            sso = self.steady_state_output
            sso_10_per = 0.1 * sso
            sso_90_per = 0.9 * sso
            f_10_per = lambda t: self.u.evaluate(t) - sso_10_per
            f_90_per = lambda t: self.u.evaluate(t) - sso_90_per
            t_10_per = root_scalar(f_10_per, bracket=[0.0, self.t_inf]).root
            t_90_per = root_scalar(f_90_per, bracket=[0.0, self.t_inf]).root
            t_rise = t_90_per - t_10_per
            return t_rise
        return float('nan')

    @property
    def settling_time(self) -> float:
        """
        Returns the settling time of the system based on a unit step response.
        If, however, the system is undamped, NaN is returned.

        The settling time is defined as the time required for the transient's
        damped oscillations to reach and stay within +/- 2% of the steady-state
        value.
        """
        response_type = self.get_natural_response_type()
        if response_type != NaturalResponseType.UNDAMPED:
            if response_type != NaturalResponseType.OVERDAMPED and self.zeta < 1:
                n = -math.log(0.02 * math.sqrt(1 - self.zeta ** 2))
                d = self.zeta * self.omega_nat
                Ts = n / d
                return Ts
            else:
                sso = self.steady_state_output
                sso_98_per = 0.98 * sso
                sso_102_per = 1.02 * sso
                time, output = self.unit_step_response(upper_limit=self.t_inf)
                mask = (output >= sso_98_per) & (output <= sso_102_per)
                i = len(mask) - 1
                t_settling = float('nan')
                while i >= 0:
                    if not mask[i]:
                        t_settling = time[i + 1]
                        break
                    i -= 1
                return t_settling
        return float('nan')

    @property
    def peak_time(self) -> float:
        """
        Returns the peak time of the system based on a unit step response.
        If the system is overdamped, NaN will be returned.

        The peak time is defined as the time required to reach the first or
        maximum peak.
        """
        if self.get_natural_response_type() != NaturalResponseType.OVERDAMPED:
            try:
                Tp = math.pi / self.omega_damped
            except ZeroDivisionError:
                Tp = float('inf')
            return Tp
        return float('nan')

    @property
    def percent_overshoot(self) -> float:
        """
        Returns the percent overshoot of the system based on a unit step
        response. Percent overshoot is only defined for the underdamped system.
        If the system is not underdamped, NaN will be returned.

        Percent overshoot is defined as the height the waveform overshoots the
        steady-state or final value at the peak time, expressed as a percentage
        of the steady-state value.
        """
        if self.get_natural_response_type() == NaturalResponseType.UNDERDAMPED:
            a = self.zeta * math.pi / math.sqrt(1 - self.zeta ** 2)
            per_overshoot = math.exp(-a) * 100
            return per_overshoot
        return float('nan')

    @classmethod
    def from_design_specs(
        cls,
        percent_overshoot: float,
        peak_time: float | None = None,
        settling_time: float | None = None,
        a: float | None = None,
        b: float | None = None,
        name: str = ''
    ) -> 'SecondOrderSystem':
        """
        Creates an underdamped second order system with a given percentage
        overshoot and a given peak time or a given settling time.

        Parameters
        ----------
        percent_overshoot:
            Allowable percentage overshoot (%).
        peak_time:
            Peak time in seconds.
        settling_time:
            Settling time in seconds.
        a:
            Coefficient in the denominator s ** 2 + a * s + b
        b:
            Coefficient in the denominator s ** 2 + a * s + b
        name:
            Name to identify the second-order system (e.g. in the legend of a
            diagram).

        Returns
        -------
        SecondOrderSystem
        """
        frac_overshoot = percent_overshoot / 100
        n = -math.log(frac_overshoot)
        d = math.sqrt(math.pi ** 2 + n ** 2)
        zeta = n / d
        if peak_time is not None:
            omega_nat = math.pi / (peak_time * math.sqrt(1 - zeta ** 2))
        elif settling_time is not None:
            n = -math.log(0.02 * math.sqrt(1 - zeta ** 2))
            omega_nat = n / (zeta * settling_time)
        elif a is not None:
            omega_nat = a / (2 * zeta)
        elif b is not None:
            omega_nat = math.sqrt(b)
        else:
            raise ValueError(
                "either peak time, settling time or coefficient 'a' "
                "must be specified"
            )
        a = 2 * zeta * omega_nat
        b = omega_nat ** 2
        system = cls(a, b, name)
        return system


def get_damping_ratio(percent_overshoot: float) -> float:
    overshoot = percent_overshoot / 100
    num = -math.log(overshoot)
    den = math.sqrt(math.pi**2 + math.log(overshoot)**2)
    damping_ratio = num / den
    return damping_ratio


def get_percent_overshoot(damping_ratio: float) -> float:
    num = damping_ratio * math.pi
    den = math.sqrt(1 - damping_ratio**2)
    percent_overshoot = math.exp(-num / den) * 100
    return percent_overshoot


def get_peak_time(omega_nat: float, damping_ratio: float) -> float:
    T_p = math.pi / (omega_nat * math.sqrt(1 - damping_ratio**2))
    return T_p


def get_settling_time(omega_nat: float, damping_ratio: float) -> float:
    T_s = 4 / (damping_ratio * omega_nat)
    return T_s
