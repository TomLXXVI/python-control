from python_control import s, TransferFunction


class ZieglerNicholsTuning:
    """Ziegler-Nichols tuning of a (approximated) first-order system with time
    lag.

    The choice of controller parameters is designed to result in a closed-loop
    step response transient with a decay ratio of approximately 0.25. This means
    that the transient decays to a quarter of its value after one period of
    oscillation.
    """
    def __init__(
        self,
        lag: float,
        final_value: float,
        tau: float
    ) -> None:
        """Creates a `ZieglerNicholsTuning` instance with parameters taken from
        the unit-step response of the system.

        Parameters
        ----------
        lag:
            Transportation lag (time delay) in the unit-step response of the
            system.
        final_value:
            The value of the unit-step response when steady state has been
            reached.
        tau:
            Time constant of the system measured as the time period from the
            inflection point (i.e. at the intersection of the tangent line with
            the time axis) to the time moment when the value of the unit-step
            response has reached 63 % of its final value.
        """
        self.L = lag
        self.A = final_value
        self.tau = tau
        self.R = self.A / self.tau

    def P_control(self) -> TransferFunction:
        """Returns the transfer function of the tuned P-controller."""
        K_p = 1 / (self.R * self.L)
        return TransferFunction(K_p)

    def PI_control(self) -> TransferFunction:
        """Returns the transfer function of the tuned PI-controller."""
        K_p = 0.9 / (self.R * self.L)
        T_i = self.L / 0.3
        return TransferFunction(K_p * (1 + 1 / (T_i * s)))

    def PID_control(self) -> TransferFunction:
        """Returns the transfer function of the tuned PID-controller."""
        K_p = 1.2 / (self.R * self.L)
        T_i = 2 * self.L
        T_d = 0.5 * self.L
        return TransferFunction(K_p * (1 + 1 / (T_i * s) + T_d * s))


class UltimateSensitivityMethod:
    """Tuning by evaluation at limit of stability.

    In the ultimate sensitivity method the criteria for adjusting the parameters
    are based on evaluating the amplitude and frequency of the oscillations of
    the system at the limit of stability. To use the method, the proportional
    gain is increased until the system becomes marginally stable. The
    corresponding gain is called the ultimate gain and the period of oscillation
    is called the ultimate period.
    """
    def __init__(self, K_u: float, P_u: float) -> None:
        """Creates a `UltimateSensitivityMethod` instance.

        Parameters
        ----------
        K_u:
            Ultimate gain.
        P_u:
            Ultimate period.
        """
        self.K_u = K_u
        self.P_u = P_u

    def P_control(self) -> TransferFunction:
        """Returns the transfer function of the tuned P-controller."""
        K_p = 0.5 * self.K_u
        return TransferFunction(K_p)

    def PI_control(self) -> TransferFunction:
        """Returns the transfer function of the tuned PI-controller."""
        K_p = 0.45 * self.K_u
        T_i = self.P_u / 1.2
        return TransferFunction(K_p * (1 + 1 / (T_i * s)))

    def PID_control(self) -> TransferFunction:
        """Returns the transfer function of the tuned PID-controller."""
        K_p = 1.6 * self.K_u
        T_i = 0.5 * self.P_u
        T_d = 0.125 * self.P_u
        return TransferFunction(K_p * (1 + 1 / (T_i * s) + T_d * s))
