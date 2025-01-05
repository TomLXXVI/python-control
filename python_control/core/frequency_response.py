from __future__ import annotations
import warnings
from typing import Callable
from collections.abc import Sequence
import sympy as sp
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar
from .symbols import s
from .transfer_function import TransferFunction
from .systems.feedback import FeedbackSystem


omega = sp.Symbol('omega', real=True, positive=True)
I = sp.I


class FrequencyResponse:
    """
    Class for deriving and analyzing the frequency response of a transfer 
    function.
    """
    def __init__(self, KGH: TransferFunction) -> None:
        """Creates a `FrequencyResponse` object.

        Parameters
        ----------
        KGH:
            Transfer function.
            
        Notes
        -----
        The gain of the transfer function `KGH` is ignored when creating the 
        `FrequencyResponse` object (i.e. the transfer function is divided by
        its gain `K`, so that its gain becomes 1).
        """
        self._KGH_original = KGH
        self._K_value = KGH.gain
        # Divide the original transfer function by its gain.
        self._GH = KGH / self._K_value
        self._zeros = self._GH.as_sympy.zeros()
        self._poles = self._GH.as_sympy.poles()
        # Replace `s` in the transfer function Sympy expression by `j * omega`.        
        self._GH_jw_expr = self._GH.expr.subs(s, I * omega)
        # Create function that returns the frequency response as a complex 
        # number for a given angular frequency `omega`.
        self.__GH_jw_fun__ = self.__create_GH_jw_fun()

    @property
    def expr(self) -> sp.Expr:
        """Returns the Sympy expression of the frequency response with unity 
        gain, i.e. the open-loop transfer function F(s) with gain `K` = 1 and 
        `s` being replaced by `j * omega`.
        """
        return self._GH_jw_expr

    @property
    def magnitude(self) -> sp.Expr:
        """Returns the Sympy expression of the magnitude of the frequency 
        response with unity gain (`K` = 1).
        """
        zfs = [sp.Abs(I * omega - zero) for zero in self._zeros]
        pfs = [sp.Abs(I * omega - pole) for pole in self._poles]
        magn_num, magn_den = 1, 1
        for zf in zfs: magn_num *= zf
        for pf in pfs: magn_den *= pf
        magn = magn_num / magn_den
        return magn

    @property
    def dB_magnitude(self) -> sp.Expr:
        """Returns the Sympy expression of the magnitude of the frequency 
        response with unity gain (`K` = 1) expressed in decibels.
        """
        return 20 * sp.log(self.magnitude, 10)

    @property
    def phase(self) -> sp.Expr:
        """Returns the Sympy expression of the phase angle of the frequency 
        response with unity gain (`K` = 1).
        """
        zts = [sp.arg(I * omega - zero) for zero in self._zeros]
        pts = [sp.arg(I * omega - pole) for pole in self._poles]
        z_phi, p_phi = 0, 0
        for zt in zts: z_phi += zt
        for pt in pts: p_phi += pt
        phi = z_phi - p_phi
        return phi

    @property
    def transfer_function(self) -> TransferFunction:
        """Returns the underlying transfer function of the frequency response.
        (Note that the original open-loop gain is included in the transfer
        function.)
        """
        return self._KGH_original

    def evaluate(
        self,
        omega_value: sp.Expr | int | float,
        K: int | float | None = None
    ) -> tuple[sp.Expr | float, ...]:
        """Evaluates the frequency response at the given angular frequency and
        open-loop gain.

        Parameters
        ----------
        omega_value:
            Value for the angular frequency (rad/s) at which the frequency 
            response needs to be evaluated.
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.

        Returns
        -------
        M:
            Magnitude of the frequency response.
        M_dB:
            Magnitude in decibels.
        phi:
            Phase angle of the frequency response in degrees.
        """
        K = K or self._K_value
        GH_jw = self._GH_jw_expr.subs(omega, omega_value)
        if not isinstance(omega_value, (int, float)):
            magn = K * sp.Abs(GH_jw)
            magn_dB = 20 * sp.log(magn, 10)
            phase = sp.arg(GH_jw) * 180 / sp.pi
            magn = sp.simplify(magn).evalf()
            magn_dB = sp.simplify(magn_dB).evalf()
            phase = phase.evalf()
        else:
            GH_jw = complex(GH_jw.evalf())
            magn = K * np.abs(GH_jw)
            magn_dB = 20 * np.log10(magn)
            phase = np.angle(GH_jw, deg=True)
        return magn, magn_dB, phase

    def bode_plot(
        self,
        K: float | None = None,
        omega_limits: tuple[float, float] = (0.1, 100),
        omega_num: int = 1000,
        title: str | None = '',
        **kwargs
    ) -> None:
        """Plots a Bode diagram of the frequency response. By default, the
        magnitude is returned in decibels and the phase angle in degrees.

        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        omega_limits:
            The lower and upper limit of the angular frequency (rad/s).
        omega_num:
            Number of calculation points between the lower and upper limit.
        title:
            Title to be displayed above the Bode diagram.
        **kwargs:
            Additional keyword arguments (refer to *Python Control Systems 
            Library* documentation on the function `bode_plot`).
        """
        K = K or self._K_value
        KGH = K * self._GH
        freq_resp = ct.frequency_response(
            KGH.as_ct,
            omega_limits=omega_limits,
            omega_num=omega_num
        )
        ct.bode_plot(
            freq_resp,
            omega_limits=omega_limits,
            dB=kwargs.pop('dB', True),
            deg=kwargs.pop('deg', True),
            wrap_phase=kwargs.pop('wrap_phase', True),
            title=title,
            **kwargs
        )
        plt.show()

    def nyquist_plot(
        self,
        K: float | None = None,
        omega_limits: tuple[float, float] = (0.1, 100),
        omega_num: int = 1000,
        title: str | None = '',
        **kwargs
    ) -> None:
        """Plots the Nyquist diagram of the transfer function.

        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        omega_limits:
            The lower and upper limit of the angular frequency (rad/s).
        omega_num:
            Number of calculation points between the lower and upper limit.
        title:
            Title to be displayed above the Nyquist diagram.
        **kwargs:
            Additional keyword arguments (refer to *Python Control Systems 
            Library* documentation on the function `nyquist_plot`).
        """
        K = K or self._K_value
        KGH = K * self._GH
        resp = ct.nyquist_response(
            KGH.as_ct,
            omega_limits=omega_limits,
            omega_num=omega_num,
            indent_radius=kwargs.pop('indent_radius', None),
            indent_points=kwargs.pop('indent_points', 0),
            indent_direction=kwargs.pop('indent_direction', 'right')
        )
        # noinspection PyTypeChecker
        ct.nyquist_plot(resp, title=title, **kwargs)
        plt.show()

    @property
    def marginal_stability_gain(self) -> tuple[float, float]:
        """Returns the open-loop gain `K` for which the closed-loop feedback
        system becomes unstable and the angular frequency (rad/s) at which it 
        will be oscillating then.
        """
        # Get imaginary part of the frequency response (the imaginary part is
        # the numerator in the expression of the phase angle of GH(jw); when
        # the imaginary part becomes zero, this means that the phase angle is
        # either 0° or +/- 180°).
        imag_GH_jw = sp.nsimplify(sp.im(self._GH_jw_expr))
        imag_GH_jw_num, _ = imag_GH_jw.as_numer_denom()
        # Find omegas for which the imaginary part of the frequency response is
        # zero (i.e. for which the phase angle of the frequency response is
        # +/- 180° or 0°):
        sol_w = sp.solve(imag_GH_jw_num, omega)
        # Get the magnitude (also in decibels) and phase angle of GH(jw)
        # for each omega in `sol_w`:
        tups = [self.evaluate(w.evalf(), K=1.0) for w in sol_w]
        # Get the magnitude of GH(jw) that goes with a phase angle of 180°:
        M_180, omega_ms, K_ms = None, None, None
        for i, (M, M_dB, phi) in enumerate(tups):
            if abs(phi) == 180.0:
                M_180 = float(M)
                omega_ms = float(sol_w[i])
                break
        # Get the gain `K_ms` with which the magnitude `M_180` needs to be 
        # multiplied to get 1. In that case the denominator 1 + F(jw) of the 
        # closed-loop transfer function will become zero (as the phase angle 
        # = 180°) and the system is called marginally unstable.
        if M_180 is not None: K_ms = 1 / M_180
        return K_ms, omega_ms

    @property
    def stability_gain_range(self) -> tuple[float, float] | None:
        """Returns the lower and upper limit of the values for the open-loop 
        gain `K` between which the closed-loop feedback system is stable.
        """
        K_ms = self.marginal_stability_gain[0]
        try:
            K_ms = float(K_ms)
        except TypeError:
            raise ValueError(
                "The gain for marginal stability could not be determined."
                f"Got K = {K_ms} for marginal stability."
            ) from None
        left_poles = [pole for pole in self._GH.poles if pole.real < 0]
        right_poles = [pole for pole in self._GH.poles if pole.real > 0]
        if not right_poles:
            # All open-loop poles are in the left-half plane or on the imaginary
            # axis.
            return 0, K_ms
        elif not left_poles:
            # All open-loop poles are in the right-half plane or on the
            # imaginary axis.
            return K_ms, float('inf')
        else:
            return None

    def gain_margin(self, K: float | None = None) -> tuple[float, float]:
        """Returns the margin in decibels that remains available for changes in 
        open-loop gain before the feedback system with the specified open-loop
        gain `K` becomes unstable. Also returns the angular frequency at which
        the Nyquist diagram crosses the negative real axis (i.e. the frequency
        for which the phase angle of the open-loop frequency response is
        +/- 180°).
        
        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        """
        K = K or self._K_value
        # Get imaginary part of the frequency response:
        imag_GH_jw = sp.im(self._GH_jw_expr)
        imag_GH_jw_num, _ = imag_GH_jw.as_numer_denom()
        # Find omegas for which the imaginary part of the frequency response is
        # zero (i.e. for which the phase angle of the frequency response is
        # +/- 180° or 0°):
        sol_w = sp.solve(imag_GH_jw_num, omega)
        # Get the magnitude, magnitude in decibels and phase angle of GH(jw)
        # for each omega in sol_w:
        tups = [self.evaluate(w, K) for w in sol_w]
        # Get the magnitude of GH(jw) that goes with a phase angle of 180°:
        M_180, omega_180 = None, None
        for i, (M, M_dB, phi) in enumerate(tups):
            if abs(phi) == 180.0:
                M_180 = float(M)
                omega_180 = float(sol_w[i])
                break
        # Get the value with which the magnitude `M_180` needs to be multiplied 
        # to get 1 and the system becoming unstable.
        if M_180 is not None:
            K = 1 / M_180
            return 20 * np.log10(K), omega_180

    def phase_margin(
        self,
        K: float | None = None,
        omega_limits: tuple[float, float] = (1.e-3, 1.e9)
    ) -> tuple[float, float]:
        """Returns the phase margin in degrees that remains available for
        changes in phase shift before the feedback system with the specified 
        open-loop gain `K` becomes unstable. Also returns the angular frequency 
        at which the magnitude of the open-loop frequency response is equal to 
        1 (0 dB).

        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        omega_limits:
            Lower and upper limit of the frequency range where the root-finding
            algorithm searches for the frequency at which the magnitude of the
            open-loop frequency response is equal to 1 (0 dB).
            If no solution can be found within the specified frequency range, a
            `ValueError` is raised. In that case, either the lower limit can be 
            decreased, or the upper limit can be increased.

        Returns
        -------
        phase_margin:
            Phase margin in degrees.
        omega_1:
            Angular frequency (rad/s) where the phase margin is measured, i.e.
            the frequency at which the magnitude of the frequency response is
            equal to 1 (0 dB).
        """
        K = K or self._K_value
        # Find the omega for which the magnitude of the frequency response with
        # open-loop gain K equals 1.
        M_expr = (K * self.magnitude).evalf()
        M_fun = sp.lambdify(omega, M_expr, 'numpy')

        def _objective(omega_: float) -> float:
            M = M_fun(omega_)
            return M - 1

        try:
            sol = root_scalar(
                _objective,
                bracket=(omega_limits[0], omega_limits[1])
            )
        except ValueError:
            raise ValueError(
                f"No frequency found where the magnitude is equal to 1 "
                f"between {omega_limits[0]:.3e} and {omega_limits[1]:.3e} rad/s."
            ) from None
        else:
            omega_1 = sol.root
            *_, phi = self.evaluate(omega_1, K)
            phase_margin = 180 - abs(phi)
            return phase_margin, omega_1

    def nichols_plot(
        self,
        K: float | None = None,
        omega_limits: tuple[float, float] = (0.1, 100),
        omega_num: int = 1000,
        title: str | None = '',
        **kwargs
    ) -> None:
        """Plots the frequency response on a Nichols chart, a plot of open-loop
        magnitude in dB vs. open-loop angle phase angle in degrees.

        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        omega_limits:
            The lower and upper limit of the angular frequency (rad/s).
        omega_num:
            Number of calculation points between the lower and upper limit.
        title:
            Title to be displayed above the Nichols chart.
        **kwargs:
            Additional keyword arguments, see Python Control Systems Library
            documentation about function `nichols_plot`.
        """
        K = K or self._K_value
        KGH = K * self._GH
        freq_resp = ct.frequency_response(
            KGH.as_ct,
            omega_limits=omega_limits,
            omega_num=omega_num
        )
        ct.nichols_plot(
            freq_resp,
            title=title,
            **kwargs
        )
        plt.show()

    def __create_GH_jw_fun(self) -> Callable:
        """Creates a function that takes an angular frequency (float) or range
        of angular frequencies (Numpy array) and returns the frequency response
        as a complex number or a range of complex numbers (Numpy array).
        """
        G_jw_mag = sp.lambdify(omega, self.magnitude, 'numpy')
        G_jw_phi = sp.lambdify(omega, self.phase, 'numpy')

        def f(omega_: float | np.ndarray) -> complex | np.ndarray:
            M = G_jw_mag(omega_)
            phi = G_jw_phi(omega_)
            return M * np.exp(phi * 1j)

        return f

    def __call__(
        self,
        omega_: float | np.ndarray,
        K: float | None = None
    ) -> complex | np.ndarray:
        """Returns the frequency response of the system at the specified angular
        frequency `omega_` (rad/s) or range of angular frequencies (Numpy array)
        and the specified open-loop gain `K`. If `K` is `None`, the gain of
        the original transfer function is used.
        If a single angular frequency `omega_` is passed, a single complex
        number is returned. If a Numpy array of angular frequencies is passed, a
        Numpy array of complex numbers is returned.
        """
        with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
            G_jw_values = self.__GH_jw_fun__(omega_)
            G_jw_values *= K or self._K_value
            return G_jw_values

    def get_peak_magnitude(
        self,
        K: float | None = None,
        omega_limits: tuple[float, float] = (0.01, 1000),
        omega_num: int = 1000
    ) -> tuple[float, float]:
        """Returns the peak value of the closed-loop frequency response
        magnitude within the specified angular frequency range, and the angular 
        frequency (rad/s) at which this peak value occurs.

        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        omega_limits:
            The lower and upper limit of the angular frequency (rad/s).
        omega_num:
            Number of calculating points between the lower and upper limit.

        Returns
        -------
        M_p:
            Peak value of the closed-loop frequency response magnitude.
        omega_p:
            Angular frequency at which the peak value occurs (rad/s).
        """
        K = K or self._K_value
        omega_arr = np.linspace(omega_limits[0], omega_limits[1], omega_num)
        T_jw_vals = self(omega_arr, K)
        T_jw_magn = np.abs(T_jw_vals)
        i_max = np.argmax(T_jw_magn)
        omega_1 = omega_arr[i_max - 1]
        omega_2 = omega_arr[i_max + 1]

        def _objective(omega_: float) -> float:
            T_jw_val = self(omega_, K)
            T_jw_magn = np.abs(T_jw_val)
            return -T_jw_magn

        res = minimize_scalar(_objective, bounds=(omega_1, omega_2))
        omega_p = res.x
        T_jw_val_max = self(omega_p, K)
        M_p = np.abs(T_jw_val_max)
        return M_p, omega_p

    def get_bandwidth_frequency(
        self,
        K: float | None = None,
        omega_limits: tuple[float, float] = (0.01, 1000),
        omega_num: int = 1000
    ) -> tuple[float, float]:
        """Returns the bandwidth frequency of the closed-loop frequency
        response.

        Parameters
        ----------
        K:
            Value for the gain at which the frequency response needs to be 
            evaluated. If `None` (default), the gain of the original transfer 
            function is taken.
        omega_limits:
            The lower and upper limit of the angular frequency (rad/s) range
            between which the bandwidth frequency can be searched for.
        omega_num:
            Number of calculating points between the lower and upper limit.

        Returns
        -------
        omega_bdw:
            Bandwidth frequency of the closed-loop frequency response.
        M_bdw:
            Magnitude of the closed-loop frequency response at the bandwidth
            frequency.

        Raises
        ------
        ValueError
            If the bandwith frequency cannot be found within the specified
            frequency range `omega_limits`.
        """
        K = K or self._K_value
        # Determine the dB-magnitude at the bandwidth frequency.
        T_jw_0 = self(0.0, K)
        T_jw_magn_dB_0 = 20 * np.log10(np.abs(T_jw_0))
        T_jw_magn_dB_bdw = T_jw_magn_dB_0 - 3.0

        # Calculate the dB-magnitude of the closed-loop frequency response at
        # `omega_num` points between the specified frequency limits
        # `omega_limits`.
        omega_range = np.linspace(omega_limits[0], omega_limits[1], omega_num)
        # noinspection PyTypeChecker
        T_jw_vals = self(omega_range, K)
        T_jw_magn_dB = 20 * np.log10(np.abs(T_jw_vals))

        # Determine the index of the dB-magnitude which is closest to the dB-
        # magnitude at the bandwidth frequency. Using this index a bracket can
        # be set within which the bandwidth frequency can be determined more
        # accurately using a root-finding algorithm.
        i = np.abs(T_jw_magn_dB_bdw - T_jw_magn_dB).argmin()
        k = i - 1 if i > 0 else 0
        l = i + 1 if i < omega_range.size - 1 else i
        # noinspection PyTypeChecker
        bracket = sorted([omega_range[k], omega_range[l]])

        def _objective(omega_: float) -> float:
            T_jw_val = self(omega_, K)
            T_jw_magn_dB = 20 * np.log10(np.abs(T_jw_val))
            err = T_jw_magn_dB_bdw - T_jw_magn_dB
            return err

        try:
            sol = root_scalar(_objective, bracket=bracket)
        except ValueError:
            raise ValueError(
                "Bandwidth frequency could not be determined within the"
                f"frequency range {bracket[0]}...{bracket[1]} rad/s."
            ) from None
        else:
            omega_bdw = sol.root
            T_jw_magn_bdw = 10 ** (T_jw_magn_dB_bdw / 20)
            return omega_bdw, T_jw_magn_bdw


def plot_bode_diagrams(
    transfer_functions: Sequence[TransferFunction],
    omega_limits: tuple[float, float] = (0.1, 100.0),
    omega_num: int = 1000,
    names: Sequence[str] | None = None,
    **kwargs
) -> None:
    """Plots the Bode diagram of multiple frequency responses in a single chart.

    Parameters
    ----------
    transfer_functions:
        Sequence of `TransferFunction` objects for which the Bode diagram needs
        to be plotted.
    omega_limits:
        The lower and upper limit of the angular frequency.
    omega_num:
        Number of calculating points between the lower and upper limit.
    names:
        Optional names for the systems in the same order as their frequency
        responses.
    **kwargs:
        Additional keyword arguments, see Python Control Systems Library
        documentation about function `bode_plot`.

    Returns
    -------
    None
    """
    freq_resp_data_list = []
    for i, transfer_function in enumerate(transfer_functions):
        freq_resp_data = ct.frequency_response(
            transfer_function.as_ct,
            omega_limits=omega_limits,
            omega_num=omega_num
        )
        if names is not None and len(names) == len(transfer_functions):
            freq_resp_data.sysname = names[i]
        else:
            freq_resp_data.sysname = f"sys [{i}]"
        freq_resp_data_list.append(freq_resp_data)

    ct.bode_plot(
        freq_resp_data_list,
        omega_limits=omega_limits,
        dB=kwargs.pop('dB', True),
        deg=kwargs.pop('deg', True),
        wrap_phase=kwargs.pop('wrap_phase', True),
        **kwargs
    )
    plt.show()


class ClosedLoopTransientResponse:
    """Class that encapsulates equations which represent a relationship between 
    a system's open-loop (phase margin) or closed-loop frequency response (peak
    magnitude, peak frequency, natural frequency, and bandwidth frequency) and
    the transient response of the closed-loop system (damping ratio, settling
    time, and peak time).

    Notes
    -----
    These equations are only valid on condition that the open-loop system can be
    approximated as a second-order system.
    """
    def __init__(self):
        """Creates an instance of class `ClosedLoopTransientResponse`."""
        self._M_p = sp.Symbol('M_p', real=True, positive=True)
        self._zeta = sp.Symbol('zeta', real=True, positive=True)
        self._omega_p = sp.Symbol('omega_p', real=True, positive=True)
        self._omega_nat = sp.Symbol('omega_nat', real=True, positive=True)
        self._omega_bdw = sp.Symbol('omega_bdw', real=True, positive=True)
        self._T_s = sp.Symbol('T_s', real=True, positive=True)
        self._T_p = sp.Symbol('T_p', real=True, positive=True)
        self._phi_m = sp.Symbol('phi_m', real=True)

        self._variables = {
            'M_p': {'symbol': self._M_p, 'value': None},
            'zeta': {'symbol': self._zeta, 'value': None},
            'omega_p': {'symbol': self._omega_p, 'value': None},
            'omega_nat': {'symbol': self._omega_nat, 'value': None},
            'omega_bdw': {'symbol': self._omega_bdw, 'value': None},
            'T_s': {'symbol': self._T_s, 'value': None},
            'T_p': {'symbol': self._T_p, 'value': None},
            'phi_m': {'symbol': self._phi_m, 'value': None}
        }

        self._equations = [
            {
                'eq': self._peak_magnitude(),
                'variables': (
                    self._variables['zeta'],
                    self._variables['M_p']
                ),
                'subs_eq': None,
                'unknowns': []},
            {
                'eq': self._peak_frequency(),
                'variables': (
                    self._variables['omega_nat'],
                    self._variables['zeta'],
                    self._variables['omega_p']
                ),
                'subs_eq': None,
                'unknowns': []},
            {
                'eq': self._bandwidth_frequency(),
                'variables': (
                    self._variables['zeta'],
                    self._variables['omega_nat'],
                    self._variables['omega_bdw']
                ),
                'subs_eq': None,
                'unknowns': []},
            {
                'eq': self._natural_frequency_with_settling_time(),
                'variables': (
                    self._variables['T_s'],
                    self._variables['zeta'],
                    self._variables['omega_nat']
                ),
                'subs_eq': None,
                'unknowns': []},
            {
                'eq': self._natural_frequency_with_peak_time(),
                'variables': (
                    self._variables['T_p'],
                    self._variables['zeta'],
                    self._variables['omega_nat']
                ),
                'subs_eq': None,
                'unknowns': []},
            {
                'eq': self._phase_margin(),
                'variables': (
                    self._variables['zeta'],
                    self._variables['phi_m']
                ),
                'subs_eq': None,
                'unknowns': []},
        ]

    def _peak_magnitude(self) -> sp.Eq:
        # variables: zeta, M_p
        lhs = 1 / (2 * self._zeta * sp.sqrt(1 - self._zeta ** 2))
        return sp.Eq(lhs, self._M_p)

    def _peak_frequency(self) -> sp.Eq:
        # variables: omega_nat, zeta, omega_p
        lhs = self._omega_nat * sp.sqrt(1 - 2 * self._zeta ** 2)
        return sp.Eq(lhs, self._omega_p)

    def _bandwidth_frequency(self) -> sp.Eq:
        # variables: zeta, omega_nat, omega_bdw
        a = 1 - 2 * self._zeta ** 2
        b = sp.sqrt(4 * self._zeta ** 4 - 4 * self._zeta ** 2 + 2)
        lhs = self._omega_nat * sp.sqrt(a + b)
        return sp.Eq(lhs, self._omega_bdw)

    def _natural_frequency_with_settling_time(self) -> sp.Eq:
        # variables: T_s, zeta, omega_nat
        lhs = 4 / (self._T_s * self._zeta)
        return sp.Eq(lhs, self._omega_nat)

    def _natural_frequency_with_peak_time(self) -> sp.Eq:
        # variables: T_p, zeta, omega_nat
        lhs = sp.pi / (self._T_p * sp.sqrt(1 - self._zeta ** 2))
        return sp.Eq(lhs, self._omega_nat)

    def _phase_margin(self) -> sp.Eq:
        # variables: zeta, phi_m
        n = 2 * self._zeta
        d = sp.sqrt(-2 * self._zeta**2 + sp.sqrt(1 + 4 * self._zeta**4))
        lhs = sp.atan2(n, d)
        return sp.Eq(lhs, self._phi_m)

    @staticmethod
    def _substitute(equation: dict) -> None:
        subs_dict = {}
        for variable in equation['variables']:
            if variable['value'] is not None:
                subs_dict[variable['symbol']] = variable['value']
            else:
                equation['unknowns'].append(variable)
        if subs_dict:
            equation['subs_eq'] = equation['eq'].subs(subs_dict)

    def _solve(self, equation: dict) -> None:
        if len(equation['unknowns']) == 1:
            unknown = equation['unknowns'].pop()
            sol = sp.solve(equation['subs_eq'], unknown['symbol'])
            if unknown['symbol'] is self._zeta and len(sol) > 1:
                sol = [sol_ for sol_ in sol if sol_ <= sp.sqrt(sp.S(1) / 2)]
                # a damping ratio larger than sqrt(1/2) yields no peak above
                # zero frequency --> see: equation of _peak_frequency
            try:
                unknown['value'] = float(sol[0])
            except IndexError:
                unknown['value'] = float('nan')

    def _reset(self) -> None:
        for equation in self._equations:
            equation['unknowns'] = []
            equation['subs_eq'] = None

    @classmethod
    def solve(
        cls,
        zeta: float | None = None,
        M_p: float | None = None,
        omega_p: float | None = None,
        omega_nat: float | None = None,
        omega_bdw: float | None = None,
        T_s: float | None = None,
        T_p: float | None = None,
        phi_m: float | None = None
    ) -> ClosedLoopTransientResponse:
        """Solves (if possible) the equations of `ClosedLoopTransientResponse`
        depending on the inputs that are given.

        Parameters
        ----------
        zeta:
            Damping ratio.
        M_p:
            Peak magnitude of the closed-loop frequency response.
        omega_p:
            Peak frequency, i.e. the angular frequency (rad/s) at which the
            peak magnitude occurs.
        omega_nat:
            The closed-loop feedback system's natural frequency (rad/s).
        omega_bdw:
            Bandwith frequency, i.e. the angular frequency (rad/s) at which the
            closed-loop magnitude response curve is 3 dB down from its value at
            zero frequency.
        T_s:
            Settling time (s).
        T_p:
            Peak time (s).
        phi_m:
            Phase margin (degrees) of the open-loop frequency response.

        Returns
        -------
        Instance of `ClosedLoopTransientResponse`. Solutions of the equations
        can be retrieved through the properties.

        Notes
        -----
        Given the closed-loop frequency response,
        -   the damping ratio `zeta` can be determined if the closed-loop peak
            magnitude `M_p` is known (or if the open-loop phase margin `phi_m`
            is known).
        -   the natural frequency `omega_nat` can be determined if the damping
            ratio `zeta` and the peak frequency `omega_p` are known.
        -   the settling time `T_s` can be determined if the damping ratio
            `zeta` and the natural frequency `omega_nat` are known.
        -   the peak time `T_p` can be determined if the damping ratio `zeta`
            and the natural frequency `omega_nat` are known.
        """
        self = cls()

        if zeta is not None and zeta > sp.sqrt(sp.S(1) / 2):
            warnings.warn(
                "Damping ratio 'zeta' cannot be larger than 0.707 "
                "to have a valid positive peak frequency.",
                category=UserWarning
            )

        self._variables['zeta']['value'] = zeta
        self._variables['M_p']['value'] = M_p
        self._variables['omega_p']['value'] = omega_p
        self._variables['omega_nat']['value'] = omega_nat
        self._variables['omega_bdw']['value'] = omega_bdw
        self._variables['T_s']['value'] = T_s
        self._variables['T_p']['value'] = T_p
        self._variables['phi_m']['value'] = np.radians(phi_m) if phi_m else None

        for i in range(2):
            for equation in self._equations:
                self._substitute(equation)
                self._solve(equation)
            self._reset()

        return self

    @property
    def M_p(self) -> float | None:
        """Peak magnitude of the closed-loop frequency response. If the
        value could not be determined, `None` is returned.
        """
        return self._variables['M_p']['value']

    @property
    def zeta(self) -> float | None:
        """Damping ratio of the closed-loop transient response. If the value
        could not be determined, `None` is returned.
        """
        return self._variables['zeta']['value']

    @property
    def omega_p(self) -> float | None:
        """Angular frequency (rad/s) at which the peak magnitude occurs. If the
        value could not be determined, `None` is returned.
        """
        return self._variables['omega_p']['value']

    @property
    def omega_nat(self) -> float | None:
        """Natural angular frequency (rad/s) of the closed-loop system. If the
        value could not be determined, `None` is returned.
        """
        return self._variables['omega_nat']['value']

    @property
    def omega_bdw(self) -> float | None:
        """Bandwidth frequency (rad/s) of the closed-loop system. If the value
        could not be determined, `None` is returned.
        """
        return self._variables['omega_bdw']['value']

    @property
    def T_s(self) -> float | None:
        """Settling time (s) of the closed-loop transient response. If the value
        could not be determined, `None` is returned.
        """
        return self._variables['T_s']['value']

    @property
    def T_p(self) -> float | None:
        """Peak time (s) of the closed-loop transient response. If the value
        could not be determined, `None` is returned.
        """
        return self._variables['T_p']['value']

    @property
    def phi_m(self) -> float | None:
        """Phase margin (degrees) of the open-loop frequency response. If the
        value could not be determined, `None` is returned.
        """
        if (phi_m := self._variables['phi_m']['value']) is not None:
            return np.degrees(phi_m)
        return None
