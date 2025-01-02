"""Root-Locus Design of SISO LTI Feedback Control Systems.

References
----------
Nise, N. S. (2020). Control Systems Engineering, EMEA Edition, 8th Edition.
"""
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
import sympy as sp
from ..core.symbols import K, s
from ..core.transfer_function import TransferFunction
from ..core.systems.feedback_old import FeedbackSystem
from ..core.root_locus import RootLocus, TransferFunctionVector, Vector
from .general import (
    create_feedback_system,
    get_dominant_poles,
    create_transfer_function,
    create_second_order_system
)


ControllerGains = namedtuple('ControllerGains', ('K_p', 'K_i', 'K_d'))


@dataclass
class SystemData:
    """Dataclass returned by the design functions in this module that holds
    the information about the designed feedback control system.

    Attributes
    ----------
    feedback_system: FeedbackSystem
        The designed feedback system.
    plant: TransferFunction | None
        Plant or process transfer function (without forward-path gain).
    controller: TransferFunction | None
        Controller transfer function.
    forward_path: TransferFunction | None
        Forward-path transfer function.
    feedback_path: TransferFunction | None
        Feedback-path transfer function.
    forward_gain: float
        Forward-path gain.
    feedback_gain: float | None
        Feedback-path gain.
    open_loop_gain: float | None
        Open-loop gain (product of forward gain and feedback gain).
    controller_gains: namedtuple ControllerGains | None
        The gains of each control action: `K_p` = proportional gain,
        `K_i`= integral gain, and `K_d` = differential gain.
    dominant_pole: complex
        The dominant, complex closed-loop pole in the positive left-half of the
        complex plane. (There is also a dominant pole in the negative left-half,
        which is the conjugate of the dominant pole in the positive left-half.)
    damping_ratio: float
        Damping ratio of the feedback system.
    natural_frequency: float
        Natural frequency of the feedback system (i.e. the magnitude or modulus
        of the dominant poles).
    settling_time: float
        Settling time.
    peak_time: float
        Peak time.
    rise_time: float
        Rise time.
    percent_overshoot: float
        Percent overshoot
    e_oo: float | None.
        Steady-state error of the feedback system; `None` if the system is not
        of type 0, type 1, or type 2. The value of `e_oo` will depend on the
        type of system. If the system is of type 0, it is the steady-state error
        of the unit step response. For a type 1 system, it is the steady-state
        error of the ramp response, and for a type 2 system it is the
        steady-state error of the parabola response. If you need the steady-state
        error for a specific type of response, use method `steady_state_error()`
        instead.
    root_locus: RootLocus
        Root locus of the feedback system.

    Notes
    -----
    Which attributes are filled in, will depend on the design function that
    returns the `SystemData` object.
    """
    feedback_system: FeedbackSystem | None
    damping_ratio: float
    forward_gain: float
    root_locus: RootLocus
    dominant_pole: complex
    natural_frequency: float
    plant: TransferFunction | None = None
    controller: TransferFunction | None = None
    controller_gains: ControllerGains | None = None
    open_loop_gain: float | None = None
    feedback_gain: float | None = None
    forward_path: TransferFunction | None = None
    feedback_path: TransferFunction | None = None

    def __post_init__(self):
        if self.dominant_pole:
            sosys = create_second_order_system(self.dominant_pole)
            self.settling_time = sosys.settling_time
            self.peak_time = sosys.peak_time
            self.rise_time = sosys.rise_time
            self.percent_overshoot = sosys.percent_overshoot
            self.e_oo = self._calc_steady_state_error()
        else:
            self.settling_time = float('nan')
            self.peak_time = float('nan')
            self.rise_time = float('nan')
            self.percent_overshoot = float('nan')
            self.e_oo = float('nan')

    def _calc_steady_state_error(self) -> float | None:
        if self.feedback_system.system_type == 'type_0':
            return self.steady_state_error('step')
        if self.feedback_system.system_type == 'type_1':
            return self.steady_state_error('ramp')
        if self.feedback_system.system_type == 'type_2':
            return self.steady_state_error('parabola')
        return None

    def steady_state_error(self, type_of_response: str) -> float | None:
        """Returns the steady-state error that corresponds with the specified
        type of response. `type_of_response` must be a string that can take one
        of these words: 'step', 'ramp', or 'parabola'.
        """
        if type_of_response == 'step':
            e_oo, *_ = self.feedback_system.steady_state_error(1 / s)
            return float(e_oo)
        if type_of_response == 'ramp':
            e_oo, *_ = self.feedback_system.steady_state_error(1 / s ** 2)
            return float(e_oo)
        if type_of_response == 'parabola':
            e_oo, *_ = self.feedback_system.steady_state_error(1 / s ** 3)
            return float(e_oo)
        return None

    @property
    def characteristics(self) -> str:
        """Returns the transient response and steady-state error characteristics
        of the feedback system in a single string.
        """
        return self.feedback_system.characteristics


def get_integral_gain(
    KG_c: TransferFunction,
    sysdata: SystemData
) -> tuple[float, float]:
    """Returns the gain of the integrator and the proportional gain of the
    PI-controller.

    Parameters
    ----------
    KG_c:
        Transfer function of the PI controller.
    sysdata:
        System data of the compensated system.

    Returns
    -------
    K_i:
        Integral gain.
    K_p:
        Proportional gain.
    """
    zero_c = KG_c.zeros[0]
    K_p = sysdata.forward_gain
    K_i = abs(zero_c) * K_p
    return K_i, K_p


def get_derivative_gain(
    KG_c: TransferFunction,
    sysdata: SystemData
) -> tuple[float, float]:
    """Returns the gain of the differentiator and the proportional gain of the
    PD-controller.

    Parameters
    ----------
    KG_c:
        Transfer function of the PD-controller.
    sysdata:
        System data of the compensated system.

    Returns
    -------
    K_d:
        Differential gain.
    K_p:
        Proportional gain.
    """
    zero_c = KG_c.zeros[0]
    K_d = sysdata.forward_gain
    K_p = K_d * abs(zero_c)
    return K_d, K_p


def design_proportional_feedback(
    G_plant: TransferFunction,
    damping_ratio: float,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> SystemData:
    """Designs an uncompensated unity-feedback system (i.e. a feedback system
    with only forward-path gain). Determines the required forward-path gain
    (proportional gain) such that the required damping ratio of the closed-loop
    feedback system is attained.

    Parameters
    ----------
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    damping_ratio:
        Required damping ratio of the system.
    name:
        An optional name to identify the system.
    gain_symbol:
        Sympy symbol that represents the gain in the forward path (default
        symbol is `K`).

    Returns
    -------
    sysdata:
        Instance of class `SystemData`.

    Raises
    ------
    ValueError:
        If no closed-loop pole of the feedback system can be found on the
        damping ratio line.
    """
    # Multiply the transfer function of the plant with the symbolic forward-path
    # gain K:
    KG = TransferFunction(gain_symbol) * G_plant
    # Create the root locus of the feedback system:
    root_locus = RootLocus(KG)
    # Search for points on the required damping ratio line in the positive
    # left-half of the complex plane where the root locus crosses.
    zeta_crossings = root_locus.find_damping_ratio_crossings(damping_ratio)
    # It is assumed that the element closest to the imaginary axis is the dominant
    # complex closed-loop pole of the feedback system in the positive left-half
    # of the complex plane (its conjugate in the negative left-half of the
    # complex plane is the other pole of the dominant pole pair).
    if zeta_crossings:
        real_parts = [abs(zc[0].real) for zc in zeta_crossings]
        i_dp = real_parts.index(min(real_parts))
        dominant_pole = zeta_crossings[i_dp][0]
        gain_value = zeta_crossings[i_dp][1]
        omega_nat = abs(dominant_pole)
        feedback_system = create_feedback_system(
            gain_value,
            KG,
            name,
            gain_symbol
        )
        return SystemData(
            feedback_system,
            damping_ratio,
            gain_value,
            root_locus,
            dominant_pole,
            omega_nat,
            G_plant
        )
    else:
        warnings.warn(
            "Unable to create the feedback system. Could not find a closed-loop "
            "pole on the damping ratio line.",
            category=RuntimeWarning
        )
        return SystemData(
            feedback_system=None,
            damping_ratio=damping_ratio,
            forward_gain=float('nan'),
            root_locus=root_locus,
            dominant_pole=complex(float('nan'), float('nan')),
            natural_frequency=float('nan'),
        )


def design_with_compensation(
    KG_c: TransferFunction,
    G_plant: TransferFunction,
    damping_ratio: float,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> SystemData:
    """Designs a unity-feedback system including cascade compensation (i.e. the
    transfer function of the compensator is added in series to the transfer
    function of the plant or process in the forward path of the unity-feedback
    system). Determines the required forward-path gain such that the required
    damping ratio of the closed-loop feedback system is attained.

    Parameters
    ----------
    KG_c:
        Transfer function of the compensator in which the forward-path gain is
        represented by a Sympy `Symbol` object. For this the predefined symbol
        `K` from the `python_control` package can be used (defined in module
        `python_control.core.symbols.py`).
    G_plant:
        Transfer function of the plant or process **without** gain (i.e. the
        same as that the plant gain is 1).
    damping_ratio:
        Required damping ratio of the feedback system.
    name:
        An optional name to identify the feedback system.
    gain_symbol:
        The Sympy symbol that is used to represent the forward-path gain in the
        expression of the compensator's transfer function `KG_c`. The default
        symbol is the predefined symbol `K` from the `python_control` package.
        So, if this symbol is used in the expression of the compensator's
        transfer function `KG_c`, this parameter can be omitted in the function
        call.

    Returns
    -------
    sysdata:
        Instance of dataclass `SystemData`.

    Warnings
    --------
    RuntimeWarning:
        If no closed-loop pole of the feedback system is found on the damping
        ratio line.
    """
    KG = KG_c * G_plant
    root_locus = RootLocus(KG)
    zeta_crossings = root_locus.find_damping_ratio_crossings(damping_ratio)
    if zeta_crossings:
        real_parts = [abs(zc[0].real) for zc in zeta_crossings]
        i_dp = real_parts.index(min(real_parts))
        dominant_pole = zeta_crossings[i_dp][0]
        gain_value = zeta_crossings[i_dp][1]
        omega_nat = abs(dominant_pole)
        feedback_system = create_feedback_system(
            gain_value,
            KG,
            name,
            gain_symbol
        )
        return SystemData(
            feedback_system,
            damping_ratio,
            gain_value,
            root_locus,
            dominant_pole,
            omega_nat,
            G_plant,
            KG_c
        )
    else:
        warnings.warn(
            "Unable to create the feedback system. Could not find a closed-loop "
            "pole on the damping ratio line.",
            category=RuntimeWarning
        )
        return SystemData(
            feedback_system=None,
            damping_ratio=damping_ratio,
            forward_gain=float('nan'),
            root_locus=root_locus,
            dominant_pole=complex(float('nan'), float('nan')),
            natural_frequency=float('nan'),
        )


def design_lag_compensator(
    feedback_system: FeedbackSystem,
    e_oo_reduction_factor: float,
    compensator_pole: float,
    gain_symbol: sp.Symbol = K
) -> TransferFunction:
    """Designs the transfer function for a lag compensator for a type 0, type 1,
    or type 2 feedback system. A type 0 system is without pure integrators,
    a type 1 system has one pure integrator, and a type 2 system has two pure
    integrators. Determines the compensator zero such that the required
    reduction in steady-state error is attained. Note that the gain of the lag
    compensator is not determined by this function. To determine the gain use
    the function `design_with_compensation` next.

    Parameters
    ----------
    feedback_system:
        Feedback system of which the initial forward-path gain has already been
        determined.
    e_oo_reduction_factor:
        Factor (> 1) indicating by how much the steady-state error should be
        reduced.
    compensator_pole:
        Selected pole of the compensator transfer function. Depends on the
        construction of the compensator, but should be near zero.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lag compensator.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    KG_c:
        Transfer function of the lag compensator with symbolic gain (instance
        of class `TransferFunction`).

    Raises
    ------
    ValueError:
        If the uncompensated system is neither of type 0, type 1, or
        type 2.
    """
    system_type = feedback_system.system_type
    if system_type == 'type_0':
        e_oo, *_ = feedback_system.steady_state_error(1 / s)
        e_oo_target = float(e_oo) / e_oo_reduction_factor
        Kp_target = 1 / e_oo_target - 1
        z_c_div_p_c = Kp_target / feedback_system.Kp
        zero_c = z_c_div_p_c * compensator_pole
        KG_c = TransferFunction(gain_symbol * (s - zero_c) / (s - compensator_pole))
        return KG_c
    elif system_type == 'type_1':
        e_oo, *_ = feedback_system.steady_state_error(1 / s**2)
        e_oo_target = float(e_oo) / e_oo_reduction_factor
        Kv_target = 1 / e_oo_target
        z_c_div_p_c = Kv_target / feedback_system.Kv
        zero_c = z_c_div_p_c * compensator_pole
        KG_c = TransferFunction(gain_symbol * (s - zero_c) / (s - compensator_pole))
        return KG_c
    elif system_type == 'type_2':
        e_oo, *_ = feedback_system.steady_state_error(1 / s**3)
        e_oo_target = float(e_oo) / e_oo_reduction_factor
        Ka_target = 1 / e_oo_target
        z_c_div_p_c = Ka_target / feedback_system.Ka
        zero_c = z_c_div_p_c * compensator_pole
        KG_c = TransferFunction(gain_symbol * (s - zero_c) / (s - compensator_pole))
        return KG_c
    else:
        raise ValueError('Uncompensated system is not of type 0, 1, or 2.')


def design_PD_controller(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    gain_symbol: sp.Symbol = K
) -> TransferFunction:
    """Designs the transfer function for a PD-controller. Determines the
    compensator zero such that the required transient response requirements are
    met. Note that the gain of the PD controller is not determined by this
    function. To determine the gain use the function `design_with_compensation`
    next.

    Parameters
    ----------
    settling_time:
        Required settling time of the PD-controlled system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the PD-controlled system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the PD-controlled system.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    gain_symbol:
        Sympy symbol to be used to represent the gain of the PD controller.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.
    
    Returns
    -------
    KG_c:
        Transfer function of the PD-controller with symbolic gain (instance
        of class `TransferFunction`).
    """
    dominant_poles = get_dominant_poles(settling_time, peak_time, damping_ratio)
    sigma_dp = dominant_poles[0].real
    omega_dp = dominant_poles[0].imag
    # Determine the required location of the compensator zero on the real axis,
    # such that the dominant poles are on the root locus of the compensated
    # system:
    tfv = TransferFunctionVector(G_plant)
    tfv.point = dominant_poles[0]
    alpha = np.radians(tfv.angle)
    if alpha < 0: alpha += 2 * np.pi
    theta_c = np.pi - alpha
    zero_c = sigma_dp - omega_dp / np.tan(theta_c)
    KG_c = TransferFunction(gain_symbol * (s - zero_c))
    return KG_c


def design_lead_compensator(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    compensator_zero: float,
    gain_symbol: sp.Symbol = K
) -> TransferFunction:
    """Designs the transfer function for a lead compensator. Determines the
    compensator pole such that the required transient response requirements are
    met. Note that the gain of the lead compensator is not determined by this
    function. To determine the gain use the function `design_with_compensation`
    next.

    Parameters
    ----------
    settling_time:
        Required settling time of the lead-compensated system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the lead-compensated system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the lead-compensated system.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    compensator_zero:
        Selected lead compensator zero. In general the zero is placed on the
        negative real axis at a distance from the origin in the neighborhood of
        the natural frequency of the uncompensated system's dominant pole (1/4
        to 1 times the value of the natural frequency).
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lead compensator.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    KG_c:
        Transfer function of the lead compensator with symbolic gain (instance
        of class `TransferFunction`).
    """
    dominant_pole, _ = get_dominant_poles(settling_time, peak_time, damping_ratio)
    # The sum of the angle of the vector of `zero_c` and the angles of the
    # vectors of the uncompensated system's poles and zeros with respect to the
    # required dominant pole must be equal to 180°.
    tfv = TransferFunctionVector(G_plant)
    tfv.point = dominant_pole
    zero_c_vector = Vector(
        real=dominant_pole.real - compensator_zero,
        imag=dominant_pole.imag
    )
    theta = tfv.angle + zero_c_vector.angle
    pole_c_angle = np.radians(theta - 180.0)
    if pole_c_angle < 0: pole_c_angle += 2 * np.pi
    # Using trigonometry, tan(α) = |B| / |A| with B the positive imaginary part
    # of the required dominant closed-loop pole and |A| the distance between
    # `pole_c` and the real part of the required dominant closed-loop pole:
    pole_c = abs(dominant_pole.real) + dominant_pole.imag / np.tan(pole_c_angle)
    if pole_c > 0: pole_c *= -1
    KG_c = TransferFunction(gain_symbol * (s - compensator_zero) / (s - pole_c))
    return KG_c


def design_PD_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> SystemData:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with PD-controller,
    added in series to the plant (cascade compensation). The entered transfer
    function of the plant must not have any forward-path gain; the overall
    required forward-path gain (product of controller gain and any plant gain)
    will be included into the transfer function of the PD-controller.

    Parameters
    ----------
    settling_time:
        Required settling time of the PD-controlled system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the PD-controlled system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the PD-controlled system.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    name:
        Optional name to identify the system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the PD-controller.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    sysdata:
        Instance of class `SystemData`.
    """
    KG_c_pd = design_PD_controller(
        settling_time,
        peak_time,
        damping_ratio,
        G_plant,
        gain_symbol
    )
    sysdata_pd = design_with_compensation(
        KG_c_pd,
        G_plant,
        damping_ratio,
        name,
        gain_symbol
    )
    K_d = sysdata_pd.forward_gain
    KG_c_pd_expr = KG_c_pd.expr.subs(gain_symbol, K_d)
    KG_c_pd = TransferFunction(KG_c_pd_expr)
    sysdata_pd.controller = KG_c_pd
    coeffs = KG_c_pd.num_coeffs
    K_p = coeffs[1]
    sysdata_pd.controller_gains = ControllerGains(K_p, None, K_d)
    return sysdata_pd


def design_PID_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    integrator_zero: float,
    names: Sequence[str] = ('', ''),
    gain_symbol: sp.Symbol = K
) -> tuple[SystemData, SystemData]:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with PID-controller,
    added in series to the transfer function of the plant (cascade
    compensation). The entered transfer function of the plant must not have any
    forward-path gain; the overall required forward-path gain (product of
    controller gain and any plant gain) will be included into the transfer
    function of the PID-controller.

    First, a compensated system with a PD-controller is designed to meet the
    transient response specifications. Then, an integral compensator is added
    to this PD-controlled system to yield the required steady-state error.

    Parameters
    ----------
    settling_time:
        Required settling time of the PD-controlled system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the PD-controlled system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the PD-controlled system.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    integrator_zero:
        Zero of the PI-controller (should be placed close to the origin of the
        complex plane).
    names:
        Optional names to identify the PID-compensated system and the
        PD-controlled system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the PID-controller.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    sysdata_pid :
        Instance of `SystemData` which holds the final PID-controlled feedback
        system (`FeedbackSystem` object).
    sysdata_pd:
        Instance of `SystemData` which holds the PD-controlled feedback system
        (`FeedbackSystem` object).
    """
    # Create a PD-controlled feedback system:
    sysdata_pd = design_PD_feedback(
        settling_time,
        peak_time,
        damping_ratio,
        G_plant,
        names[1],
        gain_symbol
    )
    # Add I-control action to PD-controlled feedback system:
    G_c_i = TransferFunction((s - integrator_zero) / s)
    sysdata_pid = design_with_compensation(
        G_c_i,
        sysdata_pd.feedback_system.open_loop,
        damping_ratio,
        names[0],
        gain_symbol
    )
    KG_c_pid = sysdata_pd.controller * G_c_i
    KG_c_pid_expr = KG_c_pid.expr.subs(gain_symbol, sysdata_pid.forward_gain)
    KG_c_pid = TransferFunction(KG_c_pid_expr)
    sysdata_pid.controller = KG_c_pid
    coeffs = KG_c_pid.num_coeffs
    K_d = coeffs[0]
    K_p = coeffs[1]
    K_i = coeffs[2]
    sysdata_pid.controller_gains = ControllerGains(K_p, K_i, K_d)
    return sysdata_pid, sysdata_pd


def design_lag_feedback(
    feedback_system: FeedbackSystem,
    e_oo_reduction_factor: float,
    compensator_pole: float,
    damping_ratio: float,
    settling_time: float | None = None,
    peak_time: float | None = None,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> SystemData:
    """Given the feedback system, adds a lag compensator in series to the
    transfer function of the plant (cascade compensation) to reduce the
    steady-state error.

    Parameters
    ----------
    feedback_system:
        Feedback system of which the initial forward-path gain has already been
        determined.
    e_oo_reduction_factor:
        Factor (> 1) indicating by how much the steady-state error should be
        reduced.
    compensator_pole:
        Selected pole of the compensator transfer function. Depends on the
        construction of the compensator, but should be near zero.
    damping_ratio:
        Required damping ratio of the feedback system.
    settling_time:
        Required settling time of the lead-compensated system.
    peak_time:
        Required peak time of the lead-compensated system.
    name:
        Optional name to identify the system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lag compensator.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    sysdata:
        Instance of class `SystemData`.
    """
    KG_c = design_lag_compensator(
        feedback_system,
        e_oo_reduction_factor,
        compensator_pole,
        gain_symbol
    )
    sysdata_fin = design_with_compensation(
        KG_c,
        feedback_system.G_p,
        damping_ratio,
        name,
        gain_symbol
    )
    return sysdata_fin


def design_lead_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    compensator_zero: float,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> SystemData:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with lead compensator,
    added in series to the transfer function of the plant (cascade compensation).
    The entered transfer function of the plant must not have any forward-path
    gain; the overall required forward-path gain (product of controller gain and
    any plant gain) will be included into the transfer function of the lead
    compensator.

    Parameters
    ----------
    settling_time:
        Required settling time of the lead-compensated system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the lead-compensated system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the lead-compensated system.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    compensator_zero:
        Selected lead compensator zero. In general the zero is placed on the
        negative real axis at a distance from the origin in the neighborhood of
        the natural frequency of the uncompensated system's dominant pole (1/4
        to 1 times the value of the natural frequency).
    name:
        Optional name to identify the system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lead compensator.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    sysdata:
        Instance of class `SystemData`.
    """
    KG_lead = design_lead_compensator(
        settling_time,
        peak_time,
        damping_ratio,
        G_plant,
        compensator_zero,
        gain_symbol
    )
    sysdata_lead = design_with_compensation(
        KG_lead,
        G_plant,
        damping_ratio,
        name
    )
    KG_lead = create_transfer_function(
        sysdata_lead.forward_gain,
        KG_lead,
        gain_symbol
    )
    sysdata_lead.controller = KG_lead
    return sysdata_lead


def design_lag_lead_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    e_oo_reduction: float,
    G_plant: TransferFunction,
    lead_compensator_zero: float,
    lag_compensator_pole: float,
    names: Sequence[str] = ('', ''),
    gain_symbol: sp.Symbol = K
) -> tuple[SystemData, SystemData]:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with lag-lead
    compensator, added in series to the transfer function of the plant (cascade
    compensation).
    The entered transfer function of the plant must not have any forward-path
    gain; the overall required forward-path gain (product of controller gain and
    any plant gain) will be included into the transfer function of the lag-lead
    compensator.

    First, a lead-compensated system is designed to meet the transient response
    specifications. Then, a lag compensator is added to this lead-compensated
    system to yield the required steady-state error.

    Parameters
    ----------
    settling_time:
        Required settling time of the lead-compensated system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the lead-compensated system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the lead-compensated system.
    e_oo_reduction:
        Factor (> 1) by which the steady-state error should be reduced.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    lead_compensator_zero:
        Selected zero of the lead compensator transfer function. In general the
        zero is placed on the negative real axis at a distance from the origin
        in the neighborhood of the natural frequency of the uncompensated
        system's dominant pole (1/4 to 1 times the value of the natural
        frequency).
    lag_compensator_pole:
        Selected pole of the lag compensator transfer function (should be near
        zero).
    names:
        Optional names to identify the lag-lead compensated system and the
        lead-compensated system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lag-lead
        compensator. The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    sysdata_lead_lag :
        Instance of `SystemData` which holds the final lag-lead compensated
        feedback system (`FeedbackSystem` object).
    sysdata_lead:
        Instance of `SystemData` which holds the lead compensated feedback
        system (`FeedbackSystem` object).
    """
    sysdata_uncomp = design_proportional_feedback(
        G_plant,
        damping_ratio,
        '',
        gain_symbol
    )
    sysdata_lead = design_lead_feedback(
        settling_time,
        peak_time,
        damping_ratio,
        G_plant,
        lead_compensator_zero,
        names[1],
        gain_symbol
    )
    if sysdata_uncomp.e_oo is not None:
        x = sysdata_uncomp.e_oo / sysdata_lead.e_oo
        y = e_oo_reduction / x
    else:
        y = None
    KG_lag = design_lag_compensator(
        feedback_system=sysdata_lead.feedback_system,
        e_oo_reduction_factor=y if y is not None else 1.0,
        compensator_pole=lag_compensator_pole,
        gain_symbol=gain_symbol
    )
    KG_lag_expr = KG_lag.expr.subs(gain_symbol, 1)
    KG_lag = TransferFunction(KG_lag_expr)
    sysdata_lead_lag = design_with_compensation(
        KG_lag,
        sysdata_lead.feedback_system.open_loop,
        damping_ratio,
        names[0]
    )
    return sysdata_lead_lag, sysdata_lead


def design_rate_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    name: str = '',
    fdb_gain_symbol: sp.Symbol = sp.Symbol('K_f', real=True, positive=True),
    fwd_gain_symbol: sp.Symbol = sp.Symbol('K_1', real=True, positive=True)
) -> SystemData:
    """Designs a rate feedback compensation system consisting of a minor loop
    and a major loop. The minor loop contains the transfer function of the plant
    (process) in its forward path. The transfer function $H = K_f * s$ in the
    feedback path of the minor loop is a pure differentiator (e.g. rate sensor
    or tachometer). The major loop has no transfer function in its feedback
    path (i.e. $H = 1$) and also its forward-path gain $K = 1$.

    The feedback paths of minor and major loop can be added, so that the system
    is reduced to a single-loop feedback system of which the transfer function
    in the feedback path becomes $H = K_f * s + 1 = K_f * (s + 1/K_f)$, being
    similar to the transfer function of a PD-controller.

    Parameters
    ----------
    settling_time:
        Required settling time of the PD-compensated system. If `None`,
        parameter `peak_time` must be set.
    peak_time:
        Required peak time of the PD-compensated system. If `None`, parameter
        `settling_time` must be set.
    damping_ratio:
        Required damping ratio of the PD-compensated system.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    name:
        Optional name to identify the system.
    fdb_gain_symbol
        Sympy symbol to be used to represent the differentiator gain in the
        feedback path of the minor loop (default symbol is `K_f`).
    fwd_gain_symbol:
        Sympy symbol to be used to represent the forward-path gain of the minor
        loop (default symbol is `K_1`).

    Returns
    -------
    sysdata:
        Instance of class `SystemData`.
    """
    # Design of the PD-controller in the feedback path of the reduced system:
    H = design_PD_controller(
        settling_time=settling_time,
        peak_time=peak_time,
        damping_ratio=damping_ratio,
        G_plant=G_plant,
        gain_symbol=fdb_gain_symbol
    )
    zero_H = H.zeros[0]
    Kf_value = 1 / abs(zero_H)  # required PD-controller gain
    # Determine the open-loop gain and the required forward-path gain of the
    # reduced system:
    KGH = fwd_gain_symbol * G_plant * H  # open-loop transfer function
    root_locus = RootLocus(KGH)  # root locus of closed-loop feedback system
    zeta_crossings = root_locus.find_damping_ratio_crossings(damping_ratio)
    KGH_value = zeta_crossings[0][1]  # required open-loop gain
    K1_value = KGH_value / Kf_value   # required forward-path gain
    # Define the forward-path transfer function of the reduced system with
    # the calculated forward-path gain:
    KG = fwd_gain_symbol * G_plant
    KG_expr = KG.expr.subs(fwd_gain_symbol, K1_value)
    KG = TransferFunction(KG_expr)
    # Define the feedback-path transfer function of the reduced system with
    # the calculated PD-controller gain:
    H_expr = H.expr.subs(fdb_gain_symbol, Kf_value)
    H = TransferFunction(H_expr)
    # Create an instance of `FeedbackSystem` and return system data:
    system = FeedbackSystem(G_c=KG, H=H, name=name)
    sys_data = SystemData(
        feedback_system=system,
        damping_ratio=damping_ratio,
        forward_gain=K1_value,
        root_locus=root_locus,
        dominant_pole=zeta_crossings[0][0],
        natural_frequency=abs(zeta_crossings[0][0]),
        open_loop_gain=KGH_value,
        feedback_gain=Kf_value,
        forward_path=KG,
        feedback_path=H,
    )
    return sys_data


def design_minor_loop_feedback(
    damping_ratio_minor: float,
    damping_ratio_major: float,
    G_plant: TransferFunction,
    H_mnl: tuple[TransferFunction, sp.Symbol],
    name: str = '',
    fwd_gain_symbol: sp.Symbol = K
) -> tuple[SystemData, SystemData]:
    """Designs a minor-loop feedback compensation system consisting of a minor
    and a major loop. The minor loop contains the transfer function of the plant
    (process) in its forward path (without forward-path gain). The transfer
    function $H(s)$ of the compensator in the feedback path of the minor loop
    must be given when calling this function. The major loop has no transfer
    function in its feedback path (i.e. $H = 1$, unity-feedback) and a Sympy
    symbol to be used to represent its forward-path gain can be passed when
    calling this function.

    First, the required gain of the feedback compensator in the minor loop is
    determined and the minor loop is then reduced into a single transfer
    function, which is the forward-path transfer function of the major loop.
    Next, the required forward-path gain of the major loop can determined.

    Parameters
    ----------
    damping_ratio_minor:
        Required damping ratio of the minor loop.
    damping_ratio_major:
        Required damping ratio of the major loop.
    G_plant:
        Transfer function of the plant or process **without** forward-path gain
        (i.e. the same as that the gain of the transfer function is 1).
    H_mnl:
        Tuple with the transfer function of the feedback compensator in the
        minor loop, and the symbol that is used to represent the gain in this
        transfer function.
    name:
        Optional name to identify the system.
    fwd_gain_symbol:
        Sympy symbol that represents the forward-path gain of the major loop
        (default symbol is K).

    Returns
    -------
    sysdata_major:
        Instance of `SystemData` which holds the system data of the major loop.
    sysdata_minor:
        Instance of `SystemData` which holds the system data of the minor loop.
    """
    # Minor loop:
    G_mnl = G_plant
    minor_loop = FeedbackSystem(G_c=G_mnl, H=H_mnl[0])
    root_locus_mnl = RootLocus(minor_loop.open_loop)
    zeta_crossings_mnl = root_locus_mnl.find_damping_ratio_crossings(damping_ratio_minor)
    if zeta_crossings_mnl:
        K_f_value = zeta_crossings_mnl[0][1]
        T_mnl_expr = minor_loop.closed_loop.expr.subs(H_mnl[1], K_f_value)
        T_mnl = TransferFunction(T_mnl_expr)
        H_mnl_expr = H_mnl[0].expr.subs(H_mnl[1], K_f_value)
        H_mnl = TransferFunction(H_mnl_expr)
        minor_loop = FeedbackSystem(G_c=G_mnl, H=H_mnl, name=name + '_minor')
        sys_data_mnl = SystemData(
            feedback_system=minor_loop,
            damping_ratio=damping_ratio_minor,
            forward_gain=1,
            root_locus=root_locus_mnl,
            dominant_pole=zeta_crossings_mnl[0][0],
            natural_frequency=abs(zeta_crossings_mnl[0][0]),
            open_loop_gain=K_f_value,
            feedback_gain=K_f_value,
            forward_path=G_mnl,
            feedback_path=H_mnl
        )
    else:
        raise ValueError(
            "Unable to design the minor loop. Could not find a closed-loop "
            "pole on the specified damping ratio line."
        )

    # Major loop:
    G_mjl = fwd_gain_symbol * T_mnl
    major_loop = FeedbackSystem(G_mjl)
    root_locus_mjl = RootLocus(major_loop.open_loop)
    zeta_crossings_mjl = root_locus_mjl.find_damping_ratio_crossings(damping_ratio_major)
    if zeta_crossings_mjl:
        K_value = zeta_crossings_mjl[0][1]
        G_mjl_expr = major_loop.open_loop.expr.subs(fwd_gain_symbol, K_value)
        G_mjl = TransferFunction(G_mjl_expr)
        major_loop = FeedbackSystem(G_mjl, name=name)
        sysdata_mjl = SystemData(
            feedback_system=major_loop,
            damping_ratio=damping_ratio_major,
            forward_gain=K_value,
            root_locus=root_locus_mjl,
            dominant_pole=zeta_crossings_mjl[0][0],
            natural_frequency=abs(zeta_crossings_mjl[0][0]),
            forward_path=G_mjl
        )
    else:
        raise ValueError(
            "Unable to design the major loop. Could not find a closed-loop "
            "pole on the specified damping ratio line."
        )
    return sysdata_mjl, sys_data_mnl
