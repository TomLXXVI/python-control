"""Root-Locus Design of SISO LTI Negative Unity-Feedback Control Systems.

References
----------
Nise, N. S. (2020). Control Systems Engineering, EMEA Edition, 8th Edition.
"""
import warnings
from dataclasses import dataclass
from collections import namedtuple
import sympy as sp
import numpy as np
from ..core.symbols import K, s
from ..core.transfer_function import TransferFunction
from ..core.systems.feedback import FeedbackSystem, is_second_order_approx
from ..core.root_locus import RootLocus, TransferFunctionVector
from .general import create_second_order_system

from ..project_logging import ModuleLogger
logger = ModuleLogger.get_logger(__name__)


def _create_controller_transfer_function(
    gain_value: float,
    KG_c: TransferFunction,
    gain_symbol: sp.Symbol = K
) -> TransferFunction:
    """Replaces the symbol for gain `gain_symbol` by its value `gain_value` in
    the controller/compensator transfer function `KG_c`.
    """
    KG_c_expr = KG_c.expr.subs(gain_symbol, gain_value)
    KG_c = TransferFunction(KG_c_expr)
    return KG_c


def _create_unity_feedback_system(
    gain_value: float,
    KG_c: TransferFunction,
    G_plant: TransferFunction,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> FeedbackSystem:
    """Creates a unity-feedback system.

    Parameters
    ----------
    gain_value:
        Value of the overall forward-path gain.
    KG_c:
        Transfer function of the controller/compensator which incorporates the
        Sympy symbol for the overall forward-path gain of the unity-feedback
        system.
    G_plant:
        Transfer function of the plant or process.
    name:
        Optional name to identify the system, e.g. in the legend of a diagram.
    gain_symbol:
        Sympy symbol representing the overall forward-path gain used in the
        transfer function of the controller/compensator that will be replaced by
        the given gain value. By default the predefined Sympy symbol `K` is
        assumed (predefined in module `core.symbols.py`).
    """
    K_c = gain_value / G_plant.gain
    G_c = _create_controller_transfer_function(K_c, KG_c, gain_symbol)
    feedback_system = FeedbackSystem(G_c=G_c, G_p=G_plant, name=name)
    return feedback_system


ControllerGains = namedtuple('ControllerGains', ('K_p', 'K_i', 'K_d'))


@dataclass
class Characteristics:
    feedback_system: FeedbackSystem
    system_type: str | None = None
    steady_state_error: float | None = None
    static_error_constant: float | None = None
    natural_frequency: float | None = None
    damping_ratio: float | None = None
    rise_time: float | None = None
    peak_time: float | None = None
    settling_time: float | None = None
    percent_overshoot: float | None = None
    second_order_approximation: bool | None = None
    dc_gain: float | None = None
    steady_state_value: float | None = None
    
    def __post_init__(self):
        if (system_type := self.feedback_system.system_type) != 'None':
            self.system_type = system_type
            self.dc_gain = self.feedback_system.closed_loop.dc_gain
            self.steady_state_value = self.feedback_system.y_oo
            match system_type:
                case 'type_0':
                    sse = self.feedback_system.steady_state_error(1 / s)
                    self.steady_state_error = float(sse.e_oo)
                    self.static_error_constant = self.feedback_system.Kp
                case 'type_1':
                    sse = self.feedback_system.steady_state_error(1 / s ** 2)
                    self.steady_state_error = float(sse.e_oo)
                    self.static_error_constant = self.feedback_system.Kv
                case 'type_2':
                    sse = self.feedback_system.steady_state_error(1 / s ** 3)
                    self.steady_state_error = float(sse.e_oo)
                    self.static_error_constant = self.feedback_system.Ka
        if is_second_order_approx(self.feedback_system):
            sosys = create_second_order_system(self.feedback_system.dominant_pole_pair[0])
            self.natural_frequency = sosys.natural_frequency
            self.damping_ratio = sosys.damping_ratio
            self.second_order_approximation = True
        else:
            self.second_order_approximation = False
        if self.feedback_system.dominant_pole_pair is not None:
            self.rise_time = self.feedback_system.rise_time()
            self.peak_time = self.feedback_system.peak_time()
            self.settling_time = self.feedback_system.settling_time()
            self.percent_overshoot = self.feedback_system.percent_overshoot()

    def __str__(self) -> str:
        items = []
        if self.system_type is not None:
            items.extend([
                f"system type: {self.system_type}",
                f"steady-state error: {self.steady_state_error:.3g}",
                f"static error constant: {self.static_error_constant:.3g}",
                f"steady-state value: {self.steady_state_value:.3g}",
                f"dc-gain: {self.dc_gain:.3g}"
            ])
        items.append(f"second-order approximation: {self.second_order_approximation}")
        if self.second_order_approximation is True:
            items.extend([
                f"natural frequency: {self.natural_frequency:.3g}",
                f"damping ratio: {self.damping_ratio:.3g}",
            ])
        if self.rise_time is not None:
            items.extend([
                f"rise time: {self.rise_time:.3g}",
                f"peak time: {self.peak_time:.3g}",
                f"settling time: {self.settling_time:.3g}",
                f"percent overshoot: {self.percent_overshoot:.3g}"
            ])
        if items:
            result = '\n'.join(items)
            return result
        else:
            result = "characteristics of feedback system cannot be determined"
            return result


@dataclass
class FeedbackSystemDesign:
    """Dataclass holding information about the design of the unity-feedback
    system.

    Attributes
    ----------
    feedback_system:
        Unity-feedback system that was designed.
    forward_gain:
        Value of the overall forward-path gain (controller gain * plant gain).
        If the value could not be determined, it will be `None`.
    feedback_gain:
        Value of the feedback-path gain.
    controller:
        Transfer function of the designed controller/compensator.
    controller_gains:
        In case of a PD- or PID-controller, the specific gains of each control
        action.
    root_locus:
        Root locus of the open-loop transfer function.
    characteristics:
        Instance of dataclass `Characteristics`, holding transient response and
        steady-state error characteristics of the unity-feedback system.
    """
    feedback_system: FeedbackSystem | None = None
    forward_gain: float | None = None
    controller: TransferFunction | None = None
    controller_gains: ControllerGains | None = None
    feedback_gain: float | None = None
    root_locus: RootLocus | None = None

    @property
    def characteristics(self) -> Characteristics:
        """Returns transient response and steady-state error characteristics
        of the unity-feedback system.
        """
        return Characteristics(self.feedback_system)


def _get_dominant_poles(
    settling_time: float | None = None,
    peak_time: float | None = None,
    damping_ratio: float = 1.0
) -> tuple[complex, complex]:
    """Returns the desired location of the dominant pole pair in the complex
    plane based on transient response requirements.

    Parameters
    ----------
    settling_time:
        Required settling time.  If `None`, parameter `peak_time` must be set.
    peak_time:
        Required peak time. If `None`, parameter `settling_time` must be set.
    damping_ratio:
        Required damping ratio.
    """
    if settling_time is not None:
        # Required real part of the dominant pole(s):
        sigma = 4 / settling_time
        # Required imaginary part of the positive dominant pole:
        phi = np.pi - np.arccos(damping_ratio)
        omega = sigma * np.tan(np.pi - phi)
        dominant_pole = complex(-sigma, omega)
    elif peak_time is not None:
        omega = np.pi / peak_time
        phi = np.pi - np.arccos(damping_ratio)
        sigma = omega / np.tan(np.pi - phi)
        dominant_pole = complex(-sigma, omega)
    else:
        raise ValueError(
            "Either settling time or peak time must be specified."
        )
    return dominant_pole, dominant_pole.conjugate()


def _determine_forward_gain(
    G: TransferFunction, 
    damping_ratio: float,
    **kwargs
) -> float | None:
    """Determines the required forward gain for the unity-feedback system with
    forward-path transfer function G(s), such that the desired damping ratio is
    attained.
    
    How it works:
    The root locus of the unity-feedback system is determined. Points in the
    positive left-half of the complex s-plane with the same damping ratio lie 
    on a straight ray from the origin (on the positive imagary axis the damping 
    ratio is 0 and on the negative real axis the damping ratio is 1).
    Along the given damping ratio line, points are searched for where the root 
    locus crosses the damping ratio line. For this, the damping ratio line
    is first subdivided into a number of intervals (keyword argument `r_num`, 
    default value 100) between a minimum radius (keyword `r_min`, default value
    0.0) and a maximum radius (keyword `r_max`, default value 20.0). If there
    are intervals where the angle of the root locus passes through 180°, Scipy's
    root finding algorithm `root_scalar` is used in these intervals to find
    the crossing-point more precisely. Each crossing-point corresponds with a
    certain gain value (i.e. the multiplication factor for which the magnitude
    of the open-loop transfer function vector becomes 1). This open-loop or
    forward-path gain value is then required value for the crossing-point 
    (a complex number) to be a stable, closed-loop pole of the unity-feedback 
    system.
    If multiple crossing-points are found, the crossing-point closest to the
    imaginary axis is considered to be the "dominant pole" (its conjugate in the 
    negative left-half of the complex plane will also be closed-loop pole, and
    this is called the "dominant pole pair"). The dominant pole pair has the 
    longest effect on the transient response (long settling time).
    If no crossing-points are found, a `ValueError` exception is raised. If this
    happens, one can retry the function with a larger value for `r_max` or 
    `r_num` or a smaller value for `r_min` (however `r_min` cannot be smaller 
    than zero). If no keyword arguments `r_min`, `r_num`, and `r_max` are
    specified, the function will try 3 times to find crossing-points by 
    increasing the number of search intervals along the damping ratio line 
    between the default value of `r_min` (0) and `r_max` (20). The number of
    points is increased by a factor of 10 each time. 
        
    Parameters
    ----------
    G:
        Forward-path transfer function of the unity-feedback system (this is 
        also the open-loop transfer function of the unity-feedback system).
    damping_ratio:
        Desired damping ratio of the system.
    kwargs: optional
        - r_min (float): minimum radius. 
        - r_max (float): maximum radius. 
        - r_num (float): number of radii between minimum and maximum radius.

    Returns
    -------
    gain_value:
        Value of the required forward-path gain.

    Raises
    ------
    ValueError:
        If no closed-loop pole has been found on the damping ratio line.
    """
    # Create the root locus of the open-loop transfer function:
    root_locus = RootLocus(G)
    # Search along the required damping ratio line in the positive left-half of 
    # the complex plane where the root locus crosses the damping ratio line.
    r_min_ = kwargs.get('r_min')
    r_max_ = kwargs.get('r_max')
    r_num_ = kwargs.get('r_num')
    r_min = r_min_ if isinstance(r_min_, (float, int)) else 0.0
    r_max = r_max_ if isinstance(r_max_, (float, int)) else 20.0
    r_num = r_num_ if isinstance(r_num_, int) else 100
    if not kwargs:
        i_max = 3
        i = 0
        zeta_crossings = []
        while i < i_max:
            zeta_crossings = root_locus.find_damping_ratio_crossings(
                damping_ratio=damping_ratio,
                r_num=10**(i + 2),
                r_max=r_max
            )
            if zeta_crossings: break
            i += 1
    else:
        zeta_crossings = root_locus.find_damping_ratio_crossings(
            damping_ratio=damping_ratio,
            r_min=r_min,
            r_max=r_max,
            r_num=r_num
        )
    # If more than one crossing point is found, it's assumed that the point 
    # closest to the imaginary axis is the dominant complex closed-loop pole of 
    # the feedback system in the positive left-half of the complex plane 
    if zeta_crossings:
        real_parts = [zc[0].real for zc in zeta_crossings]
        i_dp = real_parts.index(max(real_parts))
        gain_value = zeta_crossings[i_dp][1]
        return gain_value
    raise ValueError(
        "Unable to create the feedback system. Could not find a closed-loop "
        "pole on the damping ratio line.",
    )


def design_unity_feedback(
    KG_c: TransferFunction | None,
    G_plant: TransferFunction,
    damping_ratio: float,
    name: str = '',
    gain_symbol: sp.Symbol = K,
    **kwargs
) -> tuple[FeedbackSystem, float] | None:
    """Designs a unity-feedback system with cascade compensation (i.e. the
    transfer function of the compensator is added in series to the transfer
    function of the plant or process in the forward path of the unity-feedback
    system). Determines the overall forward-path gain such that the required
    damping ratio of the feedback system is attained.

    Parameters
    ----------
    KG_c:
        Transfer function of the compensator in which the overall forward-path
        gain of the feedback system is represented by a Sympy symbol.
        If `KG_c` is `None`, only proportional control will be considered.
    G_plant:
        Transfer function of the plant or process.
    damping_ratio:
        Required damping ratio of the feedback system.
    name:
        An optional name to identify the feedback system.
    gain_symbol:
        The Sympy symbol that is used to represent the forward-path gain in the
        expression of the compensator's transfer function `KG_c`. The default
        symbol is the predefined symbol `K` from the `python_control` package.
        If another symbol is used in the expression of `KG_c`, that symbol must
        be specified here.
    kwargs: optional
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    fbsys:
        Feedback system.
    gain_value:
        Either the value of the overall forward-path gain (which is also the 
        open-loop gain in a unity-feedback system), or `None` if no closed-loop 
        pole has been found on the damping ratio line.
    """
    if KG_c is None: KG_c = TransferFunction(gain_symbol)
    G_plant_ = G_plant / G_plant.gain
    G = KG_c * G_plant_
    try:
        gain_value = _determine_forward_gain(G, damping_ratio, **kwargs)
    except ValueError:
        fbsys = FeedbackSystem(KG_c, G_plant, name=name)
        return fbsys, None
    else:
        fbsys = _create_unity_feedback_system(
            gain_value=gain_value, 
            KG_c=KG_c, 
            G_plant=G_plant, 
            name=name, 
            gain_symbol=gain_symbol
        )
        return fbsys, gain_value


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
    compensator is not determined by this function. To determine the value of
    the gain use the function `design_unity_feedback` next.

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
    function. To determine the gain use the function `design_unity_feedback` 
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
        Transfer function of the plant or process.
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
    dominant_pole, _ = _get_dominant_poles(settling_time, peak_time, damping_ratio)
    sigma_dp, omega_dp = dominant_pole.real, dominant_pole.imag
    
    logger.debug(f"dominant pole: {dominant_pole}")
    
    G_vect = TransferFunctionVector(G_plant)
    G_vect.point = dominant_pole
    theta_G = G_vect.angle
    while theta_G < 0.0: theta_G += 360.0
    if theta_G > 360.0: theta_G %= 360.0
    
    logger.debug(
        f"open-loop transfer function angle before compensation: "
        f"{theta_G}"
    )
    
    theta_G = np.radians(theta_G)
    zero_c = sigma_dp - omega_dp / np.tan(np.pi - theta_G)
    
    logger.debug(f"compensator zero: {zero_c}")
    
    KG_c = TransferFunction(gain_symbol * (s - zero_c))

    G_comp = KG_c * G_plant
    G_comp_vect = TransferFunctionVector(G_comp)
    G_comp_vect.point = dominant_pole
    theta_G_comp = G_comp_vect.angle
    while theta_G_comp < 0.0: theta_G_comp += 360.0
    if theta_G_comp > 360.0: theta_G_comp %= 360.0
    
    logger.debug(
        f"open-loop transfer function angle after compensation: "
        f"{theta_G_comp}"
    )
    
    return KG_c


def design_lead_compensator(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    zero_c: float,
    gain_symbol: sp.Symbol = K
) -> TransferFunction:
    """Designs the transfer function for a lead compensator. Determines the 
    location of the compensator pole in the s-plane such that the required 
    transient response requirements are met, while the location of the 
    compensator zero is already given. Note that the gain of the lead 
    compensator is not determined by this function. To determine the gain use 
    the function `design_unity_feedback` next.
    
    How it works:
    The transient response requirements determine the location of a dominant
    complex pole pair in the s-plane. At these two points, the angle of the 
    resulting open-loop transfer function must be 180°, so that the root locus 
    of the resulting open-loop transfer function would pass through these 
    dominant poles. 
    This is accomplished by placing an extra zero and pole in the s-plane. The 
    zero of the compensator on the negative real axis of the s-plane is already 
    selected. The location of the compensator pole on the real axis is then 
    determined such that the total angle of the resulting open-loop transfer
    function is 180°.
    
    In the case of a true lead compensator, the compensator pole lies to the 
    left of the compensator zero on the real axis of the s-plane (is more 
    negative than the compensator zero). However, when using this method, it may
    happen that the compensator pole will be to the right of the compensator 
    zero. If that's the case, the compensator is actually a lag compensator.   
    
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
        Transfer function of the plant or process.
    zero_c:
        Selected compensator zero. In general the zero is placed on the
        negative real axis at a distance from the origin in the neighborhood of
        the natural frequency of the uncompensated system's dominant pole (1/4
        to 1 times the value of the natural frequency).
    gain_symbol:
        Sympy symbol to be used to represent the gain of the compensator.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.

    Returns
    -------
    KG_c:
        Transfer function of the lead compensator with symbolic gain (instance
        of class `TransferFunction`).
    """
    dominant_pole, _ = _get_dominant_poles(settling_time, peak_time, damping_ratio)
    sigma_dp, omega_dp = dominant_pole.real, dominant_pole.imag
    
    logger.debug(f"dominant pole: {dominant_pole}")
    
    G_vect = TransferFunctionVector(G_plant)
    G_vect.point = dominant_pole
    theta_G = G_vect.angle
    while theta_G < 0.0: theta_G += 360.0
    if theta_G > 360.0: theta_G %= 360.0
    
    logger.debug(
        f"open-loop transfer function angle before compensation: {theta_G}"
    )

    theta_G = np.radians(theta_G)
    alpha = np.arctan(omega_dp / (sigma_dp - zero_c))
    pole_c = sigma_dp + omega_dp / np.tan(np.pi - theta_G - alpha)
    
    if pole_c < zero_c:
        logger.debug(
            f"LEAD compensator zero: {zero_c}, pole: {pole_c}"
        )
    else:
        logger.debug(
            f"LAG compensator zero: {zero_c}, pole: {pole_c}"
        )
    
    KG_c = TransferFunction(gain_symbol * (s - zero_c) / (s - pole_c))
    
    G_comp = KG_c * G_plant
    G_comp_vect = TransferFunctionVector(G_comp)
    G_comp_vect.point = dominant_pole
    theta_G_comp = G_comp_vect.angle
    while theta_G_comp < 0.0: theta_G_comp += 360.0
    if theta_G_comp > 360.0: theta_G_comp %= 360.0

    logger.debug(
        f"open-loop transfer function angle after compensation: "
        f"{theta_G_comp}"
    )
    
    if pole_c > 0.0:
        warnings.warn(
            "The feedback system is unstable. Transient response requirements "
            "cannot be fulfilled."
        )
    return KG_c


def design_P_feedback(
    G_plant: TransferFunction,
    damping_ratio: float,
    name: str = '',
    **kwargs
) -> FeedbackSystemDesign:
    """Given the damping ratio requirement, designs a unity-feedback system with
    P-controller, added in series to the plant (cascade compensation). The 
    calculated overall forward-path gain (product of controller gain and any 
    plant gain) will be inserted into the transfer function of the P-controller.

    Parameters
    ----------
    G_plant:
        Transfer function of the plant or process.
    damping_ratio:
        Required damping ratio of the P-controlled system.
    name:
        Optional name to identify the system.
    kwargs: optional
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    Instance of class `FeedbackSystemDesign` (see its docstring for information).
    """
    fbsys, K_fwd = design_unity_feedback(
        KG_c=None, 
        G_plant=G_plant, 
        damping_ratio=damping_ratio, 
        name=name, 
        **kwargs
    )
    design = FeedbackSystemDesign(
        feedback_system=fbsys,
        forward_gain=K_fwd,
        controller=fbsys.G_c,
        controller_gains=ControllerGains(fbsys.G_c.gain, None, None)
    )
    return design


def design_PD_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    name: str = '',
    gain_symbol: sp.Symbol = K,
    **kwargs
) -> FeedbackSystemDesign:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with PD-controller,
    added in series to the plant (cascade compensation). The calculated overall 
    forward-path gain (product of controller gain and any plant gain) will be 
    inserted into the transfer function of the PD-controller.

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
        Transfer function of the plant or process.
    name:
        Optional name to identify the system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the PD-controller.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.
    kwargs: optional
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    Instance of class `FeedbackSystemDesign` (see its docstring for information).
    """
    KG_c = design_PD_controller(
        settling_time,
        peak_time,
        damping_ratio,
        G_plant,
        gain_symbol
    )
    fbsys, K_fwd = design_unity_feedback(
        KG_c=KG_c,
        G_plant=G_plant,
        damping_ratio=damping_ratio,
        name=name,
        gain_symbol=gain_symbol,
        **kwargs
    )
    coeffs = fbsys.G_c.num_coeffs
    K_d = coeffs[0]
    try:
        K_p = coeffs[1]
    except IndexError:
        K_p = None
    design = FeedbackSystemDesign(
        feedback_system=fbsys,
        forward_gain=K_fwd,
        controller=fbsys.G_c,
        controller_gains=ControllerGains(K_p, None, K_d),
        root_locus=RootLocus(fbsys.open_loop)
    )
    return design


def design_PID_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    integrator_zero: float,
    name: str = '',
    gain_symbol: sp.Symbol = K,
    **kwargs
) -> FeedbackSystemDesign:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with PID-controller,
    added in series to the transfer function of the plant (cascade
    compensation). The calculated overall required forward-path gain (product of
    controller gain and any plant gain) will be inserted into the transfer 
    function of the PID-controller.

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
        Transfer function of the plant or process.
    integrator_zero:
        Zero of the PI-controller (should be placed close to the origin of the
        complex plane).
    name:
        Optional name to identify the system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the PID-controller.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.
    kwargs: optional
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    Instance of class `FeedbackSystemDesign` (see its docstring for information).
    """
    # Create PD-controller:
    KG_pd = design_PD_controller(
        settling_time,
        peak_time,
        damping_ratio,
        G_plant,
        gain_symbol
    )
    # Add I-control action:
    G_i = TransferFunction((s - integrator_zero) / s)
    KG_c = KG_pd * G_i
    # Design unity-feedback system:
    fbsys, K_fwd = design_unity_feedback(
        KG_c=KG_c,
        G_plant=G_plant,
        damping_ratio=damping_ratio,
        name=name,
        gain_symbol=gain_symbol,
        **kwargs
    )
    coeffs = fbsys.G_c.num_coeffs
    K_d = coeffs[0]
    K_p = coeffs[1]
    K_i = coeffs[2]
    design = FeedbackSystemDesign(
        feedback_system=fbsys,
        forward_gain=K_fwd,
        controller=fbsys.G_c,
        controller_gains=ControllerGains(K_p, K_i, K_d),
        root_locus=RootLocus(fbsys.open_loop)
    )
    return design


def design_lag_feedback(
    feedback_system: FeedbackSystem,
    e_oo_reduction_factor: float,
    compensator_pole: float,
    damping_ratio: float,
    name: str = '',
    gain_symbol: sp.Symbol = K,
    **kwargs
) -> FeedbackSystemDesign:
    """Given the unity-feedback system, adds a lag compensator in series to the
    forward-path transfer function (cascade compensation) to reduce the
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
    name:
        Optional name to identify the system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lag compensator.
        The default symbol is the predefined symbol `K` from the
        `python_control` package.
    kwargs: optional
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    Instance of class `FeedbackSystemDesign` (see its docstring for information).
    """
    KG_c = design_lag_compensator(
        feedback_system,
        e_oo_reduction_factor,
        compensator_pole,
        gain_symbol
    )
    fbsys, K_fwd = design_unity_feedback(
        KG_c=KG_c,
        G_plant=feedback_system.G_p,
        damping_ratio=damping_ratio,
        name=name,
        gain_symbol=gain_symbol,
        **kwargs
    )
    design = FeedbackSystemDesign(
        feedback_system=fbsys,
        forward_gain=K_fwd,
        controller=fbsys.G_c,
        root_locus=RootLocus(fbsys.open_loop)
    )
    return design


def design_lead_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    compensator_zero: float,
    name: str = '',
    gain_symbol: sp.Symbol = K,
    **kwargs
) -> FeedbackSystemDesign:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with lead compensator,
    added in series to the transfer function of the plant (cascade compensation).
    The calculated overall required forward-path gain (product of controller 
    gain and any plant gain) will be inserted into the transfer function of the 
    lead compensator.

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
        Transfer function of the plant or process.
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
    kwargs:
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    Instance of class `FeedbackSystemDesign` (see its docstring for information).
    """
    KG_c = design_lead_compensator(
        settling_time=settling_time,
        peak_time=peak_time,
        damping_ratio=damping_ratio,
        G_plant=G_plant,
        zero_c=compensator_zero,
        gain_symbol=gain_symbol
    )
    fbsys, K_fwd = design_unity_feedback(
        KG_c=KG_c,
        G_plant=G_plant,
        damping_ratio=damping_ratio,
        name=name,
        gain_symbol=gain_symbol,
        **kwargs
    )
    design = FeedbackSystemDesign(
        feedback_system=fbsys,
        forward_gain=K_fwd,
        controller=fbsys.G_c,
        root_locus=RootLocus(fbsys.open_loop)
    )
    return design


def design_lag_lead_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    e_oo_reduction: float,
    G_plant: TransferFunction,
    lead_compensator_zero: float,
    lag_compensator_pole: float,
    name: str = '',
    gain_symbol: sp.Symbol = K,
    **kwargs
) -> FeedbackSystemDesign:
    """Given the transient response requirements (settling time or peak time,
    and damping ratio), designs a unity-feedback system with lag-lead
    compensator, added in series to the transfer function of the plant (cascade
    compensation). The calculated overall required forward-path gain (product of
    controller gain and any plant gain) will be inserted into the transfer 
    function of the PD-controller.

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
        Transfer function of the plant or process.
    lead_compensator_zero:
        Selected zero of the lead compensator transfer function. In general the
        zero is placed on the negative real axis at a distance from the origin
        in the neighborhood of the natural frequency of the uncompensated
        system's dominant pole (1/4 to 1 times the value of the natural
        frequency).
    lag_compensator_pole:
        Selected pole of the lag compensator transfer function (should be near
        zero).
    name:
        Optional name to identify the lag-lead compensated system.
    gain_symbol:
        Sympy symbol to be used to represent the gain of the lag-lead
        compensator. The default symbol is the predefined symbol `K` from the
        `python_control` package.
    kwargs: optional
        See docstring of function `_determine_forward_gain`.

    Returns
    -------
    Instance of class `FeedbackSystemDesign` (see its docstring for information).
    """
    KG_lead = design_lead_compensator(
        settling_time=settling_time,
        peak_time=peak_time,
        damping_ratio=damping_ratio,
        G_plant=G_plant,
        zero_c=lead_compensator_zero,
        gain_symbol=gain_symbol
    )
    fbsys, _ = design_unity_feedback(
        KG_c=KG_lead,
        G_plant=G_plant,
        damping_ratio=damping_ratio,
        name=name,
        gain_symbol=gain_symbol,
        **kwargs
    )
    KG_lag = design_lag_compensator(
        feedback_system=fbsys,
        e_oo_reduction_factor=e_oo_reduction,
        compensator_pole=lag_compensator_pole,
        gain_symbol=gain_symbol
    )
    KG_lag = TransferFunction(KG_lag.expr.subs(gain_symbol, 1))
    KG_c = KG_lead * KG_lag
    fbsys, K_fwd = design_unity_feedback(
        KG_c=KG_c,
        G_plant=G_plant,
        damping_ratio=damping_ratio,
        name=name,
        gain_symbol=gain_symbol,
        **kwargs
    )
    design = FeedbackSystemDesign(
        feedback_system=fbsys,
        forward_gain=K_fwd,
        controller=fbsys.G_c
    )
    return design


def design_rate_feedback(
    settling_time: float | None,
    peak_time: float | None,
    damping_ratio: float,
    G_plant: TransferFunction,
    name: str = '',
    fdb_gain_symbol: sp.Symbol = sp.Symbol('K_f', real=True, positive=True),
    fwd_gain_symbol: sp.Symbol = sp.Symbol('K_1', real=True, positive=True)
) -> FeedbackSystemDesign:
    """Designs a rate-feedback compensation system consisting of a minor loop
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
        Transfer function of the plant or process.
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
    Instance of class `FeedbackSystemDesign`.
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
    Kf_value = float(1 / abs(zero_H))  # required PD-controller gain
    # Determine the open-loop gain and the required forward-path gain of the
    # reduced system:
    KGH = fwd_gain_symbol * G_plant * H  # open-loop transfer function
    root_locus = RootLocus(KGH)  # root locus of closed-loop feedback system
    zeta_crossings = root_locus.find_damping_ratio_crossings(damping_ratio)
    KGH_value = zeta_crossings[0][1]  # required open-loop gain
    K1_value = KGH_value / Kf_value   # required forward-path gain
    # Define the controller transfer function of the reduced system with
    # the calculated forward-path gain:
    G_c = TransferFunction(K1_value)
    # Define the feedback-path transfer function of the reduced system with
    # the calculated PD-controller gain:
    H_expr = H.expr.subs(fdb_gain_symbol, Kf_value)
    H = TransferFunction(H_expr)
    # Create an instance of `FeedbackSystem` and return system data:
    system = FeedbackSystem(G_c=G_c, G_p=G_plant, H=H, name=name)
    design = FeedbackSystemDesign(
        feedback_system=system,
        forward_gain=K1_value,
        feedback_gain=Kf_value,
        controller=H
    )
    return design


def design_minor_loop_feedback(
    damping_ratio_minor: float,
    damping_ratio_major: float,
    G_plant: TransferFunction,
    H_mnl: tuple[TransferFunction, sp.Symbol],
    name: str = '',
    fwd_gain_symbol: sp.Symbol = K
) -> tuple[FeedbackSystemDesign, FeedbackSystemDesign]:
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
        Transfer function of the plant or process.
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
    design_major:
        Instance of `FeedbackSystemDesign` which holds the system data of the
        major loop.
    design_minor:
        Instance of `FeedbackSystemDesign` which holds the system data of the
        minor loop.
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
        minor_loop = FeedbackSystem(G_p=G_mnl, H=H_mnl, name=name + '_minor')
        design_mnl = FeedbackSystemDesign(
            feedback_system=minor_loop,
            forward_gain=1,
            feedback_gain=K_f_value,
            controller=H_mnl
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
        major_loop = FeedbackSystem(G_c=TransferFunction(K_value), G_p=T_mnl, name=name)
        design_mjl = FeedbackSystemDesign(
            feedback_system=major_loop,
            forward_gain=K_value,
            controller=major_loop.G_c
        )
    else:
        raise ValueError(
            "Unable to design the major loop. Could not find a closed-loop "
            "pole on the specified damping ratio line."
        )
    return design_mjl, design_mnl
