"""Design SISO LTI feedback control systems.

Functions for general use.
"""
import sympy as sp
import numpy as np
from ..matplotlibwrapper import LineChart
from ..core.symbols import K
from ..core.transfer_function import TransferFunction
from ..core.systems.second_order import SecondOrderSystem
from ..core.systems.feedback import FeedbackSystem


def create_transfer_function(
    K_value: float,
    G: TransferFunction,
    gain_symbol: sp.Symbol = K
) -> TransferFunction:
    KG_expr = G.expr.subs(gain_symbol, K_value)
    KG = TransferFunction(KG_expr)
    return KG


def create_feedback_system(
    K_value: float,
    G: TransferFunction,
    name: str = '',
    gain_symbol: sp.Symbol = K
) -> FeedbackSystem:
    """Given the open-loop transfer function `G` containing a symbolic gain
    `K`, and the real value for this gain `K`, substitutes this value in the
    expression of `G` and then creates and returns a `FeedbackSystem` instance.
    """
    KG = create_transfer_function(K_value, G, gain_symbol)
    feedback_system = FeedbackSystem(KG, name=name)
    return feedback_system


def create_second_order_system(
    dominant_pole: complex,
    dc_gain: float | None = None,
    name: str = ''
) -> SecondOrderSystem:
    """Given one of the two dominant complex poles, creates and returns a
    `SecondOrderSystem` instance.
    """
    omega_nat = abs(dominant_pole)
    phi = np.arctan(abs(dominant_pole.imag) / abs(dominant_pole.real))
    damping_ratio = np.cos(phi)
    a = 2 * damping_ratio * omega_nat
    b = omega_nat ** 2
    K = dc_gain if dc_gain is None else dc_gain * b
    second_order_system = SecondOrderSystem(a, b, K, name=name)
    return second_order_system


def get_dominant_poles(
    settling_time: float | None = None,
    peak_time: float | None = None,
    damping_ratio: float = 1.0
) -> tuple[complex, complex]:
    """Returns the required location of the dominant poles in the complex plane
    based on transient response requirements.

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


def _plot_responses(
    type_of_response: str,
    *systems: FeedbackSystem | SecondOrderSystem,
    **kwargs
) -> None:
    """Plots the response of the given feedback systems in a single chart with
    the purpose to compare these responses.
    """
    feedback_systems = [
        sys_
        for sys_ in systems
        if isinstance(sys_, FeedbackSystem)
    ]
    second_order_systems = [
        sys_
        for sys_ in systems
        if isinstance(sys_, SecondOrderSystem)
    ]
    if type_of_response == 'step':
        responses_fbs = [
            sys_.closed_loop.unit_step_response(**kwargs)
            for sys_ in feedback_systems
        ]
        responses_sos = [
            sys_.unit_step_response(**kwargs)
            for sys_ in second_order_systems
        ]
    elif type_of_response == 'ramp':
        responses_fbs = [
            sys_.closed_loop.ramp_response(**kwargs)
            for sys_ in feedback_systems
        ]
        responses_sos = [
            sys_.ramp_response(**kwargs)
            for sys_ in second_order_systems
        ]
    elif type_of_response == 'parabola':
        responses_fbs = [
            sys_.closed_loop.parabola_response(**kwargs)
            for sys_ in feedback_systems
        ]
        responses_sos = [
            sys_.parabola_response(**kwargs)
            for sys_ in second_order_systems
        ]
    else:
        raise ValueError(
            "Value of parameter `type_of_response` not recognized. "
            "Supported types are: 'step', 'ramp', or 'parabola'"
        )
    c = LineChart()
    if responses_sos:
        for i, response in enumerate(responses_sos):
            if label := second_order_systems[i].name:
                label = label
            else:
                label = f"second-order system {i}"
            c.add_xy_data(
                label=label,
                x1_values=response[0],
                y1_values=response[1]
            )
    if responses_fbs:
        for i, response in enumerate(responses_fbs):
            if label := feedback_systems[i].name:
                label = label
            else:
                label = f"feedback system {i}"
            c.add_xy_data(
                label=label,
                x1_values=response[0],
                y1_values=response[1]
            )
    c.add_legend()
    c.x1.add_title('time, s')
    c.y1.add_title('response')
    c.show()


def plot_step_responses(
    *systems: FeedbackSystem | SecondOrderSystem,
    **kwargs
) -> None:
    """Plots the unit step response of the systems.

    kwargs
    ------
    See docstring of `TransferFunction.unit_step_response()` in module
    core.transfer_function.py
    """
    _plot_responses('step', *systems, **kwargs)


def plot_ramp_responses(
    *systems: FeedbackSystem | SecondOrderSystem,
    **kwargs
) -> None:
    """Plots the ramp response of the systems.

    kwargs
    ------
    See docstring of `TransferFunction.ramp_response()` in module
    core.transfer_function.py
    """
    _plot_responses('ramp', *systems, **kwargs)


def plot_parabola_responses(
    *systems: FeedbackSystem | SecondOrderSystem,
    **kwargs
) -> None:
    """Plots the parabola response of the systems.

    kwargs
    ------
    See docstring of `TransferFunction.parabola_response()` in module
    core.transfer_function.py
    """
    _plot_responses('parabola', *systems, **kwargs)
