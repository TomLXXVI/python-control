"""Frequency Response Design of SISO LTI Negative Unity-Feedback Control Systems.

References
----------
Nise, N. S. (2020). Control Systems Engineering, EMEA Edition, 8th Edition.
"""
import control as ct
import numpy as np
from scipy.optimize import root_scalar
from ..core.frequency_response import FrequencyResponse
from ..core.transfer_function import TransferFunction
from ..core.symbols import s


def get_open_loop_gain(
    K_stat_err: float,
    G_jw: FrequencyResponse
) -> float:
    """Given the required static error constant, returns the required open-loop
    gain K of the system.

    Parameters
    ----------
    K_stat_err:
        Static error constant (either position constant, velocity constant, or
        acceleration constant depending on the type of system).
    G_jw:
        Frequency response of the open-loop transfer function.

    Returns
    -------
    Required open-loop gain.
    """
    zeros = G_jw.transfer_function.zeros
    poles = [pole for pole in G_jw.transfer_function.poles if pole != 0.0]
    prod_zeros = np.prod(zeros)
    prod_poles = np.prod(poles)
    K = (K_stat_err / (prod_zeros / prod_poles)).real
    return K


def find_frequency_for_phase_angle(
    phi_req: float,
    G_jw: FrequencyResponse,
    omega_limits: tuple[float, float] = (1.e-3, 1.e3),
    omega_num: int = 1000
) -> float:
    """Finds the angular frequency at which the open-loop frequency response has
    the required phase angle.

    Parameters
    ----------
    phi_req:
        Required phase angle in degrees.
    G_jw:
        Frequency response of the open-loop transfer function.
    omega_limits:
        Lower and upper limit of the frequency range where the root-finding
        algorithm searches for the frequency at which the required phase angle
        exists. If no solution can be found within the specified frequency
        range, a `ValueError` is raised. In that case, either the lower limit of
        the frequency range should be decreased, or the upper limit should be
        increased.
    omega_num:
        Number of calculation points within the specified range of frequencies.
    """
    omega_range = np.linspace(omega_limits[0], omega_limits[1], omega_num)
    G_jw_values = G_jw(omega_range)
    G_jw_phi = ct.unwrap(np.angle(G_jw_values)) * 180.0 / np.pi
    i = np.abs(G_jw_phi - phi_req).argmin()
    k = i - 1 if i > 0 else 0
    l = i + 1 if i < omega_range.size - 1 else i
    # noinspection PyTypeChecker
    bracket = sorted([omega_range[k], omega_range[l]])

    def _objective(omega: float) -> float:
        *_, phi = G_jw.evaluate(omega)
        err = phi_req - phi
        return err

    try:
        sol = root_scalar(
            _objective,
            bracket=bracket
        )
    except ValueError:
        raise ValueError(
            f"No frequency found where the phase angle is equal to {phi_req}° "
            f"between {omega_limits[0]:.3e} and {omega_limits[1]:.3e} rad/s."
        )
    else:
        omega = sol.root
        return omega


def find_frequency_for_magnitude(
    M_dB_req: float,
    G_jw: FrequencyResponse,
    K: float,
    omega_limits: tuple[float, float] = (1.e-3, 1.e3),
    omega_num: int = 1000
) -> float:
    """Finds the angular frequency at which the open-loop frequency response has
    the required magnitude in decibels.

    Parameters
    ----------
    M_dB_req:
        Required magnitude in decibels.
    G_jw:
        Frequency response of the open-loop transfer function (without
        open-loop gain).
    K:
        Open-loop gain that goes with the open-loop transfer function.
    omega_limits:
        Lower and upper limit of the frequency range where the root-finding
        algorithm searches for the frequency at which the required phase angle
        exists. If no solution can be found within the specified frequency
        range, a `ValueError` is raised. In that case, either the lower limit of
        the frequency range should be decreased, or the upper limit should be
        increased.
    omega_num:
        Number of calculation points within the specified range of frequencies.

    Returns
    -------
    omega:
        The frequency that goes with the required magnitude of the open-loop
        frequency response.
    """
    omega_range = np.linspace(omega_limits[0], omega_limits[1], omega_num)
    G_jw_values = G_jw(omega_range, K)
    G_jw_mag_dB = 20 * np.log10(np.abs(G_jw_values))
    i = np.abs(G_jw_mag_dB - M_dB_req).argmin()
    k = i - 1 if i > 0 else 0
    l = i + 1 if i < omega_range.size - 1 else i
    # noinspection PyTypeChecker
    bracket = sorted([omega_range[k], omega_range[l]])

    def _objective(omega: float) -> float:
        _, M_dB, _ = G_jw.evaluate(omega, K)
        err = M_dB_req - M_dB
        return err

    try:
        sol = root_scalar(
            _objective,
            bracket=bracket
        )
    except ValueError:
        raise ValueError(
            f"No frequency found where the magnitude is equal to {M_dB_req} dB "
            f"between {omega_limits[0]:.3e} and {omega_limits[1]:.3e} rad/s."
        )
    else:
        omega = sol.root
        return omega


def gain_adjustment_for_magnitude(
    M_dB_req: float,
    omega: float,
    G_jw: FrequencyResponse,
    K_range: tuple[float, float]
) -> float:
    """Finds the open-loop gain K so that at the given frequency the magnitude
    of the open-loop frequency response is equal to the required magnitude in
    decibels.

    Parameters
    ----------
    M_dB_req:
        Required magnitude in decibels.
    omega:
        Angular frequency (rad/s) where the required magnitude is asked.
    G_jw:
        Frequency response of the open-loop transfer function of the feedback
        system.
    K_range:
        Lower and upper limit of the range between which the open-loop gain K is
        to be searched for by the root finding algorithm.

    Returns
    -------
    K:
        Open-loop gain needed to attain the required magnitude at the frequency
        asked.
    """
    def _objective(K: float):
        _, M_dB, _ = G_jw.evaluate(omega, K)
        err = M_dB_req - M_dB
        return err

    try:
        sol = root_scalar(_objective, bracket=(K_range[0], K_range[1]))
    except ValueError:
        raise ValueError(
            f"No solution exist within K-range = [{K_range[0]}, {K_range[1]}]"
        )
    else:
        K = sol.root
        return K


def gain_adjustment_for_phase_margin(
    phi_m_req: float,
    G_jw: FrequencyResponse,
    omega_limits: tuple[float, float] = (1.e-3, 1.e9)
) -> tuple[float, float]:
    """Finds the open-loop gain K for which the feedback system has the required
    phase margin.

    Parameters
    ----------
    phi_m_req:
        Required phase margin.
    G_jw:
        Frequency response of the open-loop transfer function.
    omega_limits:
        Lower and upper limit of the frequency range where the root-finding
        algorithm searches for the frequency at which the magnitude of
        the open-loop frequency response is equal to 1 (0 dB). If no solution
        can be found within the specified frequency range, a `ValueError` is
        raised. In that case, either the lower limit of the frequency range
        should be decreased, or the upper limit should be increased.

    Returns
    -------
    K:
        Open-loop gain that corresponds with the required phase margin.
    omega_1:
        The angular frequency (rad/s) where the magnitude of the open-loop
        frequency response is equal to 1 (i.e. the frequency at which the
        phase margin is determined).
    """
    # Rough root-finding to determine a bracket for precise root-finding.
    def _get_phase_margin(K):
        try:
            phi_m, _ = G_jw.phase_margin(K, omega_limits)
        except ValueError:
            # If a phase margin cannot be determined, return nan instead of
            # raising the ValueError
            return np.nan
        else:
            return phi_m

    # Get the range of Ks for which the system is stable.
    K_range = G_jw.stability_gain_range
    if K_range is None:
        raise ValueError(
            "The stability range of the system could not be determined."
        )
    elif K_range[1] != float('inf'):
        K_values = np.linspace(K_range[1] / 1e9, K_range[1], 100)
    else:
        K_values = np.linspace(K_range[0], K_range[0] * 1e9, 100)
    # Determine the phase margins that go with the K-values in the stability
    # range.
    phi_m_values = np.array(list(map(_get_phase_margin, K_values)))
    # If certain K-values in the array returned a nan for the phase margin,
    # remove these K-values from the K-values array and remove the nans from
    # the phi_m_values array.
    nan_indexes = np.argwhere(np.isnan(phi_m_values))
    K_values = np.delete(K_values, nan_indexes)
    phi_m_values = np.delete(phi_m_values, nan_indexes)
    # Determine the index of the phase margin value in the phi_m_values array
    # which is closest to our required phase margin value.
    i = np.abs(phi_m_values - phi_m_req).argmin()
    # From the K-values array take the K-value to the left and the K-value to
    # the right from the K-value whose index is i (i.e. the K-value that gives
    # the phase margin which is closest to our target phase margin).
    i_max = phi_m_values.size - 1
    k = i - 1 if i >= 1 else 0
    l = i + 1 if i < i_max else i_max
    K_bracket = (K_values[k], K_values[l])

    # Precise root-finding: within the K-bracket search for the K that gives
    # a phase margin equal to our target phase margin.
    def _objective(K: float):
        phi_m, _ = G_jw.phase_margin(K, omega_limits)
        err = phi_m_req - phi_m
        return err

    try:
        sol = root_scalar(_objective, bracket=K_bracket)
    except ValueError:
        raise ValueError(
            f"No solution exist within K-range = "
            f"[{K_range[0]:.3e}, {K_range[1]:.3e}]"
        )
    else:
        K = sol.root
        _, omega_1 = G_jw.phase_margin(K, omega_limits)
        return K, omega_1


def create_lag_compensator(
    phi_m_req: float,
    K_req: float,
    G_jw: FrequencyResponse,
    phi_m_compensation: float = 10.0,
    omega_limits: tuple[float, float] = (1.e-3, 1.e3),
    omega_num: int = 1000
) -> TransferFunction:
    """Creates the transfer function for a lag compensator. The general form of
    this transfer function looks like:
                                   1            s + 1/T
                        G_c(s) = ------ * ------------------
                                 alpha      s + 1/(alpha*T)
    where alpha > 1. 1/T is the upper break frequency of the lag compensator and
    1/(alpha * T) is the lower break frequency. The gain of the lag compensator
    is 1/alpha. When the frequency is zero (steady-state), the dc gain of the
    lag compensator is therefore equal to one.

    Parameters
    ----------
    phi_m_req:
        Required phase margin (depends on transient response requirement).
    K_req:
        Required open-loop gain (depends on steady-state error requirement).
    G_jw:
        Open-loop frequency response of the feedback system.
    phi_m_compensation:
        Phase margin correction term to compensate for the phase angle
        contribution of the lag compensator. An angle between 5° and 12° is
        suggested. The default value is set at 10°.
    omega_limits:
        Lower and upper limit of the frequency range where the root-finding
        algorithm searches for the frequency at which the required phase margin
        is located. If no solution can be found within the specified frequency
        range, a `ValueError` is raised. In that case, either the lower limit of
        the frequency range should be decreased, or the upper limit should be
        increased.
    omega_num:
        Number of calculation points within the specified range of frequencies.

    Returns
    -------
    The transfer function G_c(s) of the lag compensator.
    """
    # Required phase margin plus a compensation term.
    phi_m_req += phi_m_compensation
    # Phase angle that corresponds with this phase margin.
    phi_req = -180.0 + phi_m_req
    # Find frequency of uncompensated system where this phase angle occurs.
    omega = find_frequency_for_phase_angle(
        phi_req, G_jw,
        omega_limits, omega_num
    )
    # Magnitude of uncompensated system at this frequency. The negative of this
    # magnitude is the attenuation the lag compensator needs to provide, so that
    # the magnitude of the compensated system's frequency response crosses 0 dB
    # around frequency `omega`
    _, M_dB, _ = G_jw.evaluate(omega, K_req)
    # The upper break frequency of the lag compensator is selected one decade
    # below frequency `omega`.
    upper_omega_break = omega / 10
    # The lower break frequency of the lag compensator is determined by drawing
    # a straight line through the upper break frequency with a slope of
    # -20 dB/decade. The frequency at the intersection with the 0 dB-line is
    # the lower break frequency.
    dY_log10 = abs(-M_dB)  # required attenuation (logarithmic height in Bode diagram)
    S_log10 = 20  # slope of 20 dB/decade
    dX_log10 = dY_log10 / S_log10  # logarithmic distance in Bode diagram between lower and upper break frequency
    dX = 10 ** dX_log10  # multiplication factor between upper and lower break frequency
    lower_omega_break = upper_omega_break / dX
    # The dc gain of the lag compensator should be 1 in order not to alter the
    # steady-state error which was already met by adjusting the open-loop gain.
    # The dc gain of 1 determines the gain of the lag compensator.
    K_c = lower_omega_break / upper_omega_break
    # Transfer function.
    G_c = TransferFunction(
        K_c * (s + upper_omega_break) / (s + lower_omega_break)
    )
    return G_c


def create_lead_compensator(
    phi_m_req: float,
    K_req: float,
    G_jw: FrequencyResponse,
    phi_m_compensation: float = 10.0,
    omega_limits: tuple[float, float] = (1.e-3, 1.e3),
    omega_num: int = 1000
) -> TransferFunction:
    """Creates the transfer function for a lead compensator. The general form
    of this transfer function looks like:
                               1            s + 1/T
                    G_c(s) = ------ * -------------------
                              beta      s + 1/(beta * T)
    where beta < 1. 1/T is the lower break frequency of the lead compensator and
    1/(beta * T) is the upper break frequency. The gain of the lead compensator
    is 1/beta. When the frequency is zero (steady-state), the dc gain of the
    lead compensator is therefore equal to one.

    Parameters
    ----------
    phi_m_req:
        Required phase margin (depends on transient response requirement).
    K_req:
        Required open-loop gain (depends on steady-state error requirement).
    G_jw:
        Open-loop frequency response of the feedback system.
    phi_m_compensation:
        Phase margin correction term that is added to compensate for the lower
        uncompensated system's phase angle at the higher phase-margin frequency,
        caused by inserting the lead compensator.
    omega_limits:
        Lower and upper limit of the frequency range where the root-finding
        algorithm searches for the frequency at which the required phase margin
        is located. If no solution can be found within the specified frequency
        range, a `ValueError` is raised. In that case, either the lower limit of
        the frequency range should be decreased, or the upper limit should be
        increased.
    omega_num:
        Number of calculation points within the specified range of frequencies.

    Returns
    -------
    The transfer function G_c(s) of the lead compensator.
    """
    # Phase margin of the uncompensated system.
    phi_m, _ = G_jw.phase_margin(K_req, omega_limits)
    # Required increase of the phase margin (by using the peak phi_max of the
    # lead compensator's phase curve).
    phi_max = phi_m_req - phi_m + phi_m_compensation
    # Determine beta.
    a = np.sin(np.radians(phi_max))
    beta = (1 - a) / (1 + a)
    # Determine the compensator's magnitude at the peak of the phase curve.
    M_dB = 20 * np.log10(1 / np.sqrt(beta))
    # Aim is to position the peak of the lead compensator's phase curve at the
    # frequency where the magnitude of the uncompensated system is M_dB below
    # the 0 dB line (i.e. where its magnitude is -M_dB), so that at this
    # frequency the compensated system's magnitude crosses the 0 dB line. So we
    # need to find the frequency where the magnitude of the uncompensated system
    # is equal to -M_dB
    omega_max = find_frequency_for_magnitude(-M_dB, G_jw, K_req, omega_limits, omega_num)
    # The upper break frequency of the lead compensator follows from:
    lower_omega_break = omega_max * np.sqrt(beta)  # 1/T
    # The lower break frequency of the lead compensator follows from:
    upper_omega_break = lower_omega_break / beta   # 1/(beta * T)
    # The gain of the lead compensator is determined such that its dc gain
    # equals one in order not to change the steady-state error behaviour
    # determined by the open-loop gain that was already designed.
    K_c = 1 / beta
    G_c = TransferFunction(
        K_c * (s + lower_omega_break) / (s + upper_omega_break)
    )
    return G_c


def create_lag_lead_compensator(
    phi_m_req: float,
    K_req: float,
    G_jw: FrequencyResponse,
    omega_phi_m: float,
    phi_m_compensation: float = 5.0,
) -> TransferFunction:
    """Creates the transfer function for a lag-lead compensator. The general
    form of this transfer function looks like:
                         s + 1/T1             s + 1/T2
            G_c(s) =  -------------- * ----------------------
                       s + gamma/T1     s + 1 / (gamma * T2)
    where gamma > 1. The first term produces the lead compensation, and the
    second term produces the lag compensation. 1/T1 is the lower break frequency
    of the lead compensator and gamma/T1 is the upper break frequency. 1/T2 is
    the upper break frequency of the lag compensator and 1/(gamma * T2) is the
    lower break frequency. The gain of the lag-lead compensator is unity.

    Parameters
    ----------
    phi_m_req:
        Required phase margin (depends on transient response requirement).
    K_req:
        Required open-loop gain (depends on steady-state error requirement).
    G_jw:
        Open-loop frequency response of the feedback system.
    omega_phi_m:
        Phase-margin frequency (should be close near the required closed-loop
        bandwidth frequency).
    phi_m_compensation:
        Phase margin correction term in degrees that is added to compensate for
        the addition of the lag compensator.

    Returns
    -------
    The transfer function G_c(s) of the lag-lead compensator.
    """
    # Determine the additional amount of phase lead to meet the phase-margin
    # requirement at the given phase-margin frequency.
    *_, phi = G_jw.evaluate(omega_phi_m, K_req)
    phi_req = -180.0 + phi_m_req
    delta_phi = phi_req - phi + phi_m_compensation
    # Design the lag compensator by selecting the higher break frequency one
    # decade below the new phase-margin frequency.
    upper_omega_break_lag = omega_phi_m / 10
    # Find the value of gamma from the additional amount of phase lead.
    a = np.sin(np.radians(delta_phi))
    beta = (1 - a) / (1 + a)
    gamma = 1 / beta
    # Determine the lag's lower break frequency.
    lower_omega_break_lag = upper_omega_break_lag / gamma
    # Transfer function of the lag compensator.
    G_lag = TransferFunction(
        (1 / gamma) * (s + upper_omega_break_lag) / (s + lower_omega_break_lag)
    )
    # Design the lead compensator. Using gamma and the phase-margin frequency,
    # find the lower and upper break frequencies for the lead compensator.
    T = 1 / (omega_phi_m * np.sqrt(beta))
    upper_omega_break_lead = 1 / (beta * T)
    lower_omega_break_lead = 1 / T
    # Transfer function of the lead compensator.
    G_lead = TransferFunction(
        (1 / beta) * (s + lower_omega_break_lead) / (s + upper_omega_break_lead)
    )
    # Transfer function of the lag-lead compensator.
    G_c = G_lag * G_lead
    return G_c
