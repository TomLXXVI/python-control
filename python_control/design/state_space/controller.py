from collections.abc import Sequence
import numpy as np
import sympy as sp
from ...core import s, StateSpace


def get_closed_loop_characteristic_coeffs(poles: Sequence[complex | float]) -> list[sp.Float]:
    """Returns the desired coefficients of the characteristic equation of the
    closed-loop system (i.e. the denominator of the closed-loop system).

    Parameters
    ----------
    poles:
        The desired poles of the closed-loop system derived from the transient
        response requirements. The number of closed-loop poles must agree with
        the order of the system.
    """
    char_eq = sp.Poly.from_expr(sp.expand(sp.prod(s - p for p in poles)))
    return char_eq.all_coeffs()


def _create_controller_equations(ss_G: StateSpace) -> list[sp.Expr]:
    """Returns expressions from which the feedback gains ki can be solved by
    equating these expressions to the desired coefficients of the closed-loop
    characteristic equation (the solving is done by calling the function
    `solve_controller_vector` next).

    Parameters
    ----------
    ss_G:
        State-space representation of the open-loop system in phase-variable
        form.
    """
    m = ss_G.B.shape[1]
    n = ss_G.B.shape[0]
    if m == 1:
        K = sp.symarray('k', n)
    else:
        K = sp.symarray('k', (m, n))
    A = ss_G._sympy_A - ss_G._sympy_B * K
    I = sp.eye(n)
    char_eq = sp.det(s * I - A)
    poly = sp.Poly.from_expr(char_eq, s)
    eqs = poly.all_coeffs()
    return eqs


def _create_integral_controller_equations(ss_G: StateSpace):
    n = ss_G.B.shape[0]
    m = ss_G.B.shape[1]
    p = ss_G.C.shape[0]
    if m == 1:
        K = sp.Matrix(sp.symarray('k', n)).transpose()
    else:
        K = sp.Matrix(sp.symarray('k', (m, n)))
    K_e = sp.Symbol('K_e')
    M1 = ss_G._sympy_A - ss_G._sympy_B * K
    M2 = ss_G._sympy_B * K_e
    M3 = -ss_G._sympy_C
    M4 = sp.zeros(p, m)
    A = sp.BlockMatrix([[M1, M2], [M3, M4]]).as_explicit()
    shape_A = sp.shape(A)
    I = sp.eye(shape_A[0])
    char_eq = sp.det(s * I - A)
    poly = sp.Poly.from_expr(char_eq, s)
    eqs = poly.all_coeffs()
    return eqs


def solve_controller_gain_vector(
    ss_G: StateSpace,
    poles: Sequence[complex | float],
    integral_control: bool
) -> tuple[np.ndarray, float | None]:
    """Solves for the feedback gains ki in order to get the desired
    coefficients in the characteristic equation of the closed-loop system.

    Parameters
    ----------
    ss_G:
        State-space representation of the open-loop system in phase-variable
        form.
    poles:
        The desired poles of the closed-loop system derived from the transient
        response requirements. The number of closed-loop poles must agree with
        the order of the system.
    integral_control:
        Indicates whether an integrator is to be added to the closed-loop
        feedback system or not.
    """
    desired_coeffs = get_closed_loop_characteristic_coeffs(poles)
    if integral_control:
        eqs = _create_integral_controller_equations(ss_G)
    else:
        eqs = _create_controller_equations(ss_G)
    solutions = {}
    for i, eq in enumerate(eqs):
        if len(eq.free_symbols) == 1:
            k = list(eq.free_symbols)[0]
            sol = sp.solve(eq - desired_coeffs[i], k)[0]
            solutions[str(k)] = sol
    K_e = solutions.pop('K_e', None)
    _, k_values = zip(*sorted(solutions.items(), key=lambda item: str(item[0])))
    K_vector = np.array([[float(k_value) for k_value in k_values]])
    # noinspection PyTypeChecker
    return K_vector, K_e


def _create_closed_loop_system(
    ss_G: StateSpace,
    K_vector: np.ndarray,
    K_e: float | None
) -> StateSpace:
    """Creates the state-space representation of the closed-loop system.

    Parameters
    ----------
    ss_G:
        State-space representation of the open-loop system in phase-variable
        form.
    K_vector:
        The feedback gains for the closed-loop system. See function
        `solve_controller_gain_vector`.
    """
    if K_e is not None:
        p = ss_G.C.shape[0]
        m = ss_G.B.shape[1]
        M1 = ss_G.A - ss_G.B @ K_vector
        M2 = ss_G.B * float(K_e)
        M3 = -ss_G.C
        M4 = np.zeros((p, m))
        A = np.block([[M1, M2], [M3, M4]])
        B = np.vstack((np.zeros((M1.shape[0], 1)), np.ones((M3.shape[0], 1))))
        C = np.hstack((ss_G.C, np.zeros((p, m))))
        ss_T = StateSpace(A, B, C)
    else:
        A = np.array(ss_G.A)
        A[-1:] -= K_vector.flatten()
        ss_T = StateSpace(A, ss_G.B, ss_G.C, ss_G.D, ss_G.x0)
    return ss_T


def design_controller(
    ss_G: StateSpace,
    closed_loop_poles: Sequence[complex | float],
    integral_control: bool = False
) -> StateSpace:
    """Creates a closed-loop feedback system with the specified poles, i.e.
    pole-placement design through state-variable feedback.

    Parameters
    ----------
    ss_G:
        State-space representation of the open-loop system.
    closed_loop_poles:
        The desired poles of the closed-loop system derived from the transient
        response requirements. The number of closed-loop poles must agree with
        the order of the system.
    integral_control:
        Indicates whether an integrator is to be added to the closed-loop
        feedback system (`integral_control=True`) or not
        (`integral_control=False`).

    Returns
    -------
    ss_T:
        State-space representation of the controlled closed-loop system.

    Raises
    ------
    ValueError:
        If the open-loop system is not completely controllable, the closed-loop
        poles cannot be placed at the desired locations on the complex plane.
    """
    if ss_G.is_controllable:
        ss_G = ss_G.transform('phase-variable')
        K_vector, K_e = solve_controller_gain_vector(ss_G, closed_loop_poles, integral_control)
        ss_T = _create_closed_loop_system(ss_G, K_vector, K_e)
        return ss_T
    else:
        raise ValueError("The system is not completely controllable.")
