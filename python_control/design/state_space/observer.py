from collections.abc import Sequence
import sympy as sp
import numpy as np
from ...core import s, StateSpace


def _get_closed_loop_characteristic_coeffs(poles: Sequence[complex | float]) -> list[sp.Float]:
    """Returns the coefficients of the desired closed-loop observer
    characteristic equation given the closed-loop poles.

    Parameters
    ----------
    poles:
        The desired closed-loop poles from transient response requirements.
        The number of closed-loop poles must agree with the order of the
        original system.
    """
    char_eq = sp.Poly.from_expr(sp.expand(sp.prod(s - p for p in poles)))
    return char_eq.all_coeffs()


def _create_observer_equations(ss_G: StateSpace) -> list[sp.Expr]:
    """Returns expressions from which the observer gains li can be solved by
    equating these expressions to the desired coefficients of the closed-loop
    characteristic equation (the solving is done by calling the function
    `solve_observer_gain_vector` next).

    Parameters
    ----------
    ss_G:
        State-space representation of the open-loop system in observer canonical
        form.
    """
    n = ss_G.A.shape[0]
    L = sp.Matrix(sp.symarray('l', n))
    I = sp.eye(n)
    A_star = ss_G._sympy_A - L * ss_G._sympy_C
    expr = sp.det(s * I - A_star)
    poly = sp.Poly.from_expr(expr, s)
    eqs = poly.all_coeffs()
    return eqs


def solve_observer_gain_vector(
    ss_G: StateSpace,
    poles: Sequence[complex | float],
    transform: bool = True
) -> np.ndarray:
    """Solves for the observer gains li in order to get the desired coefficients
     in the characteristic equation of the closed-loop observer.

    Parameters
    ----------
    ss_G:
        State-space representation of the open-loop system.
    poles:
        The desired closed-loop poles from transient response requirements.
        The number of closed-loop poles must agree with the order of the
        original system.
    transform:
        Indicates whether the observer gain vector needs to be transformed back
        to the form of the original system (`transform=True`), or can be
        returned in the observer canonical form (`transform=False`).

    Returns
    -------
    L_vector:
        numpy 2D-array

    Raises
    ------
    ValueError:
        If the open-loop system is not completely observable, the closed-loop
        poles cannot be placed at the desired locations on the complex plane.
    """
    if ss_G.is_observable:
        OM_original = ss_G.observability_matrix
        ss_G = ss_G.transform('observer')
        OM_observer = ss_G.observability_matrix
        desired_coeffs = _get_closed_loop_characteristic_coeffs(poles)
        eqs = _create_observer_equations(ss_G)
        solutions = {}
        for i, eq in enumerate(eqs):
            if len(eq.free_symbols) == 1:
                l = list(eq.free_symbols)[0]
                sol = sp.solve(eq - desired_coeffs[i], l)[0]
                solutions[l] = sol
        _, l_values = zip(*sorted(solutions.items(), key=lambda item: str(item[0])))
        L_vector = np.array([[float(l_value) for l_value in l_values]]).transpose()
        if transform:
            P = np.linalg.inv(OM_original) @ OM_observer
            L_vector = P @ L_vector
        return L_vector
    else:
        raise ValueError("The system is not completely observable.")
