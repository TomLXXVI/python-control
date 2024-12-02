from typing import TYPE_CHECKING
from collections import namedtuple
import sympy as sp
import numpy as np
import control as ct
from .laplace_transform import LaplaceTransform, InverseLaplaceTransform
from .symbols import s, t

if TYPE_CHECKING:
    from .transfer_function import TransferFunction


TimeSolution = namedtuple('TimeSolution', ('state', 'output'))
# state -> list[InverseLaplaceTransform]
# output -> list[InverseLaplaceTransform]


FrequencySolution = namedtuple('FrequencySolution', ('state', 'output'))
# state -> list[LaplaceTransform]
# output -> list[LaplaceTransform]


InputVector = list[str | sp.Expr]


class StateSpace:
    """
    State-space representation of a linear time invariant (LTI), n-th order
    system.

    State equation: x_dot = A * x + B * u
    Output equation: y = C * x + D * u
    x: state vector, (n x 1) with n the number of state variables.
    u: input or control vector, (m x 1) with m the number of input variables.
    y: output vector, (p x 1) with p the number of output variables.
    A: system matrix, (n x n).
    B: input or control matrix, (n x m)
    C: output matrix, (p x n)
    D: feedforward matrix, (p x m)
    """
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray | None = None,
        D: np.ndarray | None = None,
        x0: np.ndarray | None = None
    ) -> None:
        """
        Creates a `StateSpace` object.

        Parameters
        ----------
        A:
            System matrix.
        B:
            Input or control matrix.
        C: optional
            Output matrix.
        D: optional
            Feedforward matrix.
        x0: optional
            Initial values vector.
        """
        # System matrix:
        self.A = A
        self._sympy_A = sp.Matrix(A)

        # Input matrix:
        self.B = B
        self._sympy_B = sp.Matrix(B)

        # Output matrix:
        if C is not None:
            self.C = C
        else:
            n_rows = 1
            n_cols = self.A.shape[0]
            self.C = np.zeros((n_rows, n_cols))
        self._sympy_C = sp.Matrix(self.C)

        # Feedforward matrix:
        if D is not None:
            self.D = D
        else:
            n_rows = self.C.shape[0]
            n_cols = self.B.shape[1]
            self.D = np.zeros((n_rows, n_cols))
        self._sympy_D = sp.Matrix(self.D)

        # Initial state vector:
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((self.A.shape[0], ))
        self._sympy_x0 = sp.Matrix(self.x0)

        # "Python Control Systems Library" implementation of state-space:
        self._ct_state_space = ct.ss(self.A, self.B, self.C, self.D)

        # State-transition matrix:
        self.FI: sp.Matrix | None = None

    @classmethod
    def from_transfer_function(cls, tf: 'TransferFunction'):
        """
        Creates the state-space representation from a transfer function.
        """
        ct_ss = ct.tf2ss(tf.num_coeffs, tf.den_coeffs)
        A = np.round(ct_ss.A, 8)
        B = np.round(ct_ss.B, 8)
        C = np.round(ct_ss.C, 8)
        D = np.round(ct_ss.D, 8)
        return cls(A, B, C, D)

    @property
    def as_ct(self) -> ct.StateSpace:
        """
        Returns the "Python Control Systems Toolbox" implementation of the
        state-space representation.
        """
        return self._ct_state_space

    @property
    def poles(self) -> np.ndarray:
        return self._ct_state_space.poles()

    @property
    def zeros(self) -> np.ndarray:
        return self._ct_state_space.zeros()

    def _solve_for_laplace_state_vector(self, sympy_U: sp.Matrix) -> sp.Matrix:
        """
        Calculates the Laplace transform solution of the state vector X for the
        given Laplace transformed input vector U.
        """
        I = sp.eye(self.A.shape[0])
        M = (s * I - self._sympy_A)
        M_adj = M.adjugate()
        M_det = M.det()
        X = (M_adj / M_det) * (self._sympy_x0 + self._sympy_B * sympy_U)
        return X

    def _get_laplace_solution(
        self,
        U: list[sp.Expr]
    ) -> FrequencySolution:
        """
        Returns the Laplace transform solution of the state vector X and the
        output vector Y for the given Laplace transformed input vector U.

        Parameters
        ----------
        U:
            Laplace transform of the input vector, i.e., a list of the Laplace
            transforms of each input signal, either as strings, or as Sympy
            expressions.

        Returns
        -------
        A `FrequencySolution` namedtuple with two members, `state` and `output`.
        These are lists of `LaplaceTransform` objects, representing the Laplace
        transform of each state variable in the state vector and each output
        variable in the output vector.
        """
        U = sp.Matrix(np.array(U))

        X = self._solve_for_laplace_state_vector(U)
        n_rows, _ = sp.shape(X)
        lap_X_list = [LaplaceTransform(X[i, 0]) for i in range(n_rows)]

        Y = self._sympy_C * X + self._sympy_D * U
        n_rows, _ = sp.shape(Y)
        lap_Y_list = [LaplaceTransform(Y[i, 0]) for i in range(n_rows)]

        return FrequencySolution(lap_X_list, lap_Y_list)

    def _create_state_transition_matrix(self) -> sp.Matrix:
        I = sp.eye(self.A.shape[0])
        M = (s * I - self._sympy_A)
        inv_M = M.inv()
        n_rows, n_cols = sp.shape(inv_M)
        f = lambda i, j: sp.inverse_laplace_transform(inv_M[i, j], s, t)
        FI = sp.Matrix(n_rows, n_cols, f)
        return FI

    def _zero_input_response(self, FI: sp.Matrix) -> sp.Matrix:
        x_zi = FI * self._sympy_x0
        return x_zi

    def _zero_state_response(self, FI: sp.Matrix, u: sp.Matrix) -> sp.Matrix:
        tau = sp.Symbol('tau')
        FI_tau = FI.subs(t, t - tau)
        u_tau = u.subs(t, tau)
        M = FI_tau * self._sympy_B * u_tau
        x_zs = sp.integrate(M, (tau, 0, t))
        return x_zs

    def _get_time_solution(
        self,
        u: list[sp.Expr]
    ) -> TimeSolution:
        """
        Returns the time domain solution for the state vector x and the output
        vector y for the given input vector u.

        Parameters
        ----------
        u:
            input vector, i.e., a list of the input signals, either as strings,
            or as Sympy expressions.

        Returns
        -------
        A `TimeSolution` namedtuple with two members, `state` and `output`.
        These are two lists of `InverseLaplaceTransform` objects, representing
        each state variable in the state vector as a function of time, and each
        output variable in the output vector as a function of time.
        """
        if self.FI is None:
            self.FI = self._create_state_transition_matrix()
        u = sp.Matrix(np.array(u))
        x_zi = self._zero_input_response(self.FI)
        x_zs = self._zero_state_response(self.FI, u)
        x = x_zi + x_zs
        y = self._sympy_C * x + self._sympy_D * u
        n_rows, _ = sp.shape(x)
        x_list = [InverseLaplaceTransform(x[i, 0].evalf()) for i in range(n_rows)]
        n_rows, _ = sp.shape(y)
        y_list = [InverseLaplaceTransform(y[i, 0].evalf()) for i in range(n_rows)]
        return TimeSolution(x_list, y_list)

    def solve(self, u: InputVector) -> TimeSolution | FrequencySolution:
        """
        Returns the solution for the system's state vector `x` and output vector
        `y` when the input vector `u` is given. The system is solved either in
        the Laplace domain (s-domain) or in the time domain, depending on the
        form of the input vector u.

        Parameters
        ----------
        u:
            Input vector, i.e., a list of the input signals to the system,
            either as strings or as Sympy expressions. If the expressions are
            Laplace transforms (i.e., they contain the free symbol *s*), the
            solution in the Laplace domain is returned. Otherwise, the solution
            in the time domain is returned.

        Returns
        -------
        If the system is solved in the Laplace domain, a `FrequencySolution`
        (of type `namedtuple`) is returned with two lists of `LaplaceTransform`
        objects. The first list (i.e., the member `state` of the namedtuple) is
        the state vector `x`. The second list (i.e., the member `output` of the
        namedtuple) is the output vector `y`.

        If the system is solved in the time domain, a `TimeSolution` (of type
        `namedtuple`) is returned with two lists of `InverseLaplaceTransform`
        objects. The first list (i.e., the member `state` of the namedtuple)
        contains each state variable in the state vector as a function of time.
        The second list (i.e., the member `output` of the namedtuple) contains
        each output variable in the output vector as a function of time.
        """
        if isinstance(u[0], str):
            if 's' in u[0]:
                for i, expr in enumerate(u):
                    expr = sp.parse_expr(expr)
                    expr = expr.subs('s', s)
                    u[i] = expr
            elif 't' in u[0]:
                for i, expr in enumerate(u):
                    expr = sp.parse_expr(expr)
                    expr = expr.subs('t', t)
                    u[i] = expr

        if s in u[0].free_symbols:
            freq_sol = self._get_laplace_solution(u)
            return freq_sol
        else:
            time_sol = self._get_time_solution(u)
            return time_sol

    def transform(self, form: str) -> 'StateSpace':
        """
        Returns a new, reformed `StateSpace` object.

        Parameters
        ----------
        form: str, {'controller', 'diagonal', 'modal', 'observer', 'phase-variable'}
            The requested form of the state-space representation:
            -   'controller' --> controller canonical form.
            -   'diagonal' or 'modal' --> diagonalized or decoupled form.
            -   'observer' --> observable canonical form.
            -   'phase-variable' --> phase-variable form.

        Raises
        ------
        If the form of the state-space representation is not recognized, a
        `ValueError` exception is raised.
        """
        if form.lower() in ['controller', 'phase-variable']:
            ct_ss, _ = ct.canonical_form(
                self._ct_state_space
            )
        elif form.lower() in ['diagonal', 'modal']:
            ct_ss, _ = ct.canonical_form(
                self._ct_state_space,
                form='modal'
            )
        elif form.lower() == 'observer':
            ct_ss, _ = ct.canonical_form(
                self._ct_state_space,
                form='observable'
            )
        else:
            raise ValueError(f'Unknown state-space representation: {form}')

        A = np.round(ct_ss.A, 8)
        B = np.round(ct_ss.B, 8)
        C = np.round(ct_ss.C, 8)
        D = np.round(ct_ss.D, 8)

        if form.lower() == 'phase-variable':
            A = np.flip(A)
            B = np.flip(B)
            C = np.flip(C)
            D = np.flip(D)

        ss = StateSpace(A, B, C, D)
        return ss

    def diagonalize(self) -> 'StateSpace':
        """
        Transforms the state-space representation so that the system matrix
        becomes a diagonal matrix. In this representation each state equation
        is a function of only one state variable, and hence each differential
        equation can be solved independently of the other equations (i.e., the
        equations are decoupled).

        Notes
        -----
        1.  The diagonal elements are the eigenvalues of the system matrix A,
            which are also the poles of the corresponding transfer function.
        2.  A call to method `diagonalize` is equivalent with a call to method
            `transform` with parameter form set to 'diagonal' or 'modal'.
        """
        _, P = np.linalg.eig(self.A)
        # The columns of P are the normalized eigenvectors of A. The method
        # `eig` also returns the eigenvalues of A, which correspond with the
        # poles of the transfer function.
        inv_P = np.linalg.inv(P)
        A_new = np.round(inv_P @ self.A @ P, 8)
        B_new = np.round(inv_P @ self.B, 8)
        C_new = np.round(self.C @ P, 8)
        D_new = self.D
        x0_new = self.x0
        # noinspection PyTypeChecker
        ss_new = StateSpace(A_new, B_new, C_new, D_new, x0_new)
        return ss_new

    def steady_state_error(self, u: str | sp.Expr) -> sp.Number | sp.Expr:
        """
        Returns the steady-state error between the output `y` and the input `u`
        of the system.

        Parameters
        ----------
        u:
            Input signal of the system, either a string, or a Sympy expression.
            The expression can be a Laplace transform (i.e., it contains the
            free symbol *s*), or an expression as function of time.

        Returns
        -------
        A Sympy `Number` or `Expr`.
        """
        sol = self.solve([u])
        if isinstance(sol, FrequencySolution):
            Y = sol.output[0].expr
            E = u - Y
            e_oo = sp.limit(s * E, s, 0)
            return e_oo
        if isinstance(sol, TimeSolution):
            y = sol.output[0].expr
            e = u - y
            e_oo = sp.limit(e, t, sp.oo)
            return e_oo

    @property
    def controllability_matrix(self) -> np.ndarray:
        """Returns the controllability matrix of the system."""
        n = self.A.shape[0]
        matrices = [
            np.linalg.matrix_power(self.A, i) @ self.B
            for i in range(n)
        ]
        CM = np.hstack(matrices)
        return CM

    @property
    def is_controllable(self) -> bool:
        """Returns `True` if the system is completely controllable."""
        CM = self.controllability_matrix
        n = self.A.shape[0]
        rank = np.linalg.matrix_rank(CM)
        if rank == n:
            return True
        return False

    @property
    def observability_matrix(self) -> np.ndarray:
        """Returns the observability matrix of the system."""
        n = self.A.shape[0]
        matrices = [
            self.C @ np.linalg.matrix_power(self.A, i)
            for i in range(n)
        ]
        OM = np.vstack(matrices)
        return OM

    @property
    def is_observable(self) -> bool:
        """Returns `True` if the system is completely observable."""
        OM = self.observability_matrix
        n = self.A.shape[0]
        rank = np.linalg.matrix_rank(OM)
        if rank == n:
            return True
        return False
