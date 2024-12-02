"""
Implementation of the Routh-Hurwitz Criterion to generate a Routh table.

The Routh-Hurwitz criterion basically states that the number of roots of the
denominator of a transfer function that are in the right half-plane is equal to
the number of sign changes in the first column. Thus, a system is stable if
there are no sign changes in the first column of the Routh table.

Two special cases can occur: (1) the Routh table sometimes will have a zero only
in the first column of a row, or (2) the Routh table sometimes will have an
entire row that consists of zeros.

If the first element of a row is zero, division by zero would be required to
form the next row. To avoid this phenomenon, an epsilon, ε, is assigned to
replace the zero in the first column. The value ε is then allowed to approach
zero from either the positive or the negative side, after which the signs of
the entries in the first column can be determined.

Sometimes while making a Routh table, we find that an entire row consists of
zeros because there is an even polynomial that is a factor of the original
polynomial. Even polynomials only have roots that are symmetrical about the
origin of the s-plane. This symmetry can occur under three conditions of root
position:
(1) The roots are symmetrical and real,
(2) the roots are symmetrical and imaginary, or
(3) the roots are quadrantal and symmetrical about the origin.
Another characteristic of the Routh table is that the row previous to the row
of zeros contains the coefficients of the even polynomial that is a factor of
the original polynomial. In the returned Routh table a dashed line is drawn
above this row.
Everything from the row containing the even polynomial down to the end of the
Routh table is a test of only the even polynomial. Therefore, the number of sign
changes from the even polynomial to the end of the table equals the number of
right-half-plane roots of the even polynomial. Because of the symmetry of roots
about the origin, the even polynomial must have the same number of
left-half-plane roots as it does right-half-plane roots. Having accounted for
the roots in the right and left half-planes, we know the remaining roots must be
on the jω-axis.
Every row in the Routh table from the beginning of the chart to the row
containing the even polynomial applies only to the other factor of the original
polynomial. For this factor, the number of sign changes, from the beginning of
the table down to the even polynomial, equals the number of right-half-plane
roots. The remaining roots are left-half-plane roots. There can be no jω roots
contained in the other polynomial.

Usage
-----
To retrieve the Routh table of the denominator of a transfer function, call the
function `routh_hurwitz` of this module with the denominator of the transfer
function as Sympy `Poly` object. This function will return an object of class
`RouthHurwitz`, that returns a string representation of the Routh table when
`print` is called on it, i.e. `print(routh_hurwitz(tf.denominator.as_poly))`
with `tf` an instance of the `TransferFunction` class.

References
----------
Nise, N. S. (2020). Control Systems Engineering, EMEA Edition, 8th Edition.
"""
import sympy as sp
import numpy as np
from python_control.core.symbols import s


class RouthHurwitz:
    """
    Class that encapsulates the creation of a Routh table.

    Attributes
    ----------
    polynomial: sp.Poly
        The polynomial (i.e., the denominator of a transfer function) for which
        the routh table will be created.
    routh_matrix: sp.Matrix
        The Routh table of the polynomial, represented as a Sympy `Matrix`
        object.
    """
    def __init__(self, p: sp.Poly):
        """
        Creates a `RouthHurwitz` object.

        Parameters
        ----------
        p:
            Polynomial (i.e., the denominator of a transfer function) for which
            the routh table will be created.
        """
        self.polynomial = p
        self.routh_matrix: sp.Matrix | None = None
        self._zero_row_index: int | None = None

    def __call__(self) -> sp.Matrix:
        """
        Constructs the Routh-Hurwitz matrix given the `Poly` object in
        `self.polynomial`.

        Returns
        -------
        The Routh-Hurwitz matrix, being a Sympy `Matrix` object.
        """
        self.coeffs = self.polynomial.all_coeffs()
        N = len(self.coeffs)
        M = sp.zeros(N, (N + 1) // 2 + 1)
        r1 = self.coeffs[0::2]
        r2 = self.coeffs[1::2]
        M[0, :len(r1)] = self.__simplify_row(r1)
        M[1, :len(r2)] = self.__simplify_row(r2)
        for i in range(2, N):
            for j in range(N // 2):
                if M[i - 1, 0] == 0:
                    # Special case 1: Zero only in the first column
                    M[i - 1, 0] = sp.Float(1.e-12)
                S = M[[i - 2, i - 1], [0, j + 1]]
                M[i, j] = sp.simplify(-S.det() / M[i - 1, 0])
            if all(elem == 0 for elem in M[i, :]):
                # Special case 2: Entire row is zero
                self._zero_row_index = i
                M[i, :] = self.__replace_row_of_zeros(M[i - 1, :], N - i)
            M[i, :] = self.__simplify_row(M[i, :])
        self.routh_matrix = M[:, :-1]
        return self.routh_matrix

    def __str__(self) -> str:
        """
        Returns the Routh table as a single string.
        """
        # Convert the elements of the table to a string:
        table = self.routh_matrix.tolist()
        num_rows = len(table) - 1
        width = 0
        table_ = []
        for i, row in enumerate(table):
            row_ = [f's^{num_rows - i}']
            for elem in row:
                if isinstance(elem, sp.Float):
                    elem = str(sp.N(elem, 4))
                else:
                    elem = str(elem)
                if len(elem) > width:
                    width = len(elem)
                row_.append(elem)
            table_.append(row_)
        width += 3
        # Insert a line where a zero row was present:
        if isinstance(self._zero_row_index, int):
            num_cols = len(table_[0])
            table_.insert(self._zero_row_index, ['-' * width] * num_cols)
        # Write table into a single string:
        table__ = ""
        for row_ in table_:
            for elem_ in row_:
                table__ += f"{elem_:>{width}}"
            table__ += '\n'
        return table__

    @staticmethod
    def __replace_row_of_zeros(coeffs: list, order: int) -> list:
        """
        Replaces a row of which all elements are zero, according to the method
        explained in Nise (2020), § 6.3 - Entire Row is Zero.
        """
        exponents = [i for i in range(0, order + 1)]
        exponents.reverse()
        exponents = exponents[::2]
        p = sum(coeff * s ** exp for coeff, exp in zip(coeffs, exponents))
        p = p.as_poly(s)
        der = p.diff(s)
        new_coeffs = der.coeffs()
        new_row = []
        for i in range(len(coeffs)):
            try:
                new_row.append(new_coeffs[i])
            except IndexError:
                new_row.append(0)
        return [new_row]

    def __simplify_row(self, row: list) -> list:
        """
        Divides the numbers in list `row` by their greatest common divisor.
        """
        if all(not isinstance(elem, sp.Expr) for elem in self.coeffs):
            row = np.array([elem for elem in row])
            gcd = np.gcd.reduce(row)
            row = row / gcd
            row = row.tolist()
        return [row]


def routh_hurwitz(p: sp.Poly) -> RouthHurwitz:
    """
    Creates a `RouthHurwitz` object and returns it.

    Parameters
    ----------
    p:
        Denominator of the transfer function as a Sympy Poly object.
    """
    rh = RouthHurwitz(p)
    rh()
    return rh
