import math
import cmath
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root_scalar, minimize_scalar
import sympy as sp
import control as ct
from ..matplotlibwrapper import LineChart
from .transfer_function import TransferFunction
from .symbols import s, K


@dataclass
class Vector:
    """
    Class that encapsulates the attributes of the vector representation of
    a complex number.

    Attributes
    ----------
    magnitude:
        The length of the vector.
    angle:
        The angle of the vector with the horizontal real axis in degrees.
    real:
        Component of the vector on the real axis (i.e., the real part of the
        complex number).
    imag:
        Component of the vector on the vertical imaginary axis (i.e., the
        imaginary part of the complex number).
    complex number:
        The actual complex number that is represented by the vector.
    """
    magnitude: float | None = None
    angle: float | None = None
    real: float | None = None
    imag: float | None = None
    complex_number: complex | None = None

    def __post_init__(self):
        if self.real is not None and self.imag is not None:
            self.complex_number = self.real + self.imag * 1j
            self.magnitude = abs(self.complex_number)
            self.angle = math.degrees(cmath.phase(self.complex_number))
        elif self.magnitude is not None and self.angle is not None:
            angle = math.radians(self.angle)
            self.complex_number = cmath.rect(self.magnitude, angle)
            self.real = self.complex_number.real
            self.imag = self.complex_number.imag
        elif self.complex_number is not None:
            self.real = self.complex_number.real
            self.imag = self.complex_number.imag
            self.magnitude = abs(self.complex_number)
            self.angle = math.degrees(cmath.phase(self.complex_number))

    def __str__(self):
        return f"{self.magnitude:.4g} < {self.angle:.4g}°"


class TransferFunctionVector:
    """
    Given a transfer function and a "test point" *s* in the complex plane, the
    result of the transfer function at that point *s* can be calculated, which
    will also be a complex number, that can be represented by a vector in the
    complex plane. Hence the name of the class: `TransferFunctionVector`.

    Given a "test point" *s* in the complex plane and only using the zeros and
    poles of the open-loop transfer function of a feedback system, it is
    possible to calculate the magnitude and angle of the open-loop transfer
    function vector (this implies that we don't need to know an exact value for
    the gain K to be able to determine the vector).

    If for a given "test point" *s* in the complex plane, the angle of the
    open-loop transfer function vector is an odd multiple of 180°, then the
    "test point" *s* can be a pole of the closed-loop transfer function of the
    feedback system depending on the value of the gain *K*.
    To determine the value of the gain *K* for which the "test point" *s* is
    actually a pole of the closed-loop transfer function, the inverse of the
    magnitude of the open-loop transfer function vector is taken.
    """
    def __init__(self, GH: TransferFunction):
        """
        Creates a `TransferFunctionVector` object.

        Parameters
        ----------
        GH:
            The open-loop transfer function of a feedback system.
        """
        self.GH = GH
        self._p: complex | None = None

    @property
    def point(self) -> complex:
        """
        Returns the point for which the vector of the transfer function is to be
        calculated.
        """
        return self._p

    @point.setter
    def point(self, p: complex) -> None:
        """
        Sets the point for which the vector of the transfer function is to be
        calculated.
        """
        self._p = p

    @property
    def zero_vectors(self) -> list[Vector]:
        """
        Determines the vectors between the zeros of the transfer function and
        the given point on the complex plane.
        """
        zero_vectors = [
            Vector(
                real=self._p.real - zero.real,
                imag=self._p.imag - zero.imag
            )
            for zero in self.GH.zeros_sympy
        ]
        return zero_vectors

    @property
    def pole_vectors(self) -> list[Vector]:
        """
        Determines the vectors between the poles of the transfer function and
        the given point on the complex plane.
        """
        pole_vectors = [
            Vector(
                real=self._p.real - pole.real,
                imag=self._p.imag - pole.imag
            )
            for pole in self.GH.poles_sympy
        ]
        return pole_vectors

    @property
    def magnitude(self) -> float:
        """
        Determines the magnitude of the transfer function vector as seen
        from the origin of the complex plane (i.e., the start point of the
        transfer function vector is the origin of the complex plane).
        """
        num = math.prod(
            abs(zero_vector.complex_number)
            for zero_vector in self.zero_vectors
        )
        den = math.prod(
            abs(pole_vector.complex_number)
            for pole_vector in self.pole_vectors
        )
        try:
            M = num / den
        except ZeroDivisionError:
            M = int(num >= 0) * float('inf')
        return M

    @property
    def angle(self) -> float:
        """
        Determines the angle in degrees of the transfer function vector with
        respect to the horizontal, real axis of the complex plane.
        """
        zero_term = sum(
            math.degrees(cmath.phase(zero_vector.complex_number))
            for zero_vector in self.zero_vectors
        )
        pole_term = sum(
            math.degrees(cmath.phase(pole_vector.complex_number))
            for pole_vector in self.pole_vectors
        )
        theta = zero_term - pole_term
        return theta

    @property
    def vector(self) -> Vector:
        """
        Returns the transfer function vector as a `Vector` object.
        """
        vector = Vector(self.magnitude, self.angle)
        return vector

    @property
    def gain(self) -> float:
        """
        Returns the gain of the transfer function vector, being the factor
        with which the magnitude of the transfer function vector must be
        multiplied to get 1.
        """
        K = 1 / self.magnitude
        return K


class RootLocus:
    """Analysis of a negative- or positive-feedback system with positive
    gain K.

    Notes
    -----
    A negative-feedback system with negative gain is equivalent to a
    positive-feedback system with a positive gain.
    """
    sigma = sp.Symbol('sigma', real=True)

    def __init__(
        self,
        GH: TransferFunction,
        positive_feedback: bool = False
    ) -> None:
        """Creates a `RootLocus` object.

        Parameters
        ----------
        GH:
            Open-loop transfer function of the feedback control system.
        positive_feedback:
            Indicates if the system uses positive feedback. Default is
            negative feedback.
        """
        self.GH = GH
        self.positive_feedback = positive_feedback
        self.tfv = TransferFunctionVector(self.GH)
        self.zeros = self.GH.zeros_sympy
        self.poles = self.GH.poles_sympy

    @staticmethod
    def _is_odd_multiple_of_pi(angle: float) -> bool:
        angle = round(angle)
        k = int((angle / 180.0 - 1.0) / 2.0)
        angle_new = (2 * k + 1) * 180.0
        if angle_new == angle:
            return True
        return False

    @staticmethod
    def _is_even_multiple_of_2pi(angle: float) -> bool:
        angle = round(angle)
        k = int(angle / 360.0)
        angle_new = k * 360.0
        if angle_new == angle:
            return True
        return False

    def asymptotes(self) -> tuple[float, list[float]]:
        """Returns the intercept with the real-axis of any asymptotes and a list
        with the angles of these asymptotes.
        """
        sum_poles = sum(self.poles)
        sum_zeros = sum(self.zeros)
        num_poles = len(self.poles)
        num_zeros = len(self.zeros)
        sigma_a = (sum_poles - sum_zeros) / (num_poles - num_zeros)
        if self.positive_feedback:
            theta_a = 2 * math.pi / (num_poles - num_zeros)
            k = int(2 * math.pi / theta_a)
            theta_a = [i * theta_a for i in range(0, k)]
        else:
            theta_a = math.pi / (num_poles - num_zeros)
            k = int((2 * math.pi / theta_a - 1) / 2)
            theta_a = [(2 * i + 1) * theta_a for i in range(0, k + 1)]
        return sigma_a, theta_a

    def find_breakaway_break_in_points(self) -> tuple[list[float], list[float]]:
        """Calculates the breakaway points on the real axis, where the root
        locus leaves the real axis, and break-in points on the real axis, where
        the root locus returns to the real axis.

        A break-in point lies between open-loop zeros, while a breakaway point
        lies between open-loop poles.

        Returns
        -------
        A list with the breakaway points and a list with the break-in points.
        Either one or both lists might be empty. If that is the case, it doesn't
        necessarily mean that there are no breakaway points nor break-in points.
        It could equally mean that the (Sympy) equation that is used to determine
        the breakaway and break-in points could not find a solution. As an
        alternative, if the bounds are known (after the root locus has been
        plot) within which a breakaway or break-in point lies, method
        `get_breakaway_point`, respectively `get_break_in_point` can also be
        used.
        """
        lhs = sum(1 / (self.sigma - z) for z in self.zeros)
        rhs = sum(1 / (self.sigma - p) for p in self.poles)
        eq = sp.Eq(lhs, rhs)
        sol = sp.solve(eq, self.sigma)

        break_in_points = []
        breakaway_points = []

        real_zeros = [zero.real for zero in self.zeros if zero.imag == 0.0]
        real_poles = [pole.real for pole in self.poles if pole.imag == 0.0]

        if real_zeros:
            real_zero_min, real_zero_max = min(real_zeros), max(real_zeros)
            if real_zero_min == real_zero_max:
                if real_zero_min <= 0.0:
                    real_zero_max = real_zero_min
                    real_zero_min = -float('inf')
                else:
                    real_zero_max = float('inf')
        else:
            real_zero_min, real_zero_max = None, None

        if real_poles:
            real_pole_min, real_pole_max = min(real_poles), max(real_poles)
            if real_pole_min == real_pole_max:
                if real_pole_min <= 0.0:
                    real_pole_max = real_pole_min
                    real_pole_min = -float('inf')
                else:
                    real_pole_max = float('inf')
        else:
            real_pole_min, real_pole_max = None, None

        for elem in sol:
            if real_zeros and elem in sp.Interval(real_zero_min, real_zero_max):
                break_in_points.append(elem)
            elif real_poles and elem in sp.Interval(real_pole_min, real_pole_max):
                breakaway_points.append(elem)
        return breakaway_points, break_in_points

    def get_breakaway_point(self, bounds: tuple[float, float]) -> tuple[float, float]:
        """Returns the breakaway point that must lie between the specified lower
        and upper bound on the real axis, i.e. the point on the real axis where
        the gain reaches a (local) maximum. Also returns the gain at this point.
        """
        def _objective(sigma: float) -> float:
            self.tfv.point = complex(sigma, 0)
            return -self.tfv.gain

        res = minimize_scalar(_objective, bounds=bounds)
        self.tfv.point = complex(res.x, 0)
        return res.x, self.tfv.gain

    def get_break_in_point(self, bounds: tuple[float, float]) -> tuple[float, float]:
        """Returns the break-in point that must lie between the specified lower
        and upper bound on the real axis, i.e. the point on the real axis where
        the gain reaches a (local) minimum. Also returns the gain at this point.
        """
        def _objective(sigma: float) -> float:
            self.tfv.point = complex(sigma, 0)
            return self.tfv.gain

        res = minimize_scalar(_objective, bounds=bounds)
        self.tfv.point = complex(res.x, 0)
        return res.x, self.tfv.gain

    def find_jw_crossings(
        self,
        imag_max: float | complex
    ) -> None | tuple[complex, complex]:
        """Searches for a point on the positive imaginary axis where the root
        locus might be crossing.

        Parameters
        ----------
        imag_max:
            The value (magnitude) of the maximum imaginary number that limits
            the search area along the positive imaginary axis (the search starts
            at 0j).

        Returns
        -------
        If a jw-crossing on the positive imaginary axis is found, returns a
        2-tuple with the positive imaginary number where the root locus crosses
        the axis, and -as the root locus is symmetrical about the real axis-
        also the negative of this number.
        If no jw-crossing is found within the specified search area, `None` is
        returned.
        """
        # Select a range of points on the positive imaginary axis.
        imag_min = 0.0
        if isinstance(imag_max, complex):
            imag_max = imag_max.imag
        values = np.linspace(imag_min, imag_max, 100)
        points = [complex(0, value) for value in values]

        # Calculate open-loop transfer function vector angles.
        angles = []
        for point in points:
            self.tfv.point = point
            angles.append(self.tfv.angle)

        if self.positive_feedback:
            # Find index of angle closest to 0°:
            i = np.array([abs(angle) for angle in angles]).argmin()
        else:
            # Find index of angle closest to 180°:
            i = np.array([abs(abs(angle) - 180.0) for angle in angles]).argmin()

        # Narrow the search area to an area where the angle is close to 0° for
        # a positive-feedback system / 180° for a negative-feedback system.
        imag_min = values[i - 1]
        imag_max = values[i + 1]

        # Find exact point on the positive imaginary axis for which the angle of
        # the open-loop transfer function vector equals 0° for a
        # positive-feedback system / 180° for a negative-feedback system, i.e.
        # where the root locus crosses the imaginary axis.
        def _objective(imag: float) -> float:
            self.tfv.point = complex(0, imag)
            angle = abs(self.tfv.angle)
            if self.positive_feedback:
                return angle
            else:
                return angle - 180.0

        try:
            res = root_scalar(_objective, bracket=(imag_min, imag_max))
        except ValueError:
            return None
        else:
            jw_crossing = res.root
            return complex(0, jw_crossing), complex(0, -jw_crossing)

    def _find_angle_of_departure(self, pole_index: int):
        poles = self.poles[:]
        pole_ = poles.pop(pole_index)
        pole_vectors = [
            Vector(
                real=pole_.real - pole.real,
                imag=pole_.imag - pole.imag
            )
            for pole in poles
        ]
        pole_angles = [v.angle for v in pole_vectors]
        zero_vectors = [
            Vector(
                real=pole_.real - zero.real,
                imag=pole_.imag - zero.imag
            )
            for zero in self.zeros
        ]
        zero_angles = [v.angle for v in zero_vectors]

        if self.positive_feedback:
            theta = sum(zero_angles) - sum(pole_angles)
        else:
            theta = -180.0 + sum(zero_angles) - sum(pole_angles)

        if theta < 0.0:
            theta = 360.0 - abs(theta)
        return theta

    def find_angles_of_departure(self) -> list[tuple[complex, float]]:
        """Returns the angles of departure of the root locus at the open-loop
        poles which are complex (i.e. which have an imaginary part different
        from zero).

        Returns
        -------
        List with the complex open-loop poles and their associated angle of
        departure as 2-tuples. In no complex open-loop poles are present an
        empty list is returned.
        """
        complex_poles = [
            (i, pole)
            for i, pole in enumerate(self.poles)
            if pole.imag != 0
        ]
        results = [
            (pole, self._find_angle_of_departure(i))
            for i, pole in complex_poles
        ]
        return results

    def _find_angle_of_arrival(self, zero_index: int):
        zeros = self.zeros[:]
        zero_ = zeros.pop(zero_index)
        zero_vectors = [
            Vector(
                real=zero_.real - zero.real,
                imag=zero_.imag - zero.imag
            )
            for zero in zeros
        ]
        zero_angles = [v.angle for v in zero_vectors]
        pole_vectors = [
            Vector(
                real=zero_.real - pole.real,
                imag=zero_.imag - pole.imag
            )
            for pole in self.poles
        ]
        pole_angles = [v.angle for v in pole_vectors]

        if self.positive_feedback:
            theta = sum(zero_angles) - sum(pole_angles)
        else:
            theta = -180.0 + sum(zero_angles) - sum(pole_angles)

        if theta < 0.0:
            theta = 360.0 - abs(theta)
        return theta

    def find_angles_of_arrival(self) -> list[tuple[complex, float]]:
        """Returns the angles of arrival of the root locus at the open-loop
        zeros which are complex (i.e. which have an imaginary part different
        from zero).

        Returns
        -------
        List with the complex open-loop zeros and their associated angle of
        arrival as 2-tuples. In no complex open-loop zeros are present an
        empty list is returned.
        """
        complex_zeros = [
            (i, zero)
            for i, zero in enumerate(self.zeros)
            if zero.imag != 0
        ]
        results = [
            (zero, self._find_angle_of_arrival(i))
            for i, zero in complex_zeros
        ]
        return results

    def find_single_damping_ratio_crossing(
        self,
        damping_ratio: float,
        r_min: float,
        r_max: float
    ) -> tuple[complex | None, float | None]:
        """Searches for a point along the specified damping ratio line where the
        root locus might be crossing, i.e., where the angle of the open-loop
        transfer function vector equals 180° in case of a negative-feedback
        system or 0° in case of a positive-feedback system.

        Parameters
        ----------
        damping_ratio:
            Damping ratio of the closed-loop feedback system.
        r_min:
            Minimum search radius.
        r_max:
            Maximum search radius.

        Returns
        -------
        If a crossing point is found, returns the coordinates of this point as
        a complex number and the gain K at this point. If a crossing point could
        not be found, `(None, None)` is returned.
        """
        phi = np.pi - np.arccos(damping_ratio)

        def _objective(r: float) -> float:
            self.tfv.point = cmath.rect(r, phi)
            angle = abs(self.tfv.angle)
            if self.positive_feedback:
                return angle
            else:
                return angle - 180.0

        try:
            res = root_scalar(_objective, bracket=(r_min, r_max))
        except ValueError:
            return None, None
        else:
            r_sol = res.root
            p_sol = cmath.rect(r_sol, phi)
            self.tfv.point = p_sol
            K = self.tfv.gain
            return p_sol, K

    def find_damping_ratio_crossings(
        self,
        damping_ratio: float,
        r_min: float = 0.0,
        r_max: float = 20.0,
        r_num: int = 100
    ) -> list[tuple[complex | None, float | None]]:
        """Searches for points along the specified damping ratio line where the
        root locus might be crossing, i.e., where the angle of the open-loop
        transfer function vector equals 180° in case of a negative-feedback
        system or 0° in case of a positive-feedback system..

        Parameters
        ----------
        damping_ratio:
            Damping ratio of the closed-loop feedback system.
        r_min:
            Minimum radius, i.e. the lower bound of the search area.
            Default value is 0.0 (the origin).
        r_max:
            Maximum radius, i.e. the upper bound of the search area.
            Default value is 100.0
        r_num:
            Number of radii between the minimum and maximum radius of the
            search area.

        Returns
        -------
        If crossing points are found, returns a list with the coordinates of
        each crossing point as a complex number and the gain K at this point.
        """
        phi = np.pi - np.arccos(damping_ratio)
        radii = np.linspace(r_min, r_max, r_num)
        points = [cmath.rect(r, phi) for r in radii]

        def _get_angle(point: complex) -> float:
            self.tfv.point = point
            return self.tfv.angle

        angles = list(map(_get_angle, points))
        abs_angles = np.abs(angles)

        crossing_indexes = []
        for i in range(len(abs_angles) - 1):
            if self.positive_feedback:
                c1 = abs_angles[i] < 0.0 < abs_angles[i + 1]
                c2 = abs_angles[i] > 0.0 > abs_angles[i + 1]
            else:
                c1 = abs_angles[i] < 180.0 < abs_angles[i + 1]
                c2 = abs_angles[i] > 180.0 > abs_angles[i + 1]
            if c1 or c2:
                crossing_indexes.append((i, i + 1))

        crossing_bounds = [
            (float(radii[crossing_index[0]]), float(radii[crossing_index[1]]))
            for crossing_index in crossing_indexes
        ]
        solutions = [
            self.find_single_damping_ratio_crossing(damping_ratio, cb[0], cb[1])
            for cb in crossing_bounds
        ]
        # Check for None-solutions and false solutions:
        for i, sol in enumerate(solutions):
            if sol[0] is None:
                solutions.pop(i)
            else:
                self.tfv.point = sol[0]
                a = round(self.tfv.angle)
                if self.positive_feedback:
                    if abs(a) != 0.0:
                        solutions.pop(i)
                else:
                    if abs(a) != 180.0:
                        solutions.pop(i)
        return solutions

    def _get_gain(self, pole: complex) -> float:
        self.tfv.point = pole
        return self.tfv.gain

    def find_closed_loop_poles(
        self,
        gain: float,
        damping_ratio: float = 1.0,
        r_min: float = 0.0,
        r_max: float = 100.0,
        r_num: int = 100
    ) -> list[complex]:
        """Searches for closed-loop poles on the root locus where the specified
        gain and damping ratio are satisfied.

        Parameters
        ----------
        gain:
            Target gain.
        damping_ratio:
            Target damping ratio. Default is 1.0, which means that the
            damping ratio line coincides with the negative real axis.
        r_min:
            Minimum radius, i.e. the lower bound of the search area.
            Default value is 0.0 (the origin).
        r_max:
            Maximum radius, i.e. the upper bound of the search area.
            Default value is 100.0
        r_num:
            Number of radii between the minimum and maximum radius of the
            search area.

        Returns
        -------
        List with closed-loop poles that correspond with the target gain and
        target damping ratio. If no closed-loop poles are found, an empty list
        is returned.
        """
        phi = np.pi - np.arccos(damping_ratio)
        radii = np.linspace(r_min, r_max, r_num)
        points = [cmath.rect(r, phi) for r in radii]

        if self.positive_feedback:
            def _is_closed_loop_pole(point: complex) -> bool:
                self.tfv.point = point
                return self._is_even_multiple_of_2pi(self.tfv.angle)
        else:
            def _is_closed_loop_pole(point: complex) -> bool:
                self.tfv.point = point
                return self._is_odd_multiple_of_pi(self.tfv.angle)

        closed_loop_poles = [
            (i, p)
            for i, p in enumerate(points)
            if _is_closed_loop_pole(p)
        ]
        gains = [
            (i, self._get_gain(p))
            for i, p in closed_loop_poles
        ]
        solutions = []
        for j in range(len(gains) - 1):
            k1, gain1 = gains[j]
            k2, gain2 = gains[j + 1]
            clp = None
            if gain1 < gain < gain2:
                r1, r2 = float(radii[k1]), float(radii[k2])
                clp = self.find_single_closed_loop_pole(gain, damping_ratio, r1, r2)
            elif gain1 > gain > gain2:
                r2, r1 = float(radii[k1]), float(radii[k2])
                clp = self.find_single_closed_loop_pole(gain, damping_ratio, r1, r2)
            elif gain1 == gain:
                clp = closed_loop_poles[j]
            if clp is not None:
                solutions.append(clp)
        return solutions

    def find_single_closed_loop_pole(
        self,
        gain: float,
        damping_ratio: float = 1.0,
        r_min: float = 0.0,
        r_max: float = 100.0
    ) -> complex | None:
        """Searches for a single closed-loop pole on the root locus where the
        specified gain and damping ratio are satisfied.

        Parameters
        ----------
        gain:
            Target gain.
        damping_ratio:
            Target damping ratio. Default is 1.0, which means that the
            damping ratio line coincides with the negative real axis.
        r_min:
            Minimum radius, i.e. the lower bound of the search area.
            Default value is 0.0 (the origin).
        r_max:
            Maximum radius, i.e. the upper bound of the search area.
            Default value is 100.0

        Returns
        -------
        The closed-loop pole if found, else `None`.
        """
        phi = np.pi - np.arccos(damping_ratio)

        def _objective(modulus: float) -> float:
            pole = cmath.rect(modulus, phi)
            gain_ = self._get_gain(pole)
            return gain_ - gain

        try:
            res = root_scalar(_objective, bracket=(r_min, r_max))
        except ValueError:
            return None
        else:
            pole = cmath.rect(res.root, phi)
            return pole

    def plot(
        self,
        real_limits: tuple[float, float, float] = (-10, 10, 1),
        imag_limits: tuple[float, float, float] = (-10, 10, 1),
        **kwargs
    ) -> None:
        """Shows a plot of the root locus.

        Parameters
        ----------
        real_limits:
            The minimum viewable value on the real axis of the complex plane,
            the maximum viewable value, and the step size between two
            successive values.
        imag_limits:
            Same as `real_limits`, but applied to the imaginary axis.
        """
        if self.positive_feedback is False:
            G_ct = ct.zpk(self.zeros, self.poles, 1)
        else:
            G_ct = ct.zpk(self.zeros, self.poles, -1)
        rl_data = ct.root_locus_map(G_ct)
        loci_x, loci_y = zip(*[
            (loc.real, loc.imag)
            for loc in rl_data.loci
        ])
        poles_x, poles_y = zip(*[
            (pole.real, pole.imag)
            for pole in rl_data.poles
        ])
        if rl_data.zeros.any():
            zeros_x, zeros_y = zip(*[
                (zero.real, zero.imag)
                for zero in rl_data.zeros
            ])
        else:
            zeros_x, zeros_y = None, None

        rlp = LineChart(size=kwargs.get('fig_size'), dpi=kwargs.get('dpi'))
        rlp.add_xy_data(
            label='poles',
            x1_values=poles_x,
            y1_values=poles_y,
            style_props={
                'color': 'tab:blue',
                'marker': 'x',
                'markersize': 7,
                'linestyle': 'none'
            }
        )
        if rl_data.zeros.any():
            rlp.add_xy_data(
                label='zeros',
                x1_values=zeros_x,
                y1_values=zeros_y,
                style_props={
                    'color': 'tab:blue',
                    'marker': 'o',
                    'markersize': 7,
                    'linestyle': 'none'
                }
            )
        rlp.add_xy_data(
            label='loci',
            x1_values=loci_x,
            y1_values=loci_y,
            style_props={'color': 'tab:blue'}
        )
        rlp.x1.axes.axvline(0, color='black', linewidth=0.25)
        rlp.x1.axes.set_aspect('equal')
        rlp.x1.scale(real_limits[0], real_limits[1], real_limits[2])
        rlp.y1.axes.axhline(0, color='black', linewidth=0.25)
        rlp.y1.scale(imag_limits[0], imag_limits[1], imag_limits[2])
        rlp.x1.add_title('real')
        rlp.y1.add_title('imaginary')
        rlp.show()


def pole_sensitivity(
    T: TransferFunction,
    pole: complex | float,
    K_value: float
) -> complex:
    """Returns the sensitivity of a closed-loop pole to gain.

    Parameters
    ----------
    T:
        Closed-loop transfer function.
    pole:
        Closed-loop pole of T for which the sensitivity is to be determined.
    K_value:
        The value of the gain which corresponds with the closed-loop pole on
        the root locus.
    """
    den = T.denominator_poly.expr
    s_ = sp.Function('s')(K)  # locally redefine s as function of gain K
    f = den.subs(s, s_)
    der_f = f.diff(K)
    eq = sp.Eq(der_f, 0)
    sol = sp.solve(eq, sp.Derivative(s_, K))
    S_expr = K / s_ * sol[0]
    S = S_expr.evalf(subs={s_: pole, K: K_value})
    return complex(S)