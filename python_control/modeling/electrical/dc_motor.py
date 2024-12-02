from python_control import Quantity
from python_control.core import TransferFunction
from python_control.modeling.mechanical.rotation import RotationalImpedance
from python_control.modeling.electrical import ElectricalImpedance


class DCMotor:
    """Models an armature-controlled dc servomotor."""

    def __init__(
        self,
        Z_Ra: ElectricalImpedance,
        Z_Jm: RotationalImpedance,
        Z_Dm: RotationalImpedance,
        Kb: Quantity,
        Kt: Quantity,
        Z_La: ElectricalImpedance | None = None
    ) -> None:
        """
        Creates a `DCMotor` object.

        Parameters
        ----------
        Z_Ra:
            The armature resistance (= resistor impedance)
        Z_Jm:
            The equivalent inertia at the armature, including both the inertia
            of the armature and the load inertia reflected to the armature.
        Z_Dm:
            The equivalent viscous damping at the armature, including both the
            armature viscous damping and the load viscous damping reflected to
            the armature.
        Kb:
            The back emf constant; its value can be deduced from the
            torque-speed curve as `Ua / omega_nl`, with `Ua` the applied DC
            armature voltage and `omega_nl` the no-load speed, i.e., the angular
            velocity when the torque is zero.
        Kt:
            The motor torque constant; its value can be deduced from the
            torque-speed curve as `Kt / Ra` = `T_stall / Ua`, with `Ra` the
            armature resistance, `T_stall` the stall torque, i.e., the torque
            when the motor speed reaches zero, and `Ua` the applied DC armature
            voltage.
        Z_La: optional
            The armature inductance (= inductor impedance), which is for a DC
            motor usually much smaller compared to the armature resistance.
            By default, `Z_La` is set to `None`, which means it will be ignored.
        """
        self.Z_Ra = Z_Ra
        self.Z_La = Z_La
        self.Z_Jm = Z_Jm
        self.Z_Dm = Z_Dm
        self.Kb = Kb.to_base_units().m
        self.Kt = Kt.to_base_units().m

    def trf_fun_voltage_current(self) -> TransferFunction:
        """
        Returns the transfer function Ia / (Ua - Eb), with Ia the armature
        current, Ua the applied armature DC voltage, and Eb the back-emf.
        """
        Z_Ra = self.Z_Ra.expr
        if self.Z_La is not None:
            Z_La = self.Z_La.expr
            H = TransferFunction(1 / (Z_Ra + Z_La))
        else:
            H = TransferFunction(1 / Z_Ra)
        return H

    def trf_fun_current_torque(self) -> TransferFunction:
        """
        Returns the transfer function Tm / Ia, with Tm the electromechanical
        torque and Ia the armature current.
        """
        H = TransferFunction(self.Kt)
        return H

    def trf_fun_torque_angle(self) -> TransferFunction:
        """
        Returns the transfer function theta_m / Tm, with theta_m the angular
        displacement and Tm the electromechanical torque.
        """
        Z_Jm = self.Z_Jm.expr
        Z_Dm = self.Z_Dm.expr
        H = TransferFunction(1 / (Z_Jm + Z_Dm))
        return H

    def trf_fun_angle_back_emf(self) -> TransferFunction:
        """
        Returns the transfer function Eb / theta_m, with Eb the back-emf and
        theta_m the angular displacement.
        """
        H = TransferFunction(f'{self.Kb} * s')
        return H

    def trf_fun_angle_voltage(self) -> TransferFunction:
        """
        Returns the transfer function theta_m / Ua, with theta_m the angular
        displacement and Ua the applied armature DC voltage.
        """
        H1 = self.trf_fun_voltage_current()  # Ia = H1 * (Ua - Ea)
        H2 = self.trf_fun_current_torque()   # Tm = H2 * Ia
        H3 = self.trf_fun_torque_angle()     # theta_m = H3 * Tm
        H_fwd = H1 * H2 * H3
        H4 = self.trf_fun_angle_back_emf()   # Eb = H4 * theta_m
        H = H_fwd.feedback(H4)
        return H
