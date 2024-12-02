"""
Implements mesh analysis of an electrical network based on
"Control Systems Engineering, EMEA Edition, 8th Edition", § 2.4 "Electrical
Network Transfer Functions" - "Complex Circuits via Mesh Analysis".

The same mesh analysis can also be used with mechanical systems in which case a
mesh corresponds with a body in the mechanical system (see "Control Systems
Engineering", § 2.9). A force counteracting the movement of a body is
represented by the product of a mechanical impedance and the velocity of the
body (F(s) = Z * V(s)). Impedances between two bodies are also common to two
meshes. External forces applied to a body can be represented as a voltage
source in the mesh of the body. The mesh current represents the velocity of the
body.
"""
import sympy as sp
from python_control.core import TransferFunction
from python_control.modeling.mechanical import MechanicalComponent
from .components import ElectricalComponent


class Mesh(list):

    def __init__(self, name: str, *args, **kwargs):
        """
        Create a mesh with identifier `name`.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self._current = sp.Symbol(f'I_{name}')
        self._velocity = sp.Symbol(f'V_{name}')
        self.voltage_sources = []

    # def add_component(self, component: ElectricalComponent | MechanicalComponent):
    #     """
    #     Add an electric or mechanical component to the mesh.
    #     """
    #     if isinstance(component, ElectricalComponent):
    #         self.append(component)
    #     else:
    #         # if `component` is a mechanical component (translational or
    #         # torsional), we replace the impedance Z of the component, which
    #         # is defined with reference to displacement (Z = F(s)/X(s)), by
    #         # Z/s, which is the velocity impedance or the transfer function of
    #         # force F(s) to velocity V(s).
    #         component.Z = component.Z_v
    #         self.append(component)

    def add_component(self, component: ElectricalComponent | MechanicalComponent):
        """
        Add an electric or mechanical component to the mesh.
        """
        self.append(component)

    def add_voltage_source(self, v: sp.Expr):
        """
        Add a voltage source to the mesh.

        Notes
        -----
        A voltage source in electricity is equivalent to an external active
        force applied to a body in mechanics.
        """
        self.voltage_sources.append(v)

    @property
    def impedance_objects(self) -> set[TransferFunction]:
        """
        Returns a set of the impedance objects in the mesh.
        """
        if isinstance(self[0], ElectricalComponent):
            return set([component.Z for component in self])
        else:
            # MechanicalComponent -> velocity impedance
            return set([component.Z_v for component in self])
        # Note: using a Python `set`, we can easily get the impedances which
        # are common to two meshes using the `intersection` method of `set`.

    @property
    def current(self) -> sp.Symbol:
        """
        Returns the mesh current or the velocity in the case of an electrical
        analog of a mechanical system.
        """
        if isinstance(self[0], ElectricalComponent):
            return self._current
        else:
            return self._velocity


class Circuit(list[Mesh]):

    def __init__(self, name: str | None = None, *args, **kwargs):
        """
        Creates an empty circuit (`Circuit` object).
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.equations = []

    def add_mesh(self, mesh: Mesh):
        """
        Add a mesh (`Mesh` object) to the circuit.
        """
        self.append(mesh)

    def _analyze_mesh(self, index: int):
        # Get a temporary list of the meshes in the network except the mesh
        # under consideration, i.e., the mesh with index `index`:
        temp_list = [self[i] for i in range(len(self)) if i != index]

        present_mesh = self[index]
        terms = [
            sum(
                Z.expr
                for Z in present_mesh.impedance_objects
            ) * present_mesh.current
        ]
        for mesh in temp_list:
            # Get the impedance objects (`TransferFunction` objects) of each
            # mesh which are common to the mesh currently under consideration:
            common_imp_objs = present_mesh.impedance_objects.intersection(mesh.impedance_objects)
            if common_imp_objs:
                Z_common = sum(Z.expr for Z in common_imp_objs)
            else:
                Z_common = 0
            term = -Z_common * mesh.current
            terms.append(term)
        # Build the loop equation of the current mesh:
        lhs = sum(terms)
        rhs = sum(present_mesh.voltage_sources)
        eq = sp.Eq(lhs, rhs)
        # Add the loop equation to the loop equations of the circuit:
        self.equations.append(eq)

    def solve(self) -> dict[str, sp.Expr]:
        """
        Solves the circuit equations algebraically for the loop currents in each
        mesh.

        Returns
        -------
        A dictionary with for each loop current in the circuit its solution as
        a Sympy expression.
        """
        for i in range(len(self)):
            self._analyze_mesh(i)
        mesh_currents = (mesh.current for mesh in self)
        sol = sp.solve(self.equations, *mesh_currents)
        return sol
