"""
Example 12.5 - Observer Design for Observer Canonical Form

Determine the observer gain vector for the plant:
``G = TransferFunction((s + 4) / ((s + 1) * (s + 2) * (s + 5)))``
The observer should respond 10 times faster than the controlled loop designed
in example 12.4.

Notes
-----
An observer, being a model of the plant, estimates the values of the state
variables based on the input signals `u` and also the output signals `y` of the
system, which are fed back to the observer.
This implies that an observer has two input sides: on one side the input signals,
and on the other side the output signals of the system to be controlled. This
means that the observer is in fact a kind of MIMO system. MIMO systems are not
treated in the book "Control Systems Engineering" by Norman S. Nise. The book
only explains how to determine the required gains of the observer system.
"""
from python_control import s, TransferFunction, StateSpace, get_damping_ratio
from python_control.design import get_dominant_poles
from python_control.design.state_space.observer import solve_observer_gain_vector


G = TransferFunction((s + 4) / ((s + 1) * (s + 2) * (s + 5)))
ss_G = StateSpace.from_transfer_function(G)

# Determine the dominant poles of the observer from the transient response
# requirements of the closed-loop system:
dominant_poles = get_dominant_poles(
    settling_time=4.0,
    damping_ratio=get_damping_ratio(20.8)
)

# We want our observer to be 10 times faster than the controlled closed-loop
# system. As the system has order 3, we need 3 closed-loop poles for our
# observer.
poles = [10 * pole for pole in dominant_poles]
# We select the third pole to be 10 times the real part of the dominant
# second-order poles.
poles.append(10 * poles[0].real)

# Determine the required observer gains in observer canonical form:
L_vector = solve_observer_gain_vector(ss_G, poles, transform=False)
print(L_vector)
