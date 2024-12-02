"""
Example 12.3 - Controller Design by Matching Coefficients

Given a plant ``G = TransferFunction(10 / ((s + 1) * (s + 2)))``, design state
feedback for the plant to yield 15% overshoot with a settling time of 0.5
second.

--> Analogous to example 12.1. In this example the system is of second order, so
it suffices to determine only the two dominant poles from the transient response
requirements.
"""
from python_control import s, TransferFunction, StateSpace, get_damping_ratio
from python_control.design import get_dominant_poles
from python_control.design.state_space import design_controller


G = TransferFunction(10 / ((s + 1) * (s + 2)))
ss_G = StateSpace.from_transfer_function(G)


dominant_poles = get_dominant_poles(
    settling_time=0.5,
    damping_ratio=get_damping_ratio(15)
)

ss_T = design_controller(ss_G, dominant_poles)
print(ss_T.A)

T = TransferFunction.from_state_space(ss_T)
print(T)
