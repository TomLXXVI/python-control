"""
Example 12.4 - Controller Design by Transformation.

Design a state-variable feedback controller to yield a 20.8% overshoot and a
settling time of 4 seconds for a plant:
``G = TransferFunction((s + 4) / ((s + 1) * (s + 2) * (s + 5)))``

--> Analogous to example 12.1
"""
from python_control import s, TransferFunction, StateSpace, get_damping_ratio
from python_control.design import get_dominant_poles
from python_control.design.state_space import design_controller


G = TransferFunction((s + 4) / ((s + 1) * (s + 2) * (s + 5)))
ss_G = StateSpace.from_transfer_function(G)

# Get dominant poles from transient response requirements:
dominant_poles = get_dominant_poles(
    settling_time=4.0,
    damping_ratio=get_damping_ratio(20.8)
)

# We need a third closed-loop pole as G is a third-order system. We take -4 for
# the third pole, i.e. the zero of G, to cancel the closed-loop zero.
ss_T = design_controller(
    ss_G,
    closed_loop_poles=[*dominant_poles, -4.0]
)

T = TransferFunction.from_state_space(ss_T)
print(T)
