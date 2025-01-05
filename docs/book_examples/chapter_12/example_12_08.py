"""
Example 12.8 - Observer Design by Transformation.

Determine the observer gain vector for the plant:
```
ss_G = StateSpace(
    A=np.array([[-5, 1, 0], [0, -2, 1], [0, 0, -1]]),
    B=np.array([[0], [0], [1]]),
    C=np.array([[1, 0, 0]])
)
```
"""
import numpy as np
from python_control import (
    StateSpace,
    get_damping_ratio
)
from python_control.design import get_dominant_poles
from python_control.design.state_space.observer import solve_observer_gain_vector


ss_G = StateSpace(
    A=np.array([[-5, 1, 0], [0, -2, 1], [0, 0, -1]]),
    B=np.array([[0], [0], [1]]),
    C=np.array([[1, 0, 0]])
)

# Determine the dominant poles of the observer based on the transient response
# requirements of system G:
dominant_poles = get_dominant_poles(
    settling_time=4.0,
    damping_ratio=get_damping_ratio(20.8)
)

# We want our observer to be 10 times faster than the controlled closed-loop
# system. As the system has order 3, we also need 3 closed-loop poles.
poles = [10 * pole for pole in dominant_poles]
poles.append(10 * poles[0].real)

# Get the observer gains in the observer canonical form:
L_vec_observ = solve_observer_gain_vector(ss_G, poles, transform=False)
print(L_vec_observ)

# Get the observer gains in the original state-space representation:
L_vector_orig = solve_observer_gain_vector(ss_G, poles, transform=True)
print(L_vector_orig)
