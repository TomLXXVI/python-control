"""
Example 12.10 - Design of Integral Control.

Consider the plant:
```
ss_G = StateSpace(
    A=np.array([[0, 1], [-3, -5]]),
    B=np.array([[0], [1]]),
    C=np.array([[1, 0]])
)
```
Design a controller using integral control to yield 10% overshoot, a settling
time of 0.5 second, and zero steady-state error for a unit-step input.
"""
import numpy as np
import sympy as sp
from python_control import (
    StateSpace,
    get_damping_ratio,
    TransferFunction,
    t,
    LineChart
)
from python_control.design import get_dominant_poles
from python_control.design.state_space.controller import design_controller


ss_G = StateSpace(
    A=np.array([[0, 1], [-3, -5]]),
    B=np.array([[0], [1]]),
    C=np.array([[1, 0]])
)

# The system matrix is a 2x2-matrix, so the system is a second-order system.

# DESIRED CHARACTERISTIC EQUATION OF THE CLOSED-LOOP SYSTEM
# ---------------------------------------------------------
# From the transient response requirements determine the required dominant poles:
zeta = get_damping_ratio(percent_overshoot=10)
dominant_poles = get_dominant_poles(settling_time=0.5, damping_ratio=zeta)

# Adding an integrator to reduce the steady-state error to zero, implies that
# also a third pole must be added. This pole is placed on the negative real axis
# in the left-half of the s-plane, and needs to be more than 5 times farther
# than the real part of the dominant poles in order that the system would
# approximate a second-order system. Here, the third pole is placed at -100.
poles = [*dominant_poles, -100]

# DESIGN CLOSED-LOOP INTEGRAL-CONTROLLED SYSTEM
# ---------------------------------------------
# In example 12.1 the controller was designed only taking the transient response
# requirements into account. To add integral control to the controller, we call
# the function `design_controller` with parameter `integral_control` set to
# `True`.
ss_T = design_controller(ss_G, poles, integral_control=True)
print(ss_T.A)

T = TransferFunction.from_state_space(ss_T)
print(T)

# UNIT STEP RESPONSE
# ------------------
# Solve the state-space representation of the controlled closed-loop system for
# its response to a unit-step input.
time_sol = ss_T.solve([sp.Heaviside(t)])
y = time_sol.output[0]

# Draw the unit-step response on a line chart.
time_axis = np.linspace(0, 2, 1000)
y_axis = y.evaluate(time_axis)
c = LineChart()
c.add_xy_data(
    label='',
    x1_values=time_axis,
    y1_values=y_axis
)
c.x1.add_title('time, s')
c.y1.add_title('unit step response')
c.show()
