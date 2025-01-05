"""
Example 12.2 - Controllability via the Controllability Matrix

Given the system below, determine its controllability.
"""
import numpy as np
from python_control import StateSpace

ss = StateSpace(
    A=np.array([[-1, 1, 0], [0, -1, 0], [0, 0, -2]]),
    B=np.array([[0], [1], [1]])
)

print(ss.controllability_matrix)
print(ss.is_controllable)
