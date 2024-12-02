"""
Example 12.6 - Observability via the Observability Matrix

Determine if the system below is observable.
"""
import numpy as np
from python_control import StateSpace

ss = StateSpace(
    A=np.array([[0, 1, 0], [0, 0, 1], [-4, -3, -2]]),
    B=np.array([[0], [0], [1]]),
    C=np.array([[0, 5, 1]])
)

print(ss.observability_matrix)
print(ss.is_observable)
