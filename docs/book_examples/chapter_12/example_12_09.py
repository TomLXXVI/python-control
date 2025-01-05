"""
Example 12.9 - Observer Design by Matching Coefficients

Given the plant transfer function:
``G = TransferFunction(407 * (s + 0.916) / ((s + 1.27) * (s + 2.69)))``
design an observer for the phase variables with a transient response described
by damping ratio = 0.7 and natural frequency = 100.
"""
from python_control import s, TransferFunction, StateSpace, ClosedLoopTransientResponse
from python_control.design import get_dominant_poles
from python_control.design.state_space.observer import solve_observer_gain_vector


G = TransferFunction(407 * (s + 0.916) / ((s + 1.27) * (s + 2.69)))

# Create state-space representation of G and put it in phase-variable form:
ss_G = StateSpace.from_transfer_function(G)
ss_G = ss_G.transform('phase-variable')

print(f"system matrix:\n{ss_G.A}")
print(f"output matrix:\n{ss_G.C}")
print(f"system observable? {ss_G.is_observable}")

# To get at the dominant poles that will result in the required transient
# response, we can first use class `ClosedTransientResponse` to find the peak
# time (or settling time) that goes with the specified natural frequency. Next,
# we can use function `get_dominant_poles`.
CLTR_obj = ClosedLoopTransientResponse.solve(zeta=0.7, omega_nat=100)
poles = get_dominant_poles(
    peak_time=CLTR_obj.T_p,
    damping_ratio=CLTR_obj.zeta
)

# By default the observer gain vector is returned in the same state-space
# representation form as the original system ss_G; in this case in
# phase-variable form.
L_vector = solve_observer_gain_vector(ss_G, poles)
print(L_vector)
