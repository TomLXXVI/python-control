"""
Example 12.1 - Controller Design for Phase-Variable Form.

Given the plant ``G = TransferFunction(20 * (s + 5) / (s * (s + 1) * (s + 4)))``,
design the phase-variable feedback gains to yield 9.5% overshoot and a settling
time of 0.74 second.
"""
from python_control import s, TransferFunction, get_damping_ratio, StateSpace
from python_control.design import get_dominant_poles
from python_control.design.state_space import design_controller


def main():
    G = TransferFunction(20 * (s + 5) / (s * (s + 1) * (s + 4)))
    ss_G = StateSpace.from_transfer_function(G)

    # Get the dominant poles from transient response requirements.
    dominant_poles = get_dominant_poles(
        settling_time=0.74,
        damping_ratio=get_damping_ratio(9.5)
    )
    # Since the system is third-order, we must select another closed-loop pole.
    # The closed-loop system will have a zero at -5, the same as the open-loop
    # system. We could select the third closed-loop pole to cancel the
    # closed-loop zero.
    closed_loop_poles = [*dominant_poles, -5.0]

    # Design the closed-loop system with state-variable feedback (each state
    # variable `x` is fed back to the input `u` of the system).
    ss_T = design_controller(ss_G, closed_loop_poles)
    print(ss_T.A)

    # Convert the state-space representation into a transfer function.
    T = TransferFunction.from_state_space(ss_T)
    print(T)

    # Plot the unit step response of the closed-loop system with state-variable
    # feedback.
    T.plot_unit_step_response(title_on=False, upper_limit=2.0)


if __name__ == '__main__':
    main()
