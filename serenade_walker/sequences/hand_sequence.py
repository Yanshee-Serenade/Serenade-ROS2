"""
Move hand gait sequence.
"""

from serenade_walker.sequences.base_sequences import CyclingSequence, GaitStep


class HandSequence(CyclingSequence):
    """Turn left gait sequence."""

    def _initialize_steps(self):
        """Initialize turn left steps."""
        # self.steps = [
        #     GaitStep(right_arm=(0.1, 0.22, 0.01)),
        #     GaitStep(right_arm=(0.1, 0.19, 0.01)),
        #     GaitStep(right_arm=(0.13, 0.19, 0.01)),
        #     GaitStep(right_arm=(0.13, 0.22, 0.01)),
        # ]
        # self.steps = [
        #     GaitStep(right_arm=(0.05, 0.18, 0.10)),
        #     GaitStep(right_arm=(0.10, 0.28, 0.10)),
        #     GaitStep(right_arm=(0.10, 0.18, 0.10)),
        #     GaitStep(right_arm=(0.05, 0.28, 0.10)),
        # ]
        # self.steps = [
        #     GaitStep(left_arm=(-0.02, 0.25, 0.10), right_arm=(0.02, 0.25, 0.10)),
        #     GaitStep(left_arm=(-0.02, 0.25, 0.10), right_arm=(0.02, 0.25, 0.10)),
        #     GaitStep(left_arm=(-0.06, 0.25, 0.10), right_arm=(0.06, 0.25, 0.10)),
        #     GaitStep(left_arm=(-0.06, 0.25, 0.10), right_arm=(0.06, 0.25, 0.10)),
        # ]
        self.steps = [
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.06, 0.23, 0.10), (0.06, 0.23, 0.10)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.06, 0.23, 0.10), (0.06, 0.23, 0.10)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.02, 0.23, 0.10), (0.02, 0.23, 0.10)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.02, 0.23, 0.10), (0.02, 0.23, 0.10)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.02, 0.25, 0.07), (0.02, 0.25, 0.07)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.02, 0.25, 0.07), (0.02, 0.25, 0.07)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.06, 0.25, 0.07), (0.06, 0.25, 0.07)).set_left_lean(30).set_right_lean(30),
            GaitStep((-0.02, 0.04, 0.02), (0.02, 0.04, 0.02), (-0.06, 0.25, 0.07), (0.06, 0.25, 0.07)).set_left_lean(30).set_right_lean(30),
        ]
