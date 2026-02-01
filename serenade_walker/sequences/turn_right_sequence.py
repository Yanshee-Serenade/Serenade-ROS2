"""
Turn right gait sequence.
"""

from serenade_walker.sequences.base_sequences import CyclingSequence, GaitStep


class TurnRightSequence(CyclingSequence):
    """Turn right gait sequence."""

    def _initialize_steps(self):
        """Initialize turn right steps."""
        self.steps = [
            GaitStep((-0.02, 0.06, 0.0), (0.02, 0.06, 0.0)),
            GaitStep((-0.04, 0.06, 0.02), (0.04, 0.06, -0.02)),
        ]

    def get_step(self, step_index: int) -> GaitStep:
        """
        Get gait step for the given step index with Y-axis offset for turn right.

        Args:
            step_index: Current step index (0-based)

        Returns:
            GaitStep instance for the step with offset applied
        """
        step = super().get_step(step_index)

        # Add Y-axis offset to right foot for turn right
        right_pos = (
            step.right_pos[0] + 0.0,
            step.right_pos[1] + 0.004,
            step.right_pos[2] + 0.0,
        )
        new_step = step.copy()
        new_step.right_pos = right_pos
        return new_step
