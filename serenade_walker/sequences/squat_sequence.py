"""
Squat gait sequence.
"""

from serenade_walker.sequences.base_sequences import GaitStep, OneShotSequence


class SquatSequence(OneShotSequence):
    """Squat gait sequence."""

    def _initialize_steps(self):
        """Initialize squat steps."""
        self.steps = [
            GaitStep((-0.02, 0.09, 0.0), (0.02, 0.09, 0.0)),
        ]
