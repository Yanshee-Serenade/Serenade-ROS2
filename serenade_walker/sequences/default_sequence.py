"""
Default (standing) gait sequence.
"""

from serenade_walker.sequences.base_sequences import GaitStep, OneShotSequence


class DefaultSequence(OneShotSequence):
    """Default (standing) gait sequence."""

    def _initialize_steps(self):
        """Initialize default steps."""
        self.steps = [
            GaitStep(),
        ]
