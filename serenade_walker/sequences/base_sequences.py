"""
Base sequence classes for gait definitions.

Provides the GaitStep class and base classes for cycling and one-shot sequences.
"""

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from serenade_walker.controller import RobotWalker


class GaitStep:
    """Represents a single gait phase with target positions and joint modifications."""

    def __init__(
        self,
        left_pos = (-0.02, 0.04, 0.0),
        right_pos = (0.02, 0.04, 0.0),
        left_arm = (-0.1, 0.17, 0.01),
        right_arm = (0.1, 0.17, 0.01),
    ):
        """
        Initialize a gait step with target foot positions.

        Args:
            left_pos: Left foot target position (x, y, z)
            right_pos: Right foot target position (x, y, z)
        """
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.modifiers: Dict[str, float] = {}
    
    def copy(self) -> "GaitStep":
        """Convenience method to create a deep copy."""
        return copy.deepcopy(self)

    def set_right_arm(self, angle: float) -> "GaitStep":
        """Set right arm forward lift angle."""
        self.modifiers["right_arm"] = angle
        return self

    def set_left_arm(self, angle: float) -> "GaitStep":
        """Set left arm forward lift angle."""
        self.modifiers["left_arm"] = angle
        return self

    def set_right_lean(self, angle: float) -> "GaitStep":
        """Set right side forward lean angle."""
        self.modifiers["right_lean"] = angle
        return self

    def set_left_lean(self, angle: float) -> "GaitStep":
        """Set left side forward lift angle."""
        self.modifiers["left_lean"] = angle
        return self

    def set_right_ankle_lean(self, angle: float) -> "GaitStep":
        """Set right ankle forward lean angle."""
        self.modifiers["right_ankle_lean"] = angle
        return self

    def set_left_ankle_lean(self, angle: float) -> "GaitStep":
        """Set left ankle forward lean angle."""
        self.modifiers["left_ankle_lean"] = angle
        return self

    def set_neck(self, angle: float) -> "GaitStep":
        """Set neck rotation angle (right rotation)."""
        self.modifiers["neck"] = angle
        return self


class BaseSequence(ABC):
    """Base class for all gait sequences."""

    def attach_walker(self, walker: "RobotWalker"):
        """
        Initialize sequence with walker reference.

        Args:
            walker: Reference to the walker instance
        """
        self.walker = walker
        self.init_pose = self.walker.camera_pose
        self.steps: List[GaitStep] = []
        self._initialize_steps()

    @abstractmethod
    def _initialize_steps(self):
        """Initialize the sequence steps. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_step(self, step_index: int) -> GaitStep:
        """
        Get gait step for the given step index.

        Args:
            step_index: Current step index (0-based)

        Returns:
            GaitStep instance for the step, or None if sequence should stop
        """
        pass


class CyclingSequence(BaseSequence):
    """Base class for cycling sequences that never return None."""

    def get_step(self, step_index: int) -> GaitStep:
        """
        Get gait step for the given step index, cycling through steps.

        Args:
            step_index: Current step index (0-based)

        Returns:
            GaitStep instance for the step (never returns None)
        """
        if not self.steps:
            return GaitStep()

        # Cycle through steps using modulo
        phase = step_index % len(self.steps)
        return self.steps[phase]


class OneShotSequence(BaseSequence):
    """Base class for one-shot sequences that return None after completion."""

    def get_step(self, step_index: int) -> GaitStep:
        """
        Get gait step for the given step index, returns None after last step.

        Args:
            step_index: Current step index (0-based)

        Returns:
            GaitStep instance for the step, or None if step_index >= len(steps)
        """
        if step_index < len(self.steps):
            return self.steps[step_index]
        return GaitStep()
