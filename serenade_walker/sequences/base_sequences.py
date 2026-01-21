"""
Base sequence classes for gait definitions.

Provides the GaitStep class and base classes for cycling and one-shot sequences.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from serenade_walker.controller import RobotWalker


class GaitStep:
    """Represents a single gait phase with target positions and joint modifications."""

    def __init__(
        self,
        left_pos: Tuple[float, float, float],
        right_pos: Tuple[float, float, float],
    ):
        """
        Initialize a gait step with target foot positions.

        Args:
            left_pos: Left foot target position (x, y, z)
            right_pos: Right foot target position (x, y, z)
        """
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.modifiers: Dict[str, float] = {}
        self.offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

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

    def __add__(self, other: "GaitStep") -> "GaitStep":
        """
        Add two GaitStep objects together.

        Args:
            other: Another GaitStep to add to this one

        Returns:
            New GaitStep with combined positions, modifiers, and offsets
        """
        # Add positions
        new_left_pos = (
            self.left_pos[0] + other.left_pos[0],
            self.left_pos[1] + other.left_pos[1],
            self.left_pos[2] + other.left_pos[2],
        )
        new_right_pos = (
            self.right_pos[0] + other.right_pos[0],
            self.right_pos[1] + other.right_pos[1],
            self.right_pos[2] + other.right_pos[2],
        )

        # Create new GaitStep
        new_step = GaitStep(new_left_pos, new_right_pos)

        # Merge modifiers (other takes precedence)
        new_step.modifiers = self.modifiers.copy()
        new_step.modifiers.update(other.modifiers)

        # Add offsets
        new_step.offset = (
            self.offset[0] + other.offset[0],
            self.offset[1] + other.offset[1],
            self.offset[2] + other.offset[2],
        )

        return new_step


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
            return GaitStep((-0.02, 0.04, 0.0), (0.02, 0.04, 0.0))

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
        return GaitStep((-0.02, 0.04, 0.0), (0.02, 0.04, 0.0))
