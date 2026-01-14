"""
Gait definitions and step generation.

This module provides the WalkerState enum for different gait types,
the GaitStep class for representing gait step data,
and base classes for gait sequences.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from .controller import RobotWalker


class WalkerState(Enum):
    """Enumeration of different walker states."""

    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    WALK = "walk"
    SQUAT = "squat"
    DEFAULT = "default"


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
        """Set left side forward lean angle."""
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

    def __init__(self, walker: "RobotWalker"):
        """
        Initialize sequence with walker reference.

        Args:
            walker: Reference to the walker instance
        """
        self.walker = walker
        self.init_pose = self.walker.get_camera_pose()
        self.steps: List[GaitStep] = []
        self._initialize_steps()

    @abstractmethod
    def _initialize_steps(self):
        """Initialize the sequence steps. Must be implemented by subclasses."""
        pass

    def get_step(self, phase: int) -> GaitStep:
        """
        Get gait step for the given phase.

        Args:
            phase: Current phase index

        Returns:
            GaitStep instance for the phase
        """
        if phase < len(self.steps):
            return self.steps[phase]
        return self.steps[0]


class TurnLeftSequence(BaseSequence):
    """Turn left gait sequence."""

    def _initialize_steps(self):
        """Initialize turn left steps."""
        self.steps = [
            GaitStep((-0.02, 0.06, 0.0), (0.02, 0.06, 0.0)),
            GaitStep((-0.04, 0.06, -0.02), (0.04, 0.06, 0.02)),
        ]

    def get_step(self, phase: int) -> GaitStep:
        """
        Get gait step for the given phase with Y-axis offset for turn left.

        Args:
            phase: Current phase index

        Returns:
            GaitStep instance for the phase with offset applied
        """
        step = super().get_step(phase)
        # Add Y-axis offset to left foot for turn left
        left_pos = (
            step.left_pos[0] + 0.0,
            step.left_pos[1] + 0.001,
            step.left_pos[2] + 0.0,
        )
        new_step = GaitStep(left_pos, step.right_pos)
        new_step.modifiers = step.modifiers.copy()
        return new_step


class TurnRightSequence(BaseSequence):
    """Turn right gait sequence."""

    def _initialize_steps(self):
        """Initialize turn right steps."""
        self.steps = [
            GaitStep((-0.02, 0.06, 0.0), (0.02, 0.06, 0.0)),
            GaitStep((-0.04, 0.06, 0.02), (0.04, 0.06, -0.02)),
        ]

    def get_step(self, phase: int) -> GaitStep:
        """
        Get gait step for the given phase with Y-axis offset for turn right.

        Args:
            phase: Current phase index

        Returns:
            GaitStep instance for the phase with offset applied
        """
        step = super().get_step(phase)
        # Add Y-axis offset to right foot for turn right
        right_pos = (
            step.right_pos[0] + 0.0,
            step.right_pos[1] + 0.004,
            step.right_pos[2] + 0.0,
        )
        new_step = GaitStep(step.left_pos, right_pos)
        new_step.modifiers = step.modifiers.copy()
        return new_step


class WalkSequence(BaseSequence):
    """Walk gait sequence."""

    def _initialize_steps(self):
        """Initialize walk steps."""
        self.steps = [
            GaitStep((-0.02, 0.06, 0.03), (0.02, 0.06, -0.01))
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(5)
            .set_right_ankle_lean(-3)
            .set_left_arm(-45)
            .set_right_arm(45),
            GaitStep((-0.02, 0.06, 0.03), (0.02, 0.06, -0.01))
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(5)
            .set_left_ankle_lean(-3)
            .set_left_arm(-45)
            .set_right_arm(45),
            GaitStep((-0.02, 0.06, -0.01), (0.02, 0.06, 0.03))
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(-5)
            .set_left_ankle_lean(-3)
            .set_left_arm(45)
            .set_right_arm(-45),
            GaitStep((-0.02, 0.06, -0.01), (0.02, 0.06, 0.03))
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(-5)
            .set_right_ankle_lean(-3)
            .set_left_arm(45)
            .set_right_arm(-45),
        ]

    def get_step(self, phase: int) -> GaitStep:
        """
        Get gait step for the given phase with Y-axis offset for walking.

        Args:
            phase: Current phase index

        Returns:
            GaitStep instance for the phase with offset applied
        """
        step = super().get_step(phase)
        right_spin = 0.0

        # Parameters (Tune these!)
        # K_yaw: How hard to correct for angle errors (Start small: 0.05 to 0.1)
        # K_lat: How hard to correct for lateral drift (Start small: 0.1 to 0.3)
        MAX_CORRECTION = 0.005  # Meters
        K_yaw = 4 * MAX_CORRECTION
        K_lat = 20 * MAX_CORRECTION

        if self.init_pose and self.walker.camera_pose_client:
            current_pose_data = self.walker.get_camera_pose()

            # Ensure we have data and it's not the exact same millisecond as init
            if current_pose_data:
                # 1. Convert Data to Scipy Format (x, y, z, w)
                # init_pose is CameraPoseData
                q_init = [
                    self.init_pose.orientation.x,
                    self.init_pose.orientation.y,
                    self.init_pose.orientation.z,
                    self.init_pose.orientation.w,
                ]
                p_init = np.array(
                    [
                        self.init_pose.position.x,
                        self.init_pose.position.y,
                        self.init_pose.position.z,
                    ]
                )

                # current_pose_data is (data, timestamp) tuple from controller.py check?
                # looking at controller.py, get_camera_pose returns CameraPoseData directly?
                # wait, controller.py says: return self.camera_pose_client.get_latest_pose()
                # Let's assume it returns the object directly based on type hinting.

                curr = current_pose_data
                q_curr = [
                    curr.orientation.x,
                    curr.orientation.y,
                    curr.orientation.z,
                    curr.orientation.w,
                ]
                p_curr = np.array([curr.position.x, curr.position.y, curr.position.z])

                # 2. Create Rotation Objects
                r_init = R.from_quat(q_init)
                r_curr = R.from_quat(q_curr)

                # 3. Calculate Relative Transform (World -> Local Error)
                # We want the current position/rotation relative to the Init frame
                r_rel = r_init.inv() * r_curr
                p_rel = r_init.inv().apply(p_curr - p_init)

                # 4. Extract Errors
                # Heading Error: Yaw angle (rotation around Y-axis)
                # as_euler returns (x, y, z) ordering. index 1 is Y (yaw).
                yaw_error = r_rel.as_euler("xyz", degrees=False)[1]

                # Lateral Error: Displacement in X axis
                lat_error = p_rel[0]

                # 5. Calculate Control Output (PD Controller - P term only here)
                # If yaw is positive (turned left), we need right_spin positive.
                # If lat is positive (drifted left), we need right_spin positive.
                right_spin = (K_yaw * yaw_error) + (K_lat * lat_error)

                # Clip limits to prevent falling
                right_spin = max(min(right_spin, MAX_CORRECTION), -MAX_CORRECTION)
                print(
                    f"Yaw Error: {yaw_error:.4f}, Lateral Error: {lat_error:.4f}, Right Spin: {right_spin:.4f}"
                )

        if phase == 0 or phase == 3:
            # Reducing the addition to `y` increases the friction between robot left foot
            # and ground, so that robot turns left
            left_pos = (
                step.left_pos[0] + 0.0,
                step.left_pos[1] + 0.005 + min(right_spin, 0),
                step.left_pos[2] + 0.0,
            )
            new_step = GaitStep(left_pos, step.right_pos)
            new_step.modifiers = step.modifiers.copy()
            return new_step
        else:
            right_pos = (
                step.right_pos[0] + 0.0,
                step.right_pos[1] + 0.005 - max(right_spin, 0),
                step.right_pos[2] + 0.0,
            )
            new_step = GaitStep(step.left_pos, right_pos)
            new_step.modifiers = step.modifiers.copy()
            return new_step


class SquatSequence(BaseSequence):
    """Squat gait sequence."""

    def _initialize_steps(self):
        """Initialize squat steps."""
        self.steps = [
            GaitStep((-0.02, 0.09, 0.0), (0.02, 0.09, 0.0)),
        ]


class DefaultSequence(BaseSequence):
    """Default (standing) gait sequence."""

    def _initialize_steps(self):
        """Initialize default steps."""
        self.steps = [
            GaitStep((-0.02, 0.04, 0.0), (0.02, 0.04, 0.0)),
        ]
