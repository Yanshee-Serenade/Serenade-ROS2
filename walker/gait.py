"""
Gait definitions and step generation.

This module provides the WalkerState enum for different gait types
and the GaitStep class for representing individual gait phases.
"""

from enum import Enum
from typing import List, Tuple

from .kinematics import KinematicsSolver


class WalkerState(Enum):
    """Enumeration of different walker states."""

    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    WALK = "walk"
    SQUAT = "squat"
    DEFAULT = "default"


class GaitStep:
    """Represents a single gait phase with all joint angles."""

    def __init__(
        self,
        solver: KinematicsSolver,
        grid_size: int,
        left_pos: Tuple[float, float, float],
        right_pos: Tuple[float, float, float],
    ):
        """
        Initialize a gait step based on foot positions.

        Note: Performs y-z coordinate swap.
        Input coordinate format: (x, y, z) -> Output coordinate: (x, z, y)

        Args:
            solver: Kinematics solver instance
            grid_size: Grid search density for IK
            left_pos: Left foot position (x, y, z)
            right_pos: Right foot position (x, y, z)
        """
        self.joint_angles = [0.0] * 17

        # Coordinate conversion: swap y and z
        left_pos_converted = (left_pos[0], left_pos[2], left_pos[1])  # (x, z, y)
        right_pos_converted = (right_pos[0], right_pos[2], right_pos[1])  # (x, z, y)

        # Solve left leg inverse kinematics
        left_result = solver.solve_leg_ik(False, left_pos_converted, grid_size)
        if left_result:
            left_angles, _ = left_result
            for i in range(5):
                self.joint_angles[11 + i] = left_angles[i]  # Left leg indices 11-15

        # Solve right leg inverse kinematics
        right_result = solver.solve_leg_ik(True, right_pos_converted, grid_size)
        if right_result:
            right_angles, _ = right_result
            for i in range(5):
                self.joint_angles[6 + i] = right_angles[i]  # Right leg indices 6-10

        # Set default arm angles
        self.joint_angles[0] = 90.0  # Right shoulder yaw
        self.joint_angles[1] = 165.0  # Right shoulder pitch
        self.joint_angles[2] = 100.0  # Right elbow
        self.joint_angles[3] = 90.0  # Left shoulder yaw
        self.joint_angles[4] = 15.0  # Left shoulder pitch
        self.joint_angles[5] = 80.0  # Left elbow

        # Set default neck angle
        self.joint_angles[16] = 90.0

    def set_right_arm(self, angle: float) -> "GaitStep":
        """Set right arm forward lift angle."""
        self.joint_angles[0] -= angle  # Right shoulder yaw
        return self

    def set_left_arm(self, angle: float) -> "GaitStep":
        """Set left arm forward lift angle."""
        self.joint_angles[3] += angle  # Left shoulder yaw
        return self

    def set_right_lean(self, angle: float) -> "GaitStep":
        """Set right side forward lean angle."""
        self.joint_angles[7] -= angle  # Right leg hip lateral swing
        return self

    def set_left_lean(self, angle: float) -> "GaitStep":
        """Set left side forward lean angle."""
        self.joint_angles[12] += angle  # Left leg hip lateral swing
        return self

    def set_right_ankle_lean(self, angle: float) -> "GaitStep":
        """Set right ankle forward lean angle."""
        self.joint_angles[9] += angle
        return self

    def set_left_ankle_lean(self, angle: float) -> "GaitStep":
        """Set left ankle forward lean angle."""
        self.joint_angles[14] -= angle
        return self

    def set_neck(self, angle: float) -> "GaitStep":
        """Set neck rotation angle (right rotation)."""
        self.joint_angles[16] += angle
        return self

    @property
    def angles(self) -> List[float]:
        """Get a copy of the joint angles."""
        return self.joint_angles.copy()
