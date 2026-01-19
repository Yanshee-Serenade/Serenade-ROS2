"""
Robot walker controller.

This module provides the RobotWalker class which manages gait sequences,
state transitions, and robot motion control.
"""

import time
from typing import List, Optional

from .gait import (
    BaseSequence,
    DefaultSequence,
    GaitStep,
)
from .kinematics import KinematicsSolver


class RobotWalker:
    """Main robot walker controller for managing gait sequences and motion."""

    def __init__(
        self,
        solver: KinematicsSolver,
        client,
        grid_size: int = 12,
        period_ms: int = 200,
    ):
        """
        Initialize robot walker controller.

        Args:
            solver: Inverse kinematics solver
            client: Robot client (e.g., JointAngleTCPClient)
            grid_size: Grid search density for IK
            period_ms: Action period in milliseconds
        """
        self.solver = solver
        self.client = client
        self.grid_size = grid_size
        self.period_ms = period_ms

        # State control
        self.current_phase = 0
        self.current_sequence = DefaultSequence()

        # Timing related
        self.last_action_time = time.time() * 1000  # Convert to milliseconds
        self.start_time = time.time()
        self.running = False

        # Camera pose state
        self._camera_pose = None

    def reset(self):
        """Reset to initial state."""
        self.current_phase = 0
        self.last_action_time = time.time() * 1000
        # Run default sequence once to reset position
        self.run_sequence(DefaultSequence())

    def _calculate_angles(self, step: GaitStep) -> List[float]:
        """
        Calculate joint angles from gait step data.

        Args:
            step: GaitStep instance with target positions and modifiers

        Returns:
            List of joint angles
        """
        joint_angles = [0.0] * 17

        # Coordinate conversion: swap y and z for IK solver
        left_pos_converted = (
            step.left_pos[0],
            step.left_pos[2],
            step.left_pos[1],
        )  # (x, z, y)

        right_pos_converted = (
            step.right_pos[0],
            step.right_pos[2],
            step.right_pos[1],
        )  # (x, z, y)

        # Solve left leg inverse kinematics
        left_result = self.solver.solve_leg_ik(
            False, left_pos_converted, self.grid_size
        )
        if left_result:
            left_angles, _ = left_result
            for i in range(5):
                joint_angles[11 + i] = left_angles[i]  # Left leg indices 11-15

        # Solve right leg inverse kinematics
        right_result = self.solver.solve_leg_ik(
            True, right_pos_converted, self.grid_size
        )
        if right_result:
            right_angles, _ = right_result
            for i in range(5):
                joint_angles[6 + i] = right_angles[i]  # Right leg indices 6-10

        # Set default arm angles
        joint_angles[0] = 90.0  # Right shoulder yaw
        joint_angles[1] = 165.0  # Right shoulder pitch
        joint_angles[2] = 100.0  # Right elbow
        joint_angles[3] = 90.0  # Left shoulder yaw
        joint_angles[4] = 15.0  # Left shoulder pitch
        joint_angles[5] = 80.0  # Left elbow

        # Set default neck angle
        joint_angles[16] = 90.0

        # Apply modifiers
        if "right_arm" in step.modifiers:
            joint_angles[0] -= step.modifiers["right_arm"]  # Right shoulder yaw
        if "left_arm" in step.modifiers:
            joint_angles[3] += step.modifiers["left_arm"]  # Left shoulder yaw
        if "right_lean" in step.modifiers:
            joint_angles[7] -= step.modifiers[
                "right_lean"
            ]  # Right leg hip lateral swing
        if "left_lean" in step.modifiers:
            joint_angles[12] += step.modifiers[
                "left_lean"
            ]  # Left leg hip lateral swing
        if "right_ankle_lean" in step.modifiers:
            joint_angles[9] += step.modifiers["right_ankle_lean"]
        if "left_ankle_lean" in step.modifiers:
            joint_angles[14] -= step.modifiers["left_ankle_lean"]
        if "neck" in step.modifiers:
            joint_angles[16] += step.modifiers["neck"]

        return joint_angles

    def _apply_angles(self, angles):
        """Send angles to robot client."""
        try:
            success, msg = self.client.set_joint_angles(angles, time_ms=self.period_ms)
            if not success:
                print(f"发送角度失败: {msg}")
        except Exception as e:
            print(f"发送角度时出错: {e}")

    def run_sequence(self, sequence: BaseSequence):
        """
        Run a gait sequence to completion.

        Args:
            sequence: BaseSequence instance to run
        """
        sequence.attach_walker(self)
        self.current_sequence = sequence
        self.current_phase = 0

        print(f"开始运行序列: {sequence.__class__.__name__}")

        while True:
            step = sequence.get_step(self.current_phase)

            if step is None:
                # Sequence completed (one-shot sequence)
                break

            # Calculate and apply angles from step
            angles = self._calculate_angles(step)
            self._apply_angles(angles)

            # Move to next phase
            self.current_phase += 1

            # Wait for period_ms
            time.sleep(self.period_ms / 1000.0)

        print(f"序列完成: {sequence.__class__.__name__}")

    def get_current_timestamp(self) -> float:
        """
        Get the current timestamp since start.

        Returns:
            float: Time in seconds since the controller started
        """
        return time.time() - self.start_time

    def get_camera_pose(self) -> Optional[object]:
        """
        Get the latest camera pose.

        Returns:
            The stored camera pose object or None if not available
        """
        return self._camera_pose

    def set_camera_pose(self, pose):
        """
        Store the camera pose.

        Args:
            pose: Camera pose object from ROS2 subscription
        """
        self._camera_pose = pose

    def set_scale(self, scale: float):
        """
        Set the scale factor for the controller.

        Args:
            scale: Scale factor to set
        """
        # This method is kept for compatibility but doesn't need to do anything
        # with the new architecture since scale was previously handled by camera_pose_client
        pass
