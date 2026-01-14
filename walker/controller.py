"""
Robot walker controller.

This module provides the RobotWalker class which manages gait sequences,
state transitions, and robot motion control.
"""

import time
from typing import List, Optional

from ros_api import CameraPoseClient, CameraPoseData

from .gait import (
    DefaultSequence,
    GaitStep,
    SquatSequence,
    TurnLeftSequence,
    TurnRightSequence,
    WalkerState,
    WalkSequence,
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
        camera_pose_client: Optional[CameraPoseClient] = None,
    ):
        """
        Initialize robot walker controller.

        Args:
            solver: Inverse kinematics solver
            client: Robot client (e.g., JointAngleTCPClient)
            grid_size: Grid search density for IK
            period_ms: Action period in milliseconds
            camera_pose_client: Optional CameraPoseClient for accessing camera pose data
        """
        self.solver = solver
        self.client = client
        self.grid_size = grid_size
        self.period_ms = period_ms
        self.camera_pose_client = camera_pose_client

        # State control
        self.current_state = WalkerState.TURN_RIGHT
        self.previous_state = None
        self.current_phase = 0
        self.current_sequence = DefaultSequence(self)

        # Timing related
        self.last_action_time = time.time() * 1000  # Convert to milliseconds
        self.start_time = time.time()
        self.running = False

        # Start camera pose streaming if client is available
        if self.camera_pose_client:
            self.camera_pose_client.start_streaming()

    def set_state(self, state: WalkerState):
        """Set the walker state."""
        if state != self.current_state:
            self.current_state = state
            if state == WalkerState.TURN_LEFT:
                self.current_sequence = TurnLeftSequence(self)
            elif state == WalkerState.TURN_RIGHT:
                self.current_sequence = TurnRightSequence(self)
            elif state == WalkerState.WALK:
                self.current_sequence = WalkSequence(self)
            elif state == WalkerState.SQUAT:
                self.current_sequence = SquatSequence(self)
            elif state == WalkerState.DEFAULT:
                self.current_sequence = DefaultSequence(self)
            self.current_phase = 0
            print(f"切换状态到: {state.value}")

    def reset(self):
        """Reset to initial state."""
        self.current_phase = 0
        self.last_action_time = time.time() * 1000
        self.run_frame(WalkerState.DEFAULT, 0)

    def update(self):
        """Update gait phase, should be called in a loop."""
        current_time = time.time() * 1000  # milliseconds

        # Check if it's time for the next phase
        if current_time - self.last_action_time >= self.period_ms:
            self._apply_current_phase()
            self.current_phase = (self.current_phase + 1) % len(
                self.current_sequence.steps
            )
            self.last_action_time = current_time

    def _apply_current_phase(self):
        """Apply current phase to the robot."""
        if self.current_phase < len(self.current_sequence.steps):
            step = self.current_sequence.get_step(self.current_phase)

            # Calculate joint angles from step data
            angles = self._calculate_angles(step)
            self._apply_angles(angles)

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

    def run_frame(self, state: WalkerState, frame: int = 0):
        """
        Run a specific frame of a gait sequence.

        Args:
            state: Gait type
            frame: Target frame number
        """
        self.set_state(state)
        step = self.current_sequence.get_step(frame)

        # Calculate and apply angles from step
        angles = self._calculate_angles(step)
        self._apply_angles(angles)

    def run_sequence(self, state: WalkerState, cycles: int = 4):
        """
        Run multiple cycles of a specific gait.

        Args:
            state: Gait type
            cycles: Number of cycles to run
        """
        self.reset()
        self.set_state(state)
        time.sleep(1)

        total_phases = len(self.current_sequence.steps) * cycles
        print(
            f"开始运行 {state.value} 步态，共 {cycles} 个周期 ({total_phases} 个相位)"
        )

        for i in range(total_phases):
            self._apply_current_phase()
            self.current_phase = (self.current_phase + 1) % len(
                self.current_sequence.steps
            )

            # Wait for period_ms
            time.sleep(self.period_ms / 1000.0)

        print(f"{state.value} 步态完成")

    def get_current_timestamp(self) -> float:
        """
        Get the current timestamp since start.

        Returns:
            float: Time in seconds since the controller started
        """
        return time.time() - self.start_time

    def get_camera_pose(self) -> Optional[CameraPoseData]:
        """
        Get the latest camera pose and timestamp since start.

        Returns:
            tuple: (pose_data, timestamp_seconds) where:
                - pose_data: CameraPoseData object or None if not available
                - timestamp_seconds: Time in seconds since the camera pose client started
        """
        if not self.camera_pose_client:
            return None

        return self.camera_pose_client.get_latest_pose()

    def set_scale(self, scale: float):
        """
        Set the scale factor for the controller.

        Args:
            scale: Scale factor to set
        """
        if self.camera_pose_client:
            self.camera_pose_client.set_scale(scale)
