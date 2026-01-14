"""
Robot walker controller.

This module provides the RobotWalker class which manages gait sequences,
state transitions, and robot motion control.
"""

import time
from typing import Dict, List, Optional, Tuple

from ros_api import CameraPoseClient, CameraPoseData

from .gait import GaitStep, WalkerState
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

        # Initialize gait sequences
        self.gait_sequences = self._initialize_gait_sequences()
        self.current_sequence = self.gait_sequences[self.current_state]

        # Timing related
        self.last_action_time = time.time() * 1000  # Convert to milliseconds
        self.start_time = time.time()
        self.running = False

        # Start camera pose streaming if client is available
        if self.camera_pose_client:
            self.camera_pose_client.start_streaming()

    def _initialize_gait_sequences(self) -> Dict[WalkerState, List[GaitStep]]:
        """Initialize all gait sequences."""
        sequences = {}

        # Turn left sequence
        sequences[WalkerState.TURN_LEFT] = [
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.061, 0.0), (0.02, 0.06, 0.0)
            ),
            GaitStep(
                self.solver, self.grid_size, (-0.04, 0.061, -0.02), (0.04, 0.06, 0.02)
            ),
        ]

        # Turn right sequence
        sequences[WalkerState.TURN_RIGHT] = [
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.06, 0.0), (0.02, 0.064, 0.0)
            ),
            GaitStep(
                self.solver, self.grid_size, (-0.04, 0.06, 0.02), (0.04, 0.064, -0.02)
            ),
        ]

        # Walk sequence
        sequences[WalkerState.WALK] = [
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.065, 0.03), (0.02, 0.06, -0.01)
            )
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(5)
            .set_right_ankle_lean(-3)
            .set_left_arm(-45)
            .set_right_arm(45),
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.06, 0.03), (0.02, 0.065, -0.01)
            )
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(5)
            .set_left_ankle_lean(-3)
            .set_left_arm(-45)
            .set_right_arm(45),
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.06, -0.01), (0.02, 0.065, 0.03)
            )
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(-5)
            .set_left_ankle_lean(-3)
            .set_left_arm(45)
            .set_right_arm(-45),
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.065, -0.01), (0.02, 0.06, 0.03)
            )
            .set_left_lean(15)
            .set_right_lean(15)
            .set_neck(-5)
            .set_right_ankle_lean(-3)
            .set_left_arm(45)
            .set_right_arm(-45),
        ]

        # Squat sequence
        sequences[WalkerState.SQUAT] = [
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.09, 0.0), (0.02, 0.09, 0.0)
            ),
        ]

        # Default sequence
        sequences[WalkerState.DEFAULT] = [
            GaitStep(
                self.solver, self.grid_size, (-0.02, 0.04, 0.0), (0.02, 0.04, 0.0)
            ),
        ]

        return sequences

    def set_state(self, state: WalkerState):
        """Set the walker state."""
        if state != self.current_state:
            self.current_state = state
            self.current_sequence = self.gait_sequences[state]
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
            self.current_phase = (self.current_phase + 1) % len(self.current_sequence)
            self.last_action_time = current_time

    def _apply_current_phase(self):
        """Apply current phase to the robot."""
        if self.current_phase < len(self.current_sequence):
            step = self.current_sequence[self.current_phase]
            angles = step.angles
            self._apply_angles(angles)

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
        step = self.gait_sequences[state][frame]
        angles = step.angles
        self._apply_angles(angles)

    def run_sequence(self, state: WalkerState, cycles: int = 4):
        """
        Run multiple cycles of a specific gait.

        Args:
            state: Gait type
            cycles: Number of cycles to run
        """
        self.set_state(state)
        self.reset()
        time.sleep(self.period_ms / 1000.0)

        total_phases = len(self.current_sequence) * cycles
        print(
            f"开始运行 {state.value} 步态，共 {cycles} 个周期 ({total_phases} 个相位)"
        )

        for i in range(total_phases):
            self._apply_current_phase()
            self.current_phase = (self.current_phase + 1) % len(self.current_sequence)

            # Print current position information
            (pose, timestamp) = self.get_latest_camera_pose()
            print(f"当前时间：{self.get_current_timestamp()}")
            print(f"相机时间：{timestamp}")
            if pose:
                print(f"相机位置：{pose.p}")
            else:
                print("相机位置：None")

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

    def get_latest_camera_pose(self) -> Tuple[Optional[CameraPoseData], float]:
        """
        Get the latest camera pose and timestamp since start.

        Returns:
            tuple: (pose_data, timestamp_seconds) where:
                - pose_data: CameraPoseData object or None if not available
                - timestamp_seconds: Time in seconds since the camera pose client started
        """
        if not self.camera_pose_client:
            return None, 0.0

        (pose, timestamp) = self.camera_pose_client.get_latest_pose()
        return pose, timestamp - self.start_time
