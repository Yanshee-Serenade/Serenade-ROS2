"""
Walk target gait sequence.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from serenade_walker.sequences.base_sequences import CyclingSequence, GaitStep
from visualization_msgs.msg import Marker


class WalkTargetSequence(CyclingSequence):
    """Walk gait sequence."""

    def __init__(self, backward=False):
        self.backward = backward
        super().__init__()

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

    def get_step(self, step_index: int) -> GaitStep:
        """
        Get gait step for the given step index with Y-axis offset for walking.

        Args:
            step_index: Current step index (0-based)

        Returns:
            GaitStep instance for the step with offset applied
        """
        step = super().get_step(step_index)
        right_spin = 0.0

        target_position = np.zeros(3)
        if self.walker.marker_array is not None:
            for marker in self.walker.marker_array.markers:
                marker: Marker = marker
                if marker.action == Marker.ADD and marker.ns == "texts":
                    # Only track teddy bear (Nailong)
                    if "teddy" in marker.text:
                        target_position[0] = marker.pose.position.x
                        target_position[1] = marker.pose.position.y
                        target_position[2] = marker.pose.position.z - 0.15

        # Below code is legacy
        # Change it to follow target_position (relative to camera frame)

        # Parameters (Tune these!)
        # K_yaw: How hard to correct for angle errors (Start small: 0.05 to 0.1)
        # K_lat: How hard to correct for lateral drift (Start small: 0.1 to 0.3)
        SCALE_FACTOR = 16  # Hard-coded scale factor for ORB-SLAM3
        MAX_CORRECTION = 0.005  # Meters
        K_yaw = 4 * MAX_CORRECTION
        K_lat = 20 * MAX_CORRECTION

        if self.init_pose:
            current_pose_data = self.walker.camera_pose

            # Ensure we have data and it's not the exact same millisecond as init
            if current_pose_data:
                # 1. Convert Data to Scipy Format (x, y, z, w)
                # init_pose is CameraPoseData
                q_init = [
                    self.init_pose._pose.orientation.x,
                    self.init_pose._pose.orientation.y,
                    self.init_pose._pose.orientation.z,
                    self.init_pose._pose.orientation.w,
                ]
                p_init = np.array(
                    [
                        self.init_pose._pose.position.x,
                        self.init_pose._pose.position.y,
                        self.init_pose._pose.position.z,
                    ]
                )

                # current_pose_data is CameraPoseData
                curr = current_pose_data
                q_curr = [
                    curr._pose.orientation.x,
                    curr._pose.orientation.y,
                    curr._pose.orientation.z,
                    curr._pose.orientation.w,
                ]
                p_curr = np.array(
                    [
                        curr._pose.position.x, 
                        curr._pose.position.y, 
                        curr._pose.position.z
                    ]
                )

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
                # If backward is True, we invert the sign of yaw_error.
                yaw_error = -yaw_error if self.backward else yaw_error
                right_spin = (K_yaw * yaw_error) + (K_lat * lat_error)
                right_spin *= SCALE_FACTOR

                # Clip limits to prevent falling
                right_spin = max(min(right_spin, MAX_CORRECTION), -MAX_CORRECTION)
                print(
                    f"Yaw Error: {yaw_error:.4f}, Lateral Error: {lat_error:.4f}, Right Spin: {right_spin:.4f}"
                )

        # Calculate phase for offset logic (0-3 for walk sequence)
        phase = step_index % 4

        if self.backward ^ (phase == 0 or phase == 3):
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
