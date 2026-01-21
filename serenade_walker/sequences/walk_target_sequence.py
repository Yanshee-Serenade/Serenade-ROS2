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
        self.last_index = 0
        self.last_has_target = False
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
        has_target = False
        if self.walker.marker_array is not None:
            for marker in self.walker.marker_array.markers:
                marker: Marker = marker
                if marker.action == Marker.ADD and marker.ns == "texts":
                    if self.walker.target in marker.text:
                        has_target = True
                        target_position[0] = marker.pose.position.x
                        target_position[1] = marker.pose.position.y
                        target_position[2] = marker.pose.position.z - 0.15
                        print(f"Found nailong at target: f{target_position}", flush=True)
                        break

        if has_target:
            # Parameters (Tune these!)
            MAX_CORRECTION = 0.005  # Meters
            K = 4
            right_spin = target_position[0] / target_position[2] * K * MAX_CORRECTION
            right_spin = max(min(right_spin, MAX_CORRECTION), -MAX_CORRECTION)
            print(f"Right spin = {right_spin}", flush=True)

            if not self.last_has_target:
                print(f"Setting last_index = 0", flush=True)
                self.last_index = 0

            # Calculate phase for offset logic (0-3 for walk sequence)
            phase = (step_index - self.last_index) % 4

            # Set last has target
            self.last_has_target = has_target

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

        # If no target, return default step
        return GaitStep((-0.02, 0.04, 0.0), (0.02, 0.04, 0.0))
