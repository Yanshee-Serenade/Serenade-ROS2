"""
Walk target gait sequence.
"""

from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String

from serenade_agent.config import INTERRUPTIBLE_FLAG, TARGET_RIGHT, TARGET_LEFT, TARGET_NONE
from serenade_walker.sequences.base_sequences import BaseSequence, GaitStep
from visualization_msgs.msg import Marker


class State(Enum):
    IDLE = 1
    WALK = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4


class WalkTargetSequence(BaseSequence):
    """Walk gait sequence."""

    def __init__(self, backward=False):
        self.backward = backward
        self.last_index = 0
        self.state = State.IDLE
        self.target_lost_timer = 999
        self.target_position = np.zeros(3)
        super().__init__()

    def _initialize_steps(self):
        """Initialize walk steps."""
        self.steps_walk = [
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
        self.steps_turn_left = [
            GaitStep((-0.02, 0.06, 0.0), (0.02, 0.06, 0.0)),
            GaitStep((-0.04, 0.06, -0.02), (0.04, 0.06, 0.02)),
        ]
        self.steps_turn_right = [
            GaitStep((-0.02, 0.06, 0.0), (0.02, 0.06, 0.0)),
            GaitStep((-0.04, 0.06, 0.02), (0.04, 0.06, -0.02)),
        ]

    def get_step(self, step_index: int) -> GaitStep:
        """
        Get gait step for the given step index with Y-axis offset for walking.

        Args:
            step_index: Current step index (0-based)

        Returns:
            GaitStep instance for the step with offset applied
        """

        # Get next state
        if self.walker.target == TARGET_LEFT:
            next_state = State.TURN_LEFT
        elif self.walker.target == TARGET_RIGHT:
            next_state = State.TURN_RIGHT
        elif self.walker.target == TARGET_NONE:
            next_state = State.IDLE
        else:
            # Check if target exists
            has_target = False
            if self.walker.marker_array is not None:
                for marker in self.walker.marker_array.markers:
                    marker: Marker = marker
                    if marker.action == Marker.ADD and marker.ns == "texts":
                        target_text = self.walker.target.strip()
                        marker_text = marker.text.split('_')[1].split('\n')[0].strip()
                        if target_text in marker_text:
                            has_target = True
                            self.target_position[0] = marker.pose.position.x
                            self.target_position[1] = marker.pose.position.y
                            self.target_position[2] = marker.pose.position.z - 0.15
                            break
            
            # Accumulate target lost timer if not exist
            if has_target:
                self.target_lost_timer = 0
            else:
                self.target_lost_timer += 1
            
            # If lost for more than 4 steps, reset self target and set to idle
            if self.target_lost_timer < 4:
                next_state = State.WALK
            else:
                target_name = self.walker.target
                question_msg = String()
                question_msg.data = f"{INTERRUPTIBLE_FLAG}你想跟踪的 {target_name} 不存在了，可能它的 ID 变了，也可能它被遮挡或移走了。请决定你的下一步行动！"
                if self.walker.node is not None:
                    self.walker.node.get_logger().info(f"Target '{target_name}' lost, publishing question")
                    self.walker.node.question_publisher.publish(question_msg)
                self.walker.target = TARGET_NONE
                next_state = State.IDLE
        
        # Transfer state
        if self.state != next_state:
            old_state = self.state
            self.state = next_state
            self.last_index = 0
            print(f"State changed from {old_state} to {next_state}", flush=True)

        # Apply state
        if self.state == State.TURN_LEFT:
            step = self.steps_turn_left[(step_index - self.last_index) % len(self.steps_turn_left)]
            return step
        elif self.state == State.TURN_RIGHT:
            step = self.steps_turn_right[(step_index - self.last_index) % len(self.steps_turn_right)]
            return step
        elif self.state == State.WALK:
            step = self.steps_walk[(step_index - self.last_index) % len(self.steps_walk)]
            MAX_CORRECTION = 0.005  # Meters
            K = 4
            right_spin = 0.0

            # 计算从原点(0,0,0)到target_position的圆弧曲率
            # 初始方向是+z轴，所以这是一个在x-z平面上的圆弧运动
            x = self.target_position[0]  # 横向偏移
            z = self.target_position[2]  # 前进距离

            if z != 0:
                # 对于初始方向为+z，终点为(x,z)的圆弧:
                # 圆心在(0,R)，半径为R
                # 满足: x² + (z-R)² = R²
                # 解得: R = (x² + z²) / (2z)
                R = (x*x + z*z) / (2*z)
                curvature = 1.0 / R if R != 0 else 0

                # 曲率符号: 如果x>0，向右转弯，曲率为正
                curvature = curvature if x >= 0 else -curvature
                print(f"Radius = {R}", flush=True)

                # 使用曲率计算right_spin
                right_spin = curvature * K * MAX_CORRECTION

            else:
                # 如果z=0，使用最大转向
                right_spin = MAX_CORRECTION if x > 0 else -MAX_CORRECTION

            # 限制在[-MAX_CORRECTION, MAX_CORRECTION]范围内
            right_spin = max(min(right_spin, MAX_CORRECTION), -MAX_CORRECTION)
            print(f"Right spin = {right_spin}", flush=True)

            # Calculate phase for offset logic (0-3 for walk sequence)
            phase = (step_index - self.last_index) % 4

            if self.backward ^ (phase == 0 or phase == 3):
                # Reducing the addition to `y` increases the friction between robot left foot
                # and ground, so that robot turns left
                left_pos = (
                    step.left_pos[0] + 0.0,
                    step.left_pos[1] + 0.005 + min(right_spin, 0),
                    step.left_pos[2] + 0.0,
                )
                new_step = step.copy()
                new_step.left_pos = left_pos
                return new_step
            else:
                right_pos = (
                    step.right_pos[0] + 0.0,
                    step.right_pos[1] + 0.005 - max(right_spin, 0),
                    step.right_pos[2] + 0.0,
                )
                new_step = step.copy()
                new_step.right_pos = right_pos
                return new_step

        # If no target, return default step
        return GaitStep()
