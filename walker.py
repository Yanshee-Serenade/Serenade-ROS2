import time
import ctypes
import math
from typing import List, Dict, Tuple, Optional
from enum import Enum
from ros_api import JointAngleTCPClient
import threading
import YanAPI

def _move_neck(angle, runtime):
    """线程函数：控制脖子舵机转动"""
    result = YanAPI.set_servos_angles({"NeckLR": angle}, runtime)
    print(result)

def move_neck(angle, runtime):
    threading.Thread(target=_move_neck, args=(round(angle), round(runtime / 2))).start()


class MockClient:
    def set_joint_angles(self, angles, time_ms=200):
        print(f"   模拟发送角度: {[round(i * 2048 / 180) for i in angles]}，时间: {time_ms}ms")
        return True, "模拟成功"

# ================================
# 1. 加载逆运动学库
# ================================
class KinematicsSolver:
    def __init__(self, lib_path: str = "./libyanshee_kinematics.so"):
        """初始化逆运动学求解器"""
        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise RuntimeError(f"无法加载库 {lib_path}: {e}")
        
        # 设置函数原型
        self.lib.inverse_kinematics.argtypes = [
            ctypes.c_int,      # is_right_leg
            ctypes.c_double,   # target_x
            ctypes.c_double,   # target_y
            ctypes.c_double,   # target_z
            ctypes.c_int,      # grid_density
            ctypes.POINTER(ctypes.c_double),  # angles数组
            ctypes.POINTER(ctypes.c_double)   # error
        ]
        self.lib.inverse_kinematics.restype = ctypes.c_int
    
    def solve_leg_ik(self, is_right_leg: bool, target_pos: Tuple[float, float, float], 
                    grid_density: int = 12) -> Optional[Tuple[List[float], float]]:
        """
        求解单腿逆运动学
        
        注意：这里不进行坐标转换，直接使用传入的坐标
        但在GaitStep中，我们会交换y和z坐标
        """
        angles = (ctypes.c_double * 5)()
        error = ctypes.c_double()
        
        x, y, z = target_pos
        
        result = self.lib.inverse_kinematics(
            ctypes.c_int(1 if is_right_leg else 0),
            ctypes.c_double(x),
            ctypes.c_double(y),
            ctypes.c_double(z),
            ctypes.c_int(grid_density),
            angles,
            ctypes.byref(error)
        )
        
        if result == 1:
            angles_degrees = [round(math.degrees(float(angles[i])), 0) for i in range(5)]
            return angles_degrees, float(error.value)
        else:
            return None

# ================================
# 2. 定义步态相关类
# ================================
class WalkerState(Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    WALK = "walk"
    SQUAT = "squat"
    DEFAULT = "default"

class GaitStep:
    """步态相位类，包含一个时刻的所有关节角度"""
    
    def __init__(self, solver: KinematicsSolver, grid_size: int, 
                 left_pos: Tuple[float, float, float], 
                 right_pos: Tuple[float, float, float]):
        """
        基于脚部位置初始化GaitStep
        
        注意：这里进行y和z坐标交换
        输入坐标格式：(x, y, z) -> 输出坐标：(x, z, y)
        """
        self.joint_angles = [0.0] * 17
        
        # 坐标转换：交换y和z
        left_pos_converted = (left_pos[0], left_pos[2], left_pos[1])  # (x, z, y)
        right_pos_converted = (right_pos[0], right_pos[2], right_pos[1])  # (x, z, y)
        
        # 解算左腿逆运动学（不转换坐标）
        left_result = solver.solve_leg_ik(False, left_pos_converted, grid_size)
        if left_result:
            left_angles, _ = left_result
            for i in range(5):
                self.joint_angles[11 + i] = left_angles[i]  # 左腿索引 11-15
        
        # 解算右腿逆运动学（不转换坐标）
        right_result = solver.solve_leg_ik(True, right_pos_converted, grid_size)
        if right_result:
            right_angles, _ = right_result
            for i in range(5):
                self.joint_angles[6 + i] = right_angles[i]  # 右腿索引 6-10
        
        # 设置默认手臂角度
        self.joint_angles[0] = 90.0   # 右肩偏航
        self.joint_angles[1] = 165.0  # 右肩俯仰
        self.joint_angles[2] = 100.0  # 右肘
        self.joint_angles[3] = 90.0   # 左肩偏航
        self.joint_angles[4] = 15.0   # 左肩俯仰
        self.joint_angles[5] = 80.0   # 左肘
        
        # 设置默认脖子角度
        self.joint_angles[16] = 90.0
    
    def set_right_arm(self, angle: float) -> 'GaitStep':
        """设置右手往前抬的角度"""
        self.joint_angles[0] -= angle  # 右肩偏航
        return self
    
    def set_left_arm(self, angle: float) -> 'GaitStep':
        """设置左手往前抬的角度"""
        self.joint_angles[3] += angle  # 左肩偏航
        return self
    
    def set_right_lean(self, angle: float) -> 'GaitStep':
        """设置右侧往前倾的角度"""
        self.joint_angles[7] -= angle  # 右腿髋部侧摆
        return self
    
    def set_left_lean(self, angle: float) -> 'GaitStep':
        """设置左侧往前倾的角度"""
        self.joint_angles[12] += angle  # 左腿髋部侧摆
        return self
    
    def set_right_ankle_lean(self, angle: float) -> 'GaitStep':
        """设置右踝往前倾的角度"""
        self.joint_angles[9] += angle
        return self
    
    def set_left_ankle_lean(self, angle: float) -> 'GaitStep':
        """设置左踝往前倾的角度"""
        self.joint_angles[14] -= angle
        return self
    
    def set_neck(self, angle: float) -> 'GaitStep':
        """设置脖子向右旋转角度"""
        self.joint_angles[16] += angle
        return self
    
    @property
    def angles(self) -> List[float]:
        return self.joint_angles.copy()

# ================================
# 3. Walker主类
# ================================
class RobotWalker:
    def __init__(self, solver: KinematicsSolver, client, grid_size: int = 12, period_ms: int = 200):
        """
        初始化机器人行走控制器
        
        Args:
            solver: 逆运动学求解器
            client: 机器人客户端（如JointAngleTCPClient）
            grid_size: 网格搜索密度
            period_ms: 动作周期（毫秒）
        """
        self.solver = solver
        self.client = client
        self.grid_size = grid_size
        self.period_ms = period_ms
        
        # 状态控制
        self.current_state = WalkerState.TURN_RIGHT
        self.previous_state = None
        self.current_phase = 0
        
        # 初始化步态序列
        self.gait_sequences = self._initialize_gait_sequences()
        self.current_sequence = self.gait_sequences[self.current_state]
        
        # 计时相关
        self.last_action_time = time.time() * 1000  # 转换为毫秒
        self.running = False

        # 角度设置 API
        YanAPI.yan_api_init("raspberrypi")
    
    def _initialize_gait_sequences(self) -> Dict[WalkerState, List[GaitStep]]:
        """初始化所有步态序列"""
        sequences = {}
        sequences[WalkerState.TURN_LEFT] = [
            GaitStep(self.solver, self.grid_size, 
                     (-0.02, 0.061, 0.0), (0.02, 0.06, 0.0)),
            GaitStep(self.solver, self.grid_size,
                     (-0.04, 0.061, -0.02), (0.04, 0.06, 0.02)),
            GaitStep(self.solver, self.grid_size,
                     (-0.06, 0.061, -0.04), (0.06, 0.06, 0.04)),
            GaitStep(self.solver, self.grid_size,
                     (-0.04, 0.061, -0.02), (0.04, 0.06, 0.02)),
        ]
        sequences[WalkerState.TURN_RIGHT] = [
            GaitStep(self.solver, self.grid_size,
                     (-0.02, 0.06, 0.0), (0.02, 0.064, 0.0)),
            GaitStep(self.solver, self.grid_size,
                     (-0.04, 0.06, 0.02), (0.04, 0.064, -0.02)),
            GaitStep(self.solver, self.grid_size,
                     (-0.06, 0.06, 0.04), (0.06, 0.064, -0.04)),
            GaitStep(self.solver, self.grid_size,
                     (-0.04, 0.06, 0.02), (0.04, 0.064, -0.02)),
        ]
        sequences[WalkerState.WALK] = [
            GaitStep(self.solver, self.grid_size,
                     (-0.02, 0.065, 0.03), (0.02, 0.06, -0.01))
                .set_left_lean(15).set_right_lean(15).set_neck(5)
                .set_right_ankle_lean(-3).set_right_arm(45).set_left_arm(-45),
            GaitStep(self.solver, self.grid_size,
                     (-0.02, 0.06, 0.03), (0.02, 0.065, -0.01))
                .set_left_lean(15).set_right_lean(15).set_neck(5)
                .set_left_ankle_lean(-3).set_left_arm(-45).set_right_arm(45),
            GaitStep(self.solver, self.grid_size,
                     (-0.02, 0.06, -0.01), (0.02, 0.065, 0.03))
                .set_left_lean(15).set_right_lean(15).set_neck(-5)
                .set_left_ankle_lean(-3).set_left_arm(45).set_right_arm(-45),
            GaitStep(self.solver, self.grid_size,
                     (-0.02, 0.065, -0.01), (0.02, 0.06, 0.03))
                .set_left_lean(15).set_right_lean(15).set_neck(-5)
                .set_right_ankle_lean(-3).set_left_arm(45).set_right_arm(-45),
        ]
        sequences[WalkerState.SQUAT] = [
            GaitStep(self.solver, self.grid_size, 
                     (-0.02, 0.09, 0.0), (0.02, 0.09, 0.0)),
        ]
        sequences[WalkerState.DEFAULT] = [
            GaitStep(self.solver, self.grid_size, 
                     (-0.02, 0.04, 0.0), (0.02, 0.04, 0.0)),
        ]
        
        return sequences
    
    def set_state(self, state: WalkerState):
        """设置行走状态"""
        if state != self.current_state:
            self.current_state = state
            self.current_sequence = self.gait_sequences[state]
            self.current_phase = 0
            print(f"切换状态到: {state.value}")
    
    def reset(self):
        """重置到初始状态"""
        self.current_phase = 0
        self.last_action_time = time.time() * 1000
        self.run_frame(WalkerState.DEFAULT, 0)
    
    def update(self):
        """更新步态相位，应在循环中调用"""
        current_time = time.time() * 1000  # 毫秒
        
        # 检查是否到达下一个周期
        if current_time - self.last_action_time >= self.period_ms:
            self._apply_current_phase()
            self.current_phase = (self.current_phase + 1) % len(self.current_sequence)
            self.last_action_time = current_time
    
    def _apply_current_phase(self):
        """应用当前相位到机器人"""
        if self.current_phase < len(self.current_sequence):
            step = self.current_sequence[self.current_phase]
            angles = step.angles
            self._apply_angles(angles)
    
    def _apply_angles(self, angles):
        """发送到机器人客户端"""
        try:
            success, msg = self.client.set_joint_angles(angles, time_ms=self.period_ms)
            if not success:
                print(f"发送角度失败: {msg}")
        except Exception as e:
            print(f"发送角度时出错: {e}")
        
        # 修复脖子不动的问题
        move_neck(angles[16], self.period_ms)
    
    def run_frame(self, state: WalkerState, frame: int = 0):
        """
        固定到指定步态的某个周期
        
        Args:
            state: 步态类型
            frame: 目标周期数
        """
        step = self.gait_sequences[state][frame]
        angles = step.angles
        self._apply_angles(angles)
    
    def run_sequence(self, state: WalkerState, cycles: int = 4):
        """
        运行指定步态的多个周期
        
        Args:
            state: 步态类型
            cycles: 运行周期数
        """
        self.set_state(state)
        self.reset()
        time.sleep(self.period_ms / 1000.0)
        
        total_phases = len(self.current_sequence) * cycles
        print(f"开始运行 {state.value} 步态，共 {cycles} 个周期 ({total_phases} 个相位)")
        
        for i in range(total_phases):
            self._apply_current_phase()
            self.current_phase = (self.current_phase + 1) % len(self.current_sequence)
            
            # 等待period_ms
            time.sleep(self.period_ms / 1000.0)
        
        print(f"{state.value} 步态完成")

# ================================
# 4. 测试主程序
# ================================
def create_walker(period_ms: int = 200) -> RobotWalker:
    # 1. 初始化逆运动学求解器
    print("1. 初始化逆运动学求解器...")
    solver = KinematicsSolver("./libyanshee_kinematics.so")
    print("   √ 逆运动学求解器初始化成功")
    
    # 2. 初始化机器人客户端（模拟）
    print("\n2. 初始化机器人客户端...")
    angle_client = JointAngleTCPClient(host='localhost', port=51120, timeout=10)
    print("   √ 客户端初始化成功")
    
    # 3. 创建Walker
    print("\n3. 创建Walker控制器...")
    walker = RobotWalker(solver, angle_client, grid_size=12, period_ms=period_ms)
    print(f"   √ Walker创建成功，周期: {walker.period_ms}ms")
    return walker