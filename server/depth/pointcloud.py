from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from ros_api.model import (
    CameraIntrinsics,
    CameraPose,
    DistortionCoefficients,
)

from ..config import config


class PointCloudValidator:
    """
    跟踪数据验证类：修复 y 轴巨大误差，优化畸变校正逻辑，匹配题目 Eigen 视觉计算场景
    """

    def __init__(
        self,
        camera_pose: CameraPose,
        intrinsics: CameraIntrinsics = config.CAMERA_INTRINSICS,
        distortion_coeffs: DistortionCoefficients = config.DISTORTION_COEFFS,
    ):
        """
        初始化验证器（适配新的带类型标注数据结构）
        Args:
            intrinsics (CameraIntrinsics): 相机内参（带类型标注）
            camera_pose (CameraPose): 相机位姿（带类型标注）
            distortion_coeffs (DistortionCoefficients): 相机畸变参数 [k1, k2, p1, p2, k3]
        """
        self.intrinsics = intrinsics
        self.camera_pose = camera_pose
        self.distortion_coeffs = distortion_coeffs
        self.K = self._build_intrinsics_matrix()
        self.Tcw, self.Twc = self._build_pose_transform_matrices()

    def _init_distortion_coeffs(
        self, distortion_coeffs: Optional[List[float]]
    ) -> np.ndarray:
        """初始化畸变参数，补全radtan模型所需的5个参数"""
        if distortion_coeffs is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        dist = np.array(distortion_coeffs, dtype=np.float64).flatten()
        if len(dist) < 5:
            dist = np.pad(
                dist, (0, 5 - len(dist)), mode="constant", constant_values=0.0
            )
        return dist[:5]

    def _build_intrinsics_matrix(self) -> np.ndarray:
        """构建相机内参矩阵 K (3x3)"""
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    def _build_pose_transform_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """构建正确的位姿变换矩阵"""
        pos = self.camera_pose.position
        ori = self.camera_pose.orientation
        t_wc = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        quaternion_wc = np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float64)

        R_wc = R.from_quat(quaternion_wc).as_matrix().astype(np.float64)
        R_cw = R_wc.T

        Twc = np.eye(4, dtype=np.float64)
        Twc[:3, :3] = R_wc
        Twc[:3, 3] = t_wc

        t_cw = -np.dot(R_cw, t_wc)
        Tcw = np.eye(4, dtype=np.float64)
        Tcw[:3, :3] = R_cw
        Tcw[:3, 3] = t_cw

        return Tcw, Twc

    def _undistort_to_normalized_coords(
        self, pixel_x: float, pixel_y: float
    ) -> Tuple[float, float]:
        """重构：畸变校正直接得到「去畸变后的归一化图像坐标」"""
        fx, fy, cx, cy = (
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
        )
        k1, k2, p1, p2, k3 = self.distortion_coeffs

        # 步骤1：像素坐标 → 原始归一化图像坐标
        u = (pixel_x - cx) / fx
        v = (pixel_y - cy) / fy

        # 步骤2：计算径向畸变参数
        r_sq = u**2 + v**2
        r_4 = r_sq**2
        r_6 = r_sq**3
        dist_radial = 1 + k1 * r_sq + k2 * r_4 + k3 * r_6

        # 步骤3：计算切向畸变校正量
        delta_u = 2 * p1 * u * v + p2 * (r_sq + 2 * u**2)
        delta_v = p1 * (r_sq + 2 * v**2) + 2 * p2 * u * v

        # 步骤4：去畸变后的归一化坐标
        u_undist = u * dist_radial + delta_u
        v_undist = v * dist_radial + delta_v

        return u_undist, v_undist

    def camera_point_to_world_point(self, camera_point: np.ndarray) -> np.ndarray:
        """核心修复：移除 Yc 计算的负号，匹配相机坐标系定义"""
        camera_point = np.array(camera_point, dtype=np.float64).flatten()
        if len(camera_point) != 3:
            raise ValueError(
                "相机点格式错误，必须为 [pixel_x, pixel_y, camera_depth_z]"
            )
        pixel_x, pixel_y, camera_depth_z = camera_point

        # 畸变校正，获取去畸变后的归一化坐标
        u_undist, v_undist = self._undistort_to_normalized_coords(pixel_x, pixel_y)

        # 反解相机坐标系下的3D点 (Xc, Yc, Zc)
        Xc = u_undist * camera_depth_z
        Yc = v_undist * camera_depth_z
        Zc = camera_depth_z

        # 转换为齐次坐标，通过Twc矩阵转换为世界坐标
        camera_point_homo = np.array([Xc, Yc, Zc, 1.0], dtype=np.float64)
        world_point_homo_pred = np.dot(self.Twc, camera_point_homo)
        world_point_pred = world_point_homo_pred[:3]

        return world_point_pred

    def validate_point_cloud(
        self, camera_point_cloud: np.ndarray, world_point_cloud: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """验证点云，返回误差指标"""
        camera_point_cloud = np.array(camera_point_cloud, dtype=np.float64)
        world_point_cloud = np.array(world_point_cloud, dtype=np.float64)

        if camera_point_cloud.shape != world_point_cloud.shape:
            raise ValueError(
                f"相机点云与世界点云形状不匹配：{camera_point_cloud.shape} vs {world_point_cloud.shape}"
            )
        if camera_point_cloud.shape[0] == 0:
            raise ValueError("点云为空，无法计算误差")

        point_count = camera_point_cloud.shape[0]
        point_errors = np.zeros(point_count, dtype=np.float64)

        for i in range(point_count):
            world_pred = self.camera_point_to_world_point(camera_point_cloud[i])
            world_actual = world_point_cloud[i]
            point_errors[i] = np.linalg.norm(world_pred - world_actual)

        error_metrics = {
            "point_count": float(point_count),
            "mean_error": float(np.mean(point_errors)),
            "max_error": float(np.max(point_errors)),
            "min_error": float(np.min(point_errors)),
            "rmse": float(np.sqrt(np.mean(np.square(point_errors)))),
        }

        return error_metrics, point_errors
