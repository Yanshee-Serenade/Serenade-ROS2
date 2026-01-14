#!/usr/bin/env python3
import logging
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from ros_api.logger import setup_client_logger
from ros_api.model import (
    CameraIntrinsics,
    CameraPose,
    ImageInfo,
    Orientation,
    PointCloudInfo,
    Position,
    TrackingResult,
)


class TrackingDataClient:
    """
    跟踪数据TCP客户端（重构版）
    特性：
    1. 基于logging模块实现日志输出，支持开关控制
    2. 提供闭环方法complete_tracking_pipeline，一键完成连接/发送/接收/关闭
    3. 所有数据结构带完整类型标注，返回值为强类型NamedTuple
    """

    def __init__(
        self,
        server_ip: str = "127.0.0.1",
        port: int = 51121,
        enable_log: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        初始化客户端
        :param server_ip: 服务器IP地址
        :param port: 服务器端口
        :param enable_log: 是否启用日志（核心：日志开关控制）
        :param log_level: 日志级别
        """
        self.server_ip: str = server_ip
        self.port: int = port
        self.socket: Optional[socket.socket] = None
        self.total_recv_size: int = 0  # 记录总接收字节数
        self.parse_cost_ms: float = 0.0  # 记录解析耗时

        # 初始化日志器（支持开关控制）
        self.enable_log: bool = enable_log
        self.logger: logging.Logger = setup_client_logger(enable_log, log_level)

    def _create_socket(self) -> None:
        """创建TCP Socket并设置超时（内部方法）"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10.0)  # 10秒超时

    def connect_to_server(self) -> bool:
        """连接到服务器（带日志输出）"""
        if self.socket is not None:
            self.logger.warning("Socket already exists, closing first")
            self.close_connection()

        try:
            self._create_socket()
            assert self.socket is not None, "Socket creation failed"

            self.logger.info(f"Connecting to {self.server_ip}:{self.port}...")
            start_time = time.time()

            self.socket.connect((self.server_ip, self.port))
            cost_time = (time.time() - start_time) * 1000

            self.logger.info(f"Connected successfully! Cost {cost_time:.2f} ms")
            return True

        except socket.timeout:
            self.logger.error("Connection timeout (10s)")
            return False
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False

    def recv_all(self, length: int, desc: str = "unknown data") -> bytes:
        """确保接收指定长度的字节数据（带日志输出）"""
        if length <= 0:
            self.logger.warning(f"{desc}: Invalid length ({length}), return empty")
            return b""

        self.logger.info(f"{desc}: Waiting for {length} bytes...")
        data = b""
        start_time = time.time()

        assert self.socket is not None, "Socket not initialized, please connect first"

        try:
            while len(data) < length:
                remaining = length - len(data)
                chunk = self.socket.recv(min(remaining, 4096))  # 分块接收，优化传输效率

                if not chunk:
                    raise ConnectionAbortedError(
                        "Server closed the connection unexpectedly"
                    )

                data += chunk

                # 打印接收进度（仅INFO级别日志输出）
                progress = round(len(data) / length * 100, 2)
                self.logger.debug(
                    f"{desc}: Received {len(data)}/{length} bytes ({progress}%)"
                )

            # 更新统计信息
            self.total_recv_size += len(data)
            cost_time = (time.time() - start_time) * 1000

            self.logger.info(
                f"{desc}: Received complete! {len(data)} bytes, cost {cost_time:.2f} ms"
            )
            return data

        except socket.timeout:
            raise TimeoutError(
                f"Recv {desc} timeout (10s), received {len(data)}/{length} bytes"
            )

    def parse_point_cloud_data(
        self, pc_data: bytes, desc: str = "unknown point cloud"
    ) -> np.ndarray:
        """解析点云原始字节数据为NumPy数组（带日志输出）"""
        self.logger.info(f"{desc}: Starting parse ({len(pc_data)} bytes)...")

        if not pc_data:
            self.logger.warning(f"{desc}: Empty data, return empty array")
            return np.array([], dtype=np.float32)

        # 单个点的字节长度（3个float32，每个4字节）
        single_point_bytes = 12
        point_count = len(pc_data) // single_point_bytes

        if len(pc_data) % single_point_bytes != 0:
            remaining_bytes = len(pc_data) % single_point_bytes
            self.logger.warning(
                f"{desc}: Data size ({len(pc_data)}) is not multiple of 12, discard remaining {remaining_bytes} bytes"
            )

        # 初始化NumPy数组（N, 3），存储x/y/z
        point_cloud_np = np.zeros((point_count, 3), dtype=np.float32)

        # 循环解析每个点的x/y/z
        for i in range(point_count):
            point_start_offset = i * single_point_bytes
            single_point_data = pc_data[
                point_start_offset : point_start_offset + single_point_bytes
            ]

            # 解析x(0偏移)、y(4偏移)、z(8偏移)，均为float32（小端序）
            x = struct.unpack_from("<f", single_point_data, offset=0)[0]
            y = struct.unpack_from("<f", single_point_data, offset=4)[0]
            z = struct.unpack_from("<f", single_point_data, offset=8)[0]

            # 存入NumPy数组
            point_cloud_np[i] = [x, y, z]

        self.logger.info(
            f"{desc}: Parse completed! {point_count} points, NumPy shape: {point_cloud_np.shape}"
        )
        return point_cloud_np

    def send_request(self) -> bool:
        """主动发送请求数据到服务器（带日志输出）"""
        try:
            assert self.socket is not None, (
                "Socket not initialized, please connect first"
            )

            request_data = b"GET_TRACKING_DATA"
            self.socket.sendall(request_data)

            self.logger.info(
                f"Sent request: {request_data.decode('utf-8')} ({len(request_data)} bytes)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to send request: {str(e)}")
            return False

    def parse_byte_stream(self) -> Optional[TrackingResult]:
        """解析字节流，返回带类型标注的TrackingResult（核心业务逻辑）"""
        self.total_recv_size = 0  # 重置总接收字节数
        self.parse_cost_ms = 0.0
        self.logger.info("Starting byte stream parsing...")

        start_time = time.time()

        try:
            # 1. 解析内参（4个float64，32字节）
            intrinsics_data = self.recv_all(32, "Intrinsics")
            fx, fy, cx, cy = struct.unpack(">dddd", intrinsics_data)
            camera_intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
            self.logger.info(
                f"Intrinsics parsed: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
            )

            # 2. 解析服务状态（1字节，uint8→bool）
            success_data = self.recv_all(1, "Service Success")
            success = struct.unpack(">B", success_data)[0] == 1
            self.logger.info(f"Service status parsed: Success={success}")

            # 3. 解析相机位姿（7个float64，56字节）
            pose_data = self.recv_all(56, "Camera Pose")
            x, y, z, qw, qx, qy, qz = struct.unpack(">ddddddd", pose_data)
            camera_pose = CameraPose(
                position=Position(x=x, y=y, z=z),
                orientation=Orientation(w=qw, x=qx, y=qy, z=qz),
            )
            self.logger.info(
                f"Camera pose parsed: Position=({x:.6f}, {y:.6f}, {z:.6f}), Orientation=({qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})"
            )

            # 4. 解析相机坐标点云
            pc_camera_len_data = self.recv_all(4, "Camera Point Cloud Length")
            pc_camera_len = struct.unpack(">I", pc_camera_len_data)[0]

            if pc_camera_len > 0:
                pc_camera_raw = self.recv_all(pc_camera_len, "Camera Point Cloud Data")
                point_cloud_camera = self.parse_point_cloud_data(
                    pc_camera_raw, "Camera Point Cloud"
                )
            else:
                point_cloud_camera = np.array([], dtype=np.float32)

            pc_camera_info = PointCloudInfo(
                total_bytes=pc_camera_len,
                point_count=point_cloud_camera.shape[0]
                if point_cloud_camera.size > 0
                else 0,
            )
            self.logger.info(
                f"Camera point cloud parsed: {pc_camera_info.point_count} points"
            )

            # 5. 解析世界坐标点云
            pc_world_len_data = self.recv_all(4, "World Point Cloud Length")
            pc_world_len = struct.unpack(">I", pc_world_len_data)[0]

            if pc_world_len > 0:
                pc_world_raw = self.recv_all(pc_world_len, "World Point Cloud Data")
                point_cloud_world = self.parse_point_cloud_data(
                    pc_world_raw, "World Point Cloud"
                )
            else:
                point_cloud_world = np.array([], dtype=np.float32)

            pc_world_info = PointCloudInfo(
                total_bytes=pc_world_len,
                point_count=point_cloud_world.shape[0]
                if point_cloud_world.size > 0
                else 0,
            )
            self.logger.info(
                f"World point cloud parsed: {pc_world_info.point_count} points"
            )

            # 6. 解析图像数据
            img_len_data = self.recv_all(4, "Image Length")
            img_len = struct.unpack(">I", img_len_data)[0]
            current_image = None

            if img_len > 0:
                img_jpg_raw = self.recv_all(img_len, "Image Data")
                nparr = np.frombuffer(img_jpg_raw, dtype=np.uint8)
                current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            img_shape: Optional[Tuple[int, ...]] = (
                tuple(current_image.shape)  # type: ignore
                if (current_image is not None and current_image.size > 0)
                else None
            )
            image_info = ImageInfo(total_bytes=img_len, shape=img_shape)
            self.logger.info(f"Image parsed: {img_len} bytes, shape: {img_shape}")

            # 计算总解析耗时
            self.parse_cost_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Parsing completed! Total received: {self.total_recv_size} bytes, cost {self.parse_cost_ms:.2f} ms"
            )

            # 封装并返回带类型标注的结果
            return TrackingResult(
                intrinsics=camera_intrinsics,
                success=success,
                camera_pose=camera_pose,
                point_cloud_camera_info=pc_camera_info,
                tracked_points_camera=point_cloud_camera,
                point_cloud_world_info=pc_world_info,
                tracked_points_world=point_cloud_world,
                image_info=image_info,
                current_image=current_image,
                total_recv_size=self.total_recv_size,
                parse_cost_ms=self.parse_cost_ms,
            )

        except TimeoutError as e:
            self.logger.error(f"Timeout error during parsing: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Parse failed: {str(e)}")
            return None

    def close_connection(self) -> None:
        """关闭客户端连接（带日志输出）"""
        if self.socket is not None:
            try:
                self.socket.close()
                self.logger.info("Connection closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing connection: {str(e)}")
            finally:
                self.socket = None

    def complete_tracking_pipeline(self) -> Optional[TrackingResult]:
        """
        核心闭环方法：一键完成「连接→发送请求→解析数据→关闭连接」
        无需分步调用，返回带完整类型标注的TrackingResult对象
        :return: 解析完成的跟踪数据结果，失败返回None
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting complete tracking data pipeline")
        self.logger.info("=" * 60)

        try:
            # 步骤1：连接服务器
            if not self.connect_to_server():
                self.logger.error("Pipeline failed: Connection to server failed")
                return None

            # 步骤2：发送请求数据
            if not self.send_request():
                self.logger.error("Pipeline failed: Failed to send request")
                self.close_connection()
                return None

            # 步骤3：解析字节流数据
            tracking_result = self.parse_byte_stream()

            # 步骤4：无论解析成功与否，最终关闭连接
            self.close_connection()

            # 步骤5：返回结果
            if tracking_result is not None:
                self.logger.info("Pipeline completed successfully")
            else:
                self.logger.error("Pipeline completed with parsing failure")

            return tracking_result

        except Exception as e:
            self.logger.error(f"Pipeline aborted due to unexpected error: {str(e)}")
            self.close_connection()
            return None


# ===================== 跟踪数据验证类（保持原有逻辑，适配新数据结构） =====================
class TrackingDataValidator:
    """
    跟踪数据验证类：修复 y 轴巨大误差，优化畸变校正逻辑，匹配题目 Eigen 视觉计算场景
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        distortion_coeffs: Optional[List[float]] = None,
    ):
        """
        初始化验证器（适配新的带类型标注数据结构）
        Args:
            intrinsics (CameraIntrinsics): 相机内参（带类型标注）
            camera_pose (CameraPose): 相机位姿（带类型标注）
            distortion_coeffs (List[float]): 相机畸变参数 [k1, k2, p1, p2, k3]
        """
        self.intrinsics = intrinsics
        self.camera_pose = camera_pose
        self.distortion_coeffs = self._init_distortion_coeffs(distortion_coeffs)
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
        Yc = v_undist * camera_depth_z  # 关键修复：去掉负号
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
