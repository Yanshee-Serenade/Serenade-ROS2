import logging
import socket
import struct
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from serenade_ros_api.logger import setup_client_logger
from serenade_ros_api.model import (
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
        port: int = 21121,
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
            # 1. 解析服务状态（1字节，uint8→bool）
            success_data = self.recv_all(1, "Service Success")
            success = struct.unpack(">B", success_data)[0] == 1
            self.logger.info(f"Service status parsed: Success={success}")

            # 2. 解析相机位姿（7个float64，56字节）
            pose_data = self.recv_all(56, "Camera Pose")
            x, y, z, qw, qx, qy, qz = struct.unpack(">ddddddd", pose_data)
            camera_pose = CameraPose(
                position=Position(x=x, y=y, z=z),
                orientation=Orientation(w=qw, x=qx, y=qy, z=qz),
            )
            self.logger.info(
                f"Camera pose parsed: Position=({x:.6f}, {y:.6f}, {z:.6f}), Orientation=({qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})"
            )

            # 3. 解析相机坐标点云
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

            # 4. 解析世界坐标点云
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

            # 5. 解析图像数据
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
