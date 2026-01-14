#!/usr/bin/env python3
import logging
import socket
from typing import List, Optional, Tuple, Union


class JointAngleTCPClient:
    """关节角度TCP客户端，连接本地51120端口的ROS服务器（重构版，带类型标注）"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 51120,
        timeout: int = 10,
        enable_log: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        初始化客户端
        :param host: 服务器地址（默认localhost）
        :param port: 服务器端口（默认51120）
        :param timeout: 连接/接收超时时间（秒）
        :param enable_log: 是否启用日志
        :param log_level: 日志级别
        """
        self.host: str = host
        self.port: int = port
        self.timeout: int = timeout
        self.socket: Optional[socket.socket] = None

        # 初始化日志器
        self.enable_log: bool = enable_log
        self.logger: logging.Logger = logging.getLogger("JointAngleTCPClient")
        if enable_log:
            self.logger.setLevel(log_level)
            if not self.logger.handlers:
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        else:
            self.logger.disabled = True

    def _connect(self) -> bool:
        """建立与服务器的TCP连接（内部方法）"""
        if self.socket:
            self._close()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.host}:{self.port}: {str(e)}")
            self.socket = None
            return False

    def _close(self) -> None:
        """关闭TCP连接（内部方法）"""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            finally:
                self.socket = None

    def set_joint_angles(
        self, angle_list: List[float], time_ms: int = 100
    ) -> Tuple[bool, str]:
        """
        设置关节角度（发送SET指令到服务器）
        :param angle_list: 长度为17的角度列表（单位：度）
        :param time_ms: 时间参数（默认100ms）
        :return: 执行结果（布尔值：是否成功，字符串：详细信息）
        """
        # 1. 校验输入参数
        if len(angle_list) != 17:
            return False, f"Invalid angle list length: {len(angle_list)} (expected 17)"

        try:
            # 2. 构建请求指令
            angle_str = " ".join(map(str, angle_list))
            request = f"SET {time_ms} {angle_str}\n"

            # 3. 建立连接并发送请求
            if not self._connect():
                return False, "Connection failed"

            if self.socket:
                self.socket.sendall(request.encode("utf-8"))

            # 4. 接收并解析响应
            if self.socket:
                response = self.socket.recv(4096).decode("utf-8").strip()
            else:
                return False, "Socket not available"
            if response.startswith("SUCCESS"):
                return True, response
            else:
                return False, response

        except Exception as e:
            return False, f"Request failed: {str(e)}"
        finally:
            self._close()

    def get_joint_angles(
        self, req_type: int = 0, req_buf: str = ""
    ) -> Tuple[bool, Union[List[int], str]]:
        """
        获取关节角度（发送GET指令到服务器）
        :param req_type: 服务请求type参数（默认0）
        :param req_buf: 服务请求buf参数（默认空字符串）
        :return: 执行结果（布尔值：是否成功，列表/字符串：角度数组或详细错误）
        """
        try:
            # 1. 构建请求指令
            request = f"GET {req_type} {req_buf}\n"

            # 2. 建立连接并发送请求
            if not self._connect():
                return False, "Connection failed"

            if self.socket:
                self.socket.sendall(request.encode("utf-8"))
            else:
                return False, "Socket not available"

            # 3. 接收并解析响应
            if self.socket:
                response = self.socket.recv(4096).decode("utf-8").strip()
            else:
                return False, "Socket not available"
            if response.startswith("SUCCESS"):
                # 拆分出角度数组并转换为整数列表
                angle_part = response.split(": ")[1]
                angle_array = list(map(int, angle_part.split(",")))
                return True, angle_array
            else:
                return False, response

        except Exception as e:
            return False, f"Request failed: {str(e)}"
        finally:
            self._close()
