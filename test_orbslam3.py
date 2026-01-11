#!/usr/bin/env python3
import requests
import json
import struct
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Point3D:
    """三维点数据结构"""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Point3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

@dataclass
class Pose:
    """位姿数据结构"""
    position: Point3D
    orientation: Tuple[float, float, float, float]  # (x, y, z, w)
    timestamp: datetime
    frame_id: str
    seq: int
    
    def __str__(self):
        pos = self.position
        orient = self.orientation
        return (f"Pose(seq={self.seq}, frame='{self.frame_id}', "
                f"position=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                f"orientation=({orient[0]:.3f}, {orient[1]:.3f}, {orient[2]:.3f}, {orient[3]:.3f}))")

class ORBSLAM3Client:
    """ORB-SLAM3数据客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 51121):
        """
        初始化客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        # 设置较短的超时时间，避免阻塞
        self.session.timeout = 2.0
    
    def get_image(self) -> Optional[bytes]:
        """
        获取最新图像
        
        Returns:
            JPEG格式的图像字节数据，失败返回None
        """
        try:
            response = self.session.get(f"{self.base_url}/image")
            if response.status_code == 200:
                return response.content
            elif response.status_code == 503:
                print("No image data available")
                return None
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image: {e}")
            return None
    
    def get_pointcloud_raw(self) -> Optional[Dict[str, Any]]:
        """
        获取原始点云数据（JSON格式）
        
        Returns:
            点云数据的字典表示，失败返回None
        """
        try:
            response = self.session.get(f"{self.base_url}/pointcloud")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                print("No pointcloud data available")
                return None
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pointcloud: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing pointcloud JSON: {e}")
            raise
    
    def get_pose_raw(self) -> Optional[Dict[str, Any]]:
        """
        获取原始位姿数据（JSON格式）
        
        Returns:
            位姿数据的字典表示，失败返回None
        """
        try:
            response = self.session.get(f"{self.base_url}/pose")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                print("No pose data available")
                return None
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pose: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing pose JSON: {e}")
            raise
    
    def parse_pointcloud(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        解析点云数据为numpy数组
        
        Args:
            raw_data: 原始点云数据字典
            
        Returns:
            Nx3的numpy数组，每行是一个点(x, y, z)
            
        Raises:
            ValueError: 数据格式错误
        """
        try:
            # 检查数据格式
            if not isinstance(raw_data, dict):
                raise ValueError("Raw data must be a dictionary")
            
            # 提取数据
            data_bytes = bytes(raw_data['data'])
            point_step = raw_data['point_step']
            width = raw_data['width']
            height = raw_data['height']
            
            # 验证数据大小
            expected_size = height * width * point_step
            if len(data_bytes) != expected_size:
                raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data_bytes)}")
            
            # 解析字段信息
            fields = raw_data['fields']
            field_offsets = {}
            for field in fields:
                if field['name'] in ['x', 'y', 'z']:
                    field_offsets[field['name']] = field['offset']
            
            # 确保有x, y, z字段
            if not all(coord in field_offsets for coord in ['x', 'y', 'z']):
                raise ValueError("Missing required fields (x, y, z)")
            
            # 解析点云数据
            points = []
            for i in range(height * width):
                offset = i * point_step
                
                # 读取x, y, z坐标（假设是FLOAT32格式）
                x_bytes = data_bytes[offset + field_offsets['x']:offset + field_offsets['x'] + 4]
                y_bytes = data_bytes[offset + field_offsets['y']:offset + field_offsets['y'] + 4]
                z_bytes = data_bytes[offset + field_offsets['z']:offset + field_offsets['z'] + 4]
                
                # 字节序处理
                is_bigendian = raw_data.get('is_bigendian', False)
                endian_char = '>' if is_bigendian else '<'
                
                x = struct.unpack(f'{endian_char}f', x_bytes)[0]
                y = struct.unpack(f'{endian_char}f', y_bytes)[0]
                z = struct.unpack(f'{endian_char}f', z_bytes)[0]
                
                points.append([x, y, z])
            
            return np.array(points, dtype=np.float32)
            
        except (KeyError, IndexError, struct.error) as e:
            raise ValueError(f"Failed to parse pointcloud data: {e}")
    
    def parse_pose(self, raw_data: Dict[str, Any]) -> Pose:
        """
        解析位姿数据
        
        Args:
            raw_data: 原始位姿数据字典
            
        Returns:
            解析后的Pose对象
            
        Raises:
            ValueError: 数据格式错误
        """
        try:
            # 提取头部信息
            header = raw_data['header']
            pose_data = raw_data['pose']
            position = pose_data['position']
            orientation = pose_data['orientation']
            
            # 创建时间戳
            timestamp = datetime.fromtimestamp(
                header['stamp']['secs'] + header['stamp']['nsecs'] / 1e9
            )
            
            # 创建Pose对象
            return Pose(
                position=Point3D(
                    x=position['x'],
                    y=position['y'],
                    z=position['z']
                ),
                orientation=(
                    orientation['x'],
                    orientation['y'],
                    orientation['z'],
                    orientation['w']
                ),
                timestamp=timestamp,
                frame_id=header['frame_id'],
                seq=header['seq']
            )
            
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse pose data: {e}")
    
    def get_pointcloud(self) -> Optional[np.ndarray]:
        """
        获取并解析点云数据
        
        Returns:
            解析后的点云numpy数组，失败返回None
        """
        raw_data = self.get_pointcloud_raw()
        if raw_data is not None:
            try:
                return self.parse_pointcloud(raw_data)
            except ValueError as e:
                print(f"Error parsing pointcloud: {e}")
        return None
    
    def get_pose(self) -> Optional[Pose]:
        """
        获取并解析位姿数据
        
        Returns:
            解析后的Pose对象，失败返回None
        """
        raw_data = self.get_pose_raw()
        if raw_data is not None:
            try:
                return self.parse_pose(raw_data)
            except ValueError as e:
                print(f"Error parsing pose: {e}")
        return None
    
    def get_all_data(self) -> Tuple[Optional[bytes], Optional[np.ndarray], Optional[Pose]]:
        """
        一次性获取所有数据
        
        Returns:
            (image, pointcloud, pose) 三元组
        """
        image = self.get_image()
        pointcloud = self.get_pointcloud()
        pose = self.get_pose()
        return image, pointcloud, pose

# 使用示例
if __name__ == '__main__':
    # 创建客户端
    client = ORBSLAM3Client("localhost", 51121)
    
    # 获取图像
    image_data = client.get_image()
    if image_data:
        print(f"Got image data: {len(image_data)} bytes")
        with open('image.jpg', 'wb') as f:
            f.write(image_data)
    
    # 获取点云
    pointcloud = client.get_pointcloud()
    if pointcloud is not None:
        print(f"Got pointcloud with {len(pointcloud)} points")
        if len(pointcloud) > 0:
            print(f"First point: {pointcloud[0]}")
    
    # 获取位姿
    pose = client.get_pose()
    if pose is not None:
        print(f"Got pose: {pose}")