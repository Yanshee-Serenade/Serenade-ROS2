#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np


class CameraIntrinsics(NamedTuple):
    """相机内参数据结构（带类型标注）"""

    fx: float
    fy: float
    cx: float
    cy: float


class DistortionCoefficients(NamedTuple):
    """相机畸变系数数据结构（带类型标注）"""

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float


class Position(NamedTuple):
    """位置数据结构（带类型标注）"""

    x: float
    y: float
    z: float


class Orientation(NamedTuple):
    """姿态四元数数据结构（带类型标注）"""

    w: float
    x: float
    y: float
    z: float


@dataclass
class CameraPoseData:
    """Camera pose data structure"""

    topic: str
    ts: float
    frame: str
    position: Position = field(default_factory=lambda: Position(0.0, 0.0, 0.0))
    orientation: Orientation = field(
        default_factory=lambda: Orientation(1.0, 0.0, 0.0, 0.0)
    )

    @classmethod
    def from_dict(cls, data: Dict) -> "CameraPoseData":
        """Create CameraPoseData from dictionary"""
        return cls(
            topic=data.get("topic", ""),
            ts=data.get("ts", 0.0),
            frame=data.get("frame", ""),
            position=Position(*data.get("p", [0.0, 0.0, 0.0])),
            orientation=Orientation(*data.get("q", [0.0, 0.0, 0.0, 1.0])),
        )

    def to_dict(self) -> Dict:
        """Convert CameraPoseData to dictionary"""
        return {
            "topic": self.topic,
            "ts": self.ts,
            "frame": self.frame,
            "p": list(self.position),
            "q": list(self.orientation),
        }


class CameraPose(NamedTuple):
    """相机位姿数据结构（带类型标注）"""

    position: Position
    orientation: Orientation


class PointCloudInfo(NamedTuple):
    """点云信息数据结构（带类型标注）"""

    total_bytes: int
    point_count: int


class ImageInfo(NamedTuple):
    """图像信息数据结构（带类型标注）"""

    total_bytes: int
    shape: Optional[Tuple[int, ...]]


class TrackingResult(NamedTuple):
    """跟踪数据最终结果（带完整类型标注，作为闭环方法返回值）"""

    success: bool
    camera_pose: CameraPose
    point_cloud_camera_info: PointCloudInfo
    tracked_points_camera: np.ndarray
    point_cloud_world_info: PointCloudInfo
    tracked_points_world: np.ndarray
    image_info: ImageInfo
    current_image: Optional[np.ndarray]
    total_recv_size: int
    parse_cost_ms: float
