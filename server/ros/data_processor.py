"""
ROS data processor module for image extraction and processing.

This module provides functions for extracting and processing images
and point cloud data from ROS tracking results.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ros_api import TrackingResult

from ..config import config


def get_image_from_ros(
    client, timestamp: str
) -> Tuple[
    Optional[Image.Image],
    str,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Get image and point cloud data from ROS client and convert to PIL Image.

    Args:
        client: TrackingClient instance
        timestamp: Timestamp for generating image filename

    Returns:
        Tuple containing:
        - PIL Image object
        - Image save path (or error message if failed)
        - Camera coordinate point cloud
        - World coordinate point cloud
        - Original OpenCV image

        If failed, returns (None, error_message, None, None, None)
    """
    if not client or not client.is_connected():
        return None, "ROS client instance invalid", None, None, None

    try:
        # ============== Core refactoring: Call complete pipeline method ==============
        print(f"[{timestamp}] 🔍 Starting ROS data pipeline acquisition...")
        tracking_result: Optional[TrackingResult] = client.get_tracking_data()

        # Validate pipeline method return result (strongly typed object)
        if not tracking_result:
            return (
                None,
                "ROS data pipeline failed (connection/parsing/request any link error)",
                None,
                None,
                None,
            )

        # ============== Extract data from strongly typed TrackingResult ==============
        # 1. Extract OpenCV image (keep original image for matching depth map size)
        cv_image = tracking_result.current_image
        if cv_image is None or not isinstance(cv_image, np.ndarray):
            return (
                None,
                "Failed to extract image from ROS pipeline result",
                None,
                None,
                None,
            )

        # 2. Extract ORB-SLAM3 point cloud data (camera/world coordinates)
        camera_point_cloud = tracking_result.tracked_points_camera
        world_point_cloud = tracking_result.tracked_points_world

        # 3. Convert OpenCV image to PIL Image (CV2: BGR → PIL: RGB)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)

        # 4. Save image locally
        image_save_path = config.get_image_path(timestamp)
        pil_image.save(image_save_path)
        print(
            f"[{timestamp}] 💾 Saved processed image to: {image_save_path} "
            f"(size: {pil_image.size}, mode: {pil_image.mode})"
        )
        print(
            f"[{timestamp}] 📊 ROS data pipeline statistics: "
            f"Total received {tracking_result.total_recv_size} bytes, "
            f"parse time {tracking_result.parse_cost_ms:.2f}ms"
        )

        return (
            pil_image,
            image_save_path,
            camera_point_cloud,
            world_point_cloud,
            cv_image,
        )

    except Exception as e:
        error_msg = f"Failed to process data from ROS pipeline: {str(e)}"
        print(f"[{timestamp}] ❌ {error_msg}")
        return None, error_msg, None, None, None


def extract_image_shape(cv_image: np.ndarray) -> Tuple[int, int]:
    """
    Extract image shape from OpenCV image.

    Args:
        cv_image: OpenCV image array

    Returns:
        Image shape as (height, width)

    Raises:
        ValueError: If cv_image is None or invalid
    """
    if cv_image is None:
        raise ValueError("cv_image is None")

    return (int(cv_image.shape[0]), int(cv_image.shape[1]))  # (h, w)


def validate_point_cloud(
    point_cloud: Optional[np.ndarray], image_shape: Tuple[int, int]
) -> bool:
    """
    Validate point cloud data.

    Args:
        point_cloud: Point cloud array or None
        image_shape: Image shape as (height, width)

    Returns:
        True if point cloud is valid, False otherwise
    """
    if point_cloud is None or len(point_cloud) == 0:
        return False

    # Check if point cloud coordinates are within image bounds
    if len(point_cloud.shape) != 2 or point_cloud.shape[1] < 2:
        return False

    return True
