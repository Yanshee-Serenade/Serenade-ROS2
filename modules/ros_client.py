"""
ROS client module for tracking and image fetching.
Handles ROS client initialization and data retrieval.
"""

import datetime
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Import ROS API and types
import ros_api
from ros_api import TrackingResult

# Import configuration
from .config import IMAGE_SAVE_PREFIX, ROS_SERVER_IP, ROS_SERVER_PORT


def init_tracking_client(
    enable_log: bool = False,
) -> Optional[ros_api.TrackingDataClient]:
    """
    Initialize ROS tracking client.

    Args:
        enable_log: Whether to enable logging

    Returns:
        TrackingDataClient instance or None on failure
    """
    try:
        print("🔍 Initializing ROS tracking client...")

        # Create client instance
        client = ros_api.TrackingDataClient(
            server_ip=ROS_SERVER_IP, port=ROS_SERVER_PORT, enable_log=enable_log
        )

        print("✅ ROS tracking client initialized successfully")
        print(f"   Server: {ROS_SERVER_IP}:{ROS_SERVER_PORT}")
        print(f"   Logging: {'Enabled' if enable_log else 'Disabled'}")

        return client

    except Exception as e:
        print(f"❌ Failed to initialize ROS tracking client: {str(e)}")
        return None


def get_image_from_ros(
    client: ros_api.TrackingDataClient, timestamp: str
) -> Union[
    Tuple[Image.Image, str, np.ndarray, np.ndarray, np.ndarray],
    Tuple[None, str, None, None, None],
]:
    """
    Get image and point cloud data from ROS client.

    Args:
        client: TrackingDataClient instance
        timestamp: Timestamp for filename generation

    Returns:
        Tuple of (PIL image, image path, camera point cloud, world point cloud, OpenCV image)
        or error tuple on failure
    """
    if not client:
        return None, "ROS client instance is invalid", None, None, None

    try:
        print(f"[{timestamp}] 🔍 Starting ROS data pipeline...")

        # Execute complete tracking pipeline
        tracking_result: Optional[TrackingResult] = client.complete_tracking_pipeline()

        # Validate tracking result
        if not tracking_result:
            return (
                None,
                "ROS data pipeline failed (connection/parsing/request error)",
                None,
                None,
                None,
            )

        # Extract OpenCV image
        cv_image = tracking_result.current_image
        if cv_image is None or not isinstance(cv_image, np.ndarray):
            return None, "Failed to extract image from ROS result", None, None, None

        # Extract point cloud data
        camera_point_cloud = tracking_result.tracked_points_camera
        world_point_cloud = tracking_result.tracked_points_world

        # Convert OpenCV BGR to PIL RGB
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)

        # Save image locally
        image_save_path = f"{IMAGE_SAVE_PREFIX}{timestamp}.jpg"
        pil_image.save(image_save_path)

        print(f"[{timestamp}] 💾 Saved processed image to: {image_save_path}")
        print(f"   Size: {pil_image.size}, Mode: {pil_image.mode}")
        print(f"[{timestamp}] 📊 ROS data statistics:")
        print(f"   Total received: {tracking_result.total_recv_size} bytes")
        print(f"   Parse time: {tracking_result.parse_cost_ms:.2f} ms")

        return (
            pil_image,
            image_save_path,
            camera_point_cloud,
            world_point_cloud,
            cv_image,
        )

    except Exception as e:
        error_msg = f"Failed to process ROS data: {str(e)}"
        print(f"[{timestamp}] ❌ {error_msg}")
        return None, error_msg, None, None, None


def validate_point_cloud(
    point_cloud: np.ndarray, image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and filter point cloud data.

    Args:
        point_cloud: Camera point cloud (N, 3) where x=w, y=h, z=depth
        image_shape: Image dimensions (height, width)

    Returns:
        Tuple of (valid pixel_w, valid pixel_h, valid depths)
    """
    if point_cloud is None or len(point_cloud) == 0:
        return np.array([]), np.array([]), np.array([])

    # Extract pixel coordinates and depths
    pixel_w = point_cloud[:, 0].astype(np.int32)  # x = column (width)
    pixel_h = point_cloud[:, 1].astype(np.int32)  # y = row (height)
    depths = point_cloud[:, 2]  # z = depth

    # Create validation mask
    height, width = image_shape
    valid_mask = np.logical_and.reduce(
        [
            depths > 0,  # Valid depth
            pixel_w >= 0,
            pixel_w < width,  # Within image width
            pixel_h >= 0,
            pixel_h < height,  # Within image height
        ]
    )

    # Return valid data
    return pixel_w[valid_mask], pixel_h[valid_mask], depths[valid_mask]


def extract_depth_at_pixels(
    depth_map: np.ndarray, pixel_w: np.ndarray, pixel_h: np.ndarray
) -> np.ndarray:
    """
    Extract depth values at specific pixel coordinates.

    Args:
        depth_map: Depth map array (height, width)
        pixel_w: Pixel width coordinates (columns)
        pixel_h: Pixel height coordinates (rows)

    Returns:
        Depth values at specified pixels
    """
    if len(pixel_w) == 0 or len(pixel_h) == 0:
        return np.array([])

    # Extract depth values (depth_map[row, column])
    depths = depth_map[pixel_h, pixel_w]

    # Filter invalid depths
    valid_mask = depths > 0
    return depths[valid_mask]


def get_current_timestamp() -> str:
    """
    Get current timestamp string for filenames.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
