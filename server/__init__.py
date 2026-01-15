"""
Server module for AI vision and depth estimation.

This module provides a modular system for:
- ROS data acquisition and processing
- AI model loading and management
- Depth estimation and comparison
- Flask API for vision queries
"""

from .api.flask_app import create_flask_app, run_server
from .config import Config
from .depth.comparison import plot_depth_comparison, save_da3_depth_with_ros_keypoints
from .depth.generator import generate_depth_map
from .factory import create_server
from .models.model_loader import (
    ModelManager,
    load_model_da3,
    load_model_sam3,
    load_model_vlm,
)
from .ros.client import TrackingClient
from .ros.data_processor import get_image_from_ros
from .text.generator import generate_text_stream

__all__ = [
    # Main entry points
    "create_server",
    "create_flask_app",
    "run_server",
    "Config",
    # Model management
    "ModelManager",
    "load_model_vlm",
    "load_model_da3",
    "load_model_sam3",
    # ROS integration
    "TrackingClient",
    "get_image_from_ros",
    # Depth processing
    "generate_depth_map",
    "plot_depth_comparison",
    "save_da3_depth_with_ros_keypoints",
    # Text generation
    "generate_text_stream",
]
