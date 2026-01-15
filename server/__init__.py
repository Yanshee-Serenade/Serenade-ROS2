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
from .image.convert import cv2_to_pil
from .models.model_loader import (
    ModelManager,
    load_model_da3,
    load_model_sam3,
    load_model_vlm,
)
from .text.generator import generate_text_stream

__all__ = [
    # Main entry points
    "create_flask_app",
    "run_server",
    "Config",
    # Model management
    "ModelManager",
    "load_model_vlm",
    "load_model_da3",
    "load_model_sam3",
    # Depth processing
    "generate_depth_map",
    "plot_depth_comparison",
    "save_da3_depth_with_ros_keypoints",
    # Text generation
    "generate_text_stream",
    # Image conversion
    "cv2_to_pil",
]
