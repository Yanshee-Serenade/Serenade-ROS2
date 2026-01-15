"""
Configuration module for server constants and settings.

This module defines all configuration constants used throughout the server,
including model paths, network settings, and generation parameters.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Server configuration with all constants and settings."""

    # ===================== Model Configuration =====================
    # Vision Language Models
    MODEL_QWEN_8B: str = "Qwen/Qwen3-VL-8B-Instruct"
    MODEL_QWEN_4B: str = "Qwen/Qwen3-VL-4B-Instruct"
    MODEL_QWEN_2B: str = "Qwen/Qwen3-VL-2B-Instruct"
    MODEL_SMOLVLM: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    # Depth Anything 3 Models
    MODEL_DA3_LARGE: str = "depth-anything/DA3-LARGE-1.1"
    MODEL_DA3_NESTED: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

    # SAM3 Model
    MODEL_SAM3_PATH: str = "/home/seqn/sam3/sam3.pt"

    # Default model selections
    MODEL_VLM_DEFAULT: str = MODEL_QWEN_4B
    MODEL_DA3_DEFAULT: str = MODEL_DA3_LARGE

    # ===================== Network Configuration =====================
    ROS_SERVER_IP: str = "127.0.0.1"
    ROS_SERVER_PORT: int = 51121
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 51122

    # ===================== Generation Configuration =====================
    MAX_NEW_TOKENS: int = 256

    # ===================== File Path Configuration =====================
    IMAGE_SAVE_PREFIX: str = "images/image_"
    DEPTH_PLOT_SAVE_PREFIX: str = "images/depth_comparison_"
    DA3_DEPTH_SAVE_PREFIX: str = "images/da3_depth_"
    DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX: str = "images/da3_depth_with_keypoints_"

    # ===================== Runtime Configuration =====================
    ROS_CLIENT_ENABLE_LOG: bool = False

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure image directory exists
        os.makedirs("images", exist_ok=True)

    @property
    def image_save_dir(self) -> str:
        """Get the image save directory."""
        return os.path.dirname(self.IMAGE_SAVE_PREFIX)

    def get_image_path(self, timestamp: str) -> str:
        """Get full image path for a timestamp."""
        return f"{self.IMAGE_SAVE_PREFIX}{timestamp}.jpg"

    def get_depth_plot_path(self, timestamp: str) -> str:
        """Get full depth plot path for a timestamp."""
        return f"{self.DEPTH_PLOT_SAVE_PREFIX}{timestamp}.png"

    def get_da3_depth_path(self, timestamp: str) -> str:
        """Get full DA3 depth image path for a timestamp."""
        return f"{self.DA3_DEPTH_SAVE_PREFIX}{timestamp}.png"

    def get_da3_depth_with_keypoints_path(self, timestamp: str) -> str:
        """Get full DA3 depth with keypoints path for a timestamp."""
        return f"{self.DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX}{timestamp}.png"


# Global configuration instance
config = Config()
