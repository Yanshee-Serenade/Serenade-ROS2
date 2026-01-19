"""
Configuration module for server constants and settings.

This module defines all configuration constants used throughout the server,
including model paths, network settings, and generation parameters.
"""

from dataclasses import dataclass

from serenade_ros_api import CameraIntrinsics, DistortionCoefficients


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
    MODEL_DA3_BASE: str = "depth-anything/DA3-BASE"
    MODEL_DA3_LARGE: str = "depth-anything/DA3-LARGE-1.1"
    MODEL_DA3_NESTED: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

    # SAM3 Model
    MODEL_SAM3_PATH: str = "/home/seqn/sam3/sam3.pt"

    # Default model selections
    MODEL_VLM_DEFAULT: str = MODEL_QWEN_2B
    MODEL_DA3_DEFAULT: str = MODEL_DA3_LARGE

    # ===================== Network Configuration =====================
    ROS_SERVER_IP: str = "127.0.0.1"
    ROS_SERVER_PORT: int = 21121
    FLASK_HOST: str = "0.0.0.0"
    VLM_PORT: int = 21122
    DA3_PORT: int = 21123
    SAM3_PORT: int = 21124

    # ===================== Generation Configuration =====================
    MAX_NEW_TOKENS: int = 256

    # ===================== File Path Configuration =====================
    IMAGE_SAVE_PREFIX: str = "images/image_"
    DEPTH_PLOT_SAVE_PREFIX: str = "images/depth_comparison_"
    DA3_DEPTH_SAVE_PREFIX: str = "images/da3_depth_"
    DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX: str = "images/da3_depth_with_keypoints_"

    # ===================== Runtime Configuration =====================
    ROS_CLIENT_ENABLE_LOG: bool = False

    # ===================== Camera Configuration =====================
    CAMERA_INTRINSICS: CameraIntrinsics = CameraIntrinsics(
        503.640273, 502.167721, 312.565456, 244.436855
    )
    DISTORTION_COEFFS: DistortionCoefficients = DistortionCoefficients(
        0.148509, -0.255395, 0.003505, 0.001639, 0.0
    )


# Global configuration instance
config = Config()
