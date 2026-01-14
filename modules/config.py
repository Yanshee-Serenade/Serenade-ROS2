"""
Configuration module for the Yanshee server.
Contains all constants and configuration values.
"""

import os
from typing import Tuple

# ===================== Model Configuration =====================
MODEL_QWEN_8B = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_QWEN_4B = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_QWEN_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_SMOLVLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_DA3_LARGE = "depth-anything/DA3-LARGE-1.1"
MODEL_DA3_NESTED = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
MODEL_VLM_DEFAULT = MODEL_QWEN_4B
MODEL_DA3_DEFAULT = MODEL_DA3_LARGE
MODEL_SAM3_PATH = (
    "/home/seqn/sam3/sam3.pt"  # Modify according to your model weight path
)

# ===================== Network Configuration =====================
ROS_SERVER_IP = "127.0.0.1"
ROS_SERVER_PORT = 51121
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 51122

# ===================== Generation Configuration =====================
MAX_NEW_TOKENS = 256
IMAGE_SAVE_PREFIX = "images/image_"
DEPTH_PLOT_SAVE_PREFIX = "images/depth_comparison_"
DA3_DEPTH_SAVE_PREFIX = "images/da3_depth_"
DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX = "images/da3_depth_with_keypoints_"

# ===================== Image Processing Configuration =====================
DEPTH_PROCESS_RES = 504
DEPTH_PROCESS_METHOD = "upper_bound_resize"
DEPTH_EXPORT_FORMAT = "glb"

# ===================== Path Configuration =====================
# Ensure image save directory exists
os.makedirs("images", exist_ok=True)

# Type aliases for better type hints
ImageShape = Tuple[int, int]
PointCloud = Tuple[float, float, float]
