"""
Client configuration for connecting to model servers.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for serenade nodes."""

    # Camera configuration
    CAMERA_INTRINSICS_FX: float = 503.640273
    CAMERA_INTRINSICS_FY: float = 502.167721
    CAMERA_INTRINSICS_CX: float = 312.565456
    CAMERA_INTRINSICS_CY: float = 244.436855
    DISTORTION_COEFFS: tuple = (0.148509, -0.255395, 0.003505, 0.001639, 0.0)

    # Vision Language Models
    MODEL_QWEN_8B: str = "Qwen/Qwen3-VL-8B-Instruct"
    MODEL_QWEN_4B: str = "Qwen/Qwen3-VL-4B-Instruct"
    MODEL_QWEN_2B: str = "Qwen/Qwen3-VL-2B-Instruct"
    MODEL_SMOLVLM: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    # Default model selections
    MODEL_VLM_DEFAULT: str = MODEL_QWEN_2B

    # Generation configuration
    MAX_NEW_TOKENS: int = 256


# Global configuration instance
config = Config()
