"""
Configuration module for server constants and settings.

This module defines all configuration constants used throughout the server,
including model paths, network settings, and generation parameters.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Server configuration with all constants and settings."""

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
