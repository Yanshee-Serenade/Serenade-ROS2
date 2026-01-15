"""
Model loader module for managing AI models.

This module provides the ModelManager class and utility functions
for loading and managing various AI models used by the server.
"""

import time
from typing import Any, Optional

import torch
from depth_anything_3.api import DepthAnything3
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ..config import config


class ModelManager:
    """Manages loading and access to all AI models."""

    def __init__(self):
        """Initialize model manager with empty model references."""
        self.processor: Optional[Any] = None
        self.model_vlm: Optional[Any] = None
        self.model_da3: Optional[DepthAnything3] = None
        self.processor_sam3: Optional[Sam3Processor] = None

    def load_vlm(self, model_path: str = config.MODEL_VLM_DEFAULT) -> None:
        """
        Load and compile Vision Language Model.

        Args:
            model_path: Path or name of the VLM model

        Raises:
            Exception: If model loading fails
        """
        try:
            # 1. Load processor
            print(
                f"{time.time()} > Loading model processor: {model_path}...", flush=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

            # 2. Load model
            print(f"{time.time()} > Loading model weights...", flush=True)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model_vlm = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.model_vlm.eval()

            print(f"{time.time()} > ✅ VLM model loaded and compiled!", flush=True)
        except Exception as e:
            raise Exception(f"VLM model loading failed: {str(e)}")

    def load_da3(self, model_path: str = config.MODEL_DA3_DEFAULT) -> None:
        """
        Load Depth Anything 3 model.

        Args:
            model_path: Path or name of the DA3 model

        Raises:
            Exception: If model loading fails
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_da3 = DepthAnything3.from_pretrained(model_path)
            self.model_da3 = self.model_da3.to(device=device, dtype=torch.float32)
            self.model_da3.eval()
            if torch.cuda.is_available():
                self.model_da3 = torch.compile(  # pyright: ignore[reportAttributeAccessIssue]
                    self.model_da3,
                    mode="reduce-overhead",
                )
            print(f"{time.time()} > ✅ DA3 model loaded and compiled!", flush=True)
        except Exception as e:
            raise Exception(f"DA3 model loading failed: {str(e)}")

    def load_sam3(self, model_path: str = config.MODEL_SAM3_PATH) -> None:
        """
        Load SAM3 model.

        Args:
            model_path: Path to SAM3 model weights

        Raises:
            Exception: If model loading fails
        """
        try:
            model = build_sam3_image_model(
                load_from_HF=False,
                checkpoint_path=model_path,
                compile=True,
            )
            self.processor_sam3 = Sam3Processor(model)
            print(f"{time.time()} > ✅ SAM3 model loaded and compiled!", flush=True)
        except Exception as e:
            raise Exception(f"SAM3 model loading failed: {str(e)}")

    def load_all_models(self) -> None:
        """Load all models with default configurations."""
        self.load_vlm()
        self.load_da3()
        self.load_sam3()

    def get_vlm(self) -> Any:
        """Get VLM model instance."""
        if self.model_vlm is None:
            raise ValueError("VLM model not loaded")
        return self.model_vlm

    def get_processor(self) -> Any:
        """Get VLM processor instance."""
        if self.processor is None:
            raise ValueError("VLM processor not loaded")
        return self.processor

    def get_da3(self) -> DepthAnything3:
        """Get DA3 model instance."""
        if self.model_da3 is None:
            raise ValueError("DA3 model not loaded")
        return self.model_da3

    def get_sam3_processor(self) -> Optional[Sam3Processor]:
        """Get SAM3 processor instance."""
        return self.processor_sam3

    def is_vlm_loaded(self) -> bool:
        """Check if VLM model is loaded."""
        return self.model_vlm is not None and self.processor is not None

    def is_da3_loaded(self) -> bool:
        """Check if DA3 model is loaded."""
        return self.model_da3 is not None

    def is_sam3_loaded(self) -> bool:
        """Check if SAM3 model is loaded."""
        return self.processor_sam3 is not None


# Convenience functions for backward compatibility
def load_model_vlm(model_path: str = config.MODEL_VLM_DEFAULT) -> ModelManager:
    """
    Load VLM model and return model manager.

    Args:
        model_path: Path or name of the VLM model

    Returns:
        ModelManager instance with VLM loaded
    """
    manager = ModelManager()
    manager.load_vlm(model_path)
    return manager


def load_model_da3(model_path: str = config.MODEL_DA3_DEFAULT) -> ModelManager:
    """
    Load DA3 model and return model manager.

    Args:
        model_path: Path or name of the DA3 model

    Returns:
        ModelManager instance with DA3 loaded
    """
    manager = ModelManager()
    manager.load_da3(model_path)
    return manager


def load_model_sam3(model_path: str = config.MODEL_SAM3_PATH) -> ModelManager:
    """
    Load SAM3 model and return model manager.

    Args:
        model_path: Path to SAM3 model weights

    Returns:
        ModelManager instance with SAM3 loaded
    """
    manager = ModelManager()
    manager.load_sam3(model_path)
    return manager
