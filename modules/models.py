"""
Models module for AI model management.
Handles loading and initialization of VLM, DA3, and SAM3 models.
"""

from typing import Optional, Tuple

import torch
from depth_anything_3.api import DepthAnything3


# visualize_depth is used in depth_processing module
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

# Global model instances
model_vlm: Optional[AutoModelForImageTextToText] = None
processor_vlm: Optional[AutoProcessor] = None
model_da3: Optional[DepthAnything3] = None
processor_sam3: Optional[Sam3Processor] = None
model_sam3: Optional[torch.nn.Module] = None


def load_model_vlm(
    model_name: str,
) -> Tuple[Optional[AutoModelForImageTextToText], Optional[AutoProcessor]]:
    """
    Load VLM (Vision Language Model) and its processor.

    Args:
        model_name: Name or path of the VLM model

    Returns:
        Tuple of (model, processor) or (None, None) on failure
    """
    global model_vlm, processor_vlm

    try:
        print(f"🔍 Loading VLM model: {model_name}")

        # Load processor and model
        processor_vlm = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        model_vlm_instance = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model_vlm = model_vlm_instance  # Assign to global variable

        print(f"✅ VLM model loaded successfully: {model_name}")
        return model_vlm, processor_vlm

    except Exception as e:
        print(f"❌ Failed to load VLM model {model_name}: {str(e)}")
        model_vlm = None
        processor_vlm = None
        return None, None


def load_model_da3(model_name: str) -> Optional[DepthAnything3]:
    """
    Load Depth Anything 3 model.

    Args:
        model_name: Name or path of the DA3 model

    Returns:
        DA3 model instance or None on failure
    """
    global model_da3

    try:
        print(f"🔍 Loading DA3 model: {model_name}")

        # Load DA3 model
        model_da3 = DepthAnything3.from_pretrained(model_name)

        print(f"✅ DA3 model loaded successfully: {model_name}")
        return model_da3

    except Exception as e:
        print(f"❌ Failed to load DA3 model {model_name}: {str(e)}")
        model_da3 = None
        return None


def load_model_sam3(
    model_path: str,
) -> Tuple[Optional[torch.nn.Module], Optional[Sam3Processor]]:
    """
    Load SAM3 model and processor.

    Args:
        model_path: Path to SAM3 model weights

    Returns:
        Tuple of (model, processor) or (None, None) on failure
    """
    global model_sam3, processor_sam3

    try:
        print(f"🔍 Loading SAM3 model from: {model_path}")

        # Load SAM3 model
        model_sam3 = build_sam3_image_model(model_path)
        processor_sam3 = Sam3Processor(model_sam3)

        print("✅ SAM3 model loaded successfully")
        return model_sam3, processor_sam3

    except Exception as e:
        print(f"❌ Failed to load SAM3 model from {model_path}: {str(e)}")
        model_sam3 = None
        processor_sam3 = None
        return None, None


def get_vlm_model() -> Optional[AutoModelForImageTextToText]:
    """Get the loaded VLM model instance."""
    return model_vlm


def get_vlm_processor() -> Optional[AutoProcessor]:
    """Get the loaded VLM processor instance."""
    return processor_vlm


def get_da3_model() -> Optional[DepthAnything3]:
    """Get the loaded DA3 model instance."""
    return model_da3


def get_sam3_model() -> Optional[torch.nn.Module]:
    """Get the loaded SAM3 model instance."""
    return model_sam3


def get_sam3_processor() -> Optional[Sam3Processor]:
    """Get the loaded SAM3 processor instance."""
    return processor_sam3


def is_vlm_loaded() -> bool:
    """Check if VLM model is loaded."""
    return model_vlm is not None and processor_vlm is not None


def is_da3_loaded() -> bool:
    """Check if DA3 model is loaded."""
    return model_da3 is not None


def is_sam3_loaded() -> bool:
    """Check if SAM3 model is loaded."""
    return model_sam3 is not None and processor_sam3 is not None


def cleanup_models():
    """Clean up model resources."""
    global model_vlm, processor_vlm, model_da3, model_sam3, processor_sam3

    print("🧹 Cleaning up model resources...")

    # Clear VLM
    if model_vlm is not None:
        del model_vlm
        model_vlm = None

    if processor_vlm is not None:
        del processor_vlm
        processor_vlm = None

    # Clear DA3
    if model_da3 is not None:
        del model_da3
        model_da3 = None

    # Clear SAM3
    if model_sam3 is not None:
        del model_sam3
        model_sam3 = None

    if processor_sam3 is not None:
        del processor_sam3
        processor_sam3 = None

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✅ Model resources cleaned up")
