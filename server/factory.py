"""
Factory module for server creation.

This module provides the create_server function which simplifies
the creation and initialization of the complete server system.
"""

import datetime
from typing import Optional

from .config import Config, config
from .models.model_loader import ModelManager


def create_server(
    vlm_model: str = config.MODEL_VLM_DEFAULT,
    da3_model: str = config.MODEL_DA3_DEFAULT,
    load_sam3: bool = False,
    sam3_model_path: str = config.MODEL_SAM3_PATH,
) -> ModelManager:
    """
    Create and initialize a complete server system.

    This factory function handles the complete initialization process:
    1. Initialize all AI models (VLM, DA3, optionally SAM3)
    2. Configure the server with proper settings
    3. Return a fully initialized ModelManager

    Args:
        vlm_model: Vision Language Model path/name
        da3_model: Depth Anything 3 model path/name
        load_sam3: Whether to load SAM3 model
        sam3_model_path: Path to SAM3 model weights

    Returns:
        Fully initialized ModelManager instance

    Raises:
        RuntimeError: If model initialization fails
    """
    print(
        f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] 🏗️  Creating server..."
    )

    # 1. Create model manager
    print("1. Initializing model manager...")
    try:
        model_manager = ModelManager()
        print("   ✓ Model manager initialized")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model manager: {e}")

    # 2. Load Vision Language Model
    print("\n2. Loading Vision Language Model...")
    try:
        model_manager.load_vlm(vlm_model)
        print(f"   ✓ VLM model loaded: {vlm_model}")
    except Exception as e:
        raise RuntimeError(f"Failed to load VLM model: {e}")

    # 3. Load Depth Anything 3 model
    print("\n3. Loading Depth Anything 3 model...")
    try:
        model_manager.load_da3(da3_model)
        print(f"   ✓ DA3 model loaded: {da3_model}")
    except Exception as e:
        raise RuntimeError(f"Failed to load DA3 model: {e}")

    # 4. Load SAM3 model (optional)
    if load_sam3:
        print("\n4. Loading SAM3 model...")
        try:
            model_manager.load_sam3(sam3_model_path)
            print(f"   ✓ SAM3 model loaded: {sam3_model_path}")
        except Exception as e:
            print(f"   ⚠ Failed to load SAM3 model: {e}")
            print("   ⚠ Continuing without SAM3 support")

    # 5. Verify all required models are loaded
    print("\n5. Verifying model status...")
    if not model_manager.is_vlm_loaded():
        raise RuntimeError("VLM model failed to load")
    if not model_manager.is_da3_loaded():
        raise RuntimeError("DA3 model failed to load")

    print("   ✓ All required models loaded successfully")

    # 6. Print server configuration
    print("\n6. Server configuration:")
    print(f"   • VLM Model: {vlm_model}")
    print(f"   • DA3 Model: {da3_model}")
    print(f"   • SAM3 Loaded: {model_manager.is_sam3_loaded()}")
    print(f"   • ROS Server: {config.ROS_SERVER_IP}:{config.ROS_SERVER_PORT}")
    print(f"   • Flask Host: {config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"   • Image Directory: {config.image_save_dir}")

    print(
        f"\n[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] ✅ Server created successfully!"
    )

    return model_manager


def create_server_with_config(custom_config: Optional[Config] = None) -> ModelManager:
    """
    Create server with custom configuration.

    Args:
        custom_config: Custom configuration instance. If None, uses default config.

    Returns:
        Fully initialized ModelManager instance
    """
    if custom_config is None:
        custom_config = config

    return create_server(
        vlm_model=custom_config.MODEL_VLM_DEFAULT,
        da3_model=custom_config.MODEL_DA3_DEFAULT,
        sam3_model_path=custom_config.MODEL_SAM3_PATH,
    )
