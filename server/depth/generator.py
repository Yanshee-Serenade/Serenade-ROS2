"""
Depth generator module for DA3 depth map generation.

This module provides functions for generating depth maps using
Depth Anything 3 (DA3) models with proper scaling and interpolation.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..models.model_loader import ModelManager


def generate_depth_map(
    image_path: str,
    target_shape: Tuple[int, int],
    model_manager: Optional[ModelManager] = None,
) -> np.ndarray:
    """
    Generate depth map of specified size using INTER_CUBIC interpolation
    to match original image size.

    Args:
        image_path: Input image path
        target_shape: Target depth map size (height, width) (matches original image)
        model_manager: ModelManager instance with DA3 loaded. If None, creates new one.

    Returns:
        Depth map array matching target size

    Raises:
        ValueError: If DA3 model not initialized
        Exception: If depth generation fails
    """
    # Get DA3 model
    if model_manager is None:
        raise ValueError("ModelManager required for depth generation")

    if not model_manager.is_da3_loaded():
        raise ValueError("DA3 model not initialized, call load_model_da3() first")

    model_da3 = model_manager.get_da3()

    try:
        # Generate depth prediction
        prediction = model_da3.inference(
            image=[image_path],
            process_res=504,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="glb",
        )

        # Extract predicted depth map and resize to target shape
        depth_map = prediction.depth[0]

        # Resize using INTER_CUBIC interpolation for accuracy
        # cv2.resize expects (width, height) but target_shape is (height, width)
        depth_map_resized = cv2.resize(
            depth_map,
            (target_shape[1], target_shape[0]),  # (width, height)
            interpolation=cv2.INTER_CUBIC,
        )

        return depth_map_resized

    except Exception as e:
        raise Exception(f"Depth map generation failed: {str(e)}")


def validate_depth_map(depth_map: np.ndarray, image_shape: Tuple[int, int]) -> bool:
    """
    Validate depth map dimensions and values.

    Args:
        depth_map: Depth map array to validate
        image_shape: Expected image shape (height, width)

    Returns:
        True if depth map is valid, False otherwise
    """
    if depth_map is None:
        return False

    if depth_map.shape != image_shape:
        print(
            f"⚠️ Depth map shape {depth_map.shape} doesn't match image shape {image_shape}"
        )
        return False

    if np.all(depth_map <= 0):
        print("⚠️ Depth map contains only non-positive values")
        return False

    return True


def normalize_depth_map(depth_map: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """
    Normalize depth map for visualization.

    Args:
        depth_map: Input depth map
        percentile: Percentile for clipping outliers

    Returns:
        Normalized depth map in [0, 1] range
    """
    if depth_map is None or len(depth_map) == 0:
        return depth_map

    # Copy to avoid modifying original
    normalized = depth_map.copy()

    # Handle valid depth values (positive)
    valid_mask = normalized > 0
    if valid_mask.sum() <= 10:
        # Not enough valid points
        return np.zeros_like(normalized)

    # Apply inverse for visualization (closer = brighter)
    normalized[valid_mask] = 1 / normalized[valid_mask]

    # Calculate percentiles for normalization
    depth_min = np.percentile(normalized[valid_mask], percentile)
    depth_max = np.percentile(normalized[valid_mask], 100 - percentile)

    # Avoid division by zero
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6

    # Normalize to [0, 1] range
    normalized = (normalized - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0, 1)

    # Invert for visualization (closer = brighter)
    normalized = 1 - normalized

    return normalized
