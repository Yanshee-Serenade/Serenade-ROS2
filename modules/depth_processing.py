"""
Depth processing module for depth map generation and analysis.
Handles DA3 depth map generation, visualization, and comparison with ROS data.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from depth_anything_3.utils.visualize import visualize_depth

# Import configuration and models
from .config import (
    DA3_DEPTH_SAVE_PREFIX,
    DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX,
    DEPTH_EXPORT_FORMAT,
    DEPTH_PLOT_SAVE_PREFIX,
    DEPTH_PROCESS_METHOD,
    DEPTH_PROCESS_RES,
    ImageShape,
)
from .models import get_da3_model


def generate_depth_map(
    image_path: str, target_shape: ImageShape
) -> Optional[np.ndarray]:
    """
    Generate depth map with specified target dimensions.

    Args:
        image_path: Path to input image
        target_shape: Target depth map dimensions (height, width)

    Returns:
        Depth map array matching target shape, or None on failure
    """
    model_da3 = get_da3_model()
    if model_da3 is None:
        raise Exception("DA3 model not initialized. Call load_model_da3() first.")

    try:
        # Generate depth prediction
        prediction = model_da3.inference(
            image=[image_path],
            process_res=DEPTH_PROCESS_RES,
            process_res_method=DEPTH_PROCESS_METHOD,
            export_dir=None,
            export_format=DEPTH_EXPORT_FORMAT,
        )

        # Extract and resize depth map
        depth_map = prediction.depth[0]
        height, width = target_shape

        # Resize to target shape using cubic interpolation for accuracy
        depth_map_resized = cv2.resize(
            depth_map,
            (width, height),  # cv2.resize uses (width, height)
            interpolation=cv2.INTER_CUBIC,
        )

        return depth_map_resized

    except Exception as e:
        print(f"❌ Failed to generate depth map: {str(e)}")
        return None


def visualize_depth_map(depth_map: np.ndarray, cmap: str = "plasma") -> np.ndarray:
    """
    Visualize depth map with color mapping.

    Args:
        depth_map: Depth map array
        cmap: Color map to use

    Returns:
        Colorized depth visualization as uint8 RGB array
    """
    # Generate colorized depth visualization
    depth_viz_result = visualize_depth(depth_map, cmap=cmap)

    # Handle return type - visualize_depth may return a tuple or array
    if isinstance(depth_viz_result, tuple):
        # Extract the visualization array from tuple
        depth_viz = (
            depth_viz_result[0] if len(depth_viz_result) > 0 else depth_viz_result
        )
    else:
        depth_viz = depth_viz_result

    # Convert to 0-255 uint8
    depth_viz = (depth_viz * 255).astype(np.uint8)

    # Convert grayscale to RGB if needed
    if len(depth_viz.shape) == 2:
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2RGB)

    return depth_viz


def save_depth_visualization(
    depth_map: np.ndarray, save_path: str, cmap: str = "plasma"
) -> bool:
    """
    Save depth map visualization to file.

    Args:
        depth_map: Depth map array
        save_path: Path to save the visualization
        cmap: Color map to use

    Returns:
        True if successful, False otherwise
    """
    try:
        depth_viz = visualize_depth_map(depth_map, cmap)
        cv2.imwrite(save_path, depth_viz)
        return True
    except Exception as e:
        print(f"❌ Failed to save depth visualization: {str(e)}")
        return False


def plot_depth_comparison(
    camera_point_cloud: np.ndarray,
    da3_depth_map: np.ndarray,
    timestamp: str,
    image_shape: ImageShape,
) -> Optional[str]:
    """
    Compare ORB-SLAM3 depths with DA3 predicted depths.

    Args:
        camera_point_cloud: Camera coordinate point cloud (N, 3) where x=w, y=h, z=depth
        da3_depth_map: DA3 generated depth map (height, width)
        timestamp: Timestamp for filename
        image_shape: Original image dimensions (height, width)

    Returns:
        Path to saved plot, or None on failure
    """
    # Validate inputs
    if camera_point_cloud is None or len(camera_point_cloud) == 0:
        print(f"[{timestamp}] ⚠️ No valid ORB-SLAM3 point cloud data, skipping plot")
        return None

    if da3_depth_map.shape != image_shape:
        print(
            f"[{timestamp}] ⚠️ Depth map shape {da3_depth_map.shape} doesn't match image shape {image_shape}, skipping plot"
        )
        return None

    # Import ROS client functions for validation
    from .ros_client import extract_depth_at_pixels, validate_point_cloud

    # Validate and filter point cloud
    pixel_w, pixel_h, orb_slam_depths = validate_point_cloud(
        camera_point_cloud, image_shape
    )

    if len(orb_slam_depths) == 0:
        print(
            f"[{timestamp}] ⚠️ No valid pixel coordinates or depth data, skipping plot"
        )
        return None

    # Extract DA3 depths at valid pixels
    da3_depths = extract_depth_at_pixels(da3_depth_map, pixel_w, pixel_h)

    if len(da3_depths) == 0:
        print(f"[{timestamp}] ⚠️ No valid DA3 depth data, skipping plot")
        return None

    # Ensure we have matching data points
    min_len = min(len(orb_slam_depths), len(da3_depths))
    if min_len == 0:
        print(f"[{timestamp}] ⚠️ No matching data points, skipping plot")
        return None

    orb_slam_depths = orb_slam_depths[:min_len]
    da3_depths = da3_depths[:min_len]

    # Calculate depth differences
    depth_diff = orb_slam_depths - da3_depths

    # Create figure
    plt.figure(figsize=(12, 6))

    # Subplot 1: Scatter plot (depth comparison)
    plt.subplot(1, 2, 1)
    plt.scatter(orb_slam_depths, da3_depths, alpha=0.7, s=8, c="royalblue")

    # Add ideal match line
    max_depth = max(np.max(orb_slam_depths), np.max(da3_depths))
    plt.plot([0, max_depth], [0, max_depth], "r--", alpha=0.8, label="Ideal Match")

    plt.xlabel("ORB-SLAM3 True Depth (m)")
    plt.ylabel("DA3 Predicted Depth (m)")
    plt.title("ORB-SLAM3 vs DA3 Depth (Pixel-wise Match)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Error histogram
    plt.subplot(1, 2, 2)
    plt.hist(
        depth_diff, bins=50, alpha=0.7, color="purple", edgecolor="black", linewidth=0.5
    )
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.8, label="Zero Error")

    plt.xlabel("Depth Error (ORB-SLAM3 - DA3) (m)")
    plt.ylabel("Point Count")
    plt.title("Depth Error Distribution Histogram")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plot_save_path = f"{DEPTH_PLOT_SAVE_PREFIX}{timestamp}.png"

    try:
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Print statistics
        print(f"[{timestamp}] 💾 Depth comparison plot saved to: {plot_save_path}")
        print(f"[{timestamp}] 📊 Valid comparison points: {min_len}")
        print(
            f"[{timestamp}] 📊 Mean absolute error: {np.mean(np.abs(depth_diff)):.6f} m"
        )
        print(
            f"[{timestamp}] 📊 Root mean square error: {np.sqrt(np.mean(depth_diff**2)):.6f} m"
        )

        return plot_save_path

    except Exception as e:
        print(f"[{timestamp}] ❌ Failed to save depth comparison plot: {str(e)}")
        plt.close()
        return None


def save_da3_depth_with_keypoints(
    da3_depth_map: np.ndarray,
    camera_point_cloud: np.ndarray,
    timestamp: str,
    image_shape: ImageShape,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Save DA3 depth map with ROS keypoints overlay.

    Args:
        da3_depth_map: DA3 depth map array
        camera_point_cloud: Camera point cloud for keypoints
        timestamp: Timestamp for filename
        image_shape: Image dimensions (height, width)

    Returns:
        Tuple of (depth_map_path, keypoints_path) or (None, None) on failure
    """
    # Validate inputs
    if da3_depth_map is None or da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ DA3 depth map invalid or shape mismatch, skipping")
        return None, None

    # Step 1: Visualize DA3 depth map
    da3_depth_viz = visualize_depth_map(da3_depth_map, cmap="plasma")

    # Step 2: Save original depth visualization
    da3_depth_path = f"{DA3_DEPTH_SAVE_PREFIX}{timestamp}.png"
    try:
        cv2.imwrite(da3_depth_path, da3_depth_viz)
        print(f"[{timestamp}] 💾 DA3 depth visualization saved to: {da3_depth_path}")
    except Exception as e:
        print(f"[{timestamp}] ❌ Failed to save DA3 depth visualization: {str(e)}")
        da3_depth_path = None

    # Step 3: Create copy for keypoints overlay
    if camera_point_cloud is not None and len(camera_point_cloud) > 0:
        keypoints_image = da3_depth_viz.copy()

        # Import ROS client functions
        from .ros_client import validate_point_cloud

        # Get valid pixel coordinates
        pixel_w, pixel_h, depths = validate_point_cloud(camera_point_cloud, image_shape)

        if len(pixel_w) > 0:
            # Draw keypoints with white border and color-coded fill
            for w, h, depth in zip(pixel_w, pixel_h, depths):
                # Draw white border (slightly larger circle)
                cv2.circle(keypoints_image, (w, h), 4, (255, 255, 255), 1)

                # Color code based on depth (normalize to colormap)
                # Simple color coding: blue for close, red for far
                depth_normalized = min(depth / 10.0, 1.0)  # Normalize to 0-10m range
                color_b = int(255 * (1 - depth_normalized))
                color_r = int(255 * depth_normalized)
                cv2.circle(keypoints_image, (w, h), 3, (color_b, 0, color_r), -1)

            # Save keypoints overlay
            keypoints_path = f"{DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX}{timestamp}.png"
            try:
                cv2.imwrite(keypoints_path, keypoints_image)
                print(
                    f"[{timestamp}] 💾 DA3 depth with keypoints saved to: {keypoints_path}"
                )
            except Exception as e:
                print(f"[{timestamp}] ❌ Failed to save keypoints overlay: {str(e)}")
                keypoints_path = None
        else:
            print(f"[{timestamp}] ⚠️ No valid keypoints to overlay")
            keypoints_path = None
    else:
        print(f"[{timestamp}] ⚠️ No point cloud data for keypoints")
        keypoints_path = None

    return da3_depth_path, keypoints_path


def calculate_depth_statistics(
    depth_map: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate depth map statistics.

    Args:
        depth_map: Depth map array
        valid_mask: Optional mask for valid pixels

    Returns:
        Dictionary of depth statistics
    """
    if valid_mask is not None:
        valid_depths = depth_map[valid_mask]
    else:
        valid_depths = depth_map[depth_map > 0]

    if len(valid_depths) == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "valid_pixels": 0,
            "total_pixels": depth_map.size,
        }

    return {
        "min": float(np.min(valid_depths)),
        "max": float(np.max(valid_depths)),
        "mean": float(np.mean(valid_depths)),
        "median": float(np.median(valid_depths)),
        "std": float(np.std(valid_depths)),
        "valid_pixels": len(valid_depths),
        "total_pixels": depth_map.size,
    }
