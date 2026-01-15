"""
Depth comparison module for analysis and visualization.

This module provides functions for comparing depth estimates from
ORB-SLAM3 and DA3, including visualization and error analysis.
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..config import config

# Non-interactive backend for headless environments
matplotlib.use("Agg")


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit y = mx using least squares (no intercept).
    Args:
        x: Input array (DA3 depths)
        y: Target array (ORB-SLAM3 depths)
    Returns:
        slope (m)
    """
    # Reshape x to (N, 1) for lstsq
    A = x[:, np.newaxis]
    m = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m[0])


def _apply_colormap(
    data: np.ndarray, vmin: float, vmax: float, cmap_name: str = "plasma"
) -> np.ndarray:
    """
    Apply colormap to depth data using fixed linear bounds, inverted.
    """
    # Normalize to 0-1
    norm_data = (data - vmin) / (vmax - vmin + 1e-6)
    norm_data = np.clip(norm_data, 0, 1)

    # Invert the normalization (0 becomes 1, 1 becomes 0)
    norm_data = 1.0 - norm_data

    # Get colormap
    cmap = matplotlib.colormaps[cmap_name]

    # Apply colormap (returns RGBA 0-1)
    colored = cmap(norm_data)

    # Convert to RGB 0-255 uint8
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)

    return colored_uint8


def plot_depth_comparison(
    camera_point_cloud: np.ndarray,
    da3_depth_map: np.ndarray,
    timestamp: str,
    image_shape: tuple[int, int],
) -> None:
    """
    Extract DA3 depth, align it to ORB-SLAM3 via linear regression (scale only), and plot comparison.
    """
    # 1. Pre-validation
    if camera_point_cloud is None or len(camera_point_cloud) == 0:
        print(f"[{timestamp}] ⚠️ No valid ORB-SLAM3 point cloud data, skipping plot")
        return

    if da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ Depth map size doesn't match image size, skipping plot")
        return

    # 2. Extract point cloud data
    pixel_w = camera_point_cloud[:, 0].astype(np.int32)
    pixel_h = camera_point_cloud[:, 1].astype(np.int32)
    orb_slam_depth = camera_point_cloud[:, 2]

    # 3. Filter invalid data
    valid_mask = (
        (orb_slam_depth > 0)
        & (pixel_w >= 0)
        & (pixel_w < image_shape[1])
        & (pixel_h >= 0)
        & (pixel_h < image_shape[0])
    )

    valid_pixel_w = pixel_w[valid_mask]
    valid_pixel_h = pixel_h[valid_mask]
    valid_orb_slam_depth = orb_slam_depth[valid_mask]

    if len(valid_orb_slam_depth) == 0:
        return

    # 4. Extract DA3 depth at exact pixel coordinates
    valid_da3_depth = da3_depth_map[valid_pixel_h, valid_pixel_w]

    # 5. Filter DA3 invalid depth
    final_valid_mask = valid_da3_depth > 0
    final_orb_slam_depth = valid_orb_slam_depth[final_valid_mask]
    final_da3_depth = valid_da3_depth[final_valid_mask]

    if len(final_orb_slam_depth) < 2:
        print(f"[{timestamp}] ⚠️ Not enough points for regression, skipping plot")
        return

    # 6. Perform Linear Regression: ORB = m * DA3 (Force intercept to 0)
    slope = _fit_linear_regression(final_da3_depth, final_orb_slam_depth)

    # 7. Convert DA3 depths to ORB reference
    aligned_da3_depth = final_da3_depth * slope

    # 8. Create comparison plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(final_orb_slam_depth, aligned_da3_depth, alpha=0.7, s=8, c="royalblue")

    # Add diagonal line (ideal match)
    max_depth = float(
        np.maximum(np.max(final_orb_slam_depth), np.max(aligned_da3_depth))
    )
    plt.plot([0, max_depth], [0, max_depth], "r--", alpha=0.8, label="Ideal Match")

    plt.xlabel("ORB-SLAM3 True Depth (m)")
    plt.ylabel(f"Aligned DA3 Depth (m)\n(Scale={slope:.2f})")
    plt.title("Depth Comparison (Scale Alignment Only)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Error histogram
    plt.subplot(1, 2, 2)
    depth_diff = final_orb_slam_depth - aligned_da3_depth
    plt.hist(
        depth_diff, bins=50, alpha=0.7, color="purple", edgecolor="black", linewidth=0.5
    )
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.8, label="Zero Error")
    plt.xlabel("Error (ORB - Aligned DA3) (m)")
    plt.ylabel("Count")
    plt.title("Depth Error Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 9. Adjust layout and save
    plt.tight_layout()
    plot_save_path = config.get_depth_plot_path(timestamp)
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{timestamp}] 💾 Depth comparison plot saved to: {plot_save_path}")
    print(f"[{timestamp}] 📊 Regression: ORB = {slope:.4f} * DA3")


def save_da3_depth_with_ros_keypoints(
    da3_depth_map: np.ndarray,
    camera_point_cloud: np.ndarray,
    timestamp: str,
    image_shape: tuple[int, int],
) -> None:
    """
    Save aligned DA3 depth map with ROS keypoints overlay using unified coloring.

    The DA3 map is first aligned to ORB-SLAM3 scale via linear regression (y=mx).
    Then, both the map and the keypoints are colored using the SAME global min/max.
    """
    # Pre-validation
    if da3_depth_map is None or da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ DA3 depth map invalid or size mismatch, skipping")
        return

    # Data extraction for regression
    slope = 1.0
    orb_points_available = False

    valid_pixel_w, valid_pixel_h, valid_orb_slam_depth = [], [], []

    if camera_point_cloud is not None and len(camera_point_cloud) > 0:
        pixel_w = camera_point_cloud[:, 0].astype(np.int32)
        pixel_h = camera_point_cloud[:, 1].astype(np.int32)
        orb_vals = camera_point_cloud[:, 2]

        mask = (
            (orb_vals > 0)
            & (pixel_w >= 0)
            & (pixel_w < image_shape[1])
            & (pixel_h >= 0)
            & (pixel_h < image_shape[0])
        )

        valid_pixel_w = pixel_w[mask]
        valid_pixel_h = pixel_h[mask]
        valid_orb_slam_depth = orb_vals[mask]

        if len(valid_orb_slam_depth) > 10:
            # Extract corresponding DA3 points
            corresponding_da3 = da3_depth_map[valid_pixel_h, valid_pixel_w]

            # Filter where DA3 is valid
            valid_corresp_mask = corresponding_da3 > 0

            if np.sum(valid_corresp_mask) > 10:
                orb_fit = valid_orb_slam_depth[valid_corresp_mask]
                da3_fit = corresponding_da3[valid_corresp_mask]

                # Calculate Linear Regression (Scale Only)
                slope = _fit_linear_regression(da3_fit, orb_fit)
                orb_points_available = True
                print(f"[{timestamp}] ℹ️ Alignment factor found: scale={slope:.3f}")

    # 1. Align the full DA3 Map
    # Apply regression to entire map to bring it to ORB-SLAM3 metric scale
    aligned_da3_map = da3_depth_map * slope

    # 2. Determine Global Colorscale Bounds (vmin, vmax)
    # We use the aligned map's range to ensure the background is fully visible.
    valid_map_pixels = aligned_da3_map[aligned_da3_map > 0]
    if len(valid_map_pixels) > 0:
        vmin = float(np.min(valid_map_pixels))
        vmax = float(np.max(valid_map_pixels))
    else:
        vmin, vmax = 0.0, 10.0

    # 3. Generate Colored Map
    # Shape: (H, W, 1) -> (H, W, 3)
    # We treat the aligned map as a 2D array for coloring
    da3_depth_viz = _apply_colormap(aligned_da3_map, vmin, vmax, cmap_name="plasma")

    # Save original colored aligned map
    da3_depth_save_path = config.get_da3_depth_path(timestamp)
    # Convert RGB (matplotlib) to BGR (OpenCV)
    cv2.imwrite(da3_depth_save_path, cv2.cvtColor(da3_depth_viz, cv2.COLOR_RGB2BGR))

    # 4. Overlay Keypoints
    da3_depth_with_keypoints = cv2.cvtColor(
        da3_depth_viz, cv2.COLOR_RGB2BGR
    )  # Working in BGR now

    if orb_points_available:
        outer_radius = 4
        inner_radius = 2
        white_color = (255, 255, 255)

        # Get colormap instance for single point conversion
        cmap = matplotlib.colormaps["plasma"]

        for i in range(len(valid_pixel_w)):
            w, h = valid_pixel_w[i], valid_pixel_h[i]
            orb_val = valid_orb_slam_depth[i]

            # Normalize ORB value using THE SAME vmin/vmax as the background map
            norm_val = (orb_val - vmin) / (vmax - vmin + 1e-6)
            norm_val = np.clip(norm_val, 0, 1)

            # Invert for point coloring too, so it matches the background map
            norm_val = 1.0 - norm_val

            # Get Color
            rgba = cmap(norm_val)
            r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            # OpenCV uses BGR
            color_bgr = (b, g, r)

            # Draw
            cv2.circle(da3_depth_with_keypoints, (w, h), outer_radius, white_color, -1)
            cv2.circle(da3_depth_with_keypoints, (w, h), inner_radius, color_bgr, -1)

        cv2.putText(
            img=da3_depth_with_keypoints,
            text=f"Aligned | Pts: {len(valid_pixel_w)} | Scale: {slope:.2f}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=white_color,
            thickness=2,
        )

    da3_depth_keypoints_save_path = config.get_da3_depth_with_keypoints_path(timestamp)
    cv2.imwrite(da3_depth_keypoints_save_path, da3_depth_with_keypoints)
    print(
        f"[{timestamp}] 💾 Aligned DA3 map with keypoints saved to: {da3_depth_keypoints_save_path}"
    )


def calculate_depth_metrics(orb_slam_depth: np.ndarray, da3_depth: np.ndarray) -> dict:
    """
    Calculate depth comparison metrics.
    """
    if len(orb_slam_depth) == 0 or len(da3_depth) == 0:
        return {}

    depth_diff = orb_slam_depth - da3_depth

    metrics = {
        "num_points": len(orb_slam_depth),
        "mean_orb_depth": float(np.mean(orb_slam_depth)),
        "mean_da3_depth": float(np.mean(da3_depth)),
        "mean_absolute_error": float(np.mean(np.abs(depth_diff))),
        "root_mean_square_error": float(np.sqrt(np.mean(depth_diff**2))),
        "correlation": float(np.corrcoef(orb_slam_depth, da3_depth)[0, 1]),
    }

    return metrics
