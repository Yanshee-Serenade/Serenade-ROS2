"""
Depth comparison module for analysis and visualization.

This module provides functions for comparing depth estimates from
ORB-SLAM3 and DA3, including visualization and error analysis.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_3.utils.visualize import visualize_depth

from ..config import config

# Non-interactive backend for headless environments
matplotlib.use("Agg")


def plot_depth_comparison(
    camera_point_cloud: np.ndarray,
    da3_depth_map: np.ndarray,
    timestamp: str,
    image_shape: tuple[int, int],
) -> None:
    """
    Extract DA3 depth at ORB-SLAM3 pixel coordinates and plot comparison.

    Args:
        camera_point_cloud: Camera coordinate point cloud (N, 3) where:
            x = pixel width, y = pixel height, z = ORB-SLAM3 actual depth
        da3_depth_map: DA3 generated depth map (height, width)
        timestamp: Timestamp for generating save filename
        image_shape: Original image size (height, width) for pixel validation

    Returns:
        None, saves plot to file
    """
    # 1. Pre-validation
    if camera_point_cloud is None or len(camera_point_cloud) == 0:
        print(f"[{timestamp}] ⚠️ No valid ORB-SLAM3 point cloud data, skipping plot")
        return

    if da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ Depth map size doesn't match image size, skipping plot")
        return

    # 2. Extract point cloud data
    pixel_w = camera_point_cloud[:, 0].astype(np.int32)  # x = image column (width)
    pixel_h = camera_point_cloud[:, 1].astype(np.int32)  # y = image row (height)
    orb_slam_depth = camera_point_cloud[:, 2]  # z = ORB-SLAM3 actual depth (true value)

    # 3. Filter invalid data
    valid_mask = np.logical_and.reduce(
        [
            orb_slam_depth > 0,  # Filter invalid depth (<=0)
            pixel_w >= 0,
            pixel_w < image_shape[1],  # Filter pixels outside image width
            pixel_h >= 0,
            pixel_h < image_shape[0],  # Filter pixels outside image height
        ]
    )

    # 4. Extract valid data
    valid_pixel_w = pixel_w[valid_mask]
    valid_pixel_h = pixel_h[valid_mask]
    valid_orb_slam_depth = orb_slam_depth[valid_mask]

    if len(valid_orb_slam_depth) == 0:
        print(
            f"[{timestamp}] ⚠️ No valid pixel coordinates or depth data, skipping plot"
        )
        return

    # 5. Extract DA3 depth at exact pixel coordinates (h, w)
    valid_da3_depth = da3_depth_map[valid_pixel_h, valid_pixel_w]

    # 6. Filter DA3 invalid depth (<=0)
    final_valid_mask = valid_da3_depth > 0
    final_orb_slam_depth = valid_orb_slam_depth[final_valid_mask]
    final_da3_depth = valid_da3_depth[final_valid_mask]

    if len(final_orb_slam_depth) == 0:
        print(f"[{timestamp}] ⚠️ No valid DA3 depth data, skipping plot")
        return

    # 7. Create comparison plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Scatter plot (shows one-to-one depth relationship)
    plt.subplot(1, 2, 1)
    plt.scatter(final_orb_slam_depth, final_da3_depth, alpha=0.7, s=8, c="royalblue")

    # Add diagonal line (ideal match)
    max_depth = np.max([np.max(final_orb_slam_depth), np.max(final_da3_depth)])
    plt.plot([0, max_depth], [0, max_depth], "r--", alpha=0.8, label="Ideal Match")

    plt.xlabel("ORB-SLAM3 True Depth (m)")
    plt.ylabel("DA3 Predicted Depth (m)")
    plt.title("ORB-SLAM3 vs DA3 Depth (Pixel-wise Match)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Error histogram
    plt.subplot(1, 2, 2)
    depth_diff = final_orb_slam_depth - final_da3_depth
    plt.hist(
        depth_diff, bins=50, alpha=0.7, color="purple", edgecolor="black", linewidth=0.5
    )
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.8, label="Zero Error")
    plt.xlabel("Depth Error (ORB-SLAM3 - DA3) (m)")
    plt.ylabel("Point Count")
    plt.title("Depth Error Distribution Histogram")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Adjust layout and save
    plt.tight_layout()
    plot_save_path = config.get_depth_plot_path(timestamp)
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 9. Print statistics
    print(f"[{timestamp}] 💾 Depth comparison plot saved to: {plot_save_path}")
    print(f"[{timestamp}] 📊 Valid comparison points: {len(final_orb_slam_depth)}")
    print(f"[{timestamp}] 📊 Mean absolute error: {np.mean(np.abs(depth_diff)):.6f} m")
    print(
        f"[{timestamp}] 📊 Root mean square error: {np.sqrt(np.mean(depth_diff**2)):.6f} m"
    )


def save_da3_depth_with_ros_keypoints(
    da3_depth_map: np.ndarray,
    camera_point_cloud: np.ndarray,
    timestamp: str,
    image_shape: tuple[int, int],
) -> None:
    """
    Save DA3 depth map with ROS keypoints overlay.

    1. Save original DA3 depth map (colored visualization)
    2. Overlay ROS keypoints on depth map: white border + ORB-SLAM3 true depth color fill

    Args:
        da3_depth_map: DA3 generated depth map (height, width)
        camera_point_cloud: Camera coordinate point cloud (N, 3)
        timestamp: Timestamp for generating save filename
        image_shape: Original image size (height, width) for validation

    Returns:
        None, saves images to files
    """
    # Pre-validation
    if da3_depth_map is None or da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ DA3 depth map invalid or size mismatch, skipping")
        return

    # Step 1: Visualize DA3 depth map (colored, matches depth_anything_3 style)
    da3_depth_viz_result = visualize_depth(da3_depth_map, cmap="plasma")

    # Handle potential tuple return from visualize_depth
    if isinstance(da3_depth_viz_result, tuple):
        da3_depth_viz = da3_depth_viz_result[0]
    else:
        da3_depth_viz = da3_depth_viz_result

    # Convert to 8-bit RGB
    da3_depth_viz = (da3_depth_viz * 255).astype(np.uint8)
    if len(da3_depth_viz.shape) == 2:  # Grayscale to RGB
        import cv2

        da3_depth_viz = cv2.cvtColor(da3_depth_viz, cv2.COLOR_GRAY2RGB)

    # Step 2: Save original colored DA3 depth map
    da3_depth_save_path = config.get_da3_depth_path(timestamp)
    import cv2

    cv2.imwrite(da3_depth_save_path, da3_depth_viz)
    print(
        f"[{timestamp}] 💾 Original DA3 colored depth map saved to: {da3_depth_save_path}"
    )

    # Step 3: Overlay ROS keypoints (white border + ORB-SLAM3 true depth color fill)
    da3_depth_with_keypoints = da3_depth_viz.copy()

    if camera_point_cloud is not None and len(camera_point_cloud) > 0:
        # Extract pixel coordinates and ORB-SLAM3 true depth
        pixel_w = camera_point_cloud[:, 0].astype(np.int32)
        pixel_h = camera_point_cloud[:, 1].astype(np.int32)
        orb_slam_depth = camera_point_cloud[:, 2]  # ORB-SLAM3 true depth

        # Filter valid pixel coordinates and depth values
        valid_mask = np.logical_and.reduce(
            [
                orb_slam_depth > 0,
                pixel_w >= 0,
                pixel_w < image_shape[1],
                pixel_h >= 0,
                pixel_h < image_shape[0],
            ]
        )

        valid_pixel_w = pixel_w[valid_mask]
        valid_pixel_h = pixel_h[valid_mask]
        valid_orb_slam_depth = orb_slam_depth[valid_mask]

        if len(valid_pixel_w) > 0:
            # Keypoint parameters
            outer_radius = 3  # White border radius
            inner_radius = 2  # ORB true depth color radius
            white_color = (255, 255, 255)  # White border (BGR format)
            percentile = 2  # Matches visualize_depth default percentile

            # Normalize ORB-SLAM3 depth for color mapping
            orb_depth_processed = valid_orb_slam_depth.copy()

            # Apply inverse for visualization (matches visualize_depth logic)
            orb_valid_mask = orb_depth_processed > 0
            orb_depth_processed[orb_valid_mask] = (
                1 / orb_depth_processed[orb_valid_mask]
            )

            # Calculate percentiles
            if orb_valid_mask.sum() <= 10:
                orb_depth_min = 0
                orb_depth_max = 0
            else:
                orb_depth_min = np.percentile(
                    orb_depth_processed[orb_valid_mask], percentile
                )
                orb_depth_max = np.percentile(
                    orb_depth_processed[orb_valid_mask], 100 - percentile
                )

            # Avoid division by zero
            if orb_depth_min == orb_depth_max:
                orb_depth_min = orb_depth_min - 1e-6
                orb_depth_max = orb_depth_max + 1e-6

            # Normalize to [0, 1] range
            normalized_orb_depth = (
                (orb_depth_processed - orb_depth_min) / (orb_depth_max - orb_depth_min)
            ).clip(0, 1)

            # Invert for visualization
            normalized_orb_depth = 1 - normalized_orb_depth

            # Get plasma colormap (matches DA3 depth map coloring)
            plasma_cmap = matplotlib.colormaps["plasma"]

            for idx, (w, h) in enumerate(zip(valid_pixel_w, valid_pixel_h)):
                # 1. Draw white solid outer circle (border, easy to identify)
                cv2.circle(
                    img=da3_depth_with_keypoints,
                    center=(w, h),
                    radius=outer_radius,
                    color=white_color,
                    thickness=-1,  # Solid fill
                )

                # 2. Extract RGB color for ORB-SLAM3 true depth
                norm_depth = normalized_orb_depth[idx]
                orb_rgb = plasma_cmap(norm_depth)[:3]  # Get RGB values (0~1 range)
                orb_rgb_255 = (np.array(orb_rgb) * 255).astype(np.uint8)

                # Convert to BGR for OpenCV
                orb_bgr = (
                    int(orb_rgb_255[2]),
                    int(orb_rgb_255[1]),
                    int(orb_rgb_255[0]),
                )

                # 3. Draw ORB true depth colored inner circle
                cv2.circle(
                    img=da3_depth_with_keypoints,
                    center=(w, h),
                    radius=inner_radius,
                    color=orb_bgr,
                    thickness=-1,
                )

            # Add keypoint count annotation
            cv2.putText(
                img=da3_depth_with_keypoints,
                text=f"Valid ROS Keypoints: {len(valid_pixel_w)}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=white_color,
                thickness=2,
            )

    # Step 4: Save DA3 depth map with keypoints overlay
    da3_depth_keypoints_save_path = config.get_da3_depth_with_keypoints_path(timestamp)
    cv2.imwrite(da3_depth_keypoints_save_path, da3_depth_with_keypoints)
    print(
        f"[{timestamp}] 💾 DA3 depth map with ROS keypoints saved to: {da3_depth_keypoints_save_path}"
    )


def calculate_depth_metrics(orb_slam_depth: np.ndarray, da3_depth: np.ndarray) -> dict:
    """
    Calculate depth comparison metrics.

    Args:
        orb_slam_depth: ORB-SLAM3 depth values
        da3_depth: DA3 depth values

    Returns:
        Dictionary containing depth comparison metrics
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
        "median_absolute_error": float(np.median(np.abs(depth_diff))),
        "max_absolute_error": float(np.max(np.abs(depth_diff))),
        "correlation": float(np.corrcoef(orb_slam_depth, da3_depth)[0, 1]),
    }

    return metrics
