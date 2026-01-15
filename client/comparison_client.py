"""
Comparison Client - Reimplemented depth comparison using DA3 client and ROS client.
Performs depth comparison analysis between DA3 and ORB-SLAM3.
"""

import datetime
from typing import Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ros_api import TrackingDataClient

from .config import client_config
from .da3_client import DA3Client

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
    A = x[:, np.newaxis]
    m = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m[0])


def _apply_colormap(
    data: np.ndarray, vmin: float, vmax: float, cmap_name: str = "plasma"
) -> np.ndarray:
    """Apply colormap to depth data using fixed linear bounds, inverted."""
    norm_data = (data - vmin) / (vmax - vmin + 1e-6)
    norm_data = np.clip(norm_data, 0, 1)
    norm_data = 1.0 - norm_data

    cmap = matplotlib.colormaps[cmap_name]
    colored = cmap(norm_data)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)

    return colored_uint8


def plot_depth_comparison(
    camera_point_cloud: np.ndarray,
    da3_depth_map: np.ndarray,
    timestamp: str,
    image_shape: Tuple[int, int],
) -> None:
    """
    Extract DA3 depth, align it to ORB-SLAM3 via linear regression, and plot comparison.
    """
    if camera_point_cloud is None or len(camera_point_cloud) == 0:
        print(f"[{timestamp}] ⚠️ No valid ORB-SLAM3 point cloud data, skipping plot")
        return

    if da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ Depth map size doesn't match image size, skipping plot")
        return

    # Extract point cloud data
    pixel_w = camera_point_cloud[:, 0].astype(np.int32)
    pixel_h = camera_point_cloud[:, 1].astype(np.int32)
    orb_slam_depth = camera_point_cloud[:, 2]

    # Filter invalid data
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

    # Extract DA3 depth at exact pixel coordinates
    valid_da3_depth = da3_depth_map[valid_pixel_h, valid_pixel_w]

    # Filter DA3 invalid depth
    final_valid_mask = valid_da3_depth > 0
    final_orb_slam_depth = valid_orb_slam_depth[final_valid_mask]
    final_da3_depth = valid_da3_depth[final_valid_mask]

    if len(final_orb_slam_depth) < 2:
        print(f"[{timestamp}] ⚠️ Not enough points for regression, skipping plot")
        return

    # Perform Linear Regression
    slope = _fit_linear_regression(final_da3_depth, final_orb_slam_depth)

    # Convert DA3 depths to ORB reference
    aligned_da3_depth = final_da3_depth * slope

    # Create comparison plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(final_orb_slam_depth, aligned_da3_depth, alpha=0.7, s=8, c="royalblue")

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

    plt.tight_layout()
    plot_save_path = client_config.get_depth_plot_path(timestamp)
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{timestamp}] 💾 Depth comparison plot saved to: {plot_save_path}")
    print(f"[{timestamp}] 📊 Regression: ORB = {slope:.4f} * DA3")


def save_da3_depth_with_ros_keypoints(
    da3_depth_map: np.ndarray,
    camera_point_cloud: np.ndarray,
    timestamp: str,
    image_shape: Tuple[int, int],
) -> None:
    """Save aligned DA3 depth map with ROS keypoints overlay."""
    if da3_depth_map is None or da3_depth_map.shape != image_shape:
        print(f"[{timestamp}] ⚠️ DA3 depth map invalid or size mismatch, skipping")
        return

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
            corresponding_da3 = da3_depth_map[valid_pixel_h, valid_pixel_w]
            valid_corresp_mask = corresponding_da3 > 0

            if np.sum(valid_corresp_mask) > 10:
                orb_fit = valid_orb_slam_depth[valid_corresp_mask]
                da3_fit = corresponding_da3[valid_corresp_mask]
                slope = _fit_linear_regression(da3_fit, orb_fit)
                orb_points_available = True
                print(f"[{timestamp}] ℹ️ Alignment factor found: scale={slope:.3f}")

    # Align the full DA3 Map
    aligned_da3_map = da3_depth_map * slope

    # Determine Global Colorscale Bounds
    valid_map_pixels = aligned_da3_map[aligned_da3_map > 0]
    if len(valid_map_pixels) > 0:
        vmin = float(np.min(valid_map_pixels))
        vmax = float(np.max(valid_map_pixels))
    else:
        vmin, vmax = 0.0, 10.0

    # Generate Colored Map
    da3_depth_viz = _apply_colormap(aligned_da3_map, vmin, vmax, cmap_name="plasma")

    # Save original colored aligned map
    da3_depth_save_path = client_config.get_da3_depth_path(timestamp)
    cv2.imwrite(da3_depth_save_path, cv2.cvtColor(da3_depth_viz, cv2.COLOR_RGB2BGR))

    # Overlay Keypoints
    da3_depth_with_keypoints = cv2.cvtColor(da3_depth_viz, cv2.COLOR_RGB2BGR)

    if orb_points_available:
        outer_radius = 4
        inner_radius = 2
        white_color = (255, 255, 255)
        cmap = matplotlib.colormaps["plasma"]

        for i in range(len(valid_pixel_w)):
            w, h = valid_pixel_w[i], valid_pixel_h[i]
            orb_val = valid_orb_slam_depth[i]

            norm_val = (orb_val - vmin) / (vmax - vmin + 1e-6)
            norm_val = np.clip(norm_val, 0, 1)
            norm_val = 1.0 - norm_val

            rgba = cmap(norm_val)
            r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            color_bgr = (b, g, r)

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

    da3_depth_keypoints_save_path = client_config.get_da3_depth_with_keypoints_path(
        timestamp
    )
    cv2.imwrite(da3_depth_keypoints_save_path, da3_depth_with_keypoints)
    print(
        f"[{timestamp}] 💾 Aligned DA3 map with keypoints saved to: {da3_depth_keypoints_save_path}"
    )


def run_comparison_client():
    """Run depth comparison client using DA3 client and ROS client."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{timestamp}] 🔍 Starting depth comparison client...")

    # Get data from ROS
    print(f"[{timestamp}] 📡 Connecting to ROS server...")
    ros_client = TrackingDataClient(
        server_ip=client_config.ROS_HOST,
        port=client_config.ROS_PORT,
        enable_log=False,
    )

    tracking_result = ros_client.complete_tracking_pipeline()
    if not tracking_result or not tracking_result.success:
        print(f"[{timestamp}] ❌ Failed to get tracking data from ROS")
        return False

    if tracking_result.current_image is None:
        print(f"[{timestamp}] ❌ No image in tracking result")
        return False

    # Save image
    image_path = client_config.get_image_path(timestamp)
    cv2.imwrite(image_path, tracking_result.current_image)
    image_shape = (
        tracking_result.current_image.shape[0],
        tracking_result.current_image.shape[1],
    )
    print(f"[{timestamp}] 💾 Image saved: {image_path}")

    # Get DA3 depth
    print(f"[{timestamp}] 📊 Requesting DA3 depth inference...")
    da3_client = DA3Client()
    prediction = da3_client.inference(image_path)

    if not prediction:
        print(f"[{timestamp}] ❌ DA3 inference failed")
        return False

    # Resize depth to match image
    da3_depth_map = cv2.resize(
        prediction.depth[0],
        (image_shape[1], image_shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    print(f"[{timestamp}] ✅ DA3 depth obtained, shape: {da3_depth_map.shape}")

    # Get point clouds
    camera_point_cloud = tracking_result.tracked_points_camera

    # Generate comparison visualizations
    if camera_point_cloud is not None and camera_point_cloud.size > 0:
        print(f"[{timestamp}] 📈 Generating comparison plots...")
        plot_depth_comparison(camera_point_cloud, da3_depth_map, timestamp, image_shape)
        save_da3_depth_with_ros_keypoints(
            da3_depth_map, camera_point_cloud, timestamp, image_shape
        )
        print(f"[{timestamp}] ✅ Comparison client completed successfully")
        return True
    else:
        print(f"[{timestamp}] ⚠️ No point cloud data available")
        return False


if __name__ == "__main__":
    success = run_comparison_client()
    exit(0 if success else 1)
