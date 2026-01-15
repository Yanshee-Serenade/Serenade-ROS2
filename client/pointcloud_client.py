"""
Pointcloud Validation Client - Reimplemented using ROS client.
Validates point cloud transformations from camera to world coordinates.
"""

import datetime
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from ros_api import (
    CameraIntrinsics,
    CameraPose,
    TrackingDataClient,
)

from .config import client_config


class PointCloudValidator:
    """Point cloud validation class for camera-to-world coordinate transformation."""

    def __init__(
        self,
        camera_pose: CameraPose,
        intrinsics: Optional[CameraIntrinsics] = None,
        distortion_coeffs: Optional[tuple] = None,
    ):
        """
        Initialize validator.

        Args:
            camera_pose: Camera pose from ROS
            intrinsics: Camera intrinsics (default: from config)
            distortion_coeffs: Camera distortion coefficients (default: from config)
        """
        self.camera_pose = camera_pose

        if intrinsics is None:
            self.intrinsics = CameraIntrinsics(
                fx=client_config.CAMERA_INTRINSICS_FX,
                fy=client_config.CAMERA_INTRINSICS_FY,
                cx=client_config.CAMERA_INTRINSICS_CX,
                cy=client_config.CAMERA_INTRINSICS_CY,
            )
        else:
            self.intrinsics = intrinsics

        if distortion_coeffs is None:
            self.distortion_coeffs = client_config.DISTORTION_COEFFS
        else:
            self.distortion_coeffs = distortion_coeffs

        self.K = self._build_intrinsics_matrix()
        self.Tcw, self.Twc = self._build_pose_transform_matrices()

    def _build_intrinsics_matrix(self) -> np.ndarray:
        """Build camera intrinsics matrix K (3x3)."""
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    def _build_pose_transform_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build pose transformation matrices."""
        pos = self.camera_pose.position
        ori = self.camera_pose.orientation
        t_wc = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        quaternion_wc = np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float64)

        R_wc = R.from_quat(quaternion_wc).as_matrix().astype(np.float64)
        R_cw = R_wc.T

        Twc = np.eye(4, dtype=np.float64)
        Twc[:3, :3] = R_wc
        Twc[:3, 3] = t_wc

        t_cw = -np.dot(R_cw, t_wc)
        Tcw = np.eye(4, dtype=np.float64)
        Tcw[:3, :3] = R_cw
        Tcw[:3, 3] = t_cw

        return Tcw, Twc

    def _undistort_to_normalized_coords(
        self, pixel_x: float, pixel_y: float
    ) -> Tuple[float, float]:
        """Distortion correction to get undistorted normalized image coordinates."""
        fx, fy, cx, cy = (
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
        )
        k1, k2, p1, p2, k3 = self.distortion_coeffs

        # Step 1: Pixel coordinates -> original normalized image coordinates
        u = (pixel_x - cx) / fx
        v = (pixel_y - cy) / fy

        # Step 2: Calculate radial distortion parameters
        r_sq = u**2 + v**2
        r_4 = r_sq**2
        r_6 = r_sq**3
        dist_radial = 1 + k1 * r_sq + k2 * r_4 + k3 * r_6

        # Step 3: Calculate tangential distortion correction
        delta_u = 2 * p1 * u * v + p2 * (r_sq + 2 * u**2)
        delta_v = p1 * (r_sq + 2 * v**2) + 2 * p2 * u * v

        # Step 4: Undistorted normalized coordinates
        u_undist = u * dist_radial + delta_u
        v_undist = v * dist_radial + delta_v

        return u_undist, v_undist

    def camera_point_to_world_point(self, camera_point: np.ndarray) -> np.ndarray:
        """Transform camera point to world point."""
        camera_point = np.array(camera_point, dtype=np.float64).flatten()
        if len(camera_point) != 3:
            raise ValueError(
                "Camera point format error, must be [pixel_x, pixel_y, camera_depth_z]"
            )
        pixel_x, pixel_y, camera_depth_z = camera_point

        # Distortion correction
        u_undist, v_undist = self._undistort_to_normalized_coords(pixel_x, pixel_y)

        # Solve 3D point in camera coordinate system (Xc, Yc, Zc)
        Xc = u_undist * camera_depth_z
        Yc = v_undist * camera_depth_z
        Zc = camera_depth_z

        # Convert to homogeneous coordinates and transform to world coordinates
        camera_point_homo = np.array([Xc, Yc, Zc, 1.0], dtype=np.float64)
        world_point_homo_pred = np.dot(self.Twc, camera_point_homo)
        world_point_pred = world_point_homo_pred[:3]

        return world_point_pred

    def validate_point_cloud(
        self, camera_point_cloud: np.ndarray, world_point_cloud: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Validate point cloud, return error metrics."""
        camera_point_cloud = np.array(camera_point_cloud, dtype=np.float64)
        world_point_cloud = np.array(world_point_cloud, dtype=np.float64)

        if camera_point_cloud.shape != world_point_cloud.shape:
            raise ValueError(
                f"Camera and world point cloud shape mismatch: {camera_point_cloud.shape} vs {world_point_cloud.shape}"
            )
        if camera_point_cloud.shape[0] == 0:
            raise ValueError("Point cloud is empty, cannot calculate error")

        point_count = camera_point_cloud.shape[0]
        point_errors = np.zeros(point_count, dtype=np.float64)

        for i in range(point_count):
            world_pred = self.camera_point_to_world_point(camera_point_cloud[i])
            world_actual = world_point_cloud[i]
            point_errors[i] = np.linalg.norm(world_pred - world_actual)

        error_metrics = {
            "point_count": float(point_count),
            "mean_error": float(np.mean(point_errors)),
            "max_error": float(np.max(point_errors)),
            "min_error": float(np.min(point_errors)),
            "rmse": float(np.sqrt(np.mean(np.square(point_errors)))),
        }

        return error_metrics, point_errors


def run_pointcloud_client():
    """Run point cloud validation client using ROS client."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{timestamp}] 🔍 Starting point cloud validation client...")

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

    # Get point clouds
    camera_point_cloud = tracking_result.tracked_points_camera
    world_point_cloud = tracking_result.tracked_points_world
    camera_pose = tracking_result.camera_pose

    if (
        camera_point_cloud is None
        or camera_point_cloud.size == 0
        or world_point_cloud is None
        or world_point_cloud.size == 0
    ):
        print(f"[{timestamp}] ❌ No point cloud data available")
        return False

    # Validate point cloud
    print(f"[{timestamp}] 🔬 Validating point cloud transformation...")
    validator = PointCloudValidator(camera_pose)

    try:
        error_metrics, point_errors = validator.validate_point_cloud(
            camera_point_cloud, world_point_cloud
        )

        print(f"[{timestamp}] ✅ Point cloud validation completed")
        print(f"  • Point count: {int(error_metrics['point_count'])}")
        print(f"  • Mean error: {error_metrics['mean_error']:.6f} m")
        print(f"  • Max error: {error_metrics['max_error']:.6f} m")
        print(f"  • Min error: {error_metrics['min_error']:.6f} m")
        print(f"  • RMSE: {error_metrics['rmse']:.6f} m")

        return True

    except Exception as e:
        print(f"[{timestamp}] ❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = run_pointcloud_client()
    exit(0 if success else 1)
