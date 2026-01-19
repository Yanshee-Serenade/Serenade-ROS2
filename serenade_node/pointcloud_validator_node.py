#!/usr/bin/env python3
"""
Pointcloud Validation ROS2 Node - Validates point cloud transformations.
Subscribes to ROS2 topics for point clouds and camera info.
"""

import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo

from serenade_node.config import config

REFERENCE_SCALE = 7.9470454


class PointCloudValidator:
    """Point cloud validation class for camera-to-world coordinate transformation."""

    def __init__(
        self,
        camera_pose: Dict,
        intrinsics: Optional[Dict] = None,
        distortion_coeffs: Optional[tuple] = None,
    ):
        """
        Initialize validator.

        Args:
            camera_pose: Camera pose dict with position and orientation
            intrinsics: Camera intrinsics dict (default: from config)
            distortion_coeffs: Camera distortion coefficients (default: from config)
        """
        self.camera_pose = camera_pose

        if intrinsics is None:
            self.intrinsics = {
                'fx': client_config.CAMERA_INTRINSICS_FX,
                'fy': client_config.CAMERA_INTRINSICS_FY,
                'cx': client_config.CAMERA_INTRINSICS_CX,
                'cy': client_config.CAMERA_INTRINSICS_CY,
            }
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
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    def _build_pose_transform_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build pose transformation matrices."""
        pos = self.camera_pose['position']
        ori = self.camera_pose['orientation']
        t_wc = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64)
        quaternion_wc = np.array([ori['x'], ori['y'], ori['z'], ori['w']], dtype=np.float64)

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
        """
        Use OpenCV to correctly solve the inverse distortion model.
        Returns normalized coordinates (x', y') such that:
        [x', y', 1] is the direction vector in camera frame.
        """
        src_pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
        dist_coeffs = np.array(self.distortion_coeffs, dtype=np.float64)
        
        dst_pt = cv2.undistortPoints(src_pt, self.K, dist_coeffs)
        
        u_undist = dst_pt[0, 0, 0]
        v_undist = dst_pt[0, 0, 1]

        return u_undist, v_undist

    def camera_point_to_world_point(self, camera_point: np.ndarray) -> np.ndarray:
        """Transform camera point to world point."""
        camera_point = np.array(camera_point, dtype=np.float64).flatten()
        if len(camera_point) != 3:
            raise ValueError(
                "Camera point format error, must be [pixel_x, pixel_y, camera_depth_z]"
            )
        pixel_x, pixel_y, camera_depth_z = camera_point

        u_undist, v_undist = self._undistort_to_normalized_coords(pixel_x, pixel_y)

        Xc = u_undist * camera_depth_z
        Yc = v_undist * camera_depth_z
        Zc = camera_depth_z

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
            point_errors[i] = (
                np.linalg.norm(world_pred - world_actual) * REFERENCE_SCALE
            )

        error_metrics = {
            "point_count": float(point_count),
            "mean_error": float(np.mean(point_errors)),
            "max_error": float(np.max(point_errors)),
            "min_error": float(np.min(point_errors)),
            "rmse": float(np.sqrt(np.mean(np.square(point_errors)))),
        }

        return error_metrics, point_errors


class PointCloudValidatorNode(Node):
    """ROS2 Node for point cloud validation"""
    
    def __init__(self):
        super().__init__('pointcloud_validator_node')
        
        self.get_logger().info("PointCloud Validator Node initialized")
        
        # Subscribe to point clouds and camera info
        self.world_points_subscription = self.create_subscription(
            PointCloud2,
            '/orb_slam3/world_points',
            self.on_world_points,
            1
        )
        
        self.camera_points_subscription = self.create_subscription(
            PointCloud2,
            '/orb_slam3/camera_points',
            self.on_camera_points,
            1
        )
        
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.on_camera_info,
            1
        )
        
        self.last_world_points = None
        self.last_camera_points = None
        self.camera_info = None
        self.camera_pose = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }
        
    def on_world_points(self, msg: PointCloud2):
        """Handle incoming world point cloud"""
        self.last_world_points = msg
        self._try_validate()
        
    def on_camera_points(self, msg: PointCloud2):
        """Handle incoming camera point cloud"""
        self.last_camera_points = msg
        self._try_validate()
        
    def on_camera_info(self, msg: CameraInfo):
        """Handle camera info"""
        self.camera_info = msg
        # Extract intrinsics
        self.camera_pose = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }
        
    def _try_validate(self):
        """Try to validate if we have all required data"""
        if self.last_world_points is None or self.last_camera_points is None:
            return
            
        try:
            # Extract points from PointCloud2 messages
            camera_points = self._pointcloud2_to_array(self.last_camera_points)
            world_points = self._pointcloud2_to_array(self.last_world_points)
            
            if camera_points.size == 0 or world_points.size == 0:
                return
                
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.get_logger().info(f"[{timestamp}] 🔬 Validating point cloud transformation...")
            
            validator = PointCloudValidator(self.camera_pose)
            error_metrics, _ = validator.validate_point_cloud(camera_points, world_points)
            
            self.get_logger().info(f"[{timestamp}] ✅ Point cloud validation completed")
            self.get_logger().info(f"  • Point count: {int(error_metrics['point_count'])}")
            self.get_logger().info(f"  • Mean error: {error_metrics['mean_error']:.6f} m")
            self.get_logger().info(f"  • Max error: {error_metrics['max_error']:.6f} m")
            self.get_logger().info(f"  • Min error: {error_metrics['min_error']:.6f} m")
            self.get_logger().info(f"  • RMSE: {error_metrics['rmse']:.6f} m")
            
        except Exception as e:
            self.get_logger().error(f"Validation error: {str(e)}")
            
    def _pointcloud2_to_array(self, pc2_msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to numpy array"""
        import struct
        
        # Extract point data
        points = []
        point_step = pc2_msg.point_step
        row_step = pc2_msg.row_step
        
        data = pc2_msg.data
        
        for y in range(pc2_msg.height):
            for x in range(pc2_msg.width):
                offset = y * row_step + x * point_step
                
                # Extract x, y, z (assuming they're the first 3 floats)
                point_data = data[offset:offset+12]
                if len(point_data) >= 12:
                    px, py, pz = struct.unpack('fff', point_data[:12])
                    points.append([px, py, pz])
                    
        return np.array(points, dtype=np.float64) if points else np.array([], dtype=np.float64).reshape(0, 3)


def main(args=None):
    """Main function to run pointcloud validator ROS2 node"""
    rclpy.init(args=args)
    
    node = PointCloudValidatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down PointCloud Validator...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
