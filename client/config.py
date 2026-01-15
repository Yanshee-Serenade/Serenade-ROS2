"""
Client configuration for connecting to model servers.
"""

from dataclasses import dataclass


@dataclass
class ClientConfig:
    """Client configuration with server connection details."""

    # Server connection settings
    VLM_HOST: str = "127.0.0.1"
    VLM_PORT: int = 21122

    DA3_HOST: str = "127.0.0.1"
    DA3_PORT: int = 21123

    SAM3_HOST: str = "127.0.0.1"
    SAM3_PORT: int = 21124

    ROS_HOST: str = "127.0.0.1"
    ROS_PORT: int = 21121

    # Timeout settings (seconds)
    CONNECTION_TIMEOUT: float = 10.0
    RECV_TIMEOUT: float = 30.0

    # File paths
    IMAGE_SAVE_PREFIX: str = "images/image_"
    DEPTH_PLOT_SAVE_PREFIX: str = "images/depth_comparison_"
    DA3_DEPTH_SAVE_PREFIX: str = "images/da3_depth_"
    DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX: str = "images/da3_depth_with_keypoints_"

    # Camera configuration
    CAMERA_INTRINSICS_FX: float = 503.640273
    CAMERA_INTRINSICS_FY: float = 502.167721
    CAMERA_INTRINSICS_CX: float = 312.565456
    CAMERA_INTRINSICS_CY: float = 244.436855

    DISTORTION_COEFFS: tuple = (0.148509, -0.255395, 0.003505, 0.001639, 0.0)

    def get_image_path(self, timestamp: str) -> str:
        """Get full image path for a timestamp."""
        return f"{self.IMAGE_SAVE_PREFIX}{timestamp}.jpg"

    def get_depth_plot_path(self, timestamp: str) -> str:
        """Get full depth plot path for a timestamp."""
        return f"{self.DEPTH_PLOT_SAVE_PREFIX}{timestamp}.png"

    def get_da3_depth_path(self, timestamp: str) -> str:
        """Get full DA3 depth image path for a timestamp."""
        return f"{self.DA3_DEPTH_SAVE_PREFIX}{timestamp}.png"

    def get_da3_depth_with_keypoints_path(self, timestamp: str) -> str:
        """Get full DA3 depth with keypoints path for a timestamp."""
        return f"{self.DA3_DEPTH_WITH_KEYPOINTS_SAVE_PREFIX}{timestamp}.png"


# Global client configuration instance
client_config = ClientConfig()
