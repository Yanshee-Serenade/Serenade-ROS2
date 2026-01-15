"""
ROS client module for tracking data acquisition.

This module provides the TrackingClient class for connecting to ROS
and retrieving tracking data including images and point clouds.
"""

import datetime
from typing import Optional

import ros_api
from ros_api import TrackingResult

from ..config import config


class TrackingClient:
    """Client for ROS tracking data acquisition."""

    def __init__(self, enable_log: bool = config.ROS_CLIENT_ENABLE_LOG):
        """
        Initialize ROS tracking client.

        Args:
            enable_log: Whether to enable client logging

        Raises:
            Exception: If client creation fails
        """
        self.enable_log = enable_log
        self._client: Optional[ros_api.TrackingDataClient] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the ROS client connection."""
        try:
            # Instantiate the ROS API client (initialization only, no pre-connection)
            self._client = ros_api.TrackingDataClient(
                server_ip=config.ROS_SERVER_IP,
                port=config.ROS_SERVER_PORT,
                enable_log=self.enable_log,  # Disable client logging to avoid conflicts with Flask
            )
        except Exception as e:
            error_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[{error_time}] ❌ Tracking client creation failed: {str(e)}")
            raise Exception(f"Tracking client creation failed: {str(e)}")

    def get_tracking_data(self) -> Optional[TrackingResult]:
        """
        Get tracking data from ROS using the complete pipeline.

        Returns:
            TrackingResult object with all tracking data, or None if failed

        Raises:
            Exception: If client is not initialized or pipeline fails
        """
        if not self._client:
            raise ValueError("ROS client not initialized")

        try:
            # Core refactoring: Call the complete pipeline method for one-step processing
            tracking_result: Optional[TrackingResult] = (
                self._client.complete_tracking_pipeline()
            )
            return tracking_result
        except Exception as e:
            error_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[{error_time}] ❌ ROS data pipeline failed: {str(e)}")
            raise Exception(f"ROS data pipeline failed: {str(e)}")

    def is_connected(self) -> bool:
        """
        Check if the client is connected.

        Returns:
            True if client is initialized, False otherwise
        """
        return self._client is not None

    def close(self) -> None:
        """Close the client connection."""
        self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
