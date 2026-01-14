"""
Mock client for testing walker functionality without a real robot.

This module provides a MockClient class that simulates the robot client
interface for testing and development purposes.
"""

from typing import List, Tuple


class MockClient:
    """Mock robot client for simulation and testing."""

    def __init__(self):
        """Initialize the mock client."""
        self.last_angles = None
        self.last_time_ms = None
        self.call_count = 0

    def set_joint_angles(
        self, angles: List[float], time_ms: int = 200
    ) -> Tuple[bool, str]:
        """
        Simulate setting joint angles.

        Args:
            angles: List of joint angles in degrees
            time_ms: Movement time in milliseconds

        Returns:
            Tuple of (success, message)
        """
        self.last_angles = angles.copy()
        self.last_time_ms = time_ms
        self.call_count += 1

        # Convert angles to servo values (2048 per 180 degrees)
        servo_values = [round(i * 2048 / 180) for i in angles]

        print(f"  [Mock] Sending angles: {servo_values}, time: {time_ms}ms")
        print(f"  [Mock] Call count: {self.call_count}")

        return True, "Mock success"

    def get_last_angles(self) -> List[float] | None:
        """
        Get the last angles that were sent.

        Returns:
            Last angles sent, or None if no angles have been sent
        """
        return self.last_angles.copy() if self.last_angles else None

    def get_last_time_ms(self) -> int | None:
        """
        Get the last time_ms that was used.

        Returns:
            Last time_ms used, or None if no angles have been sent
        """
        return self.last_time_ms

    def get_call_count(self) -> int:
        """
        Get the number of times set_joint_angles has been called.

        Returns:
            Number of calls
        """
        return self.call_count

    def reset(self):
        """Reset the mock client state."""
        self.last_angles = None
        self.last_time_ms = None
        self.call_count = 0
        print("  [Mock] Client reset")
