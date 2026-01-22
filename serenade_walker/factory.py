"""
Factory function for creating walker instances.

This module provides the create_walker function which simplifies
the creation of RobotWalker instances with proper initialization.
"""

from typing import Optional

import rclpy
from serenade_ros_api import JointAngleTCPClient

from serenade_walker.controller import RobotWalker
from serenade_walker.kinematics import KinematicsSolver


def create_walker(
    period_ms: int = 200,
    host: str = "localhost",
    port: int = 21120,
    timeout: int = 10,
    use_mock: bool = False,
) -> RobotWalker:
    """
    Create and initialize a RobotWalker instance.

    This factory function handles the complete initialization process:
    1. Initialize the kinematics solver
    2. Create the robot client (real or mock)

    Args:
        period_ms: Action period in milliseconds
        lib_path: Path to the native kinematics library. If None, will auto-detect
                  from ROS2 package installation or current directory.
        host: Robot server hostname
        port: Robot server port
        timeout: Connection timeout in seconds
        use_mock: Whether to use a mock client for testing

    Returns:
        Initialized RobotWalker instance

    Raises:
        RuntimeError: If kinematics solver initialization fails
        ConnectionError: If robot client connection fails
    """
    
    # 1. Initialize inverse kinematics solver
    print("1. Initializing inverse kinematics solver...")
    try:
        solver = KinematicsSolver()
        print("   ✓ Inverse kinematics solver initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize kinematics solver: {e}")

    # 2. Initialize robot client
    print("\n2. Initializing robot client...")

    if use_mock:
        from serenade_walker.mock_client import MockClient

        angle_client = MockClient()
        print("   ✓ Mock client initialized (simulation mode)")
    else:
        angle_client = JointAngleTCPClient(host=host, port=port, timeout=timeout)
        print(f"   ✓ Robot client connected to {host}:{port}")

    # 3. Create Walker
    print("\n3. Creating walker controller...")
    try:
        walker = RobotWalker(
            solver=solver,
            client=angle_client,
            grid_size=12,
            period_ms=period_ms,
        )
        print(f"   ✓ Walker created successfully, period: {walker.period_ms}ms")
    except Exception as e:
        raise RuntimeError(f"Failed to create walker: {e}")

    return walker
