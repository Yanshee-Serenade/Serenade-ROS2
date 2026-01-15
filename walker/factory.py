"""
Factory function for creating walker instances.

This module provides the create_walker function which simplifies
the creation of RobotWalker instances with proper initialization.
"""

from typing import Optional

from ros_api import CameraPoseClient, JointAngleTCPClient

from .controller import RobotWalker
from .kinematics import KinematicsSolver


def create_walker(
    period_ms: int = 200,
    lib_path: str = "./libyanshee_kinematics.so",
    host: str = "localhost",
    port: int = 21120,
    timeout: int = 10,
    use_mock: bool = False,
    camera_pose_host: str = "localhost",
    camera_pose_port: int = 21118,
    use_camera_pose: bool = True,
) -> RobotWalker:
    """
    Create and initialize a RobotWalker instance.

    This factory function handles the complete initialization process:
    1. Initialize the kinematics solver
    2. Create the robot client (real or mock)
    3. Create and configure the RobotWalker

    Args:
        period_ms: Action period in milliseconds
        lib_path: Path to the native kinematics library
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
        solver = KinematicsSolver(lib_path)
        print("   ✓ Inverse kinematics solver initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize kinematics solver: {e}")

    # 2. Initialize robot client
    print("\n2. Initializing robot client...")

    if use_mock:
        from .mock_client import MockClient

        angle_client = MockClient()
        print("   ✓ Mock client initialized (simulation mode)")
    else:
        angle_client = JointAngleTCPClient(host=host, port=port, timeout=timeout)
        print(f"   ✓ Robot client connected to {host}:{port}")

    # 3. Initialize camera pose client if requested
    camera_pose_client: Optional[CameraPoseClient] = None
    if use_camera_pose:
        print("\n3. Initializing camera pose client...")
        try:
            camera_pose_client = CameraPoseClient(
                host=camera_pose_host, port=camera_pose_port
            )
            if camera_pose_client.connect():
                print(
                    f"   ✓ Camera pose client connected to {camera_pose_host}:{camera_pose_port}"
                )
            else:
                print("   ⚠ Camera pose client connection failed")
                camera_pose_client = None
        except Exception as e:
            print(f"   ⚠ Failed to initialize camera pose client: {e}")
            camera_pose_client = None

    # 4. Create Walker
    print("\n4. Creating walker controller...")
    try:
        walker = RobotWalker(
            solver=solver,
            client=angle_client,
            grid_size=12,
            period_ms=period_ms,
            camera_pose_client=camera_pose_client,
        )
        print(f"   ✓ Walker created successfully, period: {walker.period_ms}ms")
    except Exception as e:
        raise RuntimeError(f"Failed to create walker: {e}")

    return walker
