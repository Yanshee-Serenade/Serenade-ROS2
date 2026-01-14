"""
Walker module for robot gait control.

This module provides a modular system for controlling robot walking gaits,
including inverse kinematics, gait generation, and motion control.
"""

from .controller import RobotWalker
from .factory import create_walker
from .gait import GaitStep, WalkerState
from .kinematics import KinematicsSolver
from .mock_client import MockClient

__all__ = [
    "KinematicsSolver",
    "WalkerState",
    "GaitStep",
    "RobotWalker",
    "create_walker",
    "MockClient",
]
