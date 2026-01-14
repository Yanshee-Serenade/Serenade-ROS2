"""
Walker module for robot gait control.

This module provides a modular system for controlling robot walking gaits,
including inverse kinematics, gait generation, and motion control.
"""

from .controller import RobotWalker
from .factory import create_walker
from .gait import (
    BaseSequence,
    DefaultSequence,
    GaitStep,
    SquatSequence,
    TurnLeftSequence,
    TurnRightSequence,
    WalkerState,
    WalkSequence,
)
from .kinematics import KinematicsSolver
from .mock_client import MockClient

__all__ = [
    "KinematicsSolver",
    "WalkerState",
    "GaitStep",
    "BaseSequence",
    "TurnLeftSequence",
    "TurnRightSequence",
    "WalkSequence",
    "SquatSequence",
    "DefaultSequence",
    "RobotWalker",
    "create_walker",
    "MockClient",
]
