"""
Walker module for robot gait control.

This module provides a modular system for controlling robot walking gaits,
including inverse kinematics, gait generation, and motion control.
"""

from serenade_walker.controller import RobotWalker
from serenade_walker.factory import create_walker
from serenade_walker.gait import (
    BaseSequence,
    DefaultSequence,
    GaitStep,
    SquatSequence,
    TurnLeftSequence,
    TurnRightSequence,
    WalkStraightSequence,
)
from serenade_walker.kinematics import KinematicsSolver
from serenade_walker.mock_client import MockClient

__all__ = [
    "KinematicsSolver",
    "GaitStep",
    "BaseSequence",
    "TurnLeftSequence",
    "TurnRightSequence",
    "WalkStraightSequence",
    "SquatSequence",
    "DefaultSequence",
    "RobotWalker",
    "create_walker",
    "MockClient",
]
