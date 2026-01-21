"""
Gait sequences module.

Provides various gait sequence implementations for robot walking, turning, squatting, etc.
"""

from serenade_walker.sequences.base_sequences import (
    BaseSequence,
    CyclingSequence,
    OneShotSequence,
)
from serenade_walker.sequences.turn_left_sequence import TurnLeftSequence
from serenade_walker.sequences.turn_right_sequence import TurnRightSequence
from serenade_walker.sequences.walk_straight_sequence import WalkStraightSequence
from serenade_walker.sequences.squat_sequence import SquatSequence
from serenade_walker.sequences.default_sequence import DefaultSequence

__all__ = [
    "BaseSequence",
    "CyclingSequence",
    "OneShotSequence",
    "TurnLeftSequence",
    "TurnRightSequence",
    "WalkStraightSequence",
    "SquatSequence",
    "DefaultSequence",
]
