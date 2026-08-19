"""Deterministic mapping from recognized gestures to abstract actions.

This module converts a `Gesture` into an `Action`. It contains no
camera, hand-tracking, landmark, OpenCV, MediaPipe, or OS-control code.
"""

from typing import Dict

from ..gestures.recognizer import Gesture
from .action import Action


_GESTURE_TO_ACTION: Dict[Gesture, Action] = {
    Gesture.OPEN_PALM: Action.PAUSE,

    # Primary pointer control
    Gesture.POINT: Action.MOVE_CURSOR,
    Gesture.THREE_FINGERS: Action.LEFT_CLICK,

    # Secondary actions
    Gesture.PEACE: Action.DOUBLE_CLICK,

    # Media/system controls
    Gesture.THUMBS_UP: Action.VOLUME_UP,
    Gesture.THUMBS_DOWN: Action.VOLUME_DOWN,

    # Unknown pose
    Gesture.UNKNOWN: Action.NONE,
}


class ActionMapper:
    """Maps a recognized Gesture to a deterministic abstract Action."""

    def map(self, gesture: Gesture) -> Action:
        """Return the action associated with the given gesture."""
        return _GESTURE_TO_ACTION.get(gesture, Action.NONE)