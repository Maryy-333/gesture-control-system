"""Deterministic mapping from recognized gestures to abstract actions.

This module converts a `Gesture` (from `gesture_control.gestures.recognizer`)
into an `Action` (from `gesture_control.actions.action`). It has no
knowledge of cameras, hand tracking, landmark math, OpenCV, MediaPipe,
or any real input-control mechanism (mouse/keyboard/OS APIs) -- it is a
pure, stateless lookup.

Dependency direction (do not reverse):
    FingerStates -> GestureRecognizer -> Gesture -> ActionMapper -> Action
`gesture_control.gestures` has no knowledge of this module or of
`Action`; only this module depends on `Gesture`.
"""

from typing import Dict

from ..gestures.recognizer import Gesture
from .action import Action

# The current gesture-to-action mapping. Intentionally simple -- one
# action per gesture -- since interaction semantics (e.g. distinguishing
# a momentary gesture from a held one) are out of scope for this layer.
_GESTURE_TO_ACTION: Dict[Gesture, Action] = {
    Gesture.OPEN_PALM: Action.PAUSE,
    Gesture.FIST: Action.LEFT_CLICK,
    Gesture.POINT: Action.MOVE_CURSOR,
    Gesture.PEACE: Action.SCREENSHOT,
    Gesture.THUMBS_UP: Action.NONE,
    Gesture.THUMBS_DOWN: Action.NONE,
    Gesture.UNKNOWN: Action.NONE,
}


class ActionMapper:
    """Maps a `Gesture` to an `Action` using a fixed, deterministic table.

    Stateless: an `ActionMapper` holds no per-call or per-instance
    state, so any number of instances (or repeated calls on the same
    instance) always agree on the result for a given `Gesture`.

    Example:
        mapper = ActionMapper()
        action = mapper.map(Gesture.POINT)
        assert action == Action.MOVE_CURSOR
    """

    def map(self, gesture: Gesture) -> Action:
        """Return the abstract action associated with a gesture.

        Args:
            gesture: The recognized gesture to map.

        Returns:
            The corresponding `Action`. Any gesture without an explicit
            mapping (including `Gesture.UNKNOWN`) resolves to
            `Action.NONE` rather than raising, so this method never
            fails on a valid `Gesture` member.
        """
        return _GESTURE_TO_ACTION.get(gesture, Action.NONE)