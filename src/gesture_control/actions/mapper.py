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

# The current gesture-to-action mapping. Chosen to make hands-on use
# of the system actually useful:
#   - OPEN_PALM -> PAUSE: freezes/resumes control (see
#     GestureControlRuntime's GestureActionGate for how PAUSE is
#     actually handled as a toggle, since no ControlBackend method
#     performs a real "pause").
#   - FIST -> LEFT_CLICK, POINT -> MOVE_CURSOR: primary pointer control.
#   - PEACE -> DOUBLE_CLICK, THUMBS_UP -> RIGHT_CLICK: secondary click
#     actions, each on a gesture distinct from FIST/POINT.
#   - THUMBS_DOWN -> SCROLL_DOWN: only reliable now that
#     GestureRecognizer can genuinely distinguish THUMBS_DOWN from
#     THUMBS_UP via landmark geometry; previously both were
#     indistinguishable from FingerStates alone, so THUMBS_DOWN was
#     unreachable and mapping it to anything would have been dead code.
#   - UNKNOWN -> NONE: no action for an unrecognized hand pose.
# SCROLL_UP and SCREENSHOT currently have no gesture mapped to them --
# see the class docstring below for why.
_GESTURE_TO_ACTION: Dict[Gesture, Action] = {
    Gesture.OPEN_PALM: Action.PAUSE,
    Gesture.FIST: Action.LEFT_CLICK,
    Gesture.POINT: Action.MOVE_CURSOR,
    Gesture.PEACE: Action.DOUBLE_CLICK,
    Gesture.THUMBS_UP: Action.RIGHT_CLICK,
    Gesture.THUMBS_DOWN: Action.SCROLL_DOWN,
    Gesture.UNKNOWN: Action.NONE,
}


class ActionMapper:
    """Maps a `Gesture` to an `Action` using a fixed, deterministic table.

    Stateless: an `ActionMapper` holds no per-call or per-instance
    state, so any number of instances (or repeated calls on the same
    instance) always agree on the result for a given `Gesture`.

    `Action.SCROLL_UP` and `Action.SCREENSHOT` currently have no
    gesture mapped to them: every gesture `FingerStates` can reliably
    distinguish is already assigned to a more immediately useful
    action, and inventing a new hand pose just to reach these two was
    judged out of scope rather than genuinely needed (see the module
    that adjusted this mapping for the full reasoning). They remain
    valid, reachable `Action` values for a future gesture to use.

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