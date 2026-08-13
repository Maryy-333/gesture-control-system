"""Rule-based gesture recognition from finger-extension states.

This module maps a `FingerStates` snapshot (which fingers are extended)
to a named `Gesture`. It has no knowledge of cameras, images, hand
tracking, or landmark coordinates -- it consumes only the boolean
per-finger output of `FingerStateDetector`, so it has no dependency on
OpenCV or the MediaPipe runtime.

The recognizer is purely deterministic: a given `FingerStates` value
always produces the same `Gesture`, with no randomness, fuzzy matching,
or hidden state.
"""

from enum import Enum
from typing import Callable, List, Tuple

from .finger_states import FingerStates


class Gesture(str, Enum):
    """The set of gestures this system can currently recognize.

    Values are plain lowercase strings (rather than auto-generated
    numbers) so they remain stable across code changes and are safe to
    use directly in logs, config files, or serialized output.
    """

    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


def _is_open_palm(states: FingerStates) -> bool:
    """All five fingers extended."""
    return states.thumb and states.index and states.middle and states.ring and states.pinky


def _is_fist(states: FingerStates) -> bool:
    """All five fingers folded."""
    return not (states.thumb or states.index or states.middle or states.ring or states.pinky)


def _is_thumbs_up(states: FingerStates) -> bool:
    """Only the thumb extended.

    NOTE: `FingerStates` records only whether the thumb is extended,
    not which direction it points. This rule matches "thumb extended,
    all other fingers folded" -- the same finger-state pattern used for
    a thumbs-down gesture. See the module docstring and the
    `GestureRecognizer` docstring for why THUMBS_DOWN is intentionally
    never returned here.
    """
    return states.thumb and not (states.index or states.middle or states.ring or states.pinky)


def _is_point(states: FingerStates) -> bool:
    """Only the index finger extended (thumb may be either state)."""
    return states.index and not states.middle and not states.ring and not states.pinky


def _is_peace(states: FingerStates) -> bool:
    """Index and middle fingers extended, ring and pinky folded."""
    return states.index and states.middle and not states.ring and not states.pinky


# Ordered (predicate, gesture) rules. The first matching rule wins. The
# five patterns above are mutually exclusive by construction (each
# constrains a different combination of index/middle/ring/pinky, or
# folds vs. extends all fingers), so rule order does not change the
# outcome for any of the gestures currently defined -- it is kept
# explicit and linear so new rules can be reasoned about individually
# rather than through a shared scoring or priority system.
_GESTURE_RULES: List[Tuple[Callable[[FingerStates], bool], Gesture]] = [
    (_is_open_palm, Gesture.OPEN_PALM),
    (_is_fist, Gesture.FIST),
    (_is_thumbs_up, Gesture.THUMBS_UP),
    (_is_peace, Gesture.PEACE),
    (_is_point, Gesture.POINT),
]


class GestureRecognizer:
    """Maps `FingerStates` to a `Gesture` using fixed, deterministic rules.

    This class performs no image processing, hand tracking, or
    landmark math itself -- it operates purely on the boolean
    per-finger output already computed by `FingerStateDetector`.

    Known limitation -- THUMBS_UP vs. THUMBS_DOWN:
        `FingerStates` records only whether the thumb is extended, not
        which direction it points, so a thumb-only-extended state is
        geometrically identical whether the hand is oriented thumbs-up
        or thumbs-down. This recognizer does NOT attempt to fake that
        distinction: it always reports `Gesture.THUMBS_UP` for a
        thumb-only-extended state, and `Gesture.THUMBS_DOWN` is never
        produced. `Gesture.THUMBS_DOWN` exists in the enum so it has a
        stable identity for a future recognizer that also considers
        landmark direction/hand orientation (e.g. thumb tip position
        relative to the wrist), which `FingerStates` alone cannot
        provide.

    Example:
        recognizer = GestureRecognizer()
        states = FingerStates(
            thumb=False, index=True, middle=False, ring=False, pinky=False
        )
        gesture = recognizer.recognize(states)
        assert gesture == Gesture.POINT
    """

    def recognize(self, finger_states: FingerStates) -> Gesture:
        """Determine the gesture represented by a `FingerStates` snapshot.

        Args:
            finger_states: The per-finger extension state to classify.

        Returns:
            The matching `Gesture`, or `Gesture.UNKNOWN` if
            `finger_states` does not match any currently supported
            pattern. Matching is exact (no fuzzy/partial matching).
        """
        for matches, gesture in _GESTURE_RULES:
            if matches(finger_states):
                return gesture
        return Gesture.UNKNOWN