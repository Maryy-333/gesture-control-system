"""Rule-based gesture recognition from finger-extension states.

This module maps a `FingerStates` snapshot (which fingers are extended)
to a named `Gesture`. It has no knowledge of cameras, images, or hand
tracking, and no dependency on OpenCV or the MediaPipe runtime.

For every gesture except THUMBS_UP/THUMBS_DOWN, `FingerStates` alone is
sufficient. Distinguishing thumbs-up from thumbs-down genuinely
requires knowing which way the thumb points, which `FingerStates` does
not record (it only records that the thumb is extended). To resolve
that -- without ever guessing or applying an arbitrary finger-state
rule -- `recognize()` optionally accepts the same `Landmarks` sequence
`FingerStateDetector` was given, and uses real landmark geometry (the
thumb tip's position relative to the wrist) to tell the two apart. This
keeps the module's only new dependency being the plain landmark
coordinate type already defined in `finger_states.py` -- still no
OpenCV, MediaPipe, or PyAutoGUI.

The recognizer is purely deterministic: a given `(FingerStates,
Landmarks)` input always produces the same `Gesture`, with no
randomness or hidden state.
"""

from enum import Enum
from typing import Callable, List, Optional, Tuple

from .finger_states import FingerStates, HandLandmark, Landmarks


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


def _is_thumb_only_extended(states: FingerStates) -> bool:
    """Only the thumb extended -- the shared pattern for both thumbs-up and thumbs-down.

    `FingerStates` cannot tell these two gestures apart by itself: a
    thumbs-up hand and a thumbs-down hand both report "thumb extended,
    everything else folded". `recognize()` resolves the direction
    separately, using landmark geometry when available (see
    `_is_thumb_pointing_up`).
    """
    return states.thumb and not (states.index or states.middle or states.ring or states.pinky)


def _is_point(states: FingerStates) -> bool:
    """Only the index finger extended (thumb may be either state)."""
    return states.index and not states.middle and not states.ring and not states.pinky


def _is_peace(states: FingerStates) -> bool:
    """Index and middle fingers extended, ring and pinky folded."""
    return states.index and states.middle and not states.ring and not states.pinky


# Ordered (predicate, gesture) rules for every gesture that
# `FingerStates` alone can determine. THUMBS_UP/THUMBS_DOWN are
# deliberately not in this table -- resolving them needs landmark
# geometry, handled separately in `recognize()`. The patterns here are
# mutually exclusive by construction (each constrains a different
# combination of index/middle/ring/pinky, or folds vs. extends all
# fingers), so rule order does not change the outcome -- it is kept
# explicit and linear so new rules can be reasoned about individually
# rather than through a shared scoring or priority system.
_GESTURE_RULES: List[Tuple[Callable[[FingerStates], bool], Gesture]] = [
    (_is_open_palm, Gesture.OPEN_PALM),
    (_is_fist, Gesture.FIST),
    (_is_peace, Gesture.PEACE),
    (_is_point, Gesture.POINT),
]

# How far (in normalized coordinate units) the thumb tip must be above
# or below the wrist, vertically, before its direction is trusted. A
# small dead zone avoids flip-flopping between THUMBS_UP and
# THUMBS_DOWN for a thumb that is roughly level with the wrist (e.g.
# a hand held sideways, where "up" and "down" are themselves
# ambiguous). Below this margin, THUMBS_UP is reported as the
# documented fallback (see `recognize()`).
THUMB_DIRECTION_DEAD_ZONE = 0.02


def _is_thumb_pointing_up(landmarks: Landmarks) -> Optional[bool]:
    """Determine whether the thumb points up or down, from landmark geometry.

    Compares the thumb tip's vertical position to the wrist's: in
    MediaPipe's normalized image coordinates, y increases downward, so
    a thumb tip with a meaningfully *smaller* y than the wrist is
    higher on screen ("up"), and a meaningfully *larger* y is lower
    ("down"). This is a deliberately simple, real geometric check --
    not a guess -- appropriate for the thumbs-up/thumbs-down gesture,
    where the hand is typically held upright with the thumb clearly
    above or below the fist.

    Args:
        landmarks: The full 21-point landmark sequence the
            corresponding `FingerStates` was computed from.

    Returns:
        `True` if the thumb points up, `False` if it points down, or
        `None` if the thumb tip is within `THUMB_DIRECTION_DEAD_ZONE`
        of the wrist's height (direction not confidently determined).

    Limitations:
        This only considers vertical position relative to the wrist.
        It does not account for overall hand/camera rotation (e.g. a
        hand held sideways or a rotated camera), and does not use
        thumb joint angles or any 3D/depth information. For the
        upright thumbs-up/thumbs-down gesture this is built for, that
        is sufficient; a hand held at a steep angle may be
        misclassified.
    """
    wrist_y = landmarks[HandLandmark.WRIST][1]
    thumb_tip_y = landmarks[HandLandmark.THUMB_TIP][1]
    vertical_offset = thumb_tip_y - wrist_y

    if abs(vertical_offset) < THUMB_DIRECTION_DEAD_ZONE:
        return None

    return vertical_offset < 0  # smaller y (thumb tip) than wrist -> pointing up


class GestureRecognizer:
    """Maps `FingerStates` (and optionally `Landmarks`) to a `Gesture`.

    This class performs no image processing or hand tracking itself --
    it operates on the boolean per-finger output already computed by
    `FingerStateDetector`, plus (optionally) the same landmark
    coordinates that detector was given.

    THUMBS_UP vs. THUMBS_DOWN:
        A thumb-only-extended `FingerStates` is ambiguous by itself.
        - If `landmarks` is provided, the thumb tip's position relative
          to the wrist decides the direction (see
          `_is_thumb_pointing_up`), so both `Gesture.THUMBS_UP` and
          `Gesture.THUMBS_DOWN` can be genuinely, geometrically
          determined -- never guessed.
        - If `landmarks` is omitted, or the thumb is too close to
          level with the wrist to confidently call a direction, this
          falls back to `Gesture.THUMBS_UP`, matching this class's
          original documented behavior. `Gesture.THUMBS_DOWN` is never
          produced without landmark evidence supporting it.

    Example:
        recognizer = GestureRecognizer()
        states = FingerStates(
            thumb=False, index=True, middle=False, ring=False, pinky=False
        )
        gesture = recognizer.recognize(states)
        assert gesture == Gesture.POINT
    """

    def recognize(
        self, finger_states: FingerStates, landmarks: Optional[Landmarks] = None
    ) -> Gesture:
        """Determine the gesture represented by a `FingerStates` snapshot.

        Args:
            finger_states: The per-finger extension state to classify.
            landmarks: The same 21-point landmark sequence
                `finger_states` was computed from, if available. Only
                consulted to disambiguate THUMBS_UP from THUMBS_DOWN;
                every other gesture is determined from `finger_states`
                alone.

        Returns:
            The matching `Gesture`, or `Gesture.UNKNOWN` if
            `finger_states` does not match any currently supported
            pattern. Matching is exact (no fuzzy/partial matching).
        """
        for matches, gesture in _GESTURE_RULES:
            if matches(finger_states):
                return gesture

        if _is_thumb_only_extended(finger_states):
            if landmarks is not None:
                pointing_up = _is_thumb_pointing_up(landmarks)
                if pointing_up is not None:
                    return Gesture.THUMBS_UP if pointing_up else Gesture.THUMBS_DOWN
            # No landmarks, or direction not confidently determined:
            # documented fallback -- never fabricate THUMBS_DOWN.
            return Gesture.THUMBS_UP

        return Gesture.UNKNOWN