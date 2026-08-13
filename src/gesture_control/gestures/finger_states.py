"""Finger-extension detection from a full set of hand landmarks.

This module answers exactly one question per finger: is it extended or
folded? It does not recognize gestures, does not detect pinches, and
has no notion of clicks, commands, or mouse/keyboard control -- those
belong to later modules that consume `FingerStates`.

Landmark representation:
    Landmarks are accepted as a generic sequence of 21 `(x, y, z)`
    coordinate tuples, indexed exactly as MediaPipe's standard 21-point
    hand model (see `HandLandmark` below). This module does not import
    or depend on the MediaPipe runtime -- any source that can produce
    21 `(x, y, z)` tuples in that order works.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from .geometry import Point3D, angle_between_points

# Angle (in degrees) at a finger's PIP joint, formed by MCP -> PIP -> DIP,
# at or above which the finger is considered straight/extended. A fully
# extended finger is close to 180 degrees (straight); a folded finger
# bends sharply at the PIP, producing a much smaller angle.
DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES: float = 160.0

# Angle (in degrees) at the thumb's IP joint, formed by MCP -> IP -> TIP,
# at or above which the thumb is considered extended. This is deliberately
# lower than the four-finger threshold -- see the module docstring on
# `is_thumb_extended` for why.
DEFAULT_THUMB_STRAIGHTNESS_THRESHOLD_DEGREES: float = 150.0

NUM_HAND_LANDMARKS = 21


class HandLandmark(IntEnum):
    """Indices into a 21-point MediaPipe-style hand landmark list."""

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


Landmarks = Sequence[Point3D]


@dataclass(frozen=True)
class FingerStates:
    """Whether each finger is extended, at a single point in time.

    Immutable by design: a detection result is a snapshot and should
    not be mutated after the fact.
    """

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool


def _validate_landmarks(landmarks: Landmarks) -> None:
    """Ensure `landmarks` contains exactly the expected 21 points.

    Args:
        landmarks: The candidate landmark sequence.

    Raises:
        ValueError: If `landmarks` does not contain exactly
            `NUM_HAND_LANDMARKS` points.
    """
    count = len(landmarks)
    if count != NUM_HAND_LANDMARKS:
        raise ValueError(
            f"Expected {NUM_HAND_LANDMARKS} hand landmarks, got {count}."
        )


def _is_long_finger_extended(
    landmarks: Landmarks,
    mcp: HandLandmark,
    pip: HandLandmark,
    dip: HandLandmark,
    threshold_degrees: float,
) -> bool:
    """Determine whether one of the four long fingers is extended.

    Uses the angle at the PIP joint, formed by MCP -> PIP -> DIP. The
    PIP joint is the primary hinge that drives finger curling: when a
    finger is straight, this angle is close to 180 degrees; when the
    finger folds (e.g. making a fist), the PIP joint bends sharply and
    the angle drops well below the threshold. This single-angle check
    is simple, deterministic, and -- because it is computed purely from
    relative joint geometry -- invariant to the hand's overall position,
    rotation, or scale in the frame.

    Args:
        landmarks: The full 21-point landmark sequence.
        mcp: Index of the finger's MCP landmark.
        pip: Index of the finger's PIP landmark.
        dip: Index of the finger's DIP landmark.
        threshold_degrees: Minimum PIP angle, in degrees, to count as
            extended.

    Returns:
        True if the finger is extended, False if it is folded.
    """
    angle = angle_between_points(landmarks[mcp], landmarks[pip], landmarks[dip])
    return angle >= threshold_degrees


def _is_thumb_extended(landmarks: Landmarks, threshold_degrees: float) -> bool:
    """Determine whether the thumb is extended.

    The thumb is handled separately from the other four fingers because
    its joint geometry and range of motion are fundamentally different:
    it has one fewer interphalangeal joint than the long fingers (no
    PIP), it is opposable and rotated relative to the palm, and its
    natural motion is as much side-to-side (ab/adduction, across the
    palm) as it is curling. Reusing the four-finger MCP->PIP->DIP rule
    verbatim would be measuring the wrong joint entirely.

    Instead, this uses the angle at the thumb's IP joint, formed by
    MCP -> IP -> TIP -- the equivalent "does the last segment bend"
    check, but for the thumb's actual joint layout. A lower threshold
    than the long fingers is used by default, because even a
    comfortably extended thumb is anatomically straighter at the base
    than at the tip, and typically reads as somewhat less than 180
    degrees at the IP joint from most camera angles.

    Args:
        landmarks: The full 21-point landmark sequence.
        threshold_degrees: Minimum IP-joint angle, in degrees, to count
            the thumb as extended.

    Returns:
        True if the thumb is extended, False if it is folded.
    """
    angle = angle_between_points(
        landmarks[HandLandmark.THUMB_MCP],
        landmarks[HandLandmark.THUMB_IP],
        landmarks[HandLandmark.THUMB_TIP],
    )
    return angle >= threshold_degrees


class FingerStateDetector:
    """Detects per-finger extension state from a 21-point hand landmark set.

    This class only determines whether each finger is extended or
    folded, using joint angles from `gestures.geometry`. It has no
    knowledge of gestures, pinches, or any downstream interpretation.

    Example:
        detector = FingerStateDetector()
        states = detector.detect(landmarks)
        if states.index and not states.middle:
            ...  # a later module decides what this means, not this one
    """

    def __init__(
        self,
        finger_threshold_degrees: float = DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES,
        thumb_threshold_degrees: float = DEFAULT_THUMB_STRAIGHTNESS_THRESHOLD_DEGREES,
    ) -> None:
        """Initialize the detector with configurable angle thresholds.

        Args:
            finger_threshold_degrees: Minimum PIP-joint angle, in
                degrees, for the index/middle/ring/pinky fingers to be
                considered extended.
            thumb_threshold_degrees: Minimum IP-joint angle, in
                degrees, for the thumb to be considered extended.
        """
        self._finger_threshold_degrees = finger_threshold_degrees
        self._thumb_threshold_degrees = thumb_threshold_degrees

    def detect(self, landmarks: Landmarks) -> FingerStates:
        """Compute the extension state of each finger from hand landmarks.

        Args:
            landmarks: A sequence of exactly 21 `(x, y, z)` coordinate
                tuples, ordered per `HandLandmark`.

        Returns:
            A `FingerStates` snapshot for this set of landmarks.

        Raises:
            ValueError: If `landmarks` does not contain exactly 21
                points.
        """
        _validate_landmarks(landmarks)

        return FingerStates(
            thumb=_is_thumb_extended(landmarks, self._thumb_threshold_degrees),
            index=_is_long_finger_extended(
                landmarks,
                HandLandmark.INDEX_MCP,
                HandLandmark.INDEX_PIP,
                HandLandmark.INDEX_DIP,
                self._finger_threshold_degrees,
            ),
            middle=_is_long_finger_extended(
                landmarks,
                HandLandmark.MIDDLE_MCP,
                HandLandmark.MIDDLE_PIP,
                HandLandmark.MIDDLE_DIP,
                self._finger_threshold_degrees,
            ),
            ring=_is_long_finger_extended(
                landmarks,
                HandLandmark.RING_MCP,
                HandLandmark.RING_PIP,
                HandLandmark.RING_DIP,
                self._finger_threshold_degrees,
            ),
            pinky=_is_long_finger_extended(
                landmarks,
                HandLandmark.PINKY_MCP,
                HandLandmark.PINKY_PIP,
                HandLandmark.PINKY_DIP,
                self._finger_threshold_degrees,
            ),
        )