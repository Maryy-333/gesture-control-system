"""Transport-agnostic hand-tracking result types.

These types represent the OUTPUT of hand detection in a form the
existing gesture pipeline already understands, with no dependency on
MediaPipe, OpenCV, or any other perception library. `DetectedHand`
carries landmarks as `Sequence[Point3D]` -- exactly the `Landmarks`
type `FingerStateDetector.detect()` already accepts (see
`gesture_control.gestures.finger_states`) -- so no new landmark model
is introduced here; this reuses the existing one.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from ..gestures.geometry import Point3D


class Handedness(str, Enum):
    """Which hand was detected, independent of any specific tracking library.

    String-valued for the same reason as `Gesture` and `Action`: stable
    values that are safe to log or serialize.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedHand:
    """One detected hand: its landmarks and (if known) its handedness.

    `landmarks` is expected to contain 21 `(x, y, z)` points ordered
    per `gesture_control.gestures.finger_states.HandLandmark` -- the
    same ordering MediaPipe's own hand model uses -- so it can be
    passed directly to `FingerStateDetector.detect()` with no
    conversion. This class does not itself validate the landmark
    count; that validation already exists in `FingerStateDetector` and
    is not duplicated here.
    """

    landmarks: Tuple[Point3D, ...]
    handedness: Handedness = Handedness.UNKNOWN


@dataclass(frozen=True)
class HandTrackingResult:
    """The result of running hand detection on a single frame.

    Immutable and safe to hold onto after the frame that produced it
    has been discarded. An empty `hands` tuple (the default)
    represents "no hand detected" -- a normal, expected outcome, not
    an error.
    """

    hands: Tuple[DetectedHand, ...] = field(default_factory=tuple)

    @property
    def num_hands(self) -> int:
        """The number of hands detected in this result."""
        return len(self.hands)

    @property
    def has_hands(self) -> bool:
        """Whether at least one hand was detected."""
        return len(self.hands) > 0