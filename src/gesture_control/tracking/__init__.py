"""Hand-tracking / perception abstraction subpackage.

Establishes the boundary between raw frame acquisition, hand landmark
detection, and the existing gesture-processing pipeline:

    frame -> HandTrackerProtocol -> HandTrackingResult -> DetectedHand.landmarks
        -> FingerStateDetector -> GestureRecognizer -> ...

`hand_tracking_result` and `hand_tracker_protocol` have no MediaPipe
dependency at all. `mediapipe_hand_tracker` is the only concrete
implementation here, and it isolates its MediaPipe usage internally
(see that module's docstring for import-safety details), so importing
this package does not require MediaPipe to be installed or usable.
"""

from .hand_tracker_protocol import HandTrackerProtocol
from .hand_tracking_result import DetectedHand, Handedness, HandTrackingResult
from .mediapipe_hand_tracker import MediaPipeHandTracker

__all__ = [
    "DetectedHand",
    "Handedness",
    "HandTrackingResult",
    "HandTrackerProtocol",
    "MediaPipeHandTracker",
]