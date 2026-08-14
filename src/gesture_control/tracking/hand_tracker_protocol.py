"""The hand-tracking abstraction: HandTrackerProtocol -> concrete implementation.

This is the boundary the rest of the project is meant to depend on.
`FingerStateDetector`, `GestureRecognizer`, `ActionMapper`, and
`ComputerController` never need to know this protocol exists -- they
only ever see `Landmarks`/`FingerStates`/`Gesture`/`Action`. This
protocol exists purely so that whatever sits *above* those layers (a
future real-time loop, not built in this milestone) can depend on
"something that detects hands in a frame" without hard-coding MediaPipe.

Any object with a matching `detect(frame) -> HandTrackingResult` method
satisfies this protocol structurally -- no inheritance required. That
includes `MediaPipeHandTracker` (the real implementation) and any fake
tracker a test constructs.
"""

from typing import Any, Protocol, runtime_checkable

from .hand_tracking_result import HandTrackingResult


@runtime_checkable
class HandTrackerProtocol(Protocol):
    """A source of hand-tracking results for a single frame at a time.

    `frame` is intentionally left as `Any`: this abstraction does not
    require a particular frame representation (e.g. a specific
    OpenCV/numpy array shape), only that whatever implementation is
    injected knows how to interpret whatever frame it's given.
    """

    def detect(self, frame: Any) -> HandTrackingResult:
        """Detect hands in a single frame and return the result.

        Args:
            frame: A single image/frame in whatever representation the
                concrete implementation expects.

        Returns:
            A `HandTrackingResult` describing the hands (if any) found
            in `frame`.
        """
        ...