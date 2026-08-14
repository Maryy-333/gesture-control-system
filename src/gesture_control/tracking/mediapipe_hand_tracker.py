"""MediaPipe implementation of the HandTrackerProtocol abstraction.

This module -- together with `gesture_control.vision.hand_tracker`,
which it wraps -- is the only place in the tracking/perception layer
that knows MediaPipe exists. It adapts the MediaPipe-specific
`HandLandmarkerResult` (returned by
`gesture_control.vision.hand_tracker.HandTracker.process()`) into the
transport-agnostic `HandTrackingResult`/`DetectedHand` types from
`hand_tracking_result.py`, so nothing downstream of this module (the
existing `FingerStateDetector`, `GestureRecognizer`, etc.) needs a
MediaPipe import or any knowledge of MediaPipe's own result types.

Import safety:
    Importing `gesture_control.vision.hand_tracker` transitively
    imports MediaPipe, which can fail in environments where MediaPipe
    is not installed or not usable. That import is wrapped here in a
    module-level `try/except`, mirroring the pattern already used in
    `gesture_control.control.pyautogui_backend` for PyAutoGUI: if it
    fails, the failure is only raised later, when `MediaPipeHandTracker`
    is actually constructed without an injected replacement -- not
    when this module (or `gesture_control.tracking`) is merely
    imported. This means the rest of the project stays importable even
    on a machine without a working MediaPipe install.

Testability:
    `MediaPipeHandTracker` accepts an optional `tracker` argument so a
    fake/mock object can be injected in place of the real
    `gesture_control.vision.hand_tracker.HandTracker`. The fake only
    needs a `process(frame)` method returning an object with
    `hand_landmarks` (list of lists of objects with `.x`/`.y`/`.z`) and
    `handedness` (list of lists of objects with `.category_name`) --
    the same shape MediaPipe's `HandLandmarkerResult` has -- so tests
    never need MediaPipe, a webcam, or a real camera frame.
"""

from typing import Any, List, Optional, Tuple

from ..gestures.geometry import Point3D
from .hand_tracking_result import DetectedHand, Handedness, HandTrackingResult

try:
    from ..vision.hand_tracker import HandTracker as _RealHandTracker
    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as _import_error:  # pragma: no cover - environment-dependent
    _RealHandTracker = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = _import_error


def _category_name_to_handedness(category_name: Optional[str]) -> Handedness:
    """Map a MediaPipe handedness category name to our `Handedness` enum."""
    if not category_name:
        return Handedness.UNKNOWN
    normalized = category_name.strip().lower()
    if normalized == "left":
        return Handedness.LEFT
    if normalized == "right":
        return Handedness.RIGHT
    return Handedness.UNKNOWN


def _convert_landmarks(raw_landmarks: Any) -> Tuple[Point3D, ...]:
    """Convert MediaPipe-style landmark objects into `(x, y, z)` tuples."""
    return tuple((float(lm.x), float(lm.y), float(lm.z)) for lm in raw_landmarks)


def _convert_hand_handedness(raw_categories_for_hand: Any) -> Handedness:
    """Convert one hand's MediaPipe handedness categories to `Handedness`.

    MediaPipe reports handedness as a ranked list of category guesses
    per hand; only the top (first) guess is used.
    """
    if not raw_categories_for_hand:
        return Handedness.UNKNOWN
    top_category = raw_categories_for_hand[0]
    return _category_name_to_handedness(getattr(top_category, "category_name", None))


def _convert_result(raw_result: Any) -> HandTrackingResult:
    """Convert a MediaPipe `HandLandmarkerResult`-shaped object to `HandTrackingResult`."""
    raw_hand_landmarks: List[Any] = list(getattr(raw_result, "hand_landmarks", []) or [])
    raw_handedness: List[Any] = list(getattr(raw_result, "handedness", []) or [])

    hands = []
    for index, raw_landmarks_for_hand in enumerate(raw_hand_landmarks):
        raw_categories_for_hand = raw_handedness[index] if index < len(raw_handedness) else None
        hands.append(
            DetectedHand(
                landmarks=_convert_landmarks(raw_landmarks_for_hand),
                handedness=_convert_hand_handedness(raw_categories_for_hand),
            )
        )
    return HandTrackingResult(hands=tuple(hands))


class MediaPipeHandTracker:
    """Adapts `gesture_control.vision.hand_tracker.HandTracker` to `HandTrackerProtocol`.

    This class performs no hand-tracking math itself; it delegates
    frame processing to an underlying tracker (real or injected) and
    converts that tracker's result into the transport-agnostic
    `HandTrackingResult`.

    Example (production use, real MediaPipe):
        tracker = MediaPipeHandTracker()
        result = tracker.detect(frame)  # frame is a BGR np.ndarray
        if result.has_hands:
            landmarks = result.hands[0].landmarks
            finger_states = FingerStateDetector().detect(landmarks)

    Example (tests, fake underlying tracker -- see
    tests/test_mediapipe_hand_tracker.py):
        fake = FakeHandTracker(fake_result)
        adapter = MediaPipeHandTracker(tracker=fake)
    """

    def __init__(self, tracker: Optional[Any] = None, **tracker_kwargs: Any) -> None:
        """Initialize the adapter.

        Args:
            tracker: An object with a `process(frame)` method returning
                a MediaPipe-`HandLandmarkerResult`-shaped object (i.e.
                `gesture_control.vision.hand_tracker.HandTracker`, or a
                compatible fake). If omitted, a real `HandTracker` is
                constructed using `tracker_kwargs`.
            **tracker_kwargs: Forwarded to `HandTracker.__init__` when
                no `tracker` is injected (e.g. `model_path`,
                `max_num_hands`).

        Raises:
            RuntimeError: If no `tracker` was given and the real
                MediaPipe-backed `HandTracker` could not be imported in
                this environment.
        """
        if tracker is not None:
            self._tracker = tracker
        elif _RealHandTracker is not None:
            self._tracker = _RealHandTracker(**tracker_kwargs)
        else:
            raise RuntimeError(
                "MediaPipe hand tracking is unavailable in this environment "
                "(gesture_control.vision.hand_tracker could not be imported, "
                "which usually means MediaPipe is not installed or not "
                "usable here). Install/configure MediaPipe to use "
                "MediaPipeHandTracker with real tracking, or pass a "
                "compatible object via the `tracker` argument (e.g. for "
                "testing)."
            ) from _IMPORT_ERROR

    def detect(self, frame: Any) -> HandTrackingResult:
        """Detect hands in `frame` and return a transport-agnostic result.

        Args:
            frame: Passed through unchanged to the underlying tracker's
                `process()` method.

        Returns:
            A `HandTrackingResult` with zero or more `DetectedHand`
            entries, containing no MediaPipe-specific types.
        """
        raw_result = self._tracker.process(frame)
        return _convert_result(raw_result)

    def close(self) -> None:
        """Release the underlying tracker's resources, if it supports that.

        Safe to call even if the underlying tracker has no `close()`
        method (e.g. a minimal test fake).
        """
        close_method = getattr(self._tracker, "close", None)
        if callable(close_method):
            close_method()

    def __enter__(self) -> "MediaPipeHandTracker":
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Support use as a context manager: releases resources on exit."""
        self.close()