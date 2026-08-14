"""Unit tests for gesture_control.tracking.mediapipe_hand_tracker.

SAFETY: No test here invokes real MediaPipe processing, opens a
webcam, or touches real camera frames. Every test injects a fake
underlying tracker (`FakeHandTracker`, defined below) whose
`process()` method returns a plain Python object shaped like MediaPipe's
`HandLandmarkerResult` (only `.hand_landmarks` / `.handedness`,
built from simple fake landmark/category objects). None of this
requires MediaPipe to be installed or usable in this environment.
"""

import os
import sys
from typing import Any, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.tracking.hand_tracker_protocol import HandTrackerProtocol
from gesture_control.tracking.hand_tracking_result import Handedness, HandTrackingResult
from gesture_control.tracking.mediapipe_hand_tracker import MediaPipeHandTracker


class FakeLandmark:
    """A minimal stand-in for MediaPipe's NormalizedLandmark: has x, y, z."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeCategory:
    """A minimal stand-in for MediaPipe's handedness Category."""

    def __init__(self, category_name: str) -> None:
        self.category_name = category_name


class FakeHandLandmarkerResult:
    """A minimal stand-in for MediaPipe's HandLandmarkerResult."""

    def __init__(self, hand_landmarks: List[List[FakeLandmark]], handedness: List[List[FakeCategory]]) -> None:
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness


class FakeHandTracker:
    """A fake stand-in for gesture_control.vision.hand_tracker.HandTracker.

    Records the frames passed to `process()` and returns a
    pre-configured fake result, so tests can assert on both what was
    passed in and what came out -- with zero MediaPipe/webcam
    involvement.
    """

    def __init__(self, result: FakeHandLandmarkerResult) -> None:
        self._result = result
        self.processed_frames: List[Any] = []
        self.closed = False

    def process(self, frame: Any) -> FakeHandLandmarkerResult:
        self.processed_frames.append(frame)
        return self._result

    def close(self) -> None:
        self.closed = True


def _make_fake_landmarks(scale: float = 0.01) -> List[FakeLandmark]:
    return [FakeLandmark(i * scale, i * scale * 2, i * scale * 3) for i in range(21)]


def _make_fake_result(
    num_hands: int = 1,
    handedness_names: Optional[List[Optional[str]]] = None,
) -> FakeHandLandmarkerResult:
    hand_landmarks = [_make_fake_landmarks(scale=0.01 * (i + 1)) for i in range(num_hands)]
    if handedness_names is None:
        handedness_names = ["Right"] * num_hands
    handedness = [
        ([FakeCategory(name)] if name is not None else [])
        for name in handedness_names
    ]
    return FakeHandLandmarkerResult(hand_landmarks=hand_landmarks, handedness=handedness)


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------

class TestConversion:
    def test_single_hand_is_converted_correctly(self) -> None:
        fake_result = _make_fake_result(num_hands=1, handedness_names=["Right"])
        fake_tracker = FakeHandTracker(fake_result)
        adapter = MediaPipeHandTracker(tracker=fake_tracker)

        result = adapter.detect(frame="fake-frame")

        assert isinstance(result, HandTrackingResult)
        assert result.num_hands == 1
        hand = result.hands[0]
        assert hand.handedness == Handedness.RIGHT
        assert len(hand.landmarks) == 21
        assert hand.landmarks[0] == (0.0, 0.0, 0.0)
        assert hand.landmarks[1] == pytest.approx((0.01, 0.02, 0.03))

    def test_landmark_values_are_plain_floats_not_mediapipe_objects(self) -> None:
        fake_tracker = FakeHandTracker(_make_fake_result(num_hands=1))
        adapter = MediaPipeHandTracker(tracker=fake_tracker)
        result = adapter.detect(frame=None)
        for coordinate in result.hands[0].landmarks[5]:
            assert isinstance(coordinate, float)

    def test_multiple_hands_are_each_converted(self) -> None:
        fake_result = _make_fake_result(num_hands=2, handedness_names=["Left", "Right"])
        fake_tracker = FakeHandTracker(fake_result)
        adapter = MediaPipeHandTracker(tracker=fake_tracker)

        result = adapter.detect(frame="fake-frame")

        assert result.num_hands == 2
        assert result.hands[0].handedness == Handedness.LEFT
        assert result.hands[1].handedness == Handedness.RIGHT

    def test_unrecognized_handedness_category_name_maps_to_unknown(self) -> None:
        fake_result = _make_fake_result(num_hands=1, handedness_names=["Ambidextrous"])
        adapter = MediaPipeHandTracker(tracker=FakeHandTracker(fake_result))
        result = adapter.detect(frame=None)
        assert result.hands[0].handedness == Handedness.UNKNOWN

    def test_missing_handedness_for_a_hand_maps_to_unknown(self) -> None:
        fake_result = _make_fake_result(num_hands=1, handedness_names=[None])
        adapter = MediaPipeHandTracker(tracker=FakeHandTracker(fake_result))
        result = adapter.detect(frame=None)
        assert result.hands[0].handedness == Handedness.UNKNOWN


# ---------------------------------------------------------------------------
# Empty / no-hand results
# ---------------------------------------------------------------------------

class TestNoHandDetected:
    def test_zero_hands_is_handled_safely(self) -> None:
        fake_result = _make_fake_result(num_hands=0)
        adapter = MediaPipeHandTracker(tracker=FakeHandTracker(fake_result))
        result = adapter.detect(frame="empty-scene")
        assert result.num_hands == 0
        assert result.has_hands is False
        assert result.hands == ()


# ---------------------------------------------------------------------------
# Frame pass-through / dependency injection
# ---------------------------------------------------------------------------

class TestDependencyInjection:
    def test_frame_is_forwarded_unchanged_to_the_underlying_tracker(self) -> None:
        fake_tracker = FakeHandTracker(_make_fake_result(num_hands=0))
        adapter = MediaPipeHandTracker(tracker=fake_tracker)
        sentinel_frame = object()
        adapter.detect(sentinel_frame)
        assert fake_tracker.processed_frames == [sentinel_frame]

    def test_two_adapters_with_different_fakes_are_independent(self) -> None:
        tracker_a = FakeHandTracker(_make_fake_result(num_hands=1))
        tracker_b = FakeHandTracker(_make_fake_result(num_hands=0))
        adapter_a = MediaPipeHandTracker(tracker=tracker_a)
        adapter_b = MediaPipeHandTracker(tracker=tracker_b)

        result_a = adapter_a.detect("frame")
        result_b = adapter_b.detect("frame")

        assert result_a.num_hands == 1
        assert result_b.num_hands == 0

    def test_close_delegates_to_underlying_tracker(self) -> None:
        fake_tracker = FakeHandTracker(_make_fake_result(num_hands=0))
        adapter = MediaPipeHandTracker(tracker=fake_tracker)
        adapter.close()
        assert fake_tracker.closed is True

    def test_close_is_safe_when_underlying_tracker_has_no_close_method(self) -> None:
        class TrackerWithoutClose:
            def process(self, frame: Any) -> FakeHandLandmarkerResult:
                return _make_fake_result(num_hands=0)

        adapter = MediaPipeHandTracker(tracker=TrackerWithoutClose())
        adapter.close()  # must not raise

    def test_context_manager_closes_the_underlying_tracker(self) -> None:
        fake_tracker = FakeHandTracker(_make_fake_result(num_hands=0))
        with MediaPipeHandTracker(tracker=fake_tracker) as adapter:
            adapter.detect("frame")
        assert fake_tracker.closed is True


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_adapter_satisfies_hand_tracker_protocol(self) -> None:
        adapter = MediaPipeHandTracker(tracker=FakeHandTracker(_make_fake_result(num_hands=0)))
        assert isinstance(adapter, HandTrackerProtocol)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_detect_calls_are_consistent(self) -> None:
        fake_tracker = FakeHandTracker(_make_fake_result(num_hands=1, handedness_names=["Left"]))
        adapter = MediaPipeHandTracker(tracker=fake_tracker)
        results = [adapter.detect("frame") for _ in range(5)]
        assert all(r.num_hands == 1 for r in results)
        assert all(r.hands[0].handedness == Handedness.LEFT for r in results)


# ---------------------------------------------------------------------------
# Import safety: the package/module must import even if MediaPipe cannot
# ---------------------------------------------------------------------------

class TestImportSafety:
    def test_importing_the_adapter_module_never_raises(self) -> None:
        import gesture_control.tracking.mediapipe_hand_tracker  # noqa: F401

    def test_importing_the_tracking_package_never_raises(self) -> None:
        import gesture_control.tracking  # noqa: F401

    def test_constructing_without_injection_raises_a_clear_error_or_succeeds(self) -> None:
        # Whether the real MediaPipe-backed HandTracker is usable here
        # is environment-dependent: it requires both a working
        # MediaPipe import AND a downloaded model file. This only
        # asserts that, if either is missing, the failure is a clear,
        # documented exception type rather than an obscure crash -- it
        # never falls back to touching a real camera or MediaPipe
        # runtime as a side effect either way.
        from gesture_control.vision.hand_tracker import HandTrackerError

        try:
            MediaPipeHandTracker()
        except RuntimeError:
            pass  # MediaPipe itself could not be imported in this environment.
        except HandTrackerError:
            pass  # MediaPipe imported fine, but e.g. the model file is missing.