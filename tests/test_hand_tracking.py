"""Unit tests for the hand-tracking abstraction's value objects and protocol.

None of these tests use a webcam, real camera frames, real MediaPipe
processing, OpenCV GUI, PyAutoGUI, or any real OS interaction -- they
operate purely on plain Python objects.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.gestures.finger_states import FingerStateDetector, HandLandmark
from gesture_control.gestures.recognizer import Gesture, GestureRecognizer
from gesture_control.tracking.hand_tracker_protocol import HandTrackerProtocol
from gesture_control.tracking.hand_tracking_result import (
    DetectedHand,
    Handedness,
    HandTrackingResult,
)


def _make_landmarks(scale: float = 0.01):
    """Build 21 simple, distinct (x, y, z) tuples for test data."""
    return tuple((i * scale, i * scale * 2, i * scale * 3) for i in range(21))


# ---------------------------------------------------------------------------
# DetectedHand / HandTrackingResult representation
# ---------------------------------------------------------------------------

class TestDetectedHandRepresentation:
    def test_stores_landmarks_and_handedness(self) -> None:
        landmarks = _make_landmarks()
        hand = DetectedHand(landmarks=landmarks, handedness=Handedness.LEFT)
        assert hand.landmarks == landmarks
        assert hand.handedness == Handedness.LEFT

    def test_handedness_defaults_to_unknown(self) -> None:
        hand = DetectedHand(landmarks=_make_landmarks())
        assert hand.handedness == Handedness.UNKNOWN

    def test_is_immutable(self) -> None:
        hand = DetectedHand(landmarks=_make_landmarks())
        with pytest.raises(Exception):
            hand.handedness = Handedness.RIGHT  # type: ignore[misc]

    def test_landmarks_are_compatible_with_finger_state_detector(self) -> None:
        # This is the core integration point: DetectedHand.landmarks
        # must be directly usable by the existing gesture pipeline with
        # no conversion step.
        hand = DetectedHand(landmarks=_make_landmarks())
        states = FingerStateDetector().detect(hand.landmarks)
        assert states is not None

    def test_landmark_count_matches_hand_landmark_enum_size(self) -> None:
        landmarks = _make_landmarks()
        assert len(landmarks) == len(list(HandLandmark))


class TestHandTrackingResult:
    def test_default_result_has_no_hands(self) -> None:
        result = HandTrackingResult()
        assert result.hands == ()
        assert result.num_hands == 0
        assert result.has_hands is False

    def test_result_with_one_hand(self) -> None:
        hand = DetectedHand(landmarks=_make_landmarks())
        result = HandTrackingResult(hands=(hand,))
        assert result.num_hands == 1
        assert result.has_hands is True
        assert result.hands[0] is hand

    def test_result_with_multiple_hands(self) -> None:
        left = DetectedHand(landmarks=_make_landmarks(0.01), handedness=Handedness.LEFT)
        right = DetectedHand(landmarks=_make_landmarks(0.02), handedness=Handedness.RIGHT)
        result = HandTrackingResult(hands=(left, right))
        assert result.num_hands == 2
        assert result.hands[0].handedness == Handedness.LEFT
        assert result.hands[1].handedness == Handedness.RIGHT

    def test_is_immutable(self) -> None:
        result = HandTrackingResult()
        with pytest.raises(Exception):
            result.hands = (DetectedHand(landmarks=_make_landmarks()),)  # type: ignore[misc]

    def test_no_hand_result_is_a_safe_empty_state_not_an_error(self) -> None:
        # Constructing and inspecting an empty result must never raise.
        result = HandTrackingResult()
        assert isinstance(result.num_hands, int)
        assert result.has_hands is False


# ---------------------------------------------------------------------------
# HandTrackerProtocol: any structurally-compatible object satisfies it
# ---------------------------------------------------------------------------

class _MinimalFakeTracker:
    """The simplest possible object satisfying HandTrackerProtocol."""

    def __init__(self, result: HandTrackingResult) -> None:
        self._result = result

    def detect(self, frame: object) -> HandTrackingResult:
        return self._result


class TestHandTrackerProtocolCompliance:
    def test_minimal_fake_tracker_satisfies_the_protocol(self) -> None:
        tracker = _MinimalFakeTracker(HandTrackingResult())
        assert isinstance(tracker, HandTrackerProtocol)

    def test_object_missing_detect_does_not_satisfy_the_protocol(self) -> None:
        class NotATracker:
            pass

        assert not isinstance(NotATracker(), HandTrackerProtocol)

    def test_fake_tracker_can_be_used_polymorphically(self) -> None:
        hand = DetectedHand(landmarks=_make_landmarks())
        tracker: HandTrackerProtocol = _MinimalFakeTracker(HandTrackingResult(hands=(hand,)))
        result = tracker.detect(frame="anything, never inspected by the fake")
        assert result.num_hands == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_detect_calls_on_fake_tracker_are_consistent(self) -> None:
        hand = DetectedHand(landmarks=_make_landmarks())
        tracker = _MinimalFakeTracker(HandTrackingResult(hands=(hand,)))
        results = [tracker.detect(None) for _ in range(10)]
        assert all(r.num_hands == 1 for r in results)
        assert all(r.hands[0] == hand for r in results)


# ---------------------------------------------------------------------------
# End-to-end decoupling: tracking result -> gesture pipeline, no MediaPipe
# types anywhere along the way.
# ---------------------------------------------------------------------------

class TestNoMediaPipeLeakageIntoGestureLayer:
    def test_open_palm_landmarks_flow_through_to_a_recognized_gesture(self) -> None:
        # Build landmarks representing a fully open palm using the same
        # straight-chain construction style as tests/test_finger_states.py,
        # then push them all the way through FingerStateDetector and
        # GestureRecognizer using only DetectedHand -- no MediaPipe type
        # appears anywhere in this test.
        def straight_chain(base, step, count):
            return [
                (base[0] + step[0] * i, base[1] + step[1] * i, base[2] + step[2] * i)
                for i in range(count)
            ]

        landmarks = [(0.0, 0.0, 0.0)] * 21

        def set_finger(mcp, pip, dip, tip, base, step):
            chain = straight_chain(base, step, 4)
            landmarks[mcp] = chain[0]
            landmarks[pip] = chain[1]
            landmarks[dip] = chain[2]
            landmarks[tip] = chain[3]

        set_finger(
            HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP,
            HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP,
            base=(0.2, 0.5, 0.0), step=(-0.1, 0.0, 0.0),
        )
        for mcp, pip, dip, tip, base in [
            (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP, (0.4, 0.5, 0.0)),
            (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP, (0.5, 0.5, 0.0)),
            (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP, (0.6, 0.5, 0.0)),
            (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP, (0.7, 0.5, 0.0)),
        ]:
            set_finger(mcp, pip, dip, tip, base=base, step=(0.0, -0.1, 0.0))

        hand = DetectedHand(landmarks=tuple(landmarks), handedness=Handedness.RIGHT)
        tracking_result = HandTrackingResult(hands=(hand,))

        finger_states = FingerStateDetector().detect(tracking_result.hands[0].landmarks)
        gesture = GestureRecognizer().recognize(finger_states)

        assert gesture == Gesture.OPEN_PALM