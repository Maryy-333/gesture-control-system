"""Unit tests for gesture_control.gestures.smoothener.

These tests exercise GestureSmoothener in isolation -- no webcam,
MediaPipe, OpenCV, PyAutoGUI, or OS involvement of any kind. They only
construct `Gesture` values and feed them to the smoothener.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.gestures.recognizer import Gesture
from gesture_control.gestures.smoothener import GestureSmoothener


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_new_smoothener_stable_gesture_is_none(self) -> None:
        smoothener = GestureSmoothener()
        assert smoothener.stable_gesture is None

    def test_new_smoothener_has_default_stability_frames(self) -> None:
        assert GestureSmoothener().stability_frames == 3


# ---------------------------------------------------------------------------
# 2. First gesture requires stability (consecutive confirmation)
# ---------------------------------------------------------------------------


class TestFirstGestureRequiresStability:
    @pytest.mark.parametrize(
        "gesture",
        [Gesture.POINT, Gesture.THREE_FINGERS, Gesture.PEACE],
    )
    def test_first_frames_hold_initial_state_then_confirm(
        self, gesture: Gesture
    ) -> None:
        smoothener = GestureSmoothener(stability_frames=3)
        assert smoothener.smooth(gesture) == gesture
        assert smoothener.stable_gesture == gesture


# ---------------------------------------------------------------------------
# 3. Flicker does not change stable gesture
# ---------------------------------------------------------------------------


class TestFlickerDoesNotChangeStableGesture:
    def _seed(
        self,
        smoothener: GestureSmoothener,
        gesture: Gesture = Gesture.POINT,
    ) -> None:
        for _ in range(3):
            smoothener.smooth(gesture)

    def test_isolated_flicker_is_absorbed(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)
        self._seed(smoothener)
        assert smoothener.stable_gesture == Gesture.POINT

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.POINT) == Gesture.POINT
        assert smoothener.stable_gesture == Gesture.POINT

    def test_non_consecutive_alternation_never_confirms(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)
        self._seed(smoothener)

        results = [
            smoothener.smooth(g)
            for g in (
                Gesture.THREE_FINGERS,
                Gesture.POINT,
                Gesture.THREE_FINGERS,
                Gesture.POINT,
            )
        ]

        assert results == [Gesture.POINT] * 4
        assert smoothener.stable_gesture == Gesture.POINT


# ---------------------------------------------------------------------------
# 4. Three consecutive new gestures transition
# ---------------------------------------------------------------------------


class TestConsecutiveTransition:
    def test_three_consecutive_frames_confirm_new_gesture(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        assert smoothener.stable_gesture == Gesture.POINT

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.THREE_FINGERS
        assert smoothener.stable_gesture == Gesture.THREE_FINGERS

    def test_spec_example_A_full_sequence(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        outputs = [
            smoothener.smooth(g)
            for g in (
                Gesture.POINT,
                Gesture.POINT,
                Gesture.THREE_FINGERS,
                Gesture.POINT,
                Gesture.THREE_FINGERS,
                Gesture.THREE_FINGERS,
                Gesture.THREE_FINGERS,
            )
        ]

        assert outputs == [
            Gesture.POINT,
            Gesture.POINT,
            Gesture.POINT,
            Gesture.POINT,
            Gesture.POINT,
            Gesture.POINT,
            Gesture.THREE_FINGERS,
        ]


# ---------------------------------------------------------------------------
# 5. Candidate counter resets when gesture changes
# ---------------------------------------------------------------------------


class TestCandidateCounterResets:
    def test_candidate_run_is_broken_by_a_different_gesture(self) -> None:
        # Stable POINT; two THREE_FINGERS frames (candidate_count: 1, 2),
        # then a POINT breaks the run; then three fresh consecutive
        # THREE_FINGERS frames finally confirm THREE_FINGERS.
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        assert smoothener.stable_gesture == Gesture.POINT

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.POINT) == Gesture.POINT

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert (
            smoothener.smooth(Gesture.THREE_FINGERS)
            == Gesture.THREE_FINGERS
        )

    def test_candidate_gesture_different_from_stable_does_not_accumulate(
        self,
    ) -> None:
        # A THREE_FINGERS then a PEACE (different from both stable POINT
        # and the THREE_FINGERS candidate) resets the candidate to PEACE;
        # then THREE_FINGERS again starts a fresh candidate run.
        # Neither run reaches 3, so stable stays POINT.
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.smooth(Gesture.PEACE) == Gesture.POINT
        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.POINT
        assert smoothener.stable_gesture == Gesture.POINT


# ---------------------------------------------------------------------------
# 6. Same stable gesture has no unnecessary delay
# ---------------------------------------------------------------------------


class TestNoDelayForStableGesture:
    def test_holding_stable_gesture_reports_immediately(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        outputs = [
            smoothener.smooth(Gesture.POINT)
            for _ in range(4)
        ]

        assert outputs == [Gesture.POINT] * 4

    def test_repeated_same_gesture_after_confirmation_is_instant(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        for _ in range(10):
            assert smoothener.smooth(Gesture.POINT) == Gesture.POINT


# ---------------------------------------------------------------------------
# 7. stability_frames == 1
# ---------------------------------------------------------------------------


class TestStabilityFramesOne:
    def test_new_gesture_confirmed_immediately(self) -> None:
        smoothener = GestureSmoothener(stability_frames=1)

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.THREE_FINGERS
        assert smoothener.smooth(Gesture.POINT) == Gesture.POINT
        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.THREE_FINGERS

    def test_isolated_flicker_still_confirmed_at_stability_1(self) -> None:
        smoothener = GestureSmoothener(stability_frames=1)

        smoothener.smooth(Gesture.POINT)

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.THREE_FINGERS
        assert smoothener.smooth(Gesture.POINT) == Gesture.POINT


# ---------------------------------------------------------------------------
# 8. Invalid stability_frames
# ---------------------------------------------------------------------------


class TestInvalidStabilityFrames:
    @pytest.mark.parametrize("value", [0, -1, -5])
    def test_non_positive_raises_value_error(self, value: int) -> None:
        with pytest.raises(ValueError):
            GestureSmoothener(stability_frames=value)

    def test_float_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            GestureSmoothener(stability_frames=2.5)

    def test_non_integer_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            GestureSmoothener(stability_frames="three")  # type: ignore[arg-type]

    def test_boolean_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            GestureSmoothener(stability_frames=True)

        with pytest.raises(ValueError):
            GestureSmoothener(stability_frames=False)


# ---------------------------------------------------------------------------
# 9. reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_stable_gesture_to_none(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        assert smoothener.stable_gesture == Gesture.POINT

        smoothener.reset()

        assert smoothener.stable_gesture is None

    def test_reset_clears_candidate_state(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        smoothener.smooth(Gesture.THREE_FINGERS)
        smoothener.smooth(Gesture.THREE_FINGERS)

        smoothener.reset()

        assert smoothener.stable_gesture is None
        assert smoothener.smooth(Gesture.POINT) == Gesture.POINT
        assert smoothener.stable_gesture == Gesture.POINT

    def test_spec_example_E_reset_transitions(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        assert smoothener.smooth(Gesture.THREE_FINGERS) == Gesture.THREE_FINGERS

        smoothener.reset()

        assert smoothener.smooth(Gesture.POINT) == Gesture.POINT


# ---------------------------------------------------------------------------
# 10. Multiple instances are independent
# ---------------------------------------------------------------------------


class TestInstanceIndependence:
    def test_two_instances_do_not_share_state(self) -> None:
        a = GestureSmoothener(stability_frames=3)
        b = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            a.smooth(Gesture.THREE_FINGERS)

        assert b.stable_gesture is None
        assert b.smooth(Gesture.POINT) == Gesture.POINT
        assert b.stable_gesture == Gesture.POINT

    def test_different_stability_frames_are_independent(self) -> None:
        fast = GestureSmoothener(stability_frames=1)
        slow = GestureSmoothener(stability_frames=3)

        assert fast.smooth(Gesture.POINT) == Gesture.POINT
        assert slow.smooth(Gesture.POINT) == Gesture.POINT

        assert fast.smooth(Gesture.THREE_FINGERS) == Gesture.THREE_FINGERS
        assert slow.smooth(Gesture.THREE_FINGERS) == Gesture.POINT


# ---------------------------------------------------------------------------
# Spec examples B and C sanity checks
# ---------------------------------------------------------------------------


class TestSpecExamples:
    def test_example_b(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        for _ in range(3):
            smoothener.smooth(Gesture.POINT)

        outputs = [
            smoothener.smooth(Gesture.THREE_FINGERS)
            for _ in range(3)
        ]

        assert outputs == [
            Gesture.POINT,
            Gesture.POINT,
            Gesture.THREE_FINGERS,
        ]

    def test_example_c(self) -> None:
        smoothener = GestureSmoothener(stability_frames=3)

        smoothener.smooth(Gesture.POINT)

        outputs = [
            smoothener.smooth(g)
            for g in (
                Gesture.THREE_FINGERS,
                Gesture.POINT,
                Gesture.THREE_FINGERS,
                Gesture.POINT,
            )
        ]

        assert outputs == [Gesture.POINT] * 4