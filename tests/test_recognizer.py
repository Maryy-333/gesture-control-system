"""Unit tests for gesture_control.gestures.recognizer.

These tests operate purely on `FingerStates` values -- no webcam,
MediaPipe, OpenCV, or image data is used or required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.gestures.finger_states import FingerStates
from gesture_control.gestures.recognizer import Gesture, GestureRecognizer


@pytest.fixture
def recognizer() -> GestureRecognizer:
    return GestureRecognizer()


# ---------------------------------------------------------------------------
# OPEN_PALM / FIST
# ---------------------------------------------------------------------------

class TestOpenPalmAndFist:
    def test_open_palm(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=True, index=True, middle=True, ring=True, pinky=True)
        assert recognizer.recognize(states) == Gesture.OPEN_PALM

    def test_fist(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=False, index=False, middle=False, ring=False, pinky=False)
        assert recognizer.recognize(states) == Gesture.FIST


# ---------------------------------------------------------------------------
# POINT (thumb state is irrelevant)
# ---------------------------------------------------------------------------

class TestPoint:
    def test_point_with_thumb_folded(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=False, index=True, middle=False, ring=False, pinky=False)
        assert recognizer.recognize(states) == Gesture.POINT

    def test_point_with_thumb_extended(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=True, index=True, middle=False, ring=False, pinky=False)
        assert recognizer.recognize(states) == Gesture.POINT


# ---------------------------------------------------------------------------
# PEACE (thumb state is irrelevant)
# ---------------------------------------------------------------------------

class TestPeace:
    def test_peace_with_thumb_folded(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=False, index=True, middle=True, ring=False, pinky=False)
        assert recognizer.recognize(states) == Gesture.PEACE

    def test_peace_with_thumb_extended(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=True, index=True, middle=True, ring=False, pinky=False)
        assert recognizer.recognize(states) == Gesture.PEACE


# ---------------------------------------------------------------------------
# THUMBS_UP
# ---------------------------------------------------------------------------

class TestThumbsUp:
    def test_thumbs_up(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=True, index=False, middle=False, ring=False, pinky=False)
        assert recognizer.recognize(states) == Gesture.THUMBS_UP


# ---------------------------------------------------------------------------
# THUMBS_DOWN limitation: FingerStates alone cannot distinguish it from
# THUMBS_UP, and the recognizer must not pretend otherwise.
# ---------------------------------------------------------------------------

class TestThumbsDownLimitation:
    def test_thumb_only_state_never_returns_thumbs_down(
        self, recognizer: GestureRecognizer
    ) -> None:
        # This is the exact same FingerStates pattern a real thumbs-down
        # hand would produce, since FingerStates has no concept of
        # thumb direction. The recognizer must consistently classify it
        # as THUMBS_UP rather than guessing or fabricating a distinction
        # it cannot actually make from finger-extension booleans alone.
        states = FingerStates(thumb=True, index=False, middle=False, ring=False, pinky=False)
        gesture = recognizer.recognize(states)
        assert gesture == Gesture.THUMBS_UP
        assert gesture != Gesture.THUMBS_DOWN

    def test_thumbs_down_is_never_produced_by_the_recognizer(
        self, recognizer: GestureRecognizer
    ) -> None:
        # Exhaustively check every possible FingerStates combination:
        # regardless of input, THUMBS_DOWN should never be the result,
        # since this recognizer has no way to detect thumb direction.
        for thumb in (True, False):
            for index in (True, False):
                for middle in (True, False):
                    for ring in (True, False):
                        for pinky in (True, False):
                            states = FingerStates(
                                thumb=thumb, index=index, middle=middle,
                                ring=ring, pinky=pinky,
                            )
                            assert recognizer.recognize(states) != Gesture.THUMBS_DOWN

    def test_thumbs_down_gesture_value_exists_in_enum(self) -> None:
        # THUMBS_DOWN must still exist as a stable identity for a future
        # orientation-aware recognizer to use.
        assert Gesture.THUMBS_DOWN.value == "thumbs_down"


# ---------------------------------------------------------------------------
# Unsupported combinations -> UNKNOWN
# ---------------------------------------------------------------------------

class TestUnknown:
    @pytest.mark.parametrize(
        "states",
        [
            # Index and ring extended, middle/pinky folded: not a defined pattern.
            FingerStates(thumb=False, index=True, middle=False, ring=True, pinky=False),
            # Three long fingers extended, thumb/pinky folded.
            FingerStates(thumb=False, index=True, middle=True, ring=True, pinky=False),
            # Only middle finger extended.
            FingerStates(thumb=False, index=False, middle=True, ring=False, pinky=False),
            # Only ring finger extended.
            FingerStates(thumb=False, index=False, middle=False, ring=True, pinky=False),
            # Only pinky extended.
            FingerStates(thumb=False, index=False, middle=False, ring=False, pinky=True),
            # Thumb + index + middle + ring extended, pinky folded.
            FingerStates(thumb=True, index=True, middle=True, ring=True, pinky=False),
        ],
    )
    def test_unsupported_combinations_return_unknown(
        self, recognizer: GestureRecognizer, states: FingerStates
    ) -> None:
        assert recognizer.recognize(states) == Gesture.UNKNOWN


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_always_yields_same_output(self, recognizer: GestureRecognizer) -> None:
        states = FingerStates(thumb=False, index=True, middle=True, ring=False, pinky=False)
        results = {recognizer.recognize(states) for _ in range(50)}
        assert results == {Gesture.PEACE}

    def test_multiple_recognizer_instances_agree(self) -> None:
        states = FingerStates(thumb=True, index=True, middle=True, ring=True, pinky=True)
        first = GestureRecognizer()
        second = GestureRecognizer()
        assert first.recognize(states) == second.recognize(states) == Gesture.OPEN_PALM

    def test_all_32_combinations_are_classified_consistently(self) -> None:
        # Running the full input space twice through fresh recognizer
        # instances must produce identical results every time.
        def classify_all():
            r = GestureRecognizer()
            results = []
            for thumb in (True, False):
                for index in (True, False):
                    for middle in (True, False):
                        for ring in (True, False):
                            for pinky in (True, False):
                                states = FingerStates(
                                    thumb=thumb, index=index, middle=middle,
                                    ring=ring, pinky=pinky,
                                )
                                results.append(r.recognize(states))
            return results

        assert classify_all() == classify_all()


# ---------------------------------------------------------------------------
# README example from the task spec
# ---------------------------------------------------------------------------

def test_readme_example(recognizer: GestureRecognizer) -> None:
    states = FingerStates(
        thumb=False,
        index=True,
        middle=False,
        ring=False,
        pinky=False,
    )
    gesture = recognizer.recognize(states)
    assert gesture == Gesture.POINT