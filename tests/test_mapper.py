"""Unit tests for gesture_control.actions.mapper.

These tests operate purely on `Gesture` and `Action` enum values -- no
webcam, MediaPipe, OpenCV, PyAutoGUI, or OS mouse/keyboard APIs are
used or required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.actions.action import Action
from gesture_control.actions.mapper import ActionMapper
from gesture_control.gestures.recognizer import Gesture


@pytest.fixture
def mapper() -> ActionMapper:
    return ActionMapper()


# ---------------------------------------------------------------------------
# Required mappings
# ---------------------------------------------------------------------------

class TestGestureToActionMapping:
    def test_open_palm_maps_to_pause(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.OPEN_PALM) == Action.PAUSE

    def test_three_fingers_maps_to_left_click(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.THREE_FINGERS) == Action.LEFT_CLICK

    def test_point_maps_to_move_cursor(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.POINT) == Action.MOVE_CURSOR

    def test_peace_maps_to_double_click(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.PEACE) == Action.DOUBLE_CLICK

    def test_thumbs_up_maps_to_volume_up(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.THUMBS_UP) == Action.VOLUME_UP

    def test_thumbs_down_maps_to_volume_down(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.THUMBS_DOWN) == Action.VOLUME_DOWN

    def test_unknown_maps_to_none(self, mapper: ActionMapper) -> None:
        assert mapper.map(Gesture.UNKNOWN) == Action.NONE


# ---------------------------------------------------------------------------
# Full coverage of the Gesture enum
# ---------------------------------------------------------------------------

class TestFullGestureCoverage:
    def test_every_gesture_has_a_deterministic_mapping(self, mapper: ActionMapper) -> None:
        expected = {
            Gesture.OPEN_PALM: Action.PAUSE,
            Gesture.THREE_FINGERS: Action.LEFT_CLICK,
            Gesture.POINT: Action.MOVE_CURSOR,
            Gesture.PEACE: Action.DOUBLE_CLICK,
            Gesture.THUMBS_UP: Action.VOLUME_UP,
            Gesture.THUMBS_DOWN: Action.VOLUME_DOWN,
            Gesture.UNKNOWN: Action.NONE,
        }

        # Every Gesture member must be covered by the expected table
        # above, so this test fails loudly if a new Gesture is added
        # without updating both the mapper and this test.
        assert set(expected.keys()) == set(Gesture)

        for gesture, expected_action in expected.items():
            assert mapper.map(gesture) == expected_action

    def test_mapping_never_raises_for_any_gesture(self, mapper: ActionMapper) -> None:
        for gesture in Gesture:
            action = mapper.map(gesture)
            assert isinstance(action, Action)


# ---------------------------------------------------------------------------
# Determinism / statelessness
# ---------------------------------------------------------------------------

class TestDeterminismAndStatelessness:
    def test_repeated_mapping_of_same_gesture_is_stable(self, mapper: ActionMapper) -> None:
        results = {mapper.map(Gesture.THREE_FINGERS) for _ in range(50)}
        assert results == {Action.LEFT_CLICK}

    def test_two_mapper_instances_agree(self) -> None:
        first = ActionMapper()
        second = ActionMapper()
        for gesture in Gesture:
            assert first.map(gesture) == second.map(gesture)

    def test_mapping_one_gesture_does_not_affect_another(self, mapper: ActionMapper) -> None:
        # A stateless mapper must not let one call influence the next.
        assert mapper.map(Gesture.THREE_FINGERS) == Action.LEFT_CLICK
        assert mapper.map(Gesture.OPEN_PALM) == Action.PAUSE
        assert mapper.map(Gesture.THREE_FINGERS) == Action.LEFT_CLICK

    def test_interleaved_calls_across_instances_are_consistent(self) -> None:
        mapper_a = ActionMapper()
        mapper_b = ActionMapper()
        assert mapper_a.map(Gesture.PEACE) == Action.DOUBLE_CLICK
        assert mapper_b.map(Gesture.POINT) == Action.MOVE_CURSOR
        assert mapper_a.map(Gesture.PEACE) == Action.DOUBLE_CLICK
        assert mapper_b.map(Gesture.UNKNOWN) == Action.NONE


# ---------------------------------------------------------------------------
# Public API example from the task spec
# ---------------------------------------------------------------------------

def test_readme_example() -> None:
    mapper = ActionMapper()
    action = mapper.map(Gesture.POINT)
    assert action == Action.MOVE_CURSOR