"""Unit tests for gesture_control.control.controller.

All tests use a fake, in-memory `ControlBackend` (`RecordingBackend`,
defined below) that only records what was called. These tests NEVER
move the real mouse, click anything, type anything, take a real
screenshot, or otherwise interact with the operating system.
"""

import os
import sys
from typing import List, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.actions.action import Action
from gesture_control.control.controller import (
    ComputerController,
    ControlBackend,
    NoOpControlBackend,
)


class RecordingBackend:
    """A fake `ControlBackend` that records calls instead of acting on them.

    Used only to observe, in tests, which backend method
    `ComputerController` dispatched to and with what arguments. It has
    no side effects on the real system whatsoever.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, tuple]] = []

    def move_cursor(self, x: int, y: int) -> None:
        self.calls.append(("move_cursor", (x, y)))

    def left_click(self) -> None:
        self.calls.append(("left_click", ()))

    def right_click(self) -> None:
        self.calls.append(("right_click", ()))

    def double_click(self) -> None:
        self.calls.append(("double_click", ()))

    def scroll_up(self) -> None:
        self.calls.append(("scroll_up", ()))

    def scroll_down(self) -> None:
        self.calls.append(("scroll_down", ()))

    def screenshot(self) -> None:
        self.calls.append(("screenshot", ()))

    def pause(self) -> None:
        self.calls.append(("pause", ()))


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def controller(backend: RecordingBackend) -> ComputerController:
    return ComputerController(backend=backend)


# ---------------------------------------------------------------------------
# Action.NONE
# ---------------------------------------------------------------------------

class TestNoneAction:
    def test_none_calls_nothing_on_the_backend(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        controller.execute(Action.NONE)
        assert backend.calls == []

    def test_none_with_extra_coordinates_still_calls_nothing(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        controller.execute(Action.NONE, x=1, y=2)
        assert backend.calls == []


# ---------------------------------------------------------------------------
# MOVE_CURSOR
# ---------------------------------------------------------------------------

class TestMoveCursor:
    def test_move_cursor_with_coordinates(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        controller.execute(Action.MOVE_CURSOR, x=100, y=200)
        assert backend.calls == [("move_cursor", (100, 200))]

    def test_move_cursor_missing_both_coordinates_raises(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        with pytest.raises(ValueError):
            controller.execute(Action.MOVE_CURSOR)
        assert backend.calls == []

    def test_move_cursor_missing_x_raises(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        with pytest.raises(ValueError):
            controller.execute(Action.MOVE_CURSOR, y=200)
        assert backend.calls == []

    def test_move_cursor_missing_y_raises(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        with pytest.raises(ValueError):
            controller.execute(Action.MOVE_CURSOR, x=100)
        assert backend.calls == []

    def test_move_cursor_accepts_zero_as_a_valid_coordinate(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        # 0 is falsy but a perfectly valid screen coordinate -- must
        # not be treated the same as "missing".
        controller.execute(Action.MOVE_CURSOR, x=0, y=0)
        assert backend.calls == [("move_cursor", (0, 0))]

    def test_move_cursor_accepts_negative_coordinates(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        controller.execute(Action.MOVE_CURSOR, x=-5, y=-10)
        assert backend.calls == [("move_cursor", (-5, -10))]


# ---------------------------------------------------------------------------
# Zero-argument actions
# ---------------------------------------------------------------------------

class TestZeroArgumentActions:
    def test_left_click(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.LEFT_CLICK)
        assert backend.calls == [("left_click", ())]

    def test_right_click(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.RIGHT_CLICK)
        assert backend.calls == [("right_click", ())]

    def test_double_click(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.DOUBLE_CLICK)
        assert backend.calls == [("double_click", ())]

    def test_scroll_up(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.SCROLL_UP)
        assert backend.calls == [("scroll_up", ())]

    def test_scroll_down(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.SCROLL_DOWN)
        assert backend.calls == [("scroll_down", ())]

    def test_screenshot(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.SCREENSHOT)
        assert backend.calls == [("screenshot", ())]

    def test_pause(self, controller: ComputerController, backend: RecordingBackend) -> None:
        controller.execute(Action.PAUSE)
        assert backend.calls == [("pause", ())]

    def test_zero_argument_actions_ignore_extraneous_coordinates(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        # Non-movement actions should not require coordinates, and
        # should behave identically whether or not stray x/y are passed.
        controller.execute(Action.LEFT_CLICK, x=1, y=2)
        assert backend.calls == [("left_click", ())]


# ---------------------------------------------------------------------------
# Invalid / unsupported actions
# ---------------------------------------------------------------------------

class TestInvalidActions:
    def test_unsupported_action_value_raises_value_error(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        with pytest.raises(ValueError):
            controller.execute("not_a_real_action")  # type: ignore[arg-type]
        assert backend.calls == []

    def test_none_object_as_action_raises_value_error(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        with pytest.raises(ValueError):
            controller.execute(None)  # type: ignore[arg-type]
        assert backend.calls == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_execution_of_same_action_is_consistent(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        for _ in range(10):
            controller.execute(Action.LEFT_CLICK)
        assert backend.calls == [("left_click", ())] * 10

    def test_repeated_move_cursor_with_same_coordinates_is_consistent(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        for _ in range(5):
            controller.execute(Action.MOVE_CURSOR, x=42, y=84)
        assert backend.calls == [("move_cursor", (42, 84))] * 5

    def test_sequence_of_distinct_actions_dispatches_in_order(
        self, controller: ComputerController, backend: RecordingBackend
    ) -> None:
        controller.execute(Action.MOVE_CURSOR, x=1, y=2)
        controller.execute(Action.LEFT_CLICK)
        controller.execute(Action.SCROLL_UP)
        assert backend.calls == [
            ("move_cursor", (1, 2)),
            ("left_click", ()),
            ("scroll_up", ()),
        ]


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

class TestDependencyInjection:
    def test_default_backend_is_a_safe_no_op(self) -> None:
        controller = ComputerController()
        # Every supported action should run without raising and without
        # requiring any real backend -- proving the default is safe.
        controller.execute(Action.NONE)
        controller.execute(Action.MOVE_CURSOR, x=10, y=10)
        controller.execute(Action.LEFT_CLICK)
        controller.execute(Action.RIGHT_CLICK)
        controller.execute(Action.DOUBLE_CLICK)
        controller.execute(Action.SCROLL_UP)
        controller.execute(Action.SCROLL_DOWN)
        controller.execute(Action.SCREENSHOT)
        controller.execute(Action.PAUSE)

    def test_no_op_backend_satisfies_the_control_backend_protocol(self) -> None:
        assert isinstance(NoOpControlBackend(), ControlBackend)

    def test_recording_backend_satisfies_the_control_backend_protocol(self) -> None:
        # A structurally-compatible fake, with no inheritance relationship
        # to ControlBackend, should still satisfy the protocol.
        assert isinstance(RecordingBackend(), ControlBackend)

    def test_two_controllers_with_different_backends_are_independent(self) -> None:
        backend_a = RecordingBackend()
        backend_b = RecordingBackend()
        controller_a = ComputerController(backend=backend_a)
        controller_b = ComputerController(backend=backend_b)

        controller_a.execute(Action.LEFT_CLICK)

        assert backend_a.calls == [("left_click", ())]
        assert backend_b.calls == []

    def test_swapping_backend_changes_dispatch_target_without_changing_controller_code(
        self,
    ) -> None:
        first_backend = RecordingBackend()
        second_backend = RecordingBackend()

        controller = ComputerController(backend=first_backend)
        controller.execute(Action.SCREENSHOT)

        controller = ComputerController(backend=second_backend)
        controller.execute(Action.SCREENSHOT)

        assert first_backend.calls == [("screenshot", ())]
        assert second_backend.calls == [("screenshot", ())]


# ---------------------------------------------------------------------------
# Public API example from the task spec
# ---------------------------------------------------------------------------

def test_readme_example(backend: RecordingBackend) -> None:
    controller = ComputerController(backend=backend)
    controller.execute(Action.MOVE_CURSOR, x=100, y=200)
    assert backend.calls == [("move_cursor", (100, 200))]