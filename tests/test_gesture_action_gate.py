"""Unit tests for gesture_control.runtime.gesture_action_gate.

These tests operate purely on `Gesture` and `Action` enum values -- no
webcam, MediaPipe, OpenCV, PyAutoGUI, or OS interaction is used or
required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.actions.action import Action
from gesture_control.gestures.recognizer import Gesture
from gesture_control.runtime.gesture_action_gate import (
    CONTINUOUS_ACTIONS,
    GestureActionGate,
)


@pytest.fixture
def gate() -> GestureActionGate:
    return GestureActionGate()


# ---------------------------------------------------------------------------
# Debouncing discrete actions
# ---------------------------------------------------------------------------

class TestDiscreteActionDebouncing:
    def test_first_occurrence_of_a_gesture_fires(self, gate: GestureActionGate) -> None:
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True

    def test_holding_the_same_gesture_suppresses_further_firing(
        self, gate: GestureActionGate
    ) -> None:
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False

    def test_switching_to_a_different_discrete_gesture_fires_again(
        self, gate: GestureActionGate
    ) -> None:
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        assert gate.should_execute(Gesture.THUMBS_UP, Action.RIGHT_CLICK) is True
        assert gate.should_execute(Gesture.THUMBS_UP, Action.RIGHT_CLICK) is False

    def test_returning_to_a_previous_gesture_after_switching_away_fires_again(
        self, gate: GestureActionGate
    ) -> None:
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True
        assert gate.should_execute(Gesture.PEACE, Action.DOUBLE_CLICK) is True
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True  # FIST is "new" again

    @pytest.mark.parametrize(
        "gesture,action",
        [
            (Gesture.FIST, Action.LEFT_CLICK),
            (Gesture.THUMBS_UP, Action.RIGHT_CLICK),
            (Gesture.PEACE, Action.DOUBLE_CLICK),
        ],
    )
    def test_every_discrete_action_is_debounced(
        self, gate: GestureActionGate, gesture: Gesture, action: Action
    ) -> None:
        assert gate.should_execute(gesture, action) is True
        assert gate.should_execute(gesture, action) is False


# ---------------------------------------------------------------------------
# Continuous actions are exempt from debouncing
# ---------------------------------------------------------------------------

class TestContinuousActionsAlwaysFire:
    def test_move_cursor_fires_every_frame_while_held(self, gate: GestureActionGate) -> None:
        for _ in range(10):
            assert gate.should_execute(Gesture.POINT, Action.MOVE_CURSOR) is True

    def test_scroll_down_fires_every_frame_while_held(self, gate: GestureActionGate) -> None:
        for _ in range(10):
            assert gate.should_execute(Gesture.THUMBS_DOWN, Action.SCROLL_DOWN) is True

    def test_scroll_up_fires_every_frame_while_held(self, gate: GestureActionGate) -> None:
        for _ in range(10):
            # SCROLL_UP has no gesture mapped to it currently, but the
            # gate's classification is action-based and independent of
            # ActionMapper's current table.
            assert gate.should_execute(Gesture.UNKNOWN, Action.SCROLL_UP) is True

    def test_continuous_actions_set_is_exactly_the_expected_three(self) -> None:
        assert CONTINUOUS_ACTIONS == frozenset(
            {Action.MOVE_CURSOR, Action.SCROLL_UP, Action.SCROLL_DOWN}
        )


# ---------------------------------------------------------------------------
# No hand / None input
# ---------------------------------------------------------------------------

class TestNoHandInput:
    def test_none_gesture_and_action_returns_false(self, gate: GestureActionGate) -> None:
        assert gate.should_execute(None, None) is False

    def test_losing_the_hand_re_arms_the_previously_held_discrete_gesture(
        self, gate: GestureActionGate
    ) -> None:
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        gate.should_execute(None, None)  # hand lost
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True  # re-armed

    def test_losing_the_hand_does_not_resume_a_paused_gate(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True
        gate.should_execute(None, None)
        assert gate.is_paused is True  # briefly losing tracking must not silently resume


# ---------------------------------------------------------------------------
# PAUSE: absorbed, never dispatched, toggles paused state
# ---------------------------------------------------------------------------

class TestPauseToggle:
    def test_pause_never_returns_true(self, gate: GestureActionGate) -> None:
        assert gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE) is False

    def test_first_open_palm_pauses(self, gate: GestureActionGate) -> None:
        assert gate.is_paused is False
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True

    def test_holding_open_palm_does_not_re_toggle(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True

    def test_second_distinct_open_palm_entry_resumes(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)  # pause
        assert gate.is_paused is True
        gate.should_execute(Gesture.FIST, Action.LEFT_CLICK)  # switch away
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)  # open palm again -> resume
        assert gate.is_paused is False

    def test_pause_never_dispatches_even_though_it_toggles_state(
        self, gate: GestureActionGate
    ) -> None:
        results = [
            gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE),
            gate.should_execute(Gesture.FIST, Action.LEFT_CLICK),
            gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE),
        ]
        # PAUSE itself is always False; only the FIST in between (while
        # not yet paused... wait, gate IS paused after step 1) reflects
        # suppression while paused.
        assert results[0] is False  # PAUSE, absorbed
        assert results[2] is False  # PAUSE, absorbed


# ---------------------------------------------------------------------------
# While paused, everything is suppressed
# ---------------------------------------------------------------------------

class TestSuppressionWhilePaused:
    def test_discrete_actions_suppressed_while_paused(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        assert gate.should_execute(Gesture.THUMBS_UP, Action.RIGHT_CLICK) is False

    def test_continuous_actions_also_suppressed_while_paused(
        self, gate: GestureActionGate
    ) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True
        # Even MOVE_CURSOR -- normally exempt from debouncing -- must
        # be frozen while paused, or "pause" would not really pause.
        assert gate.should_execute(Gesture.POINT, Action.MOVE_CURSOR) is False
        assert gate.should_execute(Gesture.POINT, Action.MOVE_CURSOR) is False

    def test_dispatch_resumes_normally_after_unpausing(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)  # pause
        gate.should_execute(Gesture.FIST, Action.LEFT_CLICK)  # suppressed
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)  # resume

        assert gate.should_execute(Gesture.POINT, Action.MOVE_CURSOR) is True
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_re_arms_discrete_actions(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.FIST, Action.LEFT_CLICK)
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        gate.reset()
        assert gate.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True

    def test_reset_clears_paused_state(self, gate: GestureActionGate) -> None:
        gate.should_execute(Gesture.OPEN_PALM, Action.PAUSE)
        assert gate.is_paused is True
        gate.reset()
        assert gate.is_paused is False


# ---------------------------------------------------------------------------
# Determinism / independence of instances
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_sequences_on_fresh_gates_agree(self) -> None:
        def run_sequence(g: GestureActionGate):
            return [
                g.should_execute(Gesture.FIST, Action.LEFT_CLICK),
                g.should_execute(Gesture.FIST, Action.LEFT_CLICK),
                g.should_execute(Gesture.POINT, Action.MOVE_CURSOR),
                g.should_execute(Gesture.POINT, Action.MOVE_CURSOR),
            ]

        assert run_sequence(GestureActionGate()) == run_sequence(GestureActionGate())

    def test_two_gate_instances_do_not_share_state(self) -> None:
        gate_a = GestureActionGate()
        gate_b = GestureActionGate()

        gate_a.should_execute(Gesture.FIST, Action.LEFT_CLICK)
        assert gate_a.should_execute(Gesture.FIST, Action.LEFT_CLICK) is False
        # gate_b has never seen FIST -- it must fire fresh.
        assert gate_b.should_execute(Gesture.FIST, Action.LEFT_CLICK) is True


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------

class TestImportSafety:
    def test_module_has_no_forbidden_imports(self) -> None:
        import ast

        import gesture_control.runtime.gesture_action_gate as gate_module

        with open(gate_module.__file__) as f:
            tree = ast.parse(f.read())

        forbidden = {"cv2", "mediapipe", "pyautogui"}
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_roots.add(node.module.split(".")[0])

        assert not (imported_roots & forbidden)