"""Unit tests for gesture_control.runtime.runtime.

SAFETY: Every test injects fake HandTrackerProtocol / FingerStateDetector /
GestureRecognizer / ActionMapper / ComputerController objects. None of
these tests open a webcam, use real MediaPipe, use real PyAutoGUI, move
the real mouse, click anything, or access the real OS in any way.
"""

import ast
import os
import sys
from typing import Any, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.actions.action import Action
from gesture_control.gestures.finger_states import HandLandmark
from gesture_control.gestures.recognizer import Gesture
from gesture_control.mapping.coordinate_mapper import ScreenPoint
from gesture_control.runtime.runtime import FrameResult, GestureControlRuntime
from gesture_control.tracking.hand_tracking_result import (
    DetectedHand,
    Handedness,
    HandTrackingResult,
)


# ---------------------------------------------------------------------------
# Fakes -- no real MediaPipe/OpenCV/PyAutoGUI/webcam/OS involvement anywhere.
# ---------------------------------------------------------------------------

class FakeHandTracker:
    """Fake HandTrackerProtocol: returns a fixed result, records frames."""

    def __init__(self, result: HandTrackingResult) -> None:
        self._result = result
        self.received_frames: List[Any] = []

    def detect(self, frame: Any) -> HandTrackingResult:
        self.received_frames.append(frame)
        return self._result


class FakeFingerStateDetector:
    """Fake FingerStateDetector: returns a fixed sentinel, records input."""

    def __init__(self, finger_states: Any = "FAKE_FINGER_STATES") -> None:
        self._finger_states = finger_states
        self.received_landmarks: List[Any] = []

    def detect(self, landmarks: Any) -> Any:
        self.received_landmarks.append(landmarks)
        return self._finger_states


class FakeGestureRecognizer:
    """Fake GestureRecognizer: returns a fixed Gesture, records input."""

    def __init__(self, gesture: Gesture) -> None:
        self._gesture = gesture
        self.received_finger_states: List[Any] = []

    def recognize(self, finger_states: Any) -> Gesture:
        self.received_finger_states.append(finger_states)
        return self._gesture


class FakeActionMapper:
    """Fake ActionMapper: returns a fixed Action, records input."""

    def __init__(self, action: Action) -> None:
        self._action = action
        self.received_gestures: List[Gesture] = []

    def map(self, gesture: Gesture) -> Action:
        self.received_gestures.append(gesture)
        return self._action


class FakeComputerController:
    """Fake ComputerController: records every execute() call."""

    def __init__(self) -> None:
        self.executed_calls: List[Tuple[Action, Optional[int], Optional[int]]] = []

    def execute(self, action: Action, x: Optional[int] = None, y: Optional[int] = None) -> None:
        self.executed_calls.append((action, x, y))


class FakeCoordinateMapper:
    """Fake CoordinateMapper: returns a fixed ScreenPoint, records input.

    Used so runtime tests verify *dependency injection and wiring*
    (that the runtime calls the mapper with the right coordinates and
    forwards its result correctly) rather than re-testing
    `CoordinateMapper`'s own conversion math, which is already covered
    by tests/test_coordinate_mapper.py.
    """

    def __init__(self, screen_point: ScreenPoint = ScreenPoint(123, 456)) -> None:
        self._screen_point = screen_point
        self.received_points: List[Tuple[float, float]] = []

    def map_point(self, x: float, y: float) -> ScreenPoint:
        self.received_points.append((x, y))
        return self._screen_point


def _make_landmarks(
    scale: float = 0.01,
    index_tip: Optional[Tuple[float, float, float]] = None,
) -> Tuple[Tuple[float, float, float], ...]:
    landmarks = [(i * scale, i * scale * 2, i * scale * 3) for i in range(21)]
    if index_tip is not None:
        landmarks[HandLandmark.INDEX_TIP] = index_tip
    return tuple(landmarks)


def _make_hand(
    scale: float = 0.01,
    handedness: Handedness = Handedness.RIGHT,
    index_tip: Optional[Tuple[float, float, float]] = None,
) -> DetectedHand:
    return DetectedHand(landmarks=_make_landmarks(scale, index_tip=index_tip), handedness=handedness)


def _build_runtime(
    tracking_result: HandTrackingResult,
    gesture: Gesture = Gesture.FIST,
    action: Action = Action.LEFT_CLICK,
    coordinate_mapper: Optional[FakeCoordinateMapper] = None,
) -> Tuple[
    GestureControlRuntime,
    FakeHandTracker,
    FakeFingerStateDetector,
    FakeGestureRecognizer,
    FakeActionMapper,
    FakeCoordinateMapper,
    FakeComputerController,
]:
    # NOTE: `coordinate_mapper` is returned BEFORE `controller` (not
    # after), specifically so existing `runtime, *_, controller = ...`
    # call sites below keep binding `controller` to the actual
    # FakeComputerController rather than silently rebinding it to the
    # new FakeCoordinateMapper.
    tracker = FakeHandTracker(tracking_result)
    finger_state_detector = FakeFingerStateDetector()
    gesture_recognizer = FakeGestureRecognizer(gesture)
    action_mapper = FakeActionMapper(action)
    controller = FakeComputerController()
    mapper = coordinate_mapper if coordinate_mapper is not None else FakeCoordinateMapper()
    runtime = GestureControlRuntime(
        hand_tracker=tracker,
        finger_state_detector=finger_state_detector,
        gesture_recognizer=gesture_recognizer,
        action_mapper=action_mapper,
        computer_controller=controller,
        coordinate_mapper=mapper,
    )
    return runtime, tracker, finger_state_detector, gesture_recognizer, action_mapper, mapper, controller


# ---------------------------------------------------------------------------
# No hand detected
# ---------------------------------------------------------------------------

class TestNoHandDetected:
    def test_no_gesture_recognition_or_mapping_or_execution_occurs(self) -> None:
        runtime, tracker, fsd, gr, am, mapper, controller = _build_runtime(HandTrackingResult())

        result = runtime.process_frame("frame")

        assert result.hand_detected is False
        assert result.selected_hand is None
        assert result.gesture is None
        assert result.action is None
        assert result.action_executed is False
        assert fsd.received_landmarks == []
        assert gr.received_finger_states == []
        assert am.received_gestures == []
        assert controller.executed_calls == []

    def test_tracker_is_still_called(self) -> None:
        runtime, tracker, *_ = _build_runtime(HandTrackingResult())
        runtime.process_frame("my-frame")
        assert tracker.received_frames == ["my-frame"]

    def test_result_exposes_the_full_tracking_result(self) -> None:
        empty_result = HandTrackingResult()
        runtime, *_ = _build_runtime(empty_result)
        result = runtime.process_frame("frame")
        assert result.tracking_result is empty_result


# ---------------------------------------------------------------------------
# One hand detected: full pipeline wiring
# ---------------------------------------------------------------------------

class TestOneHandDetected:
    def test_first_hand_landmarks_are_passed_to_finger_state_detector(self) -> None:
        hand = _make_hand()
        runtime, _, fsd, *_ = _build_runtime(HandTrackingResult(hands=(hand,)))

        result = runtime.process_frame("frame")

        assert result.hand_detected is True
        assert result.selected_hand is hand
        assert fsd.received_landmarks == [hand.landmarks]

    def test_finger_states_are_passed_to_gesture_recognizer(self) -> None:
        hand = _make_hand()
        finger_states_sentinel = object()
        tracker = FakeHandTracker(HandTrackingResult(hands=(hand,)))
        fsd = FakeFingerStateDetector(finger_states=finger_states_sentinel)
        gr = FakeGestureRecognizer(Gesture.PEACE)
        am = FakeActionMapper(Action.SCREENSHOT)
        controller = FakeComputerController()
        mapper = FakeCoordinateMapper()
        runtime = GestureControlRuntime(tracker, fsd, gr, am, controller, mapper)

        runtime.process_frame("frame")

        assert gr.received_finger_states == [finger_states_sentinel]

    def test_recognized_gesture_is_passed_to_action_mapper(self) -> None:
        hand = _make_hand()
        runtime, _, _, gr, am, _, _ = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.PEACE, action=Action.SCREENSHOT
        )

        runtime.process_frame("frame")

        assert am.received_gestures == [Gesture.PEACE]

    def test_mapped_action_is_passed_to_controller(self) -> None:
        hand = _make_hand()
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.FIST, action=Action.LEFT_CLICK
        )

        result = runtime.process_frame("frame")

        assert controller.executed_calls == [(Action.LEFT_CLICK, None, None)]
        assert result.action == Action.LEFT_CLICK
        assert result.action_executed is True

    def test_result_reflects_the_full_frame_outcome(self) -> None:
        hand = _make_hand()
        runtime, *_ = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.FIST, action=Action.LEFT_CLICK
        )
        result = runtime.process_frame("frame")

        assert result == FrameResult(
            hand_detected=True,
            tracking_result=result.tracking_result,
            selected_hand=hand,
            gesture=Gesture.FIST,
            action=Action.LEFT_CLICK,
            action_executed=True,
        )


# ---------------------------------------------------------------------------
# Multiple hands: only the first is processed
# ---------------------------------------------------------------------------

class TestMultipleHandsDetected:
    def test_only_first_hand_is_selected_and_processed(self) -> None:
        first_hand = _make_hand(scale=0.01, handedness=Handedness.RIGHT)
        second_hand = _make_hand(scale=0.05, handedness=Handedness.LEFT)
        runtime, _, fsd, *_ = _build_runtime(
            HandTrackingResult(hands=(first_hand, second_hand))
        )

        result = runtime.process_frame("frame")

        assert result.selected_hand is first_hand
        assert fsd.received_landmarks == [first_hand.landmarks]

    def test_tracking_result_still_exposes_all_detected_hands(self) -> None:
        first_hand = _make_hand(scale=0.01)
        second_hand = _make_hand(scale=0.05)
        runtime, *_ = _build_runtime(HandTrackingResult(hands=(first_hand, second_hand)))

        result = runtime.process_frame("frame")

        assert result.tracking_result.num_hands == 2
        assert result.tracking_result.hands == (first_hand, second_hand)

    def test_move_cursor_uses_first_hands_index_tip_not_second_hands(self) -> None:
        first_hand = _make_hand(index_tip=(0.1, 0.1, 0.0), handedness=Handedness.RIGHT)
        second_hand = _make_hand(index_tip=(0.9, 0.9, 0.0), handedness=Handedness.LEFT)
        runtime, *_, mapper, controller = _build_runtime(
            HandTrackingResult(hands=(first_hand, second_hand)),
            gesture=Gesture.POINT,
            action=Action.MOVE_CURSOR,
        )

        runtime.process_frame("frame")

        assert mapper.received_points == [(0.1, 0.1)]


# ---------------------------------------------------------------------------
# MOVE_CURSOR: INDEX_TIP -> CoordinateMapper -> ComputerController
# ---------------------------------------------------------------------------

class TestMoveCursorCoordinateMapping:
    def test_point_uses_index_tip_coordinates(self) -> None:
        # A hand whose INDEX_TIP is the only landmark with these exact,
        # distinctive values -- every other landmark uses a different
        # scale -- so passing this assertion proves INDEX_TIP
        # specifically was read, not some other landmark.
        hand = _make_hand(scale=0.01, index_tip=(0.42, 0.73, 0.05))
        runtime, *_, mapper, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.POINT, action=Action.MOVE_CURSOR
        )

        runtime.process_frame("frame")

        assert mapper.received_points == [(0.42, 0.73)]

    def test_normalized_index_tip_coordinates_are_passed_through_coordinate_mapper(self) -> None:
        hand = _make_hand(index_tip=(0.1, 0.9, 0.0))
        runtime, *_, mapper, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.POINT, action=Action.MOVE_CURSOR
        )

        runtime.process_frame("frame")

        # Exactly the x, y from landmarks[INDEX_TIP] -- unmodified, and
        # the z component is not passed at all.
        expected_x, expected_y, _ = hand.landmarks[HandLandmark.INDEX_TIP]
        assert mapper.received_points == [(expected_x, expected_y)]

    def test_move_cursor_reaches_computer_controller(self) -> None:
        hand = _make_hand(index_tip=(0.5, 0.5, 0.0))
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.POINT, action=Action.MOVE_CURSOR
        )

        runtime.process_frame("frame")

        assert len(controller.executed_calls) == 1
        assert controller.executed_calls[0][0] == Action.MOVE_CURSOR

    def test_mapped_screen_coordinates_are_exactly_what_controller_receives(self) -> None:
        hand = _make_hand(index_tip=(0.3, 0.6, 0.0))
        mapper = FakeCoordinateMapper(screen_point=ScreenPoint(123, 456))
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)),
            gesture=Gesture.POINT,
            action=Action.MOVE_CURSOR,
            coordinate_mapper=mapper,
        )

        runtime.process_frame("frame")

        assert controller.executed_calls == [(Action.MOVE_CURSOR, 123, 456)]

    def test_move_cursor_sets_action_executed_true(self) -> None:
        hand = _make_hand()
        runtime, *_ = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.POINT, action=Action.MOVE_CURSOR
        )

        result = runtime.process_frame("frame")

        assert result.hand_detected is True
        assert result.gesture == Gesture.POINT
        assert result.action == Action.MOVE_CURSOR
        assert result.action_executed is True

    def test_non_move_cursor_actions_are_executed_normally(self) -> None:
        hand = _make_hand()
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.PEACE, action=Action.SCREENSHOT
        )

        result = runtime.process_frame("frame")

        assert result.action_executed is True
        assert controller.executed_calls == [(Action.SCREENSHOT, None, None)]

    def test_non_move_cursor_actions_do_not_consult_coordinate_mapper(self) -> None:
        hand = _make_hand()
        runtime, *_, mapper, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.FIST, action=Action.LEFT_CLICK
        )

        runtime.process_frame("frame")

        assert mapper.received_points == []
        assert controller.executed_calls == [(Action.LEFT_CLICK, None, None)]

    def test_runtime_does_not_fabricate_coordinates(self) -> None:
        # The only coordinates that ever reach ComputerController for
        # MOVE_CURSOR are exactly what CoordinateMapper.map_point()
        # returned -- the runtime introduces no numbers of its own.
        hand = _make_hand(index_tip=(0.9, 0.1, 0.0))
        sentinel_screen_point = ScreenPoint(999, 1)
        mapper = FakeCoordinateMapper(screen_point=sentinel_screen_point)
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)),
            gesture=Gesture.POINT,
            action=Action.MOVE_CURSOR,
            coordinate_mapper=mapper,
        )

        runtime.process_frame("frame")

        assert controller.executed_calls == [
            (Action.MOVE_CURSOR, sentinel_screen_point.x, sentinel_screen_point.y)
        ]

    def test_move_cursor_processing_is_deterministic(self) -> None:
        hand = _make_hand(index_tip=(0.25, 0.75, 0.0))
        mapper = FakeCoordinateMapper(screen_point=ScreenPoint(200, 300))
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)),
            gesture=Gesture.POINT,
            action=Action.MOVE_CURSOR,
            coordinate_mapper=mapper,
        )

        for _ in range(5):
            result = runtime.process_frame("frame")
            assert result.action == Action.MOVE_CURSOR
            assert result.action_executed is True

        assert controller.executed_calls == [(Action.MOVE_CURSOR, 200, 300)] * 5
        assert mapper.received_points == [(0.25, 0.75)] * 5


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

class TestDependencyInjection:
    def test_runtime_never_constructs_its_own_dependencies(self) -> None:
        # Constructing with fakes only, and successfully processing a
        # frame, itself demonstrates that GestureControlRuntime does
        # not require any real dependency to be constructible.
        hand = _make_hand()
        runtime, *_ = _build_runtime(HandTrackingResult(hands=(hand,)))
        result = runtime.process_frame("frame")
        assert isinstance(result, FrameResult)

    def test_two_runtimes_with_different_fakes_are_independent(self) -> None:
        hand = _make_hand()
        runtime_a, _, _, _, _, _, controller_a = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.FIST, action=Action.LEFT_CLICK
        )
        runtime_b, _, _, _, _, _, controller_b = _build_runtime(
            HandTrackingResult(), gesture=Gesture.FIST, action=Action.LEFT_CLICK
        )

        runtime_a.process_frame("frame-a")
        runtime_b.process_frame("frame-b")

        assert controller_a.executed_calls == [(Action.LEFT_CLICK, None, None)]
        assert controller_b.executed_calls == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_processing_of_the_same_frame_is_consistent(self) -> None:
        hand = _make_hand()
        runtime, *_, controller = _build_runtime(
            HandTrackingResult(hands=(hand,)), gesture=Gesture.FIST, action=Action.LEFT_CLICK
        )

        for _ in range(5):
            result = runtime.process_frame("frame")
            assert result.action == Action.LEFT_CLICK
            assert result.action_executed is True

        assert controller.executed_calls == [(Action.LEFT_CLICK, None, None)] * 5

    def test_repeated_no_hand_frames_are_consistent(self) -> None:
        runtime, *_, controller = _build_runtime(HandTrackingResult())
        for _ in range(5):
            result = runtime.process_frame("frame")
            assert result.hand_detected is False
        assert controller.executed_calls == []


# ---------------------------------------------------------------------------
# Import safety: no cv2 / mediapipe / pyautogui anywhere in this module
# ---------------------------------------------------------------------------

class TestNoForbiddenImports:
    def test_runtime_module_does_not_import_cv2_mediapipe_or_pyautogui(self) -> None:
        import gesture_control.runtime.runtime as runtime_module

        source_path = runtime_module.__file__
        with open(source_path) as f:
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

    def test_importing_the_runtime_package_never_raises(self) -> None:
        import gesture_control.runtime  # noqa: F401