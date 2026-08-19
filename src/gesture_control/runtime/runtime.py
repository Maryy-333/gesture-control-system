"""Real-time runtime orchestration for a single frame.

This module contains no MediaPipe, OpenCV, or PyAutoGUI code, no
webcam access, no landmark geometry, no finger-state logic, no
gesture-recognition logic, no gesture-to-action mapping logic, and no
normalized-to-screen coordinate conversion logic (that belongs to
`CoordinateMapper`). It only orchestrates the existing components, in
order, for one frame:

    frame
        -> HandTrackerProtocol.detect(frame)
        -> HandTrackingResult
        -> DetectedHand.landmarks (first hand, if any)
        -> FingerStateDetector.detect(landmarks)
        -> GestureRecognizer.recognize(finger_states, landmarks)
        -> ActionMapper.map(gesture)
        -> GestureActionGate.should_execute(gesture, action)
        -> [Action.MOVE_CURSOR only] landmarks[HandLandmark.INDEX_TIP]
           -> CoordinateMapper.map_point(x, y) -> ScreenPoint
        -> ComputerController.execute(action, x=..., y=...)

Every dependency is injected via the constructor; `GestureControlRuntime`
never constructs a real hand tracker, detector, recognizer, mapper,
controller, or coordinate mapper itself. This is what makes it fully
testable with fakes and free of any real webcam/MediaPipe/OpenCV/
PyAutoGUI/OS involvement.

Landmarks are passed to `GestureRecognizer.recognize()` in addition to
`FingerStates` so it can distinguish `Gesture.THUMBS_UP` from
`Gesture.THUMBS_DOWN` using real thumb-direction geometry -- see
`GestureRecognizer`'s own docstring for why `FingerStates` alone cannot
do this.

Cross-frame state:
    Unlike earlier in this project's history, `GestureControlRuntime`
    is now intentionally *not* fully stateless between frames: it owns
    a `GestureActionGate` (default-constructed if none is injected,
    mirroring how `ComputerController` defaults its own backend) that
    tracks the most recently recognized gesture and a paused flag
    across calls to `process_frame()`. This is what prevents a held
    gesture from firing a discrete action (e.g. a click) on every
    single frame, and what makes `Gesture.OPEN_PALM` -> `Action.PAUSE`
    a genuine pause/resume toggle rather than a no-op. `GestureRecognizer`
    and `ActionMapper` themselves remain pure and stateless -- this
    state lives here, one layer above them, exactly where it belongs.
    Gesture *smoothing* (filtering noisy per-frame flicker before a
    gesture is even reported) is a different concern and remains out
    of scope.
"""

from dataclasses import dataclass
from typing import Any, Optional


from ..actions.action import Action
from ..actions.mapper import ActionMapper
from ..control.controller import ComputerController
from ..gestures.finger_states import FingerStateDetector, HandLandmark
from ..gestures.recognizer import Gesture, GestureRecognizer
from ..mapping.coordinate_mapper import CoordinateMapper
from ..tracking.hand_tracker_protocol import HandTrackerProtocol
from ..tracking.hand_tracking_result import DetectedHand, HandTrackingResult
from .gesture_action_gate import GestureActionGate


@dataclass(frozen=True)
class FrameResult:
    """A transport-agnostic summary of what happened processing one frame.

    `tracking_result` is always the full `HandTrackingResult` for the
    frame (even when no hands were found), so a caller can inspect
    every detected hand -- not just the one that was processed --
    without this class duplicating that representation.

    For the no-hand case, `selected_hand`, `gesture`, and `action` are
    all `None` rather than some invented/default value: no gesture was
    recognized (not even `Gesture.UNKNOWN`, which means "recognition
    ran and found no match" -- a different thing from "recognition
    never ran"), and no action was mapped.

    `action_executed` distinguishes "an action was mapped" from "the
    controller was actually told to perform it". For the no-hand case
    it is always `False`. For a detected hand, it is `False` whenever
    `GestureActionGate` suppresses dispatch this frame -- because the
    same discrete gesture is still being held (debounce), because
    `action` is `Action.PAUSE` (which is absorbed by the gate as a
    pause/resume toggle and never forwarded), or because the gate is
    currently paused -- and `True` otherwise, reflecting that
    `ComputerController.execute()` was actually called (see
    `GestureControlRuntime.process_frame` for `Action.MOVE_CURSOR`
    coordinate details).
    """

    hand_detected: bool
    tracking_result: HandTrackingResult
    selected_hand: Optional[DetectedHand]
    gesture: Optional[Gesture]
    action: Optional[Action]
    action_executed: bool


class GestureControlRuntime:
    """Orchestrates one frame through hand tracking, gestures, and control.

    All six core dependencies are injected; none are constructed here.
    This lets tests exercise the full frame -> action flow using simple
    fake objects, with no real webcam, MediaPipe, OpenCV, or PyAutoGUI
    involved anywhere. The seventh, `gesture_action_gate`, is optional
    and defaults to a real `GestureActionGate()` if omitted -- the same
    pattern `ComputerController` uses for its default `NoOpControlBackend`.

    Hand selection is deliberately simple for this milestone: if one or
    more hands are detected, only the first one
    (`tracking_result.hands[0]`) is processed. Left/right preference,
    multi-hand gestures, and tracking a hand's identity across frames
    are all out of scope here.

    Example:
        runtime = GestureControlRuntime(
            hand_tracker=hand_tracker,
            finger_state_detector=FingerStateDetector(),
            gesture_recognizer=GestureRecognizer(),
            action_mapper=ActionMapper(),
            computer_controller=ComputerController(),
            coordinate_mapper=CoordinateMapper(ScreenSize(1920, 1080)),
        )
        result = runtime.process_frame(frame)
    """

    def __init__(
        self,
        hand_tracker: HandTrackerProtocol,
        finger_state_detector: FingerStateDetector,
        gesture_recognizer: GestureRecognizer,
        action_mapper: ActionMapper,
        computer_controller: ComputerController,
        coordinate_mapper: CoordinateMapper,
        gesture_action_gate: Optional[GestureActionGate] = None,
    ) -> None:
        """Initialize the runtime with its (already-constructed) dependencies.

        Args:
            hand_tracker: Detects hands in a frame. Any object
                satisfying `HandTrackerProtocol` (e.g.
                `MediaPipeHandTracker`, or a fake for tests).
            finger_state_detector: Computes per-finger extension state
                from a detected hand's landmarks.
            gesture_recognizer: Classifies finger states (plus
                landmarks, for THUMBS_UP/THUMBS_DOWN) into a `Gesture`.
            action_mapper: Maps a `Gesture` to an abstract `Action`.
            computer_controller: Executes an `Action` via its injected
                `ControlBackend`.
            coordinate_mapper: Converts the normalized `INDEX_TIP`
                landmark into a `ScreenPoint` for `Action.MOVE_CURSOR`.
                Not consulted for any other action.
            gesture_action_gate: Decides whether a mapped action should
                actually be dispatched this frame (debouncing discrete
                actions, and handling the `Action.PAUSE` toggle). If
                omitted, defaults to a fresh `GestureActionGate()`.
        """
        self._hand_tracker = hand_tracker
        self._finger_state_detector = finger_state_detector
        self._gesture_recognizer = gesture_recognizer
        self._action_mapper = action_mapper
        self._computer_controller = computer_controller
        self._coordinate_mapper = coordinate_mapper
        self._gesture_action_gate = (
            gesture_action_gate if gesture_action_gate is not None else GestureActionGate()
        )

    @property
    def is_paused(self) -> bool:
        """Whether the runtime's `GestureActionGate` is currently paused."""
        return self._gesture_action_gate.is_paused

    def process_frame(self, frame: Any) -> FrameResult:
        """Run one frame through the full detect -> recognize -> act flow.

        MOVE_CURSOR handling:
            `ActionMapper` maps `Gesture.POINT` to `Action.MOVE_CURSOR`.
            `ActionMapper.map()` itself takes no coordinate input and
            produces none, so this method is what supplies them: it
            reads `HandLandmark.INDEX_TIP` from the selected hand's
            landmarks (the tip of the pointing finger -- no other
            landmark is used for this), passes its normalized `(x, y)`
            straight through to `CoordinateMapper.map_point()` with no
            interpretation of its own, and forwards the resulting
            `ScreenPoint` to `ComputerController.execute()`. No
            coordinates are fabricated: the only numbers used are
            exactly what `DetectedHand.landmarks` and
            `CoordinateMapper` produce.

        Debouncing and pause (`GestureActionGate`):
            Before dispatching anything, the mapped `(gesture, action)`
            pair is passed to `self._gesture_action_gate.should_execute()`.
            If it returns `False`, `ComputerController.execute()` is
            not called at all this frame -- this is what stops a held
            FIST from firing dozens of `LEFT_CLICK`s per second, and
            what makes `Action.PAUSE` a real pause/resume toggle rather
            than a no-op (see `GestureActionGate`'s docstring). This
            check applies to every action, including `Action.MOVE_CURSOR`.

        Args:
            frame: A single frame, passed through unchanged to the
                injected `HandTrackerProtocol.detect()`.

        Returns:
            A `FrameResult` describing what happened. If no hand was
            detected, no gesture is recognized, no action is mapped,
            and `computer_controller.execute()` is not called.

        Raises:
            Whatever the injected dependencies themselves raise. This
            method does not catch or suppress exceptions from its
            dependencies.
        """
        tracking_result = self._hand_tracker.detect(frame)

        if not tracking_result.has_hands:
            self._gesture_action_gate.should_execute(None, None)
            return FrameResult(
                hand_detected=False,
                tracking_result=tracking_result,
                selected_hand=None,
                gesture=None,
                action=None,
                action_executed=False,
            )

        selected_hand = tracking_result.hands[0]
        finger_states = self._finger_state_detector.detect(selected_hand.landmarks)
        gesture = self._gesture_recognizer.recognize(finger_states, selected_hand.landmarks)
        action = self._action_mapper.map(gesture)

        should_dispatch = self._gesture_action_gate.should_execute(gesture, action)

        if not should_dispatch:
            action_executed = False
        elif action == Action.MOVE_CURSOR:
            # See the "MOVE_CURSOR handling" note above: INDEX_TIP is
            # the only landmark used, and its coordinates are passed
            # through CoordinateMapper unmodified.
            index_tip_x, index_tip_y, _ = selected_hand.landmarks[HandLandmark.INDEX_TIP]
            screen_point = self._coordinate_mapper.map_point(index_tip_x, index_tip_y)
            self._computer_controller.execute(
                Action.MOVE_CURSOR, x=screen_point.x, y=screen_point.y
            )
            action_executed = True
        else:
            self._computer_controller.execute(action)
            action_executed = True

        return FrameResult(
            hand_detected=True,
            tracking_result=tracking_result,
            selected_hand=selected_hand,
            gesture=gesture,
            action=action,
            action_executed=action_executed,
        )