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
        -> GestureRecognizer.recognize(finger_states)
        -> ActionMapper.map(gesture)
        -> [Action.MOVE_CURSOR only] landmarks[HandLandmark.INDEX_TIP]
           -> CoordinateMapper.map_point(x, y) -> ScreenPoint
        -> ComputerController.execute(action, x=..., y=...)

Every dependency is injected via the constructor; `GestureControlRuntime`
never constructs a real hand tracker, detector, recognizer, mapper,
controller, or coordinate mapper itself. This is what makes it fully
testable with fakes and free of any real webcam/MediaPipe/OpenCV/
PyAutoGUI/OS involvement.

The runtime is stateless between frames: `process_frame()` never reads
or writes any state beyond its injected dependencies, so repeated calls
are deterministic and independent of call order or history. Gesture
debouncing, cooldowns, smoothing, and any other cross-frame behavior
are explicitly out of scope here.
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
    it is always `False`; for a detected hand it reflects whether
    `ComputerController.execute()` was actually called (see
    `GestureControlRuntime.process_frame` for `Action.MOVE_CURSOR`
    details).
    """

    hand_detected: bool
    tracking_result: HandTrackingResult
    selected_hand: Optional[DetectedHand]
    gesture: Optional[Gesture]
    action: Optional[Action]
    action_executed: bool


class GestureControlRuntime:
    """Orchestrates one frame through hand tracking, gestures, and control.

    All six dependencies are injected; none are constructed here. This
    lets tests exercise the full frame -> action flow using simple fake
    objects, with no real webcam, MediaPipe, OpenCV, or PyAutoGUI
    involved anywhere.

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
    ) -> None:
        """Initialize the runtime with its (already-constructed) dependencies.

        Args:
            hand_tracker: Detects hands in a frame. Any object
                satisfying `HandTrackerProtocol` (e.g.
                `MediaPipeHandTracker`, or a fake for tests).
            finger_state_detector: Computes per-finger extension state
                from a detected hand's landmarks.
            gesture_recognizer: Classifies finger states into a
                `Gesture`.
            action_mapper: Maps a `Gesture` to an abstract `Action`.
            computer_controller: Executes an `Action` via its injected
                `ControlBackend`.
            coordinate_mapper: Converts the normalized `INDEX_TIP`
                landmark into a `ScreenPoint` for `Action.MOVE_CURSOR`.
                Not consulted for any other action.
        """
        self._hand_tracker = hand_tracker
        self._finger_state_detector = finger_state_detector
        self._gesture_recognizer = gesture_recognizer
        self._action_mapper = action_mapper
        self._computer_controller = computer_controller
        self._coordinate_mapper = coordinate_mapper

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
            `CoordinateMapper` produce. Every other action is executed
            with no coordinates, as before.

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
        gesture = self._gesture_recognizer.recognize(finger_states)
        action = self._action_mapper.map(gesture)

        if action == Action.MOVE_CURSOR:
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