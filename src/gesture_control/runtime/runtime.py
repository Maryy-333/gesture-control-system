"""Real-time runtime orchestration for a single frame.

This module contains no MediaPipe, OpenCV, or PyAutoGUI code, no
webcam access, no landmark geometry, no finger-state logic, no
gesture-recognition logic, and no gesture-to-action mapping logic. It
only orchestrates the existing components, in order, for one frame:

    frame
        -> HandTrackerProtocol.detect(frame)
        -> HandTrackingResult
        -> DetectedHand.landmarks (first hand, if any)
        -> FingerStateDetector.detect(landmarks)
        -> GestureRecognizer.recognize(finger_states)
        -> ActionMapper.map(gesture)
        -> ComputerController.execute(action)

Every dependency is injected via the constructor; `GestureControlRuntime`
never constructs a real hand tracker, detector, recognizer, mapper, or
controller itself. This is what makes it fully testable with fakes and
free of any real webcam/MediaPipe/OpenCV/PyAutoGUI/OS involvement.

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
from ..gestures.finger_states import FingerStateDetector
from ..gestures.recognizer import Gesture, GestureRecognizer
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
    controller was actually told to perform it": see the
    `GestureControlRuntime.process_frame` docstring for why
    `Action.MOVE_CURSOR` is a case where `action` can be set while
    `action_executed` is `False`.
    """

    hand_detected: bool
    tracking_result: HandTrackingResult
    selected_hand: Optional[DetectedHand]
    gesture: Optional[Gesture]
    action: Optional[Action]
    action_executed: bool


class GestureControlRuntime:
    """Orchestrates one frame through hand tracking, gestures, and control.

    All five dependencies are injected; none are constructed here. This
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
        """
        self._hand_tracker = hand_tracker
        self._finger_state_detector = finger_state_detector
        self._gesture_recognizer = gesture_recognizer
        self._action_mapper = action_mapper
        self._computer_controller = computer_controller

    def process_frame(self, frame: Any) -> FrameResult:
        """Run one frame through the full detect -> recognize -> act flow.

        MOVE_CURSOR handling:
            `ActionMapper` currently maps `Gesture.POINT` to
            `Action.MOVE_CURSOR`, but nothing in the current
            architecture supplies a screen-coordinate mapping for it
            (`DetectedHand.landmarks` are normalized hand-tracking
            coordinates, not screen pixels, and `ActionMapper.map()`
            takes no coordinate input). Rather than fabricate
            coordinates -- e.g. by reinterpreting a landmark as a
            screen position -- this method does NOT call
            `ComputerController.execute()` when the mapped action is
            `Action.MOVE_CURSOR`. The mapped action is still reported
            via `FrameResult.action` (so callers/tests can observe that
            recognition and mapping happened correctly);
            `FrameResult.action_executed` is `False` in this case to
            make the skip explicit. Every other action is executed
            normally. This will need revisiting once a coordinate
            mapping is introduced in a later milestone.

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
            # No coordinate mapping is available yet -- see the
            # "MOVE_CURSOR handling" note above. Do not call the
            # controller with fabricated coordinates.
            action_executed = False
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