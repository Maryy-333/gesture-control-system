"""Real-time runtime orchestration for a single frame.

Pipeline:

    frame
        -> HandTrackerProtocol.detect(frame)
        -> HandTrackingResult
        -> DetectedHand.landmarks
        -> FingerStateDetector.detect()
        -> GestureRecognizer.recognize()
        -> GestureSmoothener.smooth()
        -> ActionMapper.map()
        -> GestureActionGate.should_execute()
        -> CoordinateMapper.map_point() [MOVE_CURSOR only]
        -> ComputerController.execute()

The runtime is responsible for orchestration and cross-frame stabilization.
It does not contain gesture-recognition, action-mapping, or coordinate-
mapping logic of its own.

Two separate stabilization mechanisms are used:

1. Gesture smoothing:
   GestureSmoothener prevents short-lived recognition flicker from
   becoming a real gesture transition. This is particularly important
   for discrete actions such as clicks.

2. Cursor smoothing:
   MOVE_CURSOR uses an exponential moving average over the normalized
   index-finger coordinates. This reduces the small frame-to-frame
   landmark jitter produced by webcam/MediaPipe tracking while keeping
   cursor movement continuous.

GestureActionGate remains responsible for discrete-action debouncing
and pause/resume behavior.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..actions.action import Action
from ..actions.mapper import ActionMapper
from ..control.controller import ComputerController
from ..gestures.finger_states import FingerStateDetector, HandLandmark
from ..gestures.recognizer import Gesture, GestureRecognizer
from ..gestures.smoothener import GestureSmoothener
from ..mapping.coordinate_mapper import CoordinateMapper
from ..tracking.hand_tracker_protocol import HandTrackerProtocol
from ..tracking.hand_tracking_result import DetectedHand, HandTrackingResult
from .gesture_action_gate import GestureActionGate


@dataclass(frozen=True)
class FrameResult:
    """A transport-agnostic summary of what happened processing one frame."""

    hand_detected: bool
    tracking_result: HandTrackingResult
    selected_hand: Optional[DetectedHand]
    gesture: Optional[Gesture]
    action: Optional[Action]
    action_executed: bool


class GestureControlRuntime:
    """Orchestrates hand tracking, gesture stabilization, and control."""

    def __init__(
        self,
        hand_tracker: HandTrackerProtocol,
        finger_state_detector: FingerStateDetector,
        gesture_recognizer: GestureRecognizer,
        action_mapper: ActionMapper,
        computer_controller: ComputerController,
        coordinate_mapper: CoordinateMapper,
        gesture_action_gate: Optional[GestureActionGate] = None,
        gesture_smoothener: Optional[GestureSmoothener] = None,
        cursor_smoothing: float = 0.35,
    ) -> None:
        """Initialize the runtime.

        Args:
            hand_tracker:
                Detects hands in each frame.

            finger_state_detector:
                Computes finger states from hand landmarks.

            gesture_recognizer:
                Converts finger states/landmarks into a raw Gesture.

            action_mapper:
                Converts a Gesture into an Action.

            computer_controller:
                Executes actions through the configured control backend.

            coordinate_mapper:
                Converts normalized coordinates into screen coordinates.

            gesture_action_gate:
                Debounces discrete actions and handles pause/resume.

            gesture_smoothener:
                Stabilizes raw gestures across consecutive frames.
                If omitted, a default GestureSmoothener is created.

            cursor_smoothing:
                Exponential moving-average factor for cursor movement.

                Higher values:
                    More responsive, less smoothing.

                Lower values:
                    Smoother, but slightly more delayed.

                Recommended starting value:
                    0.35
        """
        if not 0.0 < cursor_smoothing <= 1.0:
            raise ValueError(
                "cursor_smoothing must be > 0 and <= 1, "
                f"got {cursor_smoothing!r}."
            )

        self._hand_tracker = hand_tracker
        self._finger_state_detector = finger_state_detector
        self._gesture_recognizer = gesture_recognizer
        self._action_mapper = action_mapper
        self._computer_controller = computer_controller
        self._coordinate_mapper = coordinate_mapper

        self._gesture_action_gate = (
            gesture_action_gate
            if gesture_action_gate is not None
            else GestureActionGate()
        )

        self._gesture_smoothener = (
            gesture_smoothener
            if gesture_smoothener is not None
            else GestureSmoothener()
        )

        self._cursor_smoothing = cursor_smoothing

        # Last smoothed normalized cursor position.
        #
        # None means that no cursor position has been established yet.
        # This is deliberately separate from GestureSmoothener state.
        self._smoothed_cursor: Optional[Tuple[float, float]] = None

    @property
    def is_paused(self) -> bool:
        """Whether the runtime is currently paused."""
        return self._gesture_action_gate.is_paused

    @property
    def smoothed_cursor(self) -> Optional[Tuple[float, float]]:
        """Return the current smoothed normalized cursor position."""
        return self._smoothed_cursor

    def _smooth_cursor(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """Apply exponential moving-average smoothing to cursor coordinates.

        The new position is calculated as:

            smoothed = previous + alpha * (current - previous)

        where `alpha` is `self._cursor_smoothing`.

        This keeps cursor movement continuous while filtering tiny
        frame-to-frame tracking fluctuations.
        """
        if self._smoothed_cursor is None:
            self._smoothed_cursor = (x, y)
            return self._smoothed_cursor

        previous_x, previous_y = self._smoothed_cursor
        alpha = self._cursor_smoothing

        smoothed_x = previous_x + alpha * (x - previous_x)
        smoothed_y = previous_y + alpha * (y - previous_y)

        self._smoothed_cursor = (smoothed_x, smoothed_y)

        return self._smoothed_cursor

    def _reset_tracking_state(self) -> None:
        """Reset state that depends on the currently tracked hand."""
        self._gesture_smoothener.reset()
        self._smoothed_cursor = None

    def process_frame(self, frame: Any) -> FrameResult:
        """Process one frame through the complete runtime pipeline."""

        tracking_result = self._hand_tracker.detect(frame)

        # No hand detected.
        #
        # Reset gesture smoothing and cursor smoothing because the previous
        # hand is no longer being tracked. GestureActionGate also receives
        # the None input so discrete actions are re-armed, while its paused
        # state intentionally remains unchanged.
        if not tracking_result.has_hands:
            self._reset_tracking_state()
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

        # ---------------------------------------------------------------
        # Gesture pipeline
        # ---------------------------------------------------------------

        finger_states = self._finger_state_detector.detect(
            selected_hand.landmarks
        )

        raw_gesture = self._gesture_recognizer.recognize(
            finger_states,
            selected_hand.landmarks,
        )

        # IMPORTANT:
        # The gate sees the STABLE gesture, not the noisy raw gesture.
        #
        # This prevents a one-frame recognition flicker such as:
        #
        #     POINT -> FIST -> POINT
        #
        # from becoming:
        #
        #     MOVE_CURSOR -> LEFT_CLICK -> MOVE_CURSOR
        #
        # and therefore prevents accidental clicks caused by recognition
        # noise.
        gesture = self._gesture_smoothener.smooth(raw_gesture)

        action = self._action_mapper.map(gesture)

        should_dispatch = self._gesture_action_gate.should_execute(
            gesture,
            action,
        )

        if not should_dispatch:
            return FrameResult(
                hand_detected=True,
                tracking_result=tracking_result,
                selected_hand=selected_hand,
                gesture=gesture,
                action=action,
                action_executed=False,
            )

        # ---------------------------------------------------------------
        # Cursor movement
        # ---------------------------------------------------------------

        if action == Action.MOVE_CURSOR:
            index_tip_x, index_tip_y, _ = selected_hand.landmarks[
                HandLandmark.INDEX_TIP
            ]

            smoothed_x, smoothed_y = self._smooth_cursor(
                index_tip_x,
                index_tip_y,
            )

            screen_point = self._coordinate_mapper.map_point(
                smoothed_x,
                smoothed_y,
            )

            self._computer_controller.execute(
                Action.MOVE_CURSOR,
                x=screen_point.x,
                y=screen_point.y,
            )

            return FrameResult(
                hand_detected=True,
                tracking_result=tracking_result,
                selected_hand=selected_hand,
                gesture=gesture,
                action=action,
                action_executed=True,
            )

        # ---------------------------------------------------------------
        # All other actions
        # ---------------------------------------------------------------

        self._computer_controller.execute(action)

        return FrameResult(
            hand_detected=True,
            tracking_result=tracking_result,
            selected_hand=selected_hand,
            gesture=gesture,
            action=action,
            action_executed=True,
        )

    def reset(self) -> None:
        """Reset all runtime stabilization and gate state."""
        self._gesture_smoothener.reset()
        self._gesture_action_gate.reset()
        self._smoothed_cursor = None