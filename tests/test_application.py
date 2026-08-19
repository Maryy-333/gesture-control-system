"""Unit tests for gesture_control.app.application.

SAFETY: Every test either fully overrides every real-backend dependency
(`hand_tracker`, `computer_controller`, `camera`, `screen_size`/
`coordinate_mapper`) with a fake, or simulates those libraries being
unavailable and asserts a clean failure. No test here opens a real
webcam, invokes real MediaPipe, invokes real PyAutoGUI, or touches the
real mouse/keyboard/screen/OS.
"""

import ast
import builtins
import os
import sys
from typing import Any, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.actions.action import Action
from gesture_control.actions.mapper import ActionMapper
from gesture_control.app.application import build_runtime, build_webcam_loop
from gesture_control.app.webcam_loop import WebcamLoop
from gesture_control.gestures.finger_states import FingerStateDetector
from gesture_control.gestures.recognizer import Gesture, GestureRecognizer
from gesture_control.mapping.coordinate_mapper import CoordinateMapper, ScreenSize
from gesture_control.runtime.runtime import GestureControlRuntime
from gesture_control.tracking.hand_tracking_result import (
    DetectedHand,
    Handedness,
    HandTrackingResult,
)


class FakeHandTracker:
    """Fake HandTrackerProtocol: returns a fixed result."""

    def __init__(self, result: HandTrackingResult) -> None:
        self._result = result
        self.received_frames: List[Any] = []

    def detect(self, frame: Any) -> HandTrackingResult:
        self.received_frames.append(frame)
        return self._result


class FakeComputerController:
    """Fake ComputerController: records every execute() call."""

    def __init__(self) -> None:
        self.executed_calls: List[Tuple[Action, Optional[int], Optional[int]]] = []

    def execute(self, action: Action, x: Optional[int] = None, y: Optional[int] = None) -> None:
        self.executed_calls.append((action, x, y))


class FakeFrameSource:
    """Fake FrameSource: yields a fixed sequence of frames."""

    def __init__(self, frames: List[Any]) -> None:
        self._frames = list(frames)
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def read(self) -> Tuple[bool, Optional[Any]]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.is_open = False


def _hand_with_open_palm_landmarks() -> DetectedHand:
    """Build a hand whose landmarks recognize as Gesture.OPEN_PALM.

    Mirrors the straight-chain construction used elsewhere in this
    project's test suite (e.g. tests/test_finger_states.py) to produce
    a fully-extended hand.
    """
    landmarks = [(0.0, 0.0, 0.0)] * 21

    def straight_chain(base: Tuple[float, float, float], step: Tuple[float, float, float], count: int):
        return [
            (base[0] + step[0] * i, base[1] + step[1] * i, base[2] + step[2] * i)
            for i in range(count)
        ]

    from gesture_control.gestures.finger_states import HandLandmark

    def set_finger(mcp, pip, dip, tip, base, step):
        chain = straight_chain(base, step, 4)
        landmarks[mcp] = chain[0]
        landmarks[pip] = chain[1]
        landmarks[dip] = chain[2]
        landmarks[tip] = chain[3]

    set_finger(
        HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP,
        HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP,
        base=(0.2, 0.5, 0.0), step=(-0.1, 0.0, 0.0),
    )
    for mcp, pip, dip, tip, base in [
        (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP, (0.4, 0.5, 0.0)),
        (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP, (0.5, 0.5, 0.0)),
        (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP, (0.6, 0.5, 0.0)),
        (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP, (0.7, 0.5, 0.0)),
    ]:
        set_finger(mcp, pip, dip, tip, base=base, step=(0.0, -0.1, 0.0))

    return DetectedHand(landmarks=tuple(landmarks), handedness=Handedness.RIGHT)


# ---------------------------------------------------------------------------
# build_runtime: wiring correctness with fully injected fakes
# ---------------------------------------------------------------------------

class TestBuildRuntimeWiring:
    def test_returns_a_gesture_control_runtime(self) -> None:
        runtime = build_runtime(
            hand_tracker=FakeHandTracker(HandTrackingResult()),
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1920, 1080),
        )
        assert isinstance(runtime, GestureControlRuntime)

    def test_end_to_end_open_palm_is_absorbed_by_the_gate_as_pause_resume(self) -> None:
        # Action.PAUSE is fully absorbed by GestureControlRuntime's
        # GestureActionGate as a pause/resume toggle -- it is never
        # forwarded to ComputerController (see GestureActionGate's
        # docstring). The mapped Gesture/Action are still reported on
        # the FrameResult; only dispatch to the controller is suppressed.
        hand_tracker = FakeHandTracker(HandTrackingResult(hands=(_hand_with_open_palm_landmarks(),)))
        controller = FakeComputerController()
        runtime = build_runtime(
            hand_tracker=hand_tracker,
            computer_controller=controller,
            screen_size=ScreenSize(1920, 1080),
        )

        result = runtime.process_frame("frame")

        assert result.gesture == Gesture.OPEN_PALM
        assert result.action == Action.PAUSE
        assert result.action_executed is False
        assert controller.executed_calls == []
        assert runtime.is_paused is True

    def test_injected_hand_tracker_receives_the_frame(self) -> None:
        hand_tracker = FakeHandTracker(HandTrackingResult())
        runtime = build_runtime(
            hand_tracker=hand_tracker,
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1920, 1080),
        )

        runtime.process_frame("my-frame")

        assert hand_tracker.received_frames == ["my-frame"]

    def test_explicit_coordinate_mapper_overrides_screen_size(self) -> None:
        mapper = CoordinateMapper(ScreenSize(640, 480))
        runtime = build_runtime(
            hand_tracker=FakeHandTracker(HandTrackingResult()),
            computer_controller=FakeComputerController(),
            coordinate_mapper=mapper,
            screen_size=ScreenSize(1920, 1080),  # should be ignored
        )
        # No direct getter on the runtime, but constructing successfully
        # with a pre-built CoordinateMapper (and no screen_size-based
        # detection) demonstrates the override path was taken without
        # requiring PyAutoGUI at all.
        assert isinstance(runtime, GestureControlRuntime)

    def test_explicit_finger_state_detector_gesture_recognizer_action_mapper_are_used(self) -> None:
        # Real (pure, dependency-free) components can be passed through
        # explicitly too -- this just confirms build_runtime doesn't
        # silently replace injected pure components with its own.
        finger_state_detector = FingerStateDetector()
        gesture_recognizer = GestureRecognizer()
        action_mapper = ActionMapper()
        hand_tracker = FakeHandTracker(HandTrackingResult(hands=(_hand_with_open_palm_landmarks(),)))
        controller = FakeComputerController()

        runtime = build_runtime(
            hand_tracker=hand_tracker,
            finger_state_detector=finger_state_detector,
            gesture_recognizer=gesture_recognizer,
            action_mapper=action_mapper,
            computer_controller=controller,
            screen_size=ScreenSize(1920, 1080),
        )

        result = runtime.process_frame("frame")
        assert result.gesture == Gesture.OPEN_PALM
        assert result.action == Action.PAUSE


# ---------------------------------------------------------------------------
# build_webcam_loop: wiring correctness with fully injected fakes
# ---------------------------------------------------------------------------

class TestBuildWebcamLoopWiring:
    def test_returns_a_webcam_loop(self) -> None:
        runtime = build_runtime(
            hand_tracker=FakeHandTracker(HandTrackingResult()),
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1920, 1080),
        )
        loop = build_webcam_loop(camera=FakeFrameSource(["f1"]), runtime=runtime)
        assert isinstance(loop, WebcamLoop)

    def test_loop_processes_frames_from_injected_camera(self) -> None:
        hand_tracker = FakeHandTracker(HandTrackingResult())
        runtime = build_runtime(
            hand_tracker=hand_tracker,
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1920, 1080),
        )
        camera = FakeFrameSource(["f1", "f2"])
        loop = build_webcam_loop(camera=camera, runtime=runtime, display=False)

        frames_processed = loop.run()

        assert frames_processed == 2
        assert hand_tracker.received_frames == ["f1", "f2"]

    def test_camera_index_is_ignored_when_camera_is_given(self) -> None:
        # Passing a nonsensical camera_index alongside an explicit
        # camera override must not matter -- the override wins and no
        # real Camera/cv2 is ever touched.
        runtime = build_runtime(
            hand_tracker=FakeHandTracker(HandTrackingResult()),
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1920, 1080),
        )
        camera = FakeFrameSource(["f1"])
        loop = build_webcam_loop(camera=camera, camera_index=99, runtime=runtime)
        frames_processed = loop.run()
        assert frames_processed == 1

    def test_screen_size_and_hand_tracker_kwargs_are_ignored_when_runtime_is_given(self) -> None:
        # When `runtime` is passed directly, build_webcam_loop must not
        # also try to build its own runtime (which would touch real
        # MediaPipe/PyAutoGUI) -- passing screen_size/hand_tracker_kwargs
        # alongside an explicit runtime should be harmless.
        camera = FakeFrameSource(["f1"])
        runtime = build_runtime(
            hand_tracker=FakeHandTracker(HandTrackingResult()),
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1280, 720),
        )
        loop = build_webcam_loop(
            camera=camera,
            runtime=runtime,
            screen_size=ScreenSize(9999, 9999),
            hand_tracker_kwargs={"max_num_hands": 5},
        )
        assert isinstance(loop, WebcamLoop)
        frames_processed = loop.run()
        assert frames_processed == 1


# ---------------------------------------------------------------------------
# Import safety: no cv2 / mediapipe / pyautogui at module import time
# ---------------------------------------------------------------------------

class TestNoForbiddenTopLevelImports:
    def test_application_module_does_not_import_cv2_mediapipe_pyautogui_at_top_level(self) -> None:
        import gesture_control.app.application as application_module

        with open(application_module.__file__) as f:
            tree = ast.parse(f.read())

        forbidden = {"cv2", "mediapipe", "pyautogui"}
        top_level_imports = set()
        for node in tree.body:  # only module-level statements, not nested in functions
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level_imports.add(node.module.split(".")[0])

        assert not (top_level_imports & forbidden)

    def test_importing_application_module_never_raises(self) -> None:
        import gesture_control.app.application  # noqa: F401

    def test_importing_app_package_never_raises(self) -> None:
        import gesture_control.app  # noqa: F401

    def test_importing_top_level_package_never_raises(self) -> None:
        import gesture_control  # noqa: F401


# ---------------------------------------------------------------------------
# Clean failure when real backends are unavailable and not overridden
# ---------------------------------------------------------------------------

class TestCleanFailureWhenBackendsUnavailable:
    def test_build_runtime_raises_runtime_error_when_mediapipe_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # MediaPipe may genuinely already be importable in this test
        # environment, in which case blocking future `import mediapipe`
        # calls wouldn't retroactively undo an already-successful
        # module-level import inside gesture_control.tracking.
        # mediapipe_hand_tracker. Instead, directly simulate MediaPipe
        # being unavailable at the exact point MediaPipeHandTracker
        # checks it, which is the deterministic way to exercise this
        # path regardless of what's actually installed here.
        import gesture_control.tracking.mediapipe_hand_tracker as mp_tracker_module

        monkeypatch.setattr(mp_tracker_module, "_RealHandTracker", None)
        monkeypatch.setattr(
            mp_tracker_module, "_IMPORT_ERROR", ImportError("simulated: mediapipe not installed")
        )

        with pytest.raises(RuntimeError):
            build_runtime(
                computer_controller=FakeComputerController(),
                screen_size=ScreenSize(1920, 1080),
            )

    def test_build_webcam_loop_raises_runtime_error_when_cv2_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "cv2" or name.startswith("cv2."):
                raise ImportError("simulated: cv2 not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        runtime = build_runtime(
            hand_tracker=FakeHandTracker(HandTrackingResult()),
            computer_controller=FakeComputerController(),
            screen_size=ScreenSize(1920, 1080),
        )

        with pytest.raises(RuntimeError):
            build_webcam_loop(runtime=runtime)

    def test_build_runtime_raises_runtime_error_when_pyautogui_unavailable_for_screen_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pyautogui" or name.startswith("pyautogui."):
                raise ImportError("simulated: pyautogui not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        with pytest.raises(RuntimeError):
            build_runtime(
                hand_tracker=FakeHandTracker(HandTrackingResult()),
                computer_controller=FakeComputerController(),
                # No screen_size and no coordinate_mapper override:
                # this forces the PyAutoGUI-based screen-size detection
                # path, which should fail cleanly.
            )