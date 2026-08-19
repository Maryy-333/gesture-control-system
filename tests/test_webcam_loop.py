"""Unit tests for gesture_control.app.webcam_loop.

SAFETY: Every test injects a fake `FrameSource` and either a real
`GestureControlRuntime` wired entirely with fakes, or a fake runtime
directly. `display` defaults to False in every test that doesn't
specifically exercise the display path (and those still use a stubbed
`cv2`, never the real module), so no test here opens a real webcam,
invokes real MediaPipe, invokes real OpenCV GUI, invokes real
PyAutoGUI, or touches the real mouse/keyboard/screen/OS.
"""

import ast
import os
import sys
from typing import Any, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.actions.action import Action
from gesture_control.app.webcam_loop import FrameSource, WebcamLoop
from gesture_control.gestures.recognizer import Gesture
from gesture_control.runtime.runtime import FrameResult
from gesture_control.tracking.hand_tracking_result import HandTrackingResult


class FakeFrameSource:
    """Fake FrameSource: yields a fixed sequence of frames, then stops.

    Mirrors `Camera`'s idempotent open()/release() contract so it's a
    faithful stand-in for tests.
    """

    def __init__(self, frames: List[Any]) -> None:
        self._frames = list(frames)
        self.is_open = False
        self.open_calls = 0
        self.release_calls = 0
        self.read_calls = 0

    def open(self) -> None:
        self.open_calls += 1
        self.is_open = True

    def read(self) -> Tuple[bool, Optional[Any]]:
        self.read_calls += 1
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.release_calls += 1
        self.is_open = False


class FakeRuntime:
    """Fake GestureControlRuntime: records frames, returns a fixed result per call."""

    def __init__(self, result: Optional[FrameResult] = None) -> None:
        self._result = result if result is not None else _default_frame_result()
        self.processed_frames: List[Any] = []

    def process_frame(self, frame: Any) -> FrameResult:
        self.processed_frames.append(frame)
        return self._result


def _default_frame_result() -> FrameResult:
    return FrameResult(
        hand_detected=False,
        tracking_result=HandTrackingResult(),
        selected_hand=None,
        gesture=None,
        action=None,
        action_executed=False,
    )


# ---------------------------------------------------------------------------
# Basic loop mechanics
# ---------------------------------------------------------------------------

class TestBasicLoopMechanics:
    def test_processes_every_available_frame(self) -> None:
        frame_source = FakeFrameSource(["f1", "f2", "f3"])
        runtime = FakeRuntime()
        loop = WebcamLoop(frame_source, runtime)

        frames_processed = loop.run()

        assert frames_processed == 3
        assert runtime.processed_frames == ["f1", "f2", "f3"]

    def test_stops_when_frame_source_is_exhausted(self) -> None:
        frame_source = FakeFrameSource([])
        runtime = FakeRuntime()
        loop = WebcamLoop(frame_source, runtime)

        frames_processed = loop.run()

        assert frames_processed == 0
        assert runtime.processed_frames == []

    def test_frames_are_passed_to_runtime_unchanged(self) -> None:
        sentinel_frame = object()
        frame_source = FakeFrameSource([sentinel_frame])
        runtime = FakeRuntime()
        loop = WebcamLoop(frame_source, runtime)

        loop.run()

        assert runtime.processed_frames == [sentinel_frame]


# ---------------------------------------------------------------------------
# Resource lifecycle
# ---------------------------------------------------------------------------

class TestResourceLifecycle:
    def test_frame_source_is_opened_and_released(self) -> None:
        frame_source = FakeFrameSource(["f1"])
        loop = WebcamLoop(frame_source, FakeRuntime())

        loop.run()

        assert frame_source.open_calls == 1
        assert frame_source.release_calls == 1
        assert frame_source.is_open is False

    def test_frame_source_is_released_even_if_runtime_raises(self) -> None:
        class ExplodingRuntime:
            def process_frame(self, frame: Any) -> FrameResult:
                raise ValueError("boom")

        frame_source = FakeFrameSource(["f1"])
        loop = WebcamLoop(frame_source, ExplodingRuntime())

        with pytest.raises(ValueError):
            loop.run()

        assert frame_source.release_calls == 1

    def test_exception_from_runtime_propagates_uncaught(self) -> None:
        class ExplodingRuntime:
            def process_frame(self, frame: Any) -> FrameResult:
                raise RuntimeError("specific failure")

        frame_source = FakeFrameSource(["f1"])
        loop = WebcamLoop(frame_source, ExplodingRuntime())

        with pytest.raises(RuntimeError, match="specific failure"):
            loop.run()


# ---------------------------------------------------------------------------
# max_frames
# ---------------------------------------------------------------------------

class TestMaxFrames:
    def test_stops_after_max_frames_even_if_more_are_available(self) -> None:
        frame_source = FakeFrameSource(["f1", "f2", "f3", "f4", "f5"])
        runtime = FakeRuntime()
        loop = WebcamLoop(frame_source, runtime, max_frames=2)

        frames_processed = loop.run()

        assert frames_processed == 2
        assert runtime.processed_frames == ["f1", "f2"]

    def test_max_frames_larger_than_available_frames_is_harmless(self) -> None:
        frame_source = FakeFrameSource(["f1", "f2"])
        loop = WebcamLoop(frame_source, FakeRuntime(), max_frames=100)

        frames_processed = loop.run()

        assert frames_processed == 2

    def test_max_frames_zero_processes_nothing(self) -> None:
        frame_source = FakeFrameSource(["f1", "f2"])
        loop = WebcamLoop(frame_source, FakeRuntime(), max_frames=0)

        frames_processed = loop.run()

        assert frames_processed == 0
        assert frame_source.release_calls == 1  # still cleaned up


# ---------------------------------------------------------------------------
# on_frame_processed callback
# ---------------------------------------------------------------------------

class TestOnFrameProcessedCallback:
    def test_callback_receives_each_frame_result(self) -> None:
        result_a = FrameResult(
            hand_detected=True,
            tracking_result=HandTrackingResult(),
            selected_hand=None,
            gesture=Gesture.FIST,
            action=Action.LEFT_CLICK,
            action_executed=True,
        )
        frame_source = FakeFrameSource(["f1", "f2"])
        runtime = FakeRuntime(result=result_a)
        received: List[FrameResult] = []

        loop = WebcamLoop(frame_source, runtime, on_frame_processed=received.append)
        loop.run()

        assert received == [result_a, result_a]

    def test_no_callback_is_fine(self) -> None:
        frame_source = FakeFrameSource(["f1"])
        loop = WebcamLoop(frame_source, FakeRuntime(), on_frame_processed=None)
        frames_processed = loop.run()
        assert frames_processed == 1


# ---------------------------------------------------------------------------
# Display (stubbed cv2 -- never the real module)
# ---------------------------------------------------------------------------

class TestDisplay:
    def test_display_false_never_imports_cv2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Make any attempt to import cv2 fail loudly, then prove the
        # loop runs fine anyway when display=False.
        import builtins

        real_import = builtins.__import__

        def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "cv2":
                raise ImportError("cv2 must not be imported when display=False")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        frame_source = FakeFrameSource(["f1", "f2"])
        loop = WebcamLoop(frame_source, FakeRuntime(), display=False)
        frames_processed = loop.run()

        assert frames_processed == 2

    def test_display_true_uses_cv2_imshow_and_waitkey(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        calls: List[Tuple[str, tuple]] = []

        fake_cv2 = types.SimpleNamespace(
            imshow=lambda window, frame: calls.append(("imshow", (window, frame))),
            waitKey=lambda delay: calls.append(("waitKey", (delay,))) or -1,
            destroyAllWindows=lambda: calls.append(("destroyAllWindows", ())),
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

        frame_source = FakeFrameSource(["f1"])
        loop = WebcamLoop(frame_source, FakeRuntime(), display=True, window_name="Test Window")
        loop.run()

        assert ("imshow", ("Test Window", "f1")) in calls
        assert any(name == "waitKey" for name, _ in calls)
        assert ("destroyAllWindows", ()) in calls

    def test_quit_key_stops_the_loop_before_frame_source_is_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types

        # waitKey reports the quit key ('q' == 113) was pressed immediately.
        fake_cv2 = types.SimpleNamespace(
            imshow=lambda window, frame: None,
            waitKey=lambda delay: ord("q"),
            destroyAllWindows=lambda: None,
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

        frame_source = FakeFrameSource(["f1", "f2", "f3"])
        runtime = FakeRuntime()
        loop = WebcamLoop(frame_source, runtime, display=True, quit_key="q")

        frames_processed = loop.run()

        # Stopped after the first frame, not all three.
        assert frames_processed == 1
        assert runtime.processed_frames == ["f1"]
        assert frame_source.release_calls == 1

    def test_non_quit_key_does_not_stop_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        fake_cv2 = types.SimpleNamespace(
            imshow=lambda window, frame: None,
            waitKey=lambda delay: -1,  # no key pressed
            destroyAllWindows=lambda: None,
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

        frame_source = FakeFrameSource(["f1", "f2"])
        loop = WebcamLoop(frame_source, FakeRuntime(), display=True, quit_key="q")

        frames_processed = loop.run()

        assert frames_processed == 2


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_runs_with_fresh_fakes_produce_consistent_results(self) -> None:
        def run_once() -> int:
            frame_source = FakeFrameSource(["f1", "f2", "f3"])
            loop = WebcamLoop(frame_source, FakeRuntime())
            return loop.run()

        results = {run_once() for _ in range(5)}
        assert results == {3}


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestFrameSourceProtocol:
    def test_fake_frame_source_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeFrameSource([]), FrameSource)

    def test_object_missing_required_methods_does_not_satisfy_protocol(self) -> None:
        class NotAFrameSource:
            pass

        assert not isinstance(NotAFrameSource(), FrameSource)


# ---------------------------------------------------------------------------
# Import safety: no cv2 / mediapipe / pyautogui at module import time
# ---------------------------------------------------------------------------

class TestNoForbiddenTopLevelImports:
    def test_webcam_loop_module_does_not_import_cv2_mediapipe_pyautogui_at_top_level(self) -> None:
        import gesture_control.app.webcam_loop as webcam_loop_module

        with open(webcam_loop_module.__file__) as f:
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

    def test_importing_webcam_loop_module_never_raises(self) -> None:
        import gesture_control.app.webcam_loop  # noqa: F401