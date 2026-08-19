"""Composition root: wires the real, concrete components together.

This is the one place in the project that is *meant* to know about
every concrete implementation -- `Camera` (OpenCV), `MediaPipeHandTracker`
(MediaPipe), and `PyAutoGUIControlBackend` (PyAutoGUI) -- and assembles
them, along with the already-pure `FingerStateDetector`,
`GestureRecognizer`, `ActionMapper`, `CoordinateMapper`, and
`ComputerController`, into a working `GestureControlRuntime` and
`WebcamLoop`.

No duplicate logic lives here: this module only constructs and injects
existing components in the order the architecture already defines
(`HandTrackerProtocol -> ... -> ComputerController -> ControlBackend`,
driven by `WebcamLoop` -> `GestureControlRuntime`). It adds no new
gesture, mapping, or control behavior of its own.

Import safety: every concrete-backend import (`Camera`,
`MediaPipeHandTracker`, `PyAutoGUIControlBackend`, `pyautogui` itself
for screen-size detection) is deferred to *inside* the functions that
actually need them, not at module level. This means importing
`gesture_control.app` (or this module) never requires a webcam,
MediaPipe, or PyAutoGUI to be installed or usable -- only *calling*
`build_runtime()` / `build_webcam_loop()` / `main()` without overriding
those pieces does. Every dependency below can also be overridden with
an already-constructed object (typically a fake, in tests), which is
what lets this module be exercised in tests with zero real hardware,
MediaPipe, or PyAutoGUI involvement.
"""

from typing import Any, Dict, Optional

from ..actions.mapper import ActionMapper
from ..control.controller import ComputerController
from ..gestures.finger_states import FingerStateDetector
from ..gestures.recognizer import GestureRecognizer
from ..mapping.coordinate_mapper import CoordinateMapper, ScreenSize
from ..runtime.runtime import GestureControlRuntime
from ..tracking.hand_tracker_protocol import HandTrackerProtocol
from .webcam_loop import FrameSource, WebcamLoop


def _detect_screen_size() -> ScreenSize:
    """Determine the real screen size via PyAutoGUI.

    Raises:
        RuntimeError: If PyAutoGUI is unavailable in this environment
            (e.g. no display, or not installed). Pass an explicit
            `screen_size=` to `build_runtime()`/`build_webcam_loop()`
            to avoid needing this.
    """
    try:
        import pyautogui  # Deferred: only needed if screen_size isn't provided.
    except Exception as error:
        raise RuntimeError(
            "Could not determine the screen size because PyAutoGUI is "
            "unavailable in this environment (it typically requires a "
            "graphical display). Pass an explicit `screen_size=ScreenSize(...)` "
            "to avoid needing PyAutoGUI for this."
        ) from error

    width, height = pyautogui.size()
    return ScreenSize(width=int(width), height=int(height))


def build_runtime(
    *,
    hand_tracker: Optional[HandTrackerProtocol] = None,
    finger_state_detector: Optional[FingerStateDetector] = None,
    gesture_recognizer: Optional[GestureRecognizer] = None,
    action_mapper: Optional[ActionMapper] = None,
    computer_controller: Optional[ComputerController] = None,
    coordinate_mapper: Optional[CoordinateMapper] = None,
    screen_size: Optional[ScreenSize] = None,
    hand_tracker_kwargs: Optional[Dict[str, Any]] = None,
) -> GestureControlRuntime:
    """Build a fully-wired `GestureControlRuntime`, using real components by default.

    Every dependency can be overridden with an already-constructed
    object. This is both how a caller customizes the real application
    (e.g. a specific `screen_size`, or a tuned `FingerStateDetector`)
    and how tests build a runtime entirely out of fakes, with no real
    webcam, MediaPipe, or PyAutoGUI involved.

    Args:
        hand_tracker: Defaults to a real `MediaPipeHandTracker`.
        finger_state_detector: Defaults to `FingerStateDetector()`.
        gesture_recognizer: Defaults to `GestureRecognizer()`.
        action_mapper: Defaults to `ActionMapper()`.
        computer_controller: Defaults to a `ComputerController` backed
            by a real `PyAutoGUIControlBackend`.
        coordinate_mapper: Defaults to a `CoordinateMapper` built from
            `screen_size` (or the real detected screen size, if
            `screen_size` is also omitted).
        screen_size: Used to build the default `coordinate_mapper`.
            Ignored if `coordinate_mapper` is given directly.
        hand_tracker_kwargs: Forwarded to `MediaPipeHandTracker()` when
            no `hand_tracker` is given (e.g. `model_path`,
            `max_num_hands`).

    Returns:
        A `GestureControlRuntime` ready to process frames.
    """
    if hand_tracker is None:
        from ..tracking.mediapipe_hand_tracker import MediaPipeHandTracker

        hand_tracker = MediaPipeHandTracker(**(hand_tracker_kwargs or {}))

    if finger_state_detector is None:
        finger_state_detector = FingerStateDetector()

    if gesture_recognizer is None:
        gesture_recognizer = GestureRecognizer()

    if action_mapper is None:
        action_mapper = ActionMapper()

    if computer_controller is None:
        from ..control.pyautogui_backend import PyAutoGUIControlBackend

        computer_controller = ComputerController(backend=PyAutoGUIControlBackend())

    if coordinate_mapper is None:
        resolved_screen_size = screen_size if screen_size is not None else _detect_screen_size()
        coordinate_mapper = CoordinateMapper(resolved_screen_size)

    return GestureControlRuntime(
        hand_tracker=hand_tracker,
        finger_state_detector=finger_state_detector,
        gesture_recognizer=gesture_recognizer,
        action_mapper=action_mapper,
        computer_controller=computer_controller,
        coordinate_mapper=coordinate_mapper,
    )


def build_webcam_loop(
    *,
    camera: Optional[FrameSource] = None,
    camera_index: int = 0,
    runtime: Optional[GestureControlRuntime] = None,
    display: bool = False,
    window_name: str = "Gesture Control",
    quit_key: str = "q",
    max_frames: Optional[int] = None,
    screen_size: Optional[ScreenSize] = None,
    hand_tracker_kwargs: Optional[Dict[str, Any]] = None,
) -> WebcamLoop:
    """Build a fully-wired `WebcamLoop`, using real components by default.

    Args:
        camera: Defaults to a real `Camera(index=camera_index)`.
        camera_index: Used to build the default `camera`. Ignored if
            `camera` is given directly.
        runtime: Defaults to `build_runtime(screen_size=screen_size,
            hand_tracker_kwargs=hand_tracker_kwargs)`.
        display: See `WebcamLoop`.
        window_name: See `WebcamLoop`.
        quit_key: See `WebcamLoop`.
        max_frames: See `WebcamLoop`.
        screen_size: Forwarded to `build_runtime()` when no `runtime`
            is given. Ignored if `runtime` is given directly.
        hand_tracker_kwargs: Forwarded to `build_runtime()` when no
            `runtime` is given. Ignored if `runtime` is given directly.

    Returns:
        A `WebcamLoop` ready to `run()`.
    """
    if camera is None:
        try:
            import cv2
            from ..camera.camera import Camera
        except Exception as error:
            raise RuntimeError(
                "Could not construct the default webcam Camera because "
                "OpenCV (cv2) is unavailable in this environment. Install "
                "OpenCV, or pass an explicit `camera=` to avoid needing it here."
            ) from error

        camera = Camera(index=camera_index)

    if runtime is None:
        runtime = build_runtime(screen_size=screen_size, hand_tracker_kwargs=hand_tracker_kwargs)

    return WebcamLoop(
        frame_source=camera,
        runtime=runtime,
        display=display,
        window_name=window_name,
        quit_key=quit_key,
        max_frames=max_frames,
    )


def main() -> None:
    """Minimal CLI entry point: run the webcam loop with real components.

    Runs until the webcam stops producing frames or (with a display)
    the quit key is pressed. This is intentionally small -- no
    configuration file, gesture debouncing, or cooldown logic lives
    here; those are later concerns.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Gesture Control System webcam loop.")
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Webcam device index (default: 0)."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without an OpenCV preview window.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the MediaPipe hand_landmarker.task model file.",
    )
    args = parser.parse_args()

    hand_tracker_kwargs: Dict[str, Any] = {}
    if args.model_path:
        hand_tracker_kwargs["model_path"] = args.model_path

    loop = build_webcam_loop(
        camera_index=args.camera_index,
        display=not args.no_display,
        hand_tracker_kwargs=hand_tracker_kwargs,
    )
    frames_processed = loop.run()
    print(f"Processed {frames_processed} frame(s).")


if __name__ == "__main__":
    main()