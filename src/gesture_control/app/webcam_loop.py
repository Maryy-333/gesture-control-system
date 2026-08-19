"""The frame-source loop: repeatedly reads a frame and drives it through
`GestureControlRuntime`.

This module contains no MediaPipe, gesture-recognition, action-mapping,
or PyAutoGUI logic of its own -- all of that already lives in
`GestureControlRuntime` and the components it orchestrates. This module
is one layer above that: it only manages the "read a frame, hand it to
the runtime, repeat" cycle, an optional preview window, and resource
cleanup.

`WebcamLoop` never constructs its own frame source or runtime; both are
injected. This is what makes it fully testable with fakes -- no real
webcam, MediaPipe, or PyAutoGUI is required to exercise the loop
mechanics themselves. `cv2` is only imported (lazily, inside the
methods that need it) when `display=True` actually triggers a frame to
be shown, so constructing/using this class with `display=False` (the
default, and what every test in this project uses) never requires
OpenCV to be importable.
"""

from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

from ..runtime.runtime import FrameResult, GestureControlRuntime


@runtime_checkable
class FrameSource(Protocol):
    """The subset of `Camera`'s interface `WebcamLoop` actually needs.

    `gesture_control.camera.Camera` satisfies this structurally, as
    does any fake used in tests -- no inheritance required.
    """

    @property
    def is_open(self) -> bool:
        """Whether the underlying device is currently open."""
        ...

    def open(self) -> None:
        """Open the device. Must be safe to call when already open."""
        ...

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Read the next frame: `(success, frame)`."""
        ...

    def release(self) -> None:
        """Release the device. Must be safe to call multiple times."""
        ...


class WebcamLoop:
    """Drives `GestureControlRuntime` across successive frames from a `FrameSource`.

    Both dependencies are injected via the constructor; neither a real
    camera nor a real runtime is ever constructed inside this class.

    Example (production use):
        loop = WebcamLoop(Camera(index=0), runtime)
        frames_processed = loop.run()

    Example (tests, fully injected -- see tests/test_webcam_loop.py):
        loop = WebcamLoop(FakeFrameSource([...]), FakeRuntime())
        frames_processed = loop.run()
    """

    def __init__(
        self,
        frame_source: FrameSource,
        runtime: GestureControlRuntime,
        *,
        display: bool = False,
        window_name: str = "Gesture Control",
        quit_key: str = "q",
        max_frames: Optional[int] = None,
        on_frame_processed: Optional[Callable[[FrameResult], None]] = None,
    ) -> None:
        """Initialize the loop.

        Args:
            frame_source: Supplies frames, e.g. a `Camera` or a fake.
            runtime: Processes each frame end to end.
            display: If True, show each raw frame in an OpenCV preview
                window and allow quitting by pressing `quit_key` while
                that window is focused. `cv2` is only imported when
                this is actually True.
            window_name: Title of the preview window, when `display`
                is True.
            quit_key: The single character that, when pressed with the
                preview window focused, stops the loop. Only consulted
                when `display` is True.
            max_frames: If given, stop after processing this many
                frames, regardless of whether the frame source has
                more available. Primarily useful for tests and
                bounded/offline runs; `None` means "run until the
                frame source is exhausted" (or the quit key is
                pressed, if `display` is True).
            on_frame_processed: Optional callback invoked with each
                frame's `FrameResult` immediately after it is
                processed. Exceptions raised by this callback are not
                caught.
        """
        self._frame_source = frame_source
        self._runtime = runtime
        self._display = display
        self._window_name = window_name
        self._quit_key = quit_key
        self._max_frames = max_frames
        self._on_frame_processed = on_frame_processed

    def run(self) -> int:
        """Run the loop until stopped, then release resources.

        Stops when any of the following happens first: the frame
        source reports no more frames (`read()` returns `success=False`),
        `max_frames` frames have been processed, or (if `display` is
        True) the configured `quit_key` is pressed in the preview
        window.

        The frame source is always released before this method
        returns, including when an exception propagates out of frame
        processing.

        Returns:
            The number of frames successfully processed.

        Raises:
            Whatever `self._runtime.process_frame()` or
            `self._on_frame_processed()` themselves raise. This method
            does not catch or suppress exceptions from its
            dependencies.
        """
        frames_processed = 0
        try:
            self._frame_source.open()

            while self._max_frames is None or frames_processed < self._max_frames:
                success, frame = self._frame_source.read()
                if not success:
                    break

                result = self._runtime.process_frame(frame)
                frames_processed += 1

                if self._on_frame_processed is not None:
                    self._on_frame_processed(result)

                if self._display and self._show_frame_and_should_quit(frame):
                    break
        finally:
            self._frame_source.release()
            if self._display:
                self._close_preview_window()

        return frames_processed

    def _show_frame_and_should_quit(self, frame: Any) -> bool:
        """Show `frame` in the preview window; return True if quit was requested."""
        import cv2  # Deferred: only needed when display=True is actually used.

        cv2.imshow(self._window_name, frame)
        pressed_key = cv2.waitKey(1) & 0xFF
        return pressed_key == ord(self._quit_key)

    def _close_preview_window(self) -> None:
        """Close any OpenCV preview window(s) opened by this loop."""
        import cv2  # Deferred: only needed when display=True is actually used.

        cv2.destroyAllWindows()