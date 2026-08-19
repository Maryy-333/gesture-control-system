"""The real application entry point: wires all existing components into
a runnable webcam gesture-control loop.

Exposes:
    `WebcamLoop`, `FrameSource` -- the dependency-injected loop
        mechanics (webcam_loop.py). No MediaPipe/PyAutoGUI/real webcam
        required to construct or test this class.
    `build_runtime`, `build_webcam_loop`, `main` -- the composition
        root (application.py) that wires the real, concrete
        components (`Camera`, `MediaPipeHandTracker`,
        `PyAutoGUIControlBackend`) together, or accepts overrides for
        every piece.

Importing this package never requires a webcam, MediaPipe, or
PyAutoGUI to be installed or usable -- see `application.py`'s module
docstring for why.
"""

from .application import build_runtime, build_webcam_loop, main
from .webcam_loop import FrameSource, WebcamLoop

__all__ = ["WebcamLoop", "FrameSource", "build_runtime", "build_webcam_loop", "main"]