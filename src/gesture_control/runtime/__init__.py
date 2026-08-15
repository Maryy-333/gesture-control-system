"""Real-time runtime orchestration subpackage.

Exposes `GestureControlRuntime` (the single-frame orchestration class)
and `FrameResult` (its immutable per-frame result type). No webcam
capture, real-time loop, MediaPipe, OpenCV, or PyAutoGUI code lives
here -- see `runtime.py`'s module docstring for the exact scope.
"""

from .runtime import FrameResult, GestureControlRuntime

__all__ = ["FrameResult", "GestureControlRuntime"]