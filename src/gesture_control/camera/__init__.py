"""Camera input subpackage.

Exposes the Camera class used for webcam access.
"""

from .camera import Camera, CameraError

__all__ = ["Camera", "CameraError"]