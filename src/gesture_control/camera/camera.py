"""Webcam input handling for the Gesture Control System.

This module provides a small, focused `Camera` class responsible only for
opening a webcam, reading frames from it, and releasing it cleanly. It has
no knowledge of gestures, UI, or any other application concerns.
"""

from typing import Optional, Tuple

import numpy as np
import cv2


class CameraError(Exception):
    """Raised when the camera cannot be opened or fails to provide frames."""


class Camera:
    """Wraps an OpenCV VideoCapture device for webcam input.

    This class is intentionally minimal: it only knows how to open a
    camera device, read frames from it, and release it. It does not
    perform any image processing, gesture recognition, or UI work.

    Example:
        camera = Camera(index=0)
        camera.open()
        try:
            ok, frame = camera.read()
            if ok:
                ...  # do something with frame
        finally:
            camera.release()
    """

    def __init__(self, index: int = 0) -> None:
        """Initialize the Camera.

        Args:
            index: The index of the webcam device to use (as understood
                by OpenCV, e.g. 0 for the default camera).
        """
        self._index: int = index
        self._capture: Optional[cv2.VideoCapture] = None

    @property
    def index(self) -> int:
        """The configured camera device index."""
        return self._index

    @property
    def is_open(self) -> bool:
        """Whether the underlying camera device is currently open."""
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        """Open the webcam device.

        Raises:
            CameraError: If the device cannot be opened.
        """
        if self.is_open:
            return

        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Unable to open camera at index {self._index}."
            )

        self._capture = capture

    def read(self) -> Tuple[bool, Optional["np.ndarray"]]:
        """Read the next available frame from the camera.

        Returns:
            A tuple (success, frame). `success` is True and `frame` is a
            valid image array if a frame was read successfully.
            `success` is False and `frame` is None otherwise (including
            when the camera has not been opened).
        """
        if not self.is_open:
            return False, None

        assert self._capture is not None  # for type checkers; is_open guards this
        success, frame = self._capture.read()
        if not success:
            return False, None

        return True, frame

    def release(self) -> None:
        """Release the camera device, if it is open.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        """Support use as a context manager: opens the camera on entry."""
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Support use as a context manager: releases the camera on exit."""
        self.release()

    def __del__(self) -> None:
        """Best-effort cleanup if the camera was not explicitly released."""
        try:
            self.release()
        except Exception:
            # Never raise from __del__.
            pass