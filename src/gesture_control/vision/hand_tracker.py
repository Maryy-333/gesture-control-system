"""Hand landmark tracking for the Gesture Control System.

This module wraps MediaPipe's HandLandmarker (Tasks API) to detect and
track hands in individual video frames. It is intentionally narrow in
scope: it takes a frame in, and returns MediaPipe's hand-tracking result
out. It does not interpret gestures, draw anything, capture frames, or
run any kind of application loop.

MediaPipe API note:
    This module targets `mediapipe==1.0.0`, the version confirmed
    installed in the development environment. That build only ships
    the Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`,
    `HandLandmarkerOptions`, etc.); the older `mediapipe.solutions.hands`
    API is not available, so this module is implemented against the
    Tasks API. The class/method signatures used below were verified
    directly against the installed 1.0.0 package rather than assumed
    from older tutorials.

    The Tasks API requires a hand-landmark model bundle (a `.task`
    file) to be present on disk; it is not bundled with the pip
    package and must be downloaded separately. See `DEFAULT_MODEL_URL`
    below.
"""

import os
import time
from typing import Optional

import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)

# Official MediaPipe model bundle for hand landmark detection.
# Download it and place it at the path passed to HandTracker (or at
# DEFAULT_MODEL_PATH), e.g.:
#   wget -O models/hand_landmarker.task <DEFAULT_MODEL_URL>
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join("models", "hand_landmarker.task")


class HandTrackerError(Exception):
    """Raised when the hand tracker cannot be initialized or fails to run."""


class HandTracker:
    """Detects and tracks hand landmarks in video frames using MediaPipe.

    This class only performs hand landmark detection. It does not
    recognize or interpret gestures, and it does not draw or display
    anything -- that is left to calling code (e.g. a manual test
    script or a later gesture-recognition module).

    Example:
        tracker = HandTracker(max_num_hands=2)
        try:
            result = tracker.process(frame)  # frame is a BGR np.ndarray
            print(len(result.hand_landmarks), "hand(s) detected")
        finally:
            tracker.close()
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the hand tracker.

        Args:
            model_path: Path to the MediaPipe `hand_landmarker.task`
                model bundle on disk. See `DEFAULT_MODEL_URL` for where
                to download it.
            max_num_hands: Maximum number of hands to detect per frame.
            min_detection_confidence: Minimum confidence (0.0-1.0) for
                the initial hand detection to be considered successful.
            min_tracking_confidence: Minimum confidence (0.0-1.0) for
                hand landmarks to be considered tracked successfully
                between frames.

        Raises:
            HandTrackerError: If the model file is missing or the
                underlying MediaPipe landmarker cannot be created.
        """
        if not os.path.isfile(model_path):
            raise HandTrackerError(
                f"Hand landmark model not found at '{model_path}'. "
                f"Download it from {DEFAULT_MODEL_URL} and save it to "
                f"that path."
            )

        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._start_time = time.monotonic()
        self._closed = False

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        except Exception as error:
            raise HandTrackerError(
                f"Failed to initialize MediaPipe HandLandmarker: {error}"
            ) from error

    @property
    def is_closed(self) -> bool:
        """Whether the underlying MediaPipe resources have been released."""
        return self._closed

    def process(self, frame: "np.ndarray") -> HandLandmarkerResult:
        """Run hand landmark detection on a single frame.

        Args:
            frame: A single video frame as a BGR `numpy.ndarray`, in the
                format produced by OpenCV's `VideoCapture.read()`.

        Returns:
            The MediaPipe `HandLandmarkerResult` for this frame. It
            exposes, among other things:
                - `hand_landmarks`: per-hand list of normalized landmarks
                - `handedness`: per-hand handedness classification
                - `hand_world_landmarks`: per-hand landmarks in world
                  coordinates
            The number of detected hands is `len(result.hand_landmarks)`.

        Raises:
            HandTrackerError: If the tracker has already been closed, or
                if MediaPipe fails to process the frame.
        """
        if self._closed:
            raise HandTrackerError("Cannot process a frame: HandTracker is closed.")

        try:
            rgb_frame = frame[:, :, ::-1]
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
            return self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as error:
            raise HandTrackerError(f"Hand tracking failed on frame: {error}") from error

    def close(self) -> None:
        """Release the underlying MediaPipe resources.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> "HandTracker":
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Support use as a context manager: releases resources on exit."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup if close() was not called explicitly."""
        try:
            self.close()
        except Exception:
            # Never raise from __del__.
            pass