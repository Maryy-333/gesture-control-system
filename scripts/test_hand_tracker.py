"""Manual integration test for Camera + HandTracker.

Opens a real webcam via the Camera class, runs each frame through
HandTracker, draws any detected hand landmarks on the frame, and
displays it in a window. Exits when the user presses 'q'.

This script is for manual/visual validation only -- it is not part of
the automated test suite, and it performs no gesture recognition.

The `HandLandmarksConnections` and `drawing_utils.draw_landmarks` imports
below were verified directly against the confirmed installed version,
`mediapipe==1.0.0`; both are valid there and required no changes.

Before running, download the MediaPipe hand landmark model and save it
to models/hand_landmarker.task (relative to the project root):

    mkdir -p models
    wget -O models/hand_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Run from the project root with:
    python scripts/test_hand_tracker.py
"""

import os
import sys

import cv2

# Allow running this script directly from the repo without installing
# the package, by adding "src" to the import path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

from gesture_control.camera import Camera, CameraError
from gesture_control.vision import HandTracker, HandTrackerError


def main() -> None:
    """Read webcam frames, run hand tracking, and display the result."""
    camera = Camera(index=0)

    try:
        camera.open()
    except CameraError as error:
        print(f"Failed to open camera: {error}")
        return

    try:
        tracker = HandTracker(max_num_hands=2)
    except HandTrackerError as error:
        print(f"Failed to initialize hand tracker: {error}")
        camera.release()
        return

    print("Camera and hand tracker ready. Press 'q' in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            try:
                result = tracker.process(frame)
            except HandTrackerError as error:
                print(f"Hand tracking error: {error}")
                break

            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(
                    frame,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )

            cv2.imshow("Hand Tracker Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released, hand tracker closed, windows closed.")


if __name__ == "__main__":
    main()