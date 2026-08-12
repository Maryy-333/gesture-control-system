"""Manual integration test for the Camera class.

Opens a real webcam, displays the live feed in a window, and exits when
the user presses 'q'. This script is for manual/visual validation only —
it is not part of the automated test suite.

Run from the project root with:
    python scripts/test_camera.py
"""

import os
import sys

import cv2

# Allow running this script directly from the repo without installing
# the package, by adding "src" to the import path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.camera import Camera, CameraError


def main() -> None:
    """Open the default webcam and display frames until 'q' is pressed."""
    camera = Camera(index=0)

    try:
        camera.open()
    except CameraError as error:
        print(f"Failed to open camera: {error}")
        return

    print("Camera opened. Press 'q' in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")


if __name__ == "__main__":
    main()