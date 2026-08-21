# Gesture-Controlled Computer Vision System

A real-time computer vision system that lets you control a computer using hand gestures captured through a webcam.

The system detects hand landmarks with MediaPipe, recognizes predefined gestures, stabilizes recognition across frames, maps gestures to computer actions, and controls the computer through PyAutoGUI.

The project was built with a modular architecture so that hand tracking, gesture recognition, action mapping, coordinate mapping, and computer control remain separate and independently testable.

## Demo

A short demonstration of the system is :


[▶️ Watch the demo on LinkedIn](https://www.linkedin.com/posts/mary-k-ai_computervision-python-ai-ugcPost-7496186092556029952-bGbM/?utm_source=share&utm_medium=member_android&rcm=ACoAAGsUk7cBGutMsPDJ3sXKJMpgaEe2X7hY4PI)

The system recognizes hand gestures and translates them into computer controls such as cursor movement, clicking, scrolling, pause/resume, and volume control.

## Features

- Real-time hand tracking through a webcam
- MediaPipe-based 21-point hand landmark detection
- Rule-based gesture recognition
- Cross-frame gesture stabilization
- Cursor movement using index-finger position
- Exponential moving-average cursor smoothing
- Gesture-based mouse actions
- Discrete-action debouncing
- Pause/resume control
- Volume up and volume down controls
- Coordinate mapping from normalized camera coordinates to screen coordinates
- Modular, dependency-injected architecture
- Automated testing with Pytest
- Hardware-independent testing through fakes and injected dependencies

## Recognized Gestures

| Gesture | Action |
|---|---|
| ☝️ Point | Move cursor |
| ✌️ Peace | double click |
| 👊 Fist | Left click |
| 🖐️ Open Palm | Pause / Resume |
| 👍 Thumbs Up | Volume up |
| 👎 Thumbs Down | Volume down |

> Gesture recognition is rule-based and uses hand landmark geometry rather than a separately trained neural-network classifier.

## How It Works

The system processes each webcam frame through a layered pipeline:

```text
Webcam Frame
     ↓
Hand Tracker
     ↓
Hand Tracking Result
     ↓
Finger State Detection
     ↓
Gesture Recognition
     ↓
Gesture Smoothing
     ↓
Action Mapping
     ↓
Gesture Action Gate
     ↓
Coordinate Mapping
     ↓
Computer Controller
     ↓
Computer Action

Gesture Stabilization

Raw gesture recognition can occasionally fluctuate between consecutive webcam frames.

For example:

POINT → FIST → POINT

could accidentally be interpreted as a click.

The GestureSmoothener stabilizes gesture recognition across frames before the result reaches the action gate.

Cursor Smoothing

Cursor movement uses an exponential moving average over the normalized index-finger coordinates:

smoothed = previous + α × (current - previous)

This reduces small frame-to-frame tracking jitter while keeping cursor movement continuous.

Action Gating

GestureActionGate handles behavior that depends on previous frames.

It:

prevents a held discrete gesture from repeatedly firing an action
allows continuous actions such as cursor movement to run every frame
handles pause/resume behavior
re-arms discrete actions when hand tracking is lost
Architecture

The project follows a modular architecture with clear separation of responsibilities.

Camera / Frame Source
        ↓
HandTrackerProtocol
        ↓
HandTrackingResult
        ↓
FingerStateDetector
        ↓
GestureRecognizer
        ↓
GestureSmoothener
        ↓
ActionMapper
        ↓
GestureActionGate
        ↓
CoordinateMapper
        ↓
ComputerController
        ↓
ControlBackend

The runtime acts as the orchestration layer. It coordinates the components but does not contain gesture-recognition, action-mapping, or coordinate-mapping rules of its own.

This separation makes individual components easier to test, replace, and maintain.

## Project Structure
gesture-controlled-computer-vision/
│
├── src/
│   └── gesture_control/
│       ├── actions/
│       │   ├── action.py
│       │   └── mapper.py
│       │
│       ├── camera/
│       │   └── camera.py
│       │
│       ├── control/
│       │   ├── controller.py
│       │   └── pyautogui_backend.py
│       │
│       ├── gestures/
│       │   ├── finger_states.py
│       │   ├── recognizer.py
│       │   └── smoothener.py
│       │
│       ├── mapping/
│       │   └── coordinate_mapper.py
│       │
│       ├── runtime/
│       │   ├── runtime.py
│       │   └── gesture_action_gate.py
│       │
│       ├── tracking/
│       │   ├── hand_tracker_protocol.py
│       │   ├── hand_tracking_result.py
│       │   └── mediapipe_hand_tracker.py
│       │
│       └── app/
│           ├── application.py
│           └── webcam_loop.py
│
├── tests/
│   ├── test_controller.py
│   ├── test_gesture_action_gate.py
│   ├── test_gesture_smoothener.py
│   ├── test_mapper.py
│   ├── test_recognizer.py
│   ├── test_runtime.py
│   ├── test_webcam_loop.py
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
## Tech Stack
Python
MediaPipe — hand landmark detection
OpenCV — webcam/frame handling
PyAutoGUI — computer control
Pytest — automated testing

## Installation

Clone the repository:
```bash
git clone https://github.com/Maryy-333/gesture-control-system
cd gesture-control-system

## Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

## Install dependencies:

pip install -r requirements.txt

MediaPipe Model

The hand-tracking component uses the MediaPipe Hand Landmarker task model.

Place the required .task model file in the location expected by the application, or provide its path when starting the application.

## Running the Application

Run the application with the default webcam:

$env:PYTHONPATH="$PWD\src"
python -m gesture_control

You can also specify a camera index:

python -m gesture_control.app.application --camera-index 0

To run without the OpenCV preview window:

python -m gesture_control.app.application --no-display

If the model path needs to be specified:

python -m gesture_control.app.application --model-path path/to/hand_landmarker.task
Testing

The project includes automated tests covering the main components and runtime behavior.

Run the complete test suite with:

pytest

The architecture allows components such as the camera, hand tracker, runtime, and controller to be replaced with test doubles, allowing most of the system to be tested without requiring a physical webcam or actual computer-control actions.

Design Principles

A major focus of this project was keeping the system modular rather than putting all behavior inside one large application loop.

Separation of concerns

Each component has a specific responsibility:

Hand Tracker → detects hands and produces landmarks
Finger State Detector → determines which fingers are extended
Gesture Recognizer → converts finger states into gestures
Gesture Smoothener → stabilizes recognition across frames
Action Mapper → maps gestures to actions
Gesture Action Gate → handles debouncing and pause/resume
Coordinate Mapper → converts camera coordinates to screen coordinates
Computer Controller → executes actions through the configured backend
Webcam Loop → manages the frame-processing cycle
Application layer → assembles the concrete components

This makes the system easier to reason about, test, and extend.

Limitations

Because the system relies on webcam-based hand tracking, performance can be affected by:

lighting conditions
camera quality
hand position and orientation
fast hand movements
temporary loss of hand tracking
natural landmark jitter

The gesture recognizer is intentionally rule-based rather than being a trained gesture-classification model.

Future Improvements

Possible future improvements include:

Support for additional gestures
More robust recognition under different hand orientations
User-configurable gesture-to-action mappings
Calibration for different screen sizes and camera setups
Improved tracking under difficult lighting conditions
Additional control backends
More advanced gesture classification

Inspiration

The idea for this project was inspired by seeing other developers build webcam-based cursor-control projects.

Rather than reproducing an existing implementation, I used the general concept as inspiration and built my own system with a modular architecture, gesture recognition, stabilization, action gating, coordinate mapping, and automated testing.

Author

Mary K

Built as a hands-on computer vision and software engineering project focused on real-time interaction, modular architecture, and testable Python development.
