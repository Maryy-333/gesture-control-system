"""Computer-control abstraction subpackage.

Exposes `ComputerController` (WHAT-to-HOW orchestration), the
`ControlBackend` protocol (the HOW interface a real backend must
satisfy), `NoOpControlBackend` (the safe default backend that performs
no real computer-control behavior), and `PyAutoGUIControlBackend` (a
real backend that drives the desktop via PyAutoGUI). See
`controller.py` and `pyautogui_backend.py` for the exact scope of
each.
"""

from .controller import ComputerController, ControlBackend, NoOpControlBackend
from .pyautogui_backend import PyAutoGUIControlBackend

__all__ = [
    "ComputerController",
    "ControlBackend",
    "NoOpControlBackend",
    "PyAutoGUIControlBackend",
]