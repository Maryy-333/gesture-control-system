"""Gesture recognition subpackage.

Exposes the full gesture-recognition pipeline implemented so far:
`geometry` (generic landmark math), `finger_states` (per-finger
extension detection), and `recognizer` (deterministic finger-states to
gesture mapping).
"""

from . import geometry
from .finger_states import FingerStateDetector, FingerStates, HandLandmark
from .recognizer import Gesture, GestureRecognizer

__all__ = [
    "geometry",
    "FingerStateDetector",
    "FingerStates",
    "HandLandmark",
    "Gesture",
    "GestureRecognizer",
]