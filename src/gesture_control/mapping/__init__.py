"""Coordinate-mapping subpackage.

Exposes `CoordinateMapper` (normalized coordinate -> screen pixel
coordinate conversion), `ScreenSize`, and `ScreenPoint`. Pure Python:
no OpenCV, MediaPipe, PyAutoGUI, GUI toolkit, or OS-specific library is
involved -- see `coordinate_mapper.py` for the exact scope.
"""

from .coordinate_mapper import CoordinateMapper, ScreenPoint, ScreenSize

__all__ = ["CoordinateMapper", "ScreenPoint", "ScreenSize"]