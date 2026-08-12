"""Vision subpackage.

Exposes the HandTracker class used for hand landmark detection.
"""

from .hand_tracker import HandTracker, HandTrackerError

__all__ = ["HandTracker", "HandTrackerError"]