"""Gesture recognition subpackage.

At this milestone, only the `geometry` module (generic math utilities
for working with hand landmark coordinates) has been implemented.
`models` and `recognizer` are part of the planned architecture but are
not yet implemented, so they are intentionally not imported here.
"""

from . import geometry

__all__ = ["geometry"]