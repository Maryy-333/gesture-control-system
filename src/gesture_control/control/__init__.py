"""Computer-control abstraction subpackage.

Exposes `ComputerController` (WHAT-to-HOW orchestration), the
`ControlBackend` protocol (the HOW interface a real backend must
satisfy), and `NoOpControlBackend` (the safe default backend that
performs no real computer-control behavior). No real mouse, keyboard,
or OS automation code lives here yet -- see `controller.py`'s module
docstring for the exact scope.
"""

from .controller import ComputerController, ControlBackend, NoOpControlBackend

__all__ = ["ComputerController", "ControlBackend", "NoOpControlBackend"]