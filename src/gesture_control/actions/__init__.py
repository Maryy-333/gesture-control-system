"""Action abstraction subpackage.

Exposes `Action` (abstract computer-control actions) and `ActionMapper`
(deterministic Gesture -> Action mapping). No real mouse, keyboard, or
OS control lives here yet -- see the module docstrings in `action.py`
and `mapper.py` for the exact scope.
"""

from .action import Action
from .mapper import ActionMapper

__all__ = ["Action", "ActionMapper"]