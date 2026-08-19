"""Cross-frame gesture stabilization: debouncing and pause/resume.

This is the "separate, higher-level concern" that keeps `GestureRecognizer`
and `ActionMapper` pure, stateless, and per-frame: it decides, given a
freshly recognized `(Gesture, Action)` pair, whether that action should
actually be dispatched to `ComputerController` *this frame* --
suppressing repeat-fire of discrete actions (e.g. clicks) while a
gesture is held, and implementing a simple pause/resume toggle. Neither
`GestureRecognizer` nor `ActionMapper` is touched or made aware this
exists; they still produce a fresh `Gesture`/`Action` every frame
regardless of what happened on previous frames.

This module has no dependency on OpenCV, MediaPipe, or PyAutoGUI --
only on the existing `Gesture` and `Action` enums.
"""

from typing import FrozenSet, Optional

from ..actions.action import Action
from ..gestures.recognizer import Gesture

# Actions with a genuinely continuous, per-frame effect (cursor
# tracking, scrolling): these are never suppressed by debouncing, since
# suppressing them would defeat their entire purpose -- POINT needs to
# keep updating the cursor position every frame it's held, not just
# once.
CONTINUOUS_ACTIONS: FrozenSet[Action] = frozenset(
    {Action.MOVE_CURSOR, Action.SCROLL_UP, Action.SCROLL_DOWN}
)


class GestureActionGate:
    """Decides, per frame, whether a mapped `Action` should reach the controller.

    Two responsibilities, both purely about timing/state *across*
    frames -- neither one is gesture recognition or action mapping:

    1. Debouncing: a discrete action (e.g. `Action.LEFT_CLICK`) fires
       once when its gesture is newly recognized, then is suppressed on
       subsequent frames for as long as the *same* gesture keeps being
       recognized -- so holding a FIST doesn't fire dozens of clicks
       per second. `CONTINUOUS_ACTIONS` are exempt and fire every frame
       their gesture is held.

    2. Pause/resume: `Action.PAUSE` (mapped from `Gesture.OPEN_PALM`)
       never itself reaches `ComputerController` -- there is no
       PyAutoGUI-level "pause" operation to call (see
       `PyAutoGUIControlBackend.pause`'s own docstring, which
       explicitly defers this to "a future, higher-level
       controller/orchestration loop that decides whether to call
       execute() at all"; this class is that loop-level component).
       Instead, each *newly* recognized PAUSE toggles an internal
       paused flag here. While paused, every other action -- including
       `MOVE_CURSOR` -- is suppressed, so showing an open palm
       genuinely freezes control until it is shown again to resume.

    Example:
        gate = GestureActionGate()
        gate.should_execute(Gesture.FIST, Action.LEFT_CLICK)   # True  (new)
        gate.should_execute(Gesture.FIST, Action.LEFT_CLICK)   # False (held)
        gate.should_execute(Gesture.POINT, Action.MOVE_CURSOR) # True  (new gesture)
        gate.should_execute(Gesture.POINT, Action.MOVE_CURSOR) # True  (continuous)
    """

    def __init__(self) -> None:
        self._last_gesture: Optional[Gesture] = None
        self._paused: bool = False

    @property
    def is_paused(self) -> bool:
        """Whether the gate is currently suppressing all actions."""
        return self._paused

    def should_execute(self, gesture: Optional[Gesture], action: Optional[Action]) -> bool:
        """Decide whether `action` should be sent to the controller this frame.

        Args:
            gesture: The gesture recognized this frame, or `None` if no
                hand was detected.
            action: The action mapped from `gesture`, or `None` if no
                hand was detected.

        Returns:
            `True` if the caller should dispatch `action` to
            `ComputerController` this frame. Always `False` for
            `Action.PAUSE` -- that action is fully absorbed here (see
            the class docstring) and never forwarded.
        """
        if gesture is None or action is None:
            # No hand: nothing is "held" anymore, so re-arm every
            # discrete action for next time. Pause state is
            # intentionally NOT reset here -- briefly losing hand
            # tracking shouldn't silently resume control.
            self._last_gesture = None
            return False

        is_new_gesture = gesture != self._last_gesture
        self._last_gesture = gesture

        if action == Action.PAUSE:
            if is_new_gesture:
                self._paused = not self._paused
            return False

        if self._paused:
            return False

        if action in CONTINUOUS_ACTIONS:
            return True

        return is_new_gesture

    def reset(self) -> None:
        """Forget held-gesture state and un-pause. Mainly useful for tests."""
        self._last_gesture = None
        self._paused = False