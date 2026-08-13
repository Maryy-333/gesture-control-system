"""Computer-control abstraction: dispatches abstract Actions to a backend.

This module sits at the very end of the pipeline:

    Landmarks -> ... -> Gesture -> ActionMapper -> Action -> ComputerController -> Backend

Three concerns are deliberately kept separate:

- `Action` (from `gesture_control.actions.action`) is WHAT should
  happen -- e.g. "move the cursor", "left click". It carries no
  knowledge of how that's actually performed.
- `ControlBackend` is HOW an action is actually carried out on a real
  machine -- the thing that would eventually wrap PyAutoGUI or another
  OS-automation library. No such real backend is implemented here.
- `ComputerController` is the orchestration layer in between: it
  validates an `Action` (and any coordinates it needs), and dispatches
  it to whatever `ControlBackend` was injected, without knowing or
  caring how that backend actually performs the action.

IMPORTANT: This module contains NO PyAutoGUI, OpenCV, MediaPipe, or
operating-system automation code. `ComputerController`'s default
backend (`NoOpControlBackend`) performs no real input-control
behavior at all, so using `ComputerController()` with no arguments is
always safe to run in tests or anywhere else -- it will never move the
real mouse, click anything, type anything, or take a real screenshot.
A real backend can be plugged in later via constructor injection
without changing this module.
"""

from typing import Dict, Optional, Protocol, runtime_checkable

from ..actions.action import Action


@runtime_checkable
class ControlBackend(Protocol):
    """The HOW of computer control: a backend that can carry out actions.

    This is a structural (duck-typed) interface -- any object that
    implements these methods with these signatures can be used as a
    backend, without needing to inherit from this class. A future
    PyAutoGUI-based backend, for example, would implement this same
    protocol and could be swapped in via `ComputerController`'s
    constructor with no changes to `ComputerController` itself.

    None of these methods have a default implementation here; this
    class only declares the shape a backend must have.
    """

    def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to the given screen coordinates."""
        ...

    def left_click(self) -> None:
        """Perform a left mouse click at the current cursor position."""
        ...

    def right_click(self) -> None:
        """Perform a right mouse click at the current cursor position."""
        ...

    def double_click(self) -> None:
        """Perform a double left-click at the current cursor position."""
        ...

    def scroll_up(self) -> None:
        """Scroll up by one unit."""
        ...

    def scroll_down(self) -> None:
        """Scroll down by one unit."""
        ...

    def screenshot(self) -> None:
        """Capture a screenshot."""
        ...

    def pause(self) -> None:
        """Pause/suspend automated control."""
        ...


class NoOpControlBackend:
    """A backend that performs no real computer-control behavior.

    This is the default backend for `ComputerController`. Every method
    is a no-op: nothing is moved, clicked, scrolled, typed, or
    captured. It exists so that `ComputerController` is safe to
    construct and use -- including in tests and in this milestone's
    demos -- without ever touching the real mouse, keyboard, or
    screen. It satisfies the `ControlBackend` protocol structurally.
    """

    def move_cursor(self, x: int, y: int) -> None:
        return None

    def left_click(self) -> None:
        return None

    def right_click(self) -> None:
        return None

    def double_click(self) -> None:
        return None

    def scroll_up(self) -> None:
        return None

    def scroll_down(self) -> None:
        return None

    def screenshot(self) -> None:
        return None

    def pause(self) -> None:
        return None


# Maps each zero-argument Action to the ControlBackend method that
# performs it. Action.NONE and Action.MOVE_CURSOR are handled
# separately in `ComputerController.execute` since NONE dispatches to
# nothing and MOVE_CURSOR requires coordinates.
_ZERO_ARGUMENT_ACTIONS: Dict[Action, str] = {
    Action.LEFT_CLICK: "left_click",
    Action.RIGHT_CLICK: "right_click",
    Action.DOUBLE_CLICK: "double_click",
    Action.SCROLL_UP: "scroll_up",
    Action.SCROLL_DOWN: "scroll_down",
    Action.SCREENSHOT: "screenshot",
    Action.PAUSE: "pause",
}


class ComputerController:
    """Dispatches abstract `Action`s to an injected `ControlBackend`.

    `ComputerController` is orchestration only: it validates inputs
    (e.g. that `MOVE_CURSOR` was given coordinates) and calls the
    corresponding method on its backend. It has no knowledge of how
    the backend actually performs any action, and performs no
    real-world side effects itself.

    Dependency injection: the backend is supplied via the constructor.
    If none is given, a safe `NoOpControlBackend` is used, so
    `ComputerController()` never performs real computer-control
    behavior. A real backend (e.g. a future PyAutoGUI-based
    implementation of `ControlBackend`) can be substituted without any
    change to this class.

    Example:
        controller = ComputerController()  # safe no-op backend
        controller.execute(Action.MOVE_CURSOR, x=100, y=200)
        controller.execute(Action.LEFT_CLICK)
    """

    def __init__(self, backend: Optional[ControlBackend] = None) -> None:
        """Initialize the controller.

        Args:
            backend: The `ControlBackend` to dispatch actions to. If
                omitted, defaults to a `NoOpControlBackend`, which
                performs no real computer-control behavior.
        """
        self._backend: ControlBackend = backend if backend is not None else NoOpControlBackend()

    def execute(self, action: Action, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Execute an abstract action via the injected backend.

        Args:
            action: The `Action` to perform.
            x: Target cursor x-coordinate. Required for
                `Action.MOVE_CURSOR`; ignored for all other actions.
            y: Target cursor y-coordinate. Required for
                `Action.MOVE_CURSOR`; ignored for all other actions.

        Raises:
            ValueError: If `action` is `Action.MOVE_CURSOR` and `x` or
                `y` is missing, or if `action` is not a supported
                `Action` value.
        """
        if action == Action.NONE:
            return

        if action == Action.MOVE_CURSOR:
            if x is None or y is None:
                raise ValueError(
                    "Action.MOVE_CURSOR requires both 'x' and 'y' coordinates."
                )
            self._backend.move_cursor(x, y)
            return

        method_name = _ZERO_ARGUMENT_ACTIONS.get(action)
        if method_name is None:
            raise ValueError(f"Unsupported action: {action!r}")

        getattr(self._backend, method_name)()