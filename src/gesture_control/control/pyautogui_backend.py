"""PyAutoGUI-based implementation of the ControlBackend protocol.

This is the only module in the codebase that imports PyAutoGUI. No
other layer (`geometry.py`, `finger_states.py`, `recognizer.py`,
`mapper.py`, `action.py`, `controller.py`) knows PyAutoGUI exists --
they all speak in terms of the `ControlBackend` protocol defined in
`controller.py`, which this module implements.

Import safety:
    PyAutoGUI can fail to import in headless/CI/server environments
    (e.g. no `DISPLAY` set), and the failure it raises is not always a
    clean `ImportError` -- depending on platform and installed
    dependencies it can surface as other exceptions during PyAutoGUI's
    own start-up probing of the display and input devices. To avoid
    that failure taking down the entire `gesture_control` package,
    the `import pyautogui` below is wrapped in a broad `except
    Exception` at *module import time*. If it fails, the failure is
    only raised later, when `PyAutoGUIControlBackend` is actually
    constructed without an injected replacement -- not when this
    module (or anything that imports it, including
    `gesture_control.control`) is merely imported.

Testability:
    `PyAutoGUIControlBackend` accepts an optional `pyautogui_module`
    argument so a fake/mock object can be injected in place of the
    real PyAutoGUI module. Tests use this to guarantee zero real
    mouse/keyboard/screen interaction, without needing PyAutoGUI to be
    importable (or even installed) in the test environment at all.
"""

from types import ModuleType
from typing import Optional

try:
    import pyautogui as _pyautogui  # type: ignore[import-untyped]
    _PYAUTOGUI_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _import_error:  # pragma: no cover - failure mode is environment-dependent
    _pyautogui = None  # type: ignore[assignment]
    _PYAUTOGUI_IMPORT_ERROR = _import_error

# One PyAutoGUI scroll "click". PyAutoGUI's own convention is that
# positive values scroll up and negative values scroll down, so
# scroll_down() below passes the negation of this amount. Chosen to be
# the smallest meaningful, clearly-documented increment; callers that
# want faster scrolling should call scroll_up()/scroll_down() multiple
# times rather than this backend guessing a "bigger" amount.
SCROLL_UNIT = 1


class PyAutoGUIControlBackend:
    """A `ControlBackend` that drives the real mouse/screen via PyAutoGUI.

    This is the first backend in the project that performs real
    computer-control side effects. Everywhere else in the codebase
    continues to depend only on the `ControlBackend` protocol, so
    `ComputerController` needs no changes to use this backend --
    it is wired in purely via constructor injection:

        controller = ComputerController(backend=PyAutoGUIControlBackend())

    screenshot() design note:
        `ControlBackend.screenshot()` returns `None` (per the existing
        protocol, which this backend does not redesign). This method
        calls `pyautogui.screenshot()` to actually capture the screen,
        but does not persist, return, or otherwise expose the captured
        image at this layer -- the image is created and then
        discarded. Saving or returning screenshots is a future
        enhancement that would need a protocol change; it is out of
        scope for this milestone.

    pause() design note:
        This method is intentionally a no-op. It does not block, loop,
        or sleep. Actual pause/resume *state* (e.g. "stop dispatching
        further actions until resumed") is a concern for a future,
        higher-level controller/orchestration loop that decides
        whether to call `execute()` at all -- it does not belong in a
        single backend method, and implementing it here as a blocking
        call would violate the "no uncontrolled blocking behavior"
        requirement for this abstraction level.
    """

    def __init__(self, pyautogui_module: Optional[ModuleType] = None) -> None:
        """Initialize the backend.

        Args:
            pyautogui_module: An object exposing the PyAutoGUI
                functions this backend uses (`moveTo`, `click`,
                `rightClick`, `doubleClick`, `scroll`, `screenshot`).
                If omitted, the real `pyautogui` module is used. This
                parameter exists primarily so tests can inject a
                fake/mock in place of PyAutoGUI.

        Raises:
            RuntimeError: If no `pyautogui_module` was given and the
                real `pyautogui` package could not be imported in this
                environment (e.g. no display available).
        """
        if pyautogui_module is not None:
            self._pyautogui: ModuleType = pyautogui_module
        elif _pyautogui is not None:
            self._pyautogui = _pyautogui
        else:
            raise RuntimeError(
                "PyAutoGUI could not be imported in this environment "
                "(it typically requires a graphical display). Install "
                "and configure PyAutoGUI to use PyAutoGUIControlBackend, "
                "or pass a compatible object via the `pyautogui_module` "
                "argument (e.g. for testing)."
            ) from _PYAUTOGUI_IMPORT_ERROR

    def move_cursor(self, x: int, y: int) -> None:
        """Move the real cursor to `(x, y)` via `pyautogui.moveTo`."""
        self._pyautogui.moveTo(x, y)

    def left_click(self) -> None:
        """Perform a real left click via `pyautogui.click`."""
        self._pyautogui.click()

    def right_click(self) -> None:
        """Perform a real right click via `pyautogui.rightClick`."""
        self._pyautogui.rightClick()

    def double_click(self) -> None:
        """Perform a real double click via `pyautogui.doubleClick`."""
        self._pyautogui.doubleClick()

    def scroll_up(self) -> None:
        """Scroll up by `SCROLL_UNIT` via `pyautogui.scroll`."""
        self._pyautogui.scroll(SCROLL_UNIT)

    def scroll_down(self) -> None:
        """Scroll down by `SCROLL_UNIT` via `pyautogui.scroll`."""
        self._pyautogui.scroll(-SCROLL_UNIT)

    def screenshot(self) -> None:
        """Capture the screen via `pyautogui.screenshot`.

        The captured image is not returned or stored; see the class
        docstring's "screenshot() design note" for why.
        """
        self._pyautogui.screenshot()

    def pause(self) -> None:
        """No-op. See the class docstring's "pause() design note"."""
        return None

    def volume_up(self) -> None:
        """Increase system volume using the Windows volume-up key."""
        self._pyautogui.press("volumeup")

    def volume_down(self) -> None:
        """Decrease system volume using the Windows volume-down key."""
        self._pyautogui.press("volumedown")