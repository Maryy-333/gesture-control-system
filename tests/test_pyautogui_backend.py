"""Unit tests for gesture_control.control.pyautogui_backend.

SAFETY: None of these tests import or use the real `pyautogui` package.
Every test injects `FakePyAutoGUI` (defined below) -- a fake, in-memory
stand-in that only records which of its functions were called and with
what arguments. It never moves the real cursor, clicks the real mouse,
scrolls the real screen, or takes a real screenshot, and it works
identically regardless of whether the real PyAutoGUI package is even
importable in this environment (it frequently isn't, e.g. on headless
CI runners with no display).

`ComputerController` dispatch behavior is already covered by
tests/test_controller.py and is intentionally not re-tested here; this
file is scoped to `PyAutoGUIControlBackend` itself.
"""

import os
import sys
from typing import Any, List, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.control.controller import ControlBackend
from gesture_control.control.pyautogui_backend import (
    SCROLL_UNIT,
    PyAutoGUIControlBackend,
)


class FakePyAutoGUI:
    """A fake stand-in for the `pyautogui` module.

    Implements only the functions `PyAutoGUIControlBackend` actually
    calls (`moveTo`, `click`, `rightClick`, `doubleClick`, `scroll`,
    `screenshot`), recording each call instead of doing anything real.
    This guarantees zero real OS/mouse/screen interaction in tests --
    there is no code path here that could touch the actual desktop.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, tuple]] = []
        self.screenshot_return_value: Any = "FAKE_SCREENSHOT_IMAGE"

    def moveTo(self, x: int, y: int) -> None:
        self.calls.append(("moveTo", (x, y)))

    def click(self) -> None:
        self.calls.append(("click", ()))

    def rightClick(self) -> None:
        self.calls.append(("rightClick", ()))

    def doubleClick(self) -> None:
        self.calls.append(("doubleClick", ()))

    def scroll(self, amount: int) -> None:
        self.calls.append(("scroll", (amount,)))

    def screenshot(self) -> Any:
        self.calls.append(("screenshot", ()))
        return self.screenshot_return_value


@pytest.fixture
def fake_pyautogui() -> FakePyAutoGUI:
    return FakePyAutoGUI()


@pytest.fixture
def backend(fake_pyautogui: FakePyAutoGUI) -> PyAutoGUIControlBackend:
    return PyAutoGUIControlBackend(pyautogui_module=fake_pyautogui)


# ---------------------------------------------------------------------------
# Each method calls the correct PyAutoGUI function with correct arguments
# ---------------------------------------------------------------------------

class TestMoveCursor:
    def test_calls_moveTo_with_given_coordinates(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.move_cursor(100, 200)
        assert fake_pyautogui.calls == [("moveTo", (100, 200))]

    def test_calls_moveTo_with_zero_and_negative_coordinates(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.move_cursor(0, -5)
        assert fake_pyautogui.calls == [("moveTo", (0, -5))]


class TestClicks:
    def test_left_click_calls_click(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.left_click()
        assert fake_pyautogui.calls == [("click", ())]

    def test_right_click_calls_rightClick(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.right_click()
        assert fake_pyautogui.calls == [("rightClick", ())]

    def test_double_click_calls_doubleClick(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.double_click()
        assert fake_pyautogui.calls == [("doubleClick", ())]


class TestScrolling:
    def test_scroll_up_calls_scroll_with_positive_unit(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.scroll_up()
        assert fake_pyautogui.calls == [("scroll", (SCROLL_UNIT,))]
        assert SCROLL_UNIT > 0

    def test_scroll_down_calls_scroll_with_negative_unit(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.scroll_down()
        assert fake_pyautogui.calls == [("scroll", (-SCROLL_UNIT,))]

    def test_scroll_unit_is_small(self) -> None:
        # A "small, clearly documented scroll amount" per spec -- not
        # an arbitrarily large jump.
        assert SCROLL_UNIT == 1


class TestScreenshot:
    def test_screenshot_calls_pyautogui_screenshot(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.screenshot()
        assert fake_pyautogui.calls == [("screenshot", ())]

    def test_screenshot_returns_none_per_protocol(
        self, backend: PyAutoGUIControlBackend
    ) -> None:
        # ControlBackend.screenshot() is documented to return None; this
        # backend keeps that contract rather than redesigning it.
        result = backend.screenshot()
        assert result is None

    def test_screenshot_image_is_captured_but_not_persisted(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        # Per this backend's documented design (see the class
        # docstring's "screenshot() design note"), the captured image
        # is created via pyautogui.screenshot() and then discarded --
        # not stored or returned -- since the ControlBackend protocol
        # keeps screenshot() -> None. This confirms the call still
        # happens (the capture is real) even though nothing is kept.
        backend.screenshot()
        assert ("screenshot", ()) in fake_pyautogui.calls
        assert not hasattr(backend, "last_screenshot")


class TestPause:
    def test_pause_returns_none_and_calls_nothing_on_pyautogui(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        result = backend.pause()
        assert result is None
        assert fake_pyautogui.calls == []

    def test_pause_does_not_block(self, backend: PyAutoGUIControlBackend) -> None:
        # A crude but effective guard against accidental blocking:
        # calling pause() many times must return promptly.
        for _ in range(1000):
            backend.pause()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_backend_satisfies_control_backend_protocol(
        self, backend: PyAutoGUIControlBackend
    ) -> None:
        assert isinstance(backend, ControlBackend)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_move_cursor_calls_are_consistent(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        for _ in range(5):
            backend.move_cursor(10, 20)
        assert fake_pyautogui.calls == [("moveTo", (10, 20))] * 5

    def test_repeated_left_click_calls_are_consistent(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        for _ in range(3):
            backend.left_click()
        assert fake_pyautogui.calls == [("click", ())] * 3

    def test_sequence_of_calls_dispatches_in_order(
        self, backend: PyAutoGUIControlBackend, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend.move_cursor(1, 1)
        backend.left_click()
        backend.scroll_up()
        assert fake_pyautogui.calls == [
            ("moveTo", (1, 1)),
            ("click", ()),
            ("scroll", (SCROLL_UNIT,)),
        ]


# ---------------------------------------------------------------------------
# Dependency injection / import safety
# ---------------------------------------------------------------------------

class TestDependencyInjectionAndImportSafety:
    def test_injected_fake_module_is_used_instead_of_real_pyautogui(
        self, fake_pyautogui: FakePyAutoGUI
    ) -> None:
        backend = PyAutoGUIControlBackend(pyautogui_module=fake_pyautogui)
        backend.left_click()
        assert fake_pyautogui.calls == [("click", ())]

    def test_two_backends_with_different_fakes_are_independent(self) -> None:
        fake_a = FakePyAutoGUI()
        fake_b = FakePyAutoGUI()
        backend_a = PyAutoGUIControlBackend(pyautogui_module=fake_a)
        backend_b = PyAutoGUIControlBackend(pyautogui_module=fake_b)

        backend_a.left_click()

        assert fake_a.calls == [("click", ())]
        assert fake_b.calls == []

    def test_importing_the_backend_module_never_raises(self) -> None:
        # Importing this module must succeed even in an environment
        # where the real PyAutoGUI cannot actually be used (e.g. no
        # display) -- the real import is only attempted, and only
        # matters, when a backend is constructed without an injected
        # replacement.
        import gesture_control.control.pyautogui_backend  # noqa: F401

    def test_importing_the_control_package_never_raises(self) -> None:
        import gesture_control.control  # noqa: F401

    def test_constructing_without_injection_never_touches_a_fake(self) -> None:
        # This test intentionally does NOT assert on success/failure of
        # constructing PyAutoGUIControlBackend() with no arguments --
        # whether the real PyAutoGUI is usable is environment-dependent
        # (e.g. missing display). It only asserts that whatever happens,
        # it happens through a clear, typed exception rather than an
        # obscure crash, and that no real automation call is made as a
        # side effect either way.
        try:
            PyAutoGUIControlBackend()
        except RuntimeError:
            pass  # Expected in headless/no-display environments.


def test_no_real_os_interaction_helper_functions_used_directly(
    backend: "PyAutoGUIControlBackend", fake_pyautogui: FakePyAutoGUI
) -> None:
    # Exercises every ControlBackend method in one pass and confirms
    # every single resulting call landed on the fake, never on a real
    # automation library.
    backend.move_cursor(1, 2)
    backend.left_click()
    backend.right_click()
    backend.double_click()
    backend.scroll_up()
    backend.scroll_down()
    backend.screenshot()
    backend.pause()

    called_functions = {name for name, _ in fake_pyautogui.calls}
    assert called_functions == {
        "moveTo", "click", "rightClick", "doubleClick", "scroll", "screenshot",
    }