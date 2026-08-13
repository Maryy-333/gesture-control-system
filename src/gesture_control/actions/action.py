"""Abstract computer-control actions.

`Action` represents WHAT should happen as a result of a recognized
gesture -- it carries no information about HOW that happens. There is
no mouse, keyboard, OS, or GUI code here, and none of these values are
wired up to any real input-control mechanism yet; that is a later,
separate layer.
"""

from enum import Enum


class Action(str, Enum):
    """The set of abstract actions the system can currently represent.

    Values are plain lowercase strings (rather than auto-generated
    numbers) so they remain stable and are safe to use directly in
    logs, config files, or serialized output.
    """

    NONE = "none"
    MOVE_CURSOR = "move_cursor"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    SCREENSHOT = "screenshot"
    PAUSE = "pause"