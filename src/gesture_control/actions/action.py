"""Abstract computer-control actions.

`Action` represents WHAT should happen as a result of a recognized
gesture -- it carries no information about HOW that happens.
"""

from enum import Enum


class Action(str, Enum):
    """The set of abstract actions the system can currently represent."""

    NONE = "none"
    MOVE_CURSOR = "move_cursor"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    SCREENSHOT = "screenshot"
    PAUSE = "pause"