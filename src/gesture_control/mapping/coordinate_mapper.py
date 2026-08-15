"""Transport-independent normalized-coordinate-to-screen-pixel mapping.

This module converts a normalized hand-tracking coordinate (e.g. from
`DetectedHand.landmarks`) into a discrete screen pixel coordinate. It
has no knowledge of `Gesture`, `Action`, `FingerStateDetector`,
`GestureRecognizer`, any hand tracker, MediaPipe, `ComputerController`,
PyAutoGUI, or webcam frames -- it only knows a numeric input coordinate,
a configured input range, a screen size, and optional axis inversion.

Pure Python: no OpenCV, MediaPipe, PyAutoGUI, GUI toolkit, or
OS-specific library is imported here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenSize:
    """The dimensions of the target screen, in pixels.

    Raises:
        ValueError: If `width` or `height` is not a positive integer.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width!r}.")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height!r}.")


@dataclass(frozen=True)
class ScreenPoint:
    """A discrete screen pixel coordinate."""

    x: int
    y: int


class CoordinateMapper:
    """Maps normalized coordinates to screen pixel coordinates.

    The valid screen coordinate range is inclusive-exclusive of the
    screen dimensions themselves: `0 <= x < screen_size.width` and
    `0 <= y < screen_size.height`. An input at `input_max` maps to
    `dimension - 1`, never to `dimension` itself.

    Example:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        mapper.map_point(0.0, 0.0)  # -> ScreenPoint(0, 0)
        mapper.map_point(1.0, 1.0)  # -> ScreenPoint(1919, 1079)
        mapper.map_point(0.5, 0.5)  # -> approximately screen center
    """

    def __init__(
        self,
        screen_size: ScreenSize,
        *,
        input_min: float = 0.0,
        input_max: float = 1.0,
        invert_x: bool = False,
        invert_y: bool = False,
    ) -> None:
        """Initialize the mapper.

        Args:
            screen_size: The target screen's dimensions.
            input_min: The smallest expected input coordinate value.
            input_max: The largest expected input coordinate value.
            invert_x: If True, `input_min` maps to the right edge and
                `input_max` maps to the left edge instead of the usual
                left-to-right direction.
            invert_y: If True, `input_min` maps to the bottom edge and
                `input_max` maps to the top edge instead of the usual
                top-to-bottom direction.

        Raises:
            ValueError: If `input_min >= input_max`.
        """
        if input_min >= input_max:
            raise ValueError(
                f"input_min ({input_min!r}) must be less than "
                f"input_max ({input_max!r})."
            )

        self._screen_size = screen_size
        self._input_min = input_min
        self._input_max = input_max
        self._invert_x = invert_x
        self._invert_y = invert_y

    def map_point(self, x: float, y: float) -> ScreenPoint:
        """Convert a normalized `(x, y)` coordinate to a `ScreenPoint`.

        Input values outside the configured `[input_min, input_max]`
        range are clamped rather than rejected, since real hand-tracking
        coordinates can occasionally fall slightly outside the expected
        range.

        Args:
            x: The input x coordinate. Converted to `float` internally.
            y: The input y coordinate. Converted to `float` internally.

        Returns:
            A `ScreenPoint` with `0 <= x < screen_size.width` and
            `0 <= y < screen_size.height`.
        """
        screen_x = self._map_axis(
            float(x), dimension=self._screen_size.width, invert=self._invert_x
        )
        screen_y = self._map_axis(
            float(y), dimension=self._screen_size.height, invert=self._invert_y
        )
        return ScreenPoint(x=screen_x, y=screen_y)

    def _map_axis(self, value: float, dimension: int, invert: bool) -> int:
        """Map one clamped, normalized axis value to a pixel index.

        Args:
            value: The raw input value for this axis.
            dimension: The screen size (width or height) for this axis.
            invert: Whether to invert this axis's direction.

        Returns:
            An integer pixel index in `[0, dimension - 1]`.
        """
        clamped = min(max(value, self._input_min), self._input_max)
        fraction = (clamped - self._input_min) / (self._input_max - self._input_min)

        if invert:
            fraction = 1.0 - fraction

        # A fraction of 1.0 must land on `dimension - 1`, never
        # `dimension` -- the last valid pixel index, not one past it.
        scaled = fraction * (dimension - 1)

        # Deterministic rounding (not Python's banker's-rounding
        # `round()`): add 0.5 and truncate toward zero.
        rounded = int(scaled + 0.5)

        # Final safety clamp against any floating-point drift at the
        # boundaries, so the result is always a valid pixel index.
        return min(max(rounded, 0), dimension - 1)