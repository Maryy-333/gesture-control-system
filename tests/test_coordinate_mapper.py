"""Unit tests for gesture_control.mapping.coordinate_mapper.

These tests are completely independent of any webcam, MediaPipe,
OpenCV, PyAutoGUI, real screen, or operating system -- they operate
purely on plain numeric input and fixed `ScreenSize` values.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.mapping import CoordinateMapper, ScreenPoint, ScreenSize


# ---------------------------------------------------------------------------
# Basic corner / center mapping
# ---------------------------------------------------------------------------

class TestBasicMapping:
    def test_minimum_coordinates_map_to_top_left(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        assert mapper.map_point(0.0, 0.0) == ScreenPoint(0, 0)

    def test_maximum_coordinates_map_to_bottom_right(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        assert mapper.map_point(1.0, 1.0) == ScreenPoint(1919, 1079)

    def test_center_coordinates_map_approximately_to_center(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(0.5, 0.5)
        assert result.x == pytest.approx(1920 / 2, abs=1)
        assert result.y == pytest.approx(1080 / 2, abs=1)

    def test_1919_1079_boundary_example_from_spec(self) -> None:
        mapper = CoordinateMapper(ScreenSize(width=1920, height=1080))
        assert mapper.map_point(0.0, 0.0) == ScreenPoint(0, 0)
        assert mapper.map_point(1.0, 1.0) == ScreenPoint(1919, 1079)

    def test_1_0_never_maps_to_width_or_height_itself(self) -> None:
        mapper = CoordinateMapper(ScreenSize(100, 50))
        result = mapper.map_point(1.0, 1.0)
        assert result.x != 100
        assert result.y != 50
        assert result.x == 99
        assert result.y == 49


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

class TestClamping:
    def test_x_below_range_is_clamped_to_minimum(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(-0.2, 0.5)
        assert result.x == 0

    def test_x_above_range_is_clamped_to_maximum(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(1.2, 0.5)
        assert result.x == 1919

    def test_y_below_range_is_clamped_to_minimum(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(0.5, -0.2)
        assert result.y == 0

    def test_y_above_range_is_clamped_to_maximum(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(0.5, 1.2)
        assert result.y == 1079

    def test_extreme_out_of_range_values_are_still_clamped(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(-1000.0, 1000.0)
        assert result == ScreenPoint(0, 1079)

    @pytest.mark.parametrize(
        "x,y",
        [
            (-5.0, -5.0),
            (5.0, 5.0),
            (-5.0, 5.0),
            (5.0, -5.0),
            (0.0, 5.0),
            (5.0, 0.0),
        ],
    )
    def test_output_is_always_inside_screen_bounds(self, x: float, y: float) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(x, y)
        assert 0 <= result.x < 1920
        assert 0 <= result.y < 1080


# ---------------------------------------------------------------------------
# Axis inversion
# ---------------------------------------------------------------------------

class TestAxisInversion:
    def test_invert_x_flips_the_x_axis(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080), invert_x=True)
        assert mapper.map_point(0.0, 0.0).x == 1919
        assert mapper.map_point(1.0, 0.0).x == 0

    def test_invert_y_flips_the_y_axis(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080), invert_y=True)
        assert mapper.map_point(0.0, 0.0).y == 1079
        assert mapper.map_point(0.0, 1.0).y == 0

    def test_both_axes_can_be_inverted_independently(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080), invert_x=True, invert_y=True)
        result = mapper.map_point(0.0, 0.0)
        assert result == ScreenPoint(1919, 1079)
        result2 = mapper.map_point(1.0, 1.0)
        assert result2 == ScreenPoint(0, 0)

    def test_inversion_defaults_to_false(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        # Not inverted: x=0.0 stays at the left edge, not the right.
        assert mapper.map_point(0.0, 0.0).x == 0
        assert mapper.map_point(0.0, 0.0).y == 0

    def test_x_inversion_does_not_affect_y(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080), invert_x=True)
        assert mapper.map_point(0.0, 1.0).y == 1079


# ---------------------------------------------------------------------------
# Custom input range
# ---------------------------------------------------------------------------

class TestCustomInputRange:
    def test_custom_range_minimum_maps_to_screen_minimum(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1000, 500), input_min=-1.0, input_max=1.0)
        assert mapper.map_point(-1.0, -1.0) == ScreenPoint(0, 0)

    def test_custom_range_maximum_maps_to_screen_maximum(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1000, 500), input_min=-1.0, input_max=1.0)
        assert mapper.map_point(1.0, 1.0) == ScreenPoint(999, 499)

    def test_custom_range_zero_maps_to_screen_center(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1000, 500), input_min=-1.0, input_max=1.0)
        result = mapper.map_point(0.0, 0.0)
        assert result.x == pytest.approx(1000 / 2, abs=1)
        assert result.y == pytest.approx(500 / 2, abs=1)

    def test_custom_range_clamps_outside_values(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1000, 500), input_min=-1.0, input_max=1.0)
        assert mapper.map_point(-2.0, 2.0) == ScreenPoint(0, 499)

    def test_arbitrary_positive_only_custom_range(self) -> None:
        mapper = CoordinateMapper(ScreenSize(200, 100), input_min=10.0, input_max=20.0)
        assert mapper.map_point(10.0, 10.0) == ScreenPoint(0, 0)
        assert mapper.map_point(20.0, 20.0) == ScreenPoint(199, 99)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_input_min_equal_to_input_max_raises(self) -> None:
        with pytest.raises(ValueError):
            CoordinateMapper(ScreenSize(1920, 1080), input_min=1.0, input_max=1.0)

    def test_input_min_greater_than_input_max_raises(self) -> None:
        with pytest.raises(ValueError):
            CoordinateMapper(ScreenSize(1920, 1080), input_min=2.0, input_max=1.0)

    def test_zero_screen_width_raises(self) -> None:
        with pytest.raises(ValueError):
            ScreenSize(0, 1080)

    def test_negative_screen_width_raises(self) -> None:
        with pytest.raises(ValueError):
            ScreenSize(-100, 1080)

    def test_zero_screen_height_raises(self) -> None:
        with pytest.raises(ValueError):
            ScreenSize(1920, 0)

    def test_negative_screen_height_raises(self) -> None:
        with pytest.raises(ValueError):
            ScreenSize(1920, -100)

    def test_non_numeric_input_raises_type_error(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        with pytest.raises((TypeError, ValueError)):
            mapper.map_point("not-a-number", 0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

class TestRounding:
    def test_rounding_is_deterministic_not_banker_rounding(self) -> None:
        # A screen size chosen so several .5 boundary cases arise, to
        # confirm consistent "round half up" behavior rather than
        # Python's round-half-to-even.
        mapper = CoordinateMapper(ScreenSize(3, 3))
        # dimension - 1 = 2; fraction 0.25 -> scaled = 0.5 -> rounds to 1
        result = mapper.map_point(0.25, 0.25)
        assert result == ScreenPoint(1, 1)

    def test_result_coordinates_are_ints(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        result = mapper.map_point(0.3333, 0.6667)
        assert isinstance(result.x, int)
        assert isinstance(result.y, int)

    def test_final_result_is_clamped_to_valid_range_after_rounding(self) -> None:
        mapper = CoordinateMapper(ScreenSize(10, 10))
        result = mapper.map_point(1.0, 1.0)
        assert 0 <= result.x < 10
        assert 0 <= result.y < 10


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_calls_with_same_input_produce_same_output(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        results = {mapper.map_point(0.3333, 0.6667) for _ in range(20)}
        assert len(results) == 1

    def test_two_mappers_with_identical_config_agree(self) -> None:
        mapper_a = CoordinateMapper(ScreenSize(1920, 1080))
        mapper_b = CoordinateMapper(ScreenSize(1920, 1080))
        assert mapper_a.map_point(0.4, 0.6) == mapper_b.map_point(0.4, 0.6)

    def test_accepts_int_and_float_input_equivalently(self) -> None:
        mapper = CoordinateMapper(ScreenSize(1920, 1080))
        assert mapper.map_point(0, 1) == mapper.map_point(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicApi:
    def test_public_imports_from_gesture_control_mapping_work(self) -> None:
        from gesture_control.mapping import CoordinateMapper as ImportedMapper
        from gesture_control.mapping import ScreenPoint as ImportedPoint
        from gesture_control.mapping import ScreenSize as ImportedSize

        mapper = ImportedMapper(ImportedSize(1920, 1080))
        assert mapper.map_point(0.0, 0.0) == ImportedPoint(0, 0)

    def test_screen_size_and_screen_point_are_immutable(self) -> None:
        size = ScreenSize(1920, 1080)
        with pytest.raises(Exception):
            size.width = 100  # type: ignore[misc]

        point = ScreenPoint(0, 0)
        with pytest.raises(Exception):
            point.x = 100  # type: ignore[misc]