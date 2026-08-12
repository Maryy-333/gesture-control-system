"""Unit tests for gesture_control.gestures.geometry."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.gestures.geometry import (
    angle_between_points,
    distance_2d,
    distance_3d,
    normalize_distance,
)


# ---------------------------------------------------------------------------
# distance_2d
# ---------------------------------------------------------------------------

class TestDistance2D:
    def test_horizontal_distance(self) -> None:
        assert distance_2d((0.0, 0.0), (3.0, 0.0)) == pytest.approx(3.0)

    def test_vertical_distance(self) -> None:
        assert distance_2d((0.0, 0.0), (0.0, 4.0)) == pytest.approx(4.0)

    def test_classic_3_4_5_triangle(self) -> None:
        assert distance_2d((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_zero_distance_same_point(self) -> None:
        assert distance_2d((1.5, -2.5), (1.5, -2.5)) == pytest.approx(0.0)

    def test_negative_coordinates(self) -> None:
        assert distance_2d((-1.0, -1.0), (2.0, 3.0)) == pytest.approx(5.0)

    def test_distance_is_symmetric(self) -> None:
        a = (1.0, 2.0)
        b = (4.0, 6.0)
        assert distance_2d(a, b) == pytest.approx(distance_2d(b, a))

    def test_distance_is_non_negative(self) -> None:
        assert distance_2d((10.0, 10.0), (0.0, 0.0)) >= 0.0


# ---------------------------------------------------------------------------
# distance_3d
# ---------------------------------------------------------------------------

class TestDistance3D:
    def test_zero_distance_same_point(self) -> None:
        assert distance_3d((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(0.0)

    def test_single_axis_distance(self) -> None:
        assert distance_3d((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)) == pytest.approx(5.0)

    def test_classic_pythagorean_triple_in_3d(self) -> None:
        # sqrt(1^2 + 2^2 + 2^2) == 3
        assert distance_3d((0.0, 0.0, 0.0), (1.0, 2.0, 2.0)) == pytest.approx(3.0)

    def test_negative_coordinates(self) -> None:
        assert distance_3d((-1.0, -1.0, -1.0), (2.0, 3.0, -1.0)) == pytest.approx(5.0)

    def test_distance_is_symmetric(self) -> None:
        a = (0.5, 1.5, 2.5)
        b = (3.5, -1.0, 4.0)
        assert distance_3d(a, b) == pytest.approx(distance_3d(b, a))

    def test_reduces_to_2d_when_z_equal(self) -> None:
        a = (0.0, 0.0, 7.0)
        b = (3.0, 4.0, 7.0)
        assert distance_3d(a, b) == pytest.approx(distance_2d((0.0, 0.0), (3.0, 4.0)))


# ---------------------------------------------------------------------------
# angle_between_points
# ---------------------------------------------------------------------------

class TestAngleBetweenPoints:
    def test_right_angle(self) -> None:
        # A=(1,0,0), vertex=(0,0,0), C=(0,1,0) -> 90 degrees
        angle = angle_between_points((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert angle == pytest.approx(90.0)

    def test_straight_line_is_180_degrees(self) -> None:
        # A, vertex, C are collinear with vertex in the middle.
        angle = angle_between_points((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert angle == pytest.approx(180.0)

    def test_zero_angle_when_points_overlap_direction(self) -> None:
        # A and C are in the same direction from the vertex -> 0 degrees.
        angle = angle_between_points((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        assert angle == pytest.approx(0.0)

    def test_45_degree_angle(self) -> None:
        angle = angle_between_points((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
        assert angle == pytest.approx(45.0)

    def test_angle_is_symmetric_in_endpoints(self) -> None:
        a = (1.0, 0.0, 0.0)
        vertex = (0.0, 0.0, 0.0)
        c = (0.0, 1.0, 1.0)
        assert angle_between_points(a, vertex, c) == pytest.approx(
            angle_between_points(c, vertex, a)
        )

    def test_angle_with_3d_component(self) -> None:
        # A directly "above" vertex on z axis, C directly "right" on x axis.
        angle = angle_between_points((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert angle == pytest.approx(90.0)

    def test_raises_when_point_a_coincides_with_vertex(self) -> None:
        with pytest.raises(ValueError):
            angle_between_points((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_raises_when_point_c_coincides_with_vertex(self) -> None:
        with pytest.raises(ValueError):
            angle_between_points((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_result_is_never_nan(self) -> None:
        # Points that could produce floating point drift just past
        # the domain of acos should still return a valid, real angle.
        a = (1.0, 1.0, 1.0)
        vertex = (0.0, 0.0, 0.0)
        c = (1.0, 1.0, 1.0)  # same direction as a -> cosine should clamp to 1.0
        angle = angle_between_points(a, vertex, c)
        assert not math.isnan(angle)
        assert angle == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# normalize_distance
# ---------------------------------------------------------------------------

class TestNormalizeDistance:
    def test_normalizes_to_expected_ratio(self) -> None:
        assert normalize_distance(5.0, 10.0) == pytest.approx(0.5)

    def test_distance_equal_to_scale_is_one(self) -> None:
        assert normalize_distance(4.0, 4.0) == pytest.approx(1.0)

    def test_zero_distance_normalizes_to_zero(self) -> None:
        assert normalize_distance(0.0, 10.0) == pytest.approx(0.0)

    def test_distance_larger_than_scale_exceeds_one(self) -> None:
        assert normalize_distance(20.0, 10.0) == pytest.approx(2.0)

    def test_raises_on_negative_distance(self) -> None:
        with pytest.raises(ValueError):
            normalize_distance(-1.0, 10.0)

    def test_raises_on_zero_reference_scale(self) -> None:
        with pytest.raises(ValueError):
            normalize_distance(5.0, 0.0)

    def test_raises_on_negative_reference_scale(self) -> None:
        with pytest.raises(ValueError):
            normalize_distance(5.0, -2.0)