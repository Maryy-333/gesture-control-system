"""Generic geometric utilities for working with hand landmark coordinates.

This module provides small, reusable, purely mathematical helpers:
Euclidean distance in 2D and 3D, the angle formed by three points, and a
generic distance-normalization helper. It has no knowledge of hands,
fingers, gestures, or any application concerns -- it operates on plain
coordinate tuples and would be equally usable for any other landmark or
point-cloud data.

There is no dependency on OpenCV or the MediaPipe runtime; callers are
expected to extract `(x, y)` / `(x, y, z)` coordinates from whatever
landmark objects they have (e.g. MediaPipe's `NormalizedLandmark`)
before calling into this module.
"""

import math
from typing import Tuple

# A 2D point as (x, y).
Point2D = Tuple[float, float]

# A 3D point as (x, y, z).
Point3D = Tuple[float, float, float]


def distance_2d(point_a: Point2D, point_b: Point2D) -> float:
    """Compute the Euclidean distance between two 2D points.

    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Args:
        point_a: The first point as (x1, y1).
        point_b: The second point as (x2, y2).

    Returns:
        The non-negative Euclidean distance between the two points.
    """
    x1, y1 = point_a
    x2, y2 = point_b
    return math.hypot(x2 - x1, y2 - y1)


def distance_3d(point_a: Point3D, point_b: Point3D) -> float:
    """Compute the Euclidean distance between two 3D points.

    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

    Args:
        point_a: The first point as (x1, y1, z1).
        point_b: The second point as (x2, y2, z2).

    Returns:
        The non-negative Euclidean distance between the two points.
    """
    x1, y1, z1 = point_a
    x2, y2, z2 = point_b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def angle_between_points(point_a: Point3D, vertex: Point3D, point_c: Point3D) -> float:
    """Compute the angle A-vertex-C, in degrees, with `vertex` as the corner.

    This treats each input as a 3D point (x, y, z). Plain 2D points can
    be passed by using z=0.0 for all three points.

    The angle is computed from the two vectors that originate at the
    vertex:
        u = point_a - vertex
        v = point_c - vertex
        angle = degrees(acos((u . v) / (|u| * |v|)))

    Args:
        point_a: One endpoint of the angle, as (x, y, z).
        vertex: The vertex of the angle (the "corner" point), as (x, y, z).
        point_c: The other endpoint of the angle, as (x, y, z).

    Returns:
        The angle A-vertex-C in degrees, in the range [0, 180].

    Raises:
        ValueError: If `point_a` or `point_c` coincides with `vertex`,
            since the angle is undefined when either vector has zero
            length.
    """
    ux, uy, uz = (point_a[i] - vertex[i] for i in range(3))
    vx, vy, vz = (point_c[i] - vertex[i] for i in range(3))

    u_magnitude = math.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    v_magnitude = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    if u_magnitude == 0.0 or v_magnitude == 0.0:
        raise ValueError(
            "Cannot compute angle: point_a or point_c coincides with vertex."
        )

    dot_product = ux * vx + uy * vy + uz * vz
    cosine = dot_product / (u_magnitude * v_magnitude)

    # Clamp to [-1, 1] to guard against floating-point drift pushing the
    # value slightly outside the domain of acos.
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def normalize_distance(distance: float, reference_scale: float) -> float:
    """Express a distance as a ratio of a reference scale.

    This is a generic helper for comparing raw distances on a scale that
    is independent of overall size (e.g. how far apart two landmarks
    are relative to a reference length such as palm width). It carries
    no knowledge of what the distance or reference represent.

    Formula: distance / reference_scale

    Args:
        distance: The raw distance to normalize. Must be non-negative.
        reference_scale: The reference length to normalize against.
            Must be strictly positive.

    Returns:
        The ratio `distance / reference_scale`.

    Raises:
        ValueError: If `distance` is negative, or if `reference_scale`
            is not strictly positive.
    """
    if distance < 0:
        raise ValueError("distance must be non-negative.")
    if reference_scale <= 0:
        raise ValueError("reference_scale must be strictly positive.")

    return distance / reference_scale