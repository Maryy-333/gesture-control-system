"""Unit tests for gesture_control.gestures.finger_states.

All landmark data here is synthetic and constructed directly from the
joint-angle geometry the detector relies on, so these tests require no
webcam, no MediaPipe runtime, and no image data.
"""

import math
import os
import sys
from typing import Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gesture_control.gestures.finger_states import (
    DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES,
    DEFAULT_THUMB_STRAIGHTNESS_THRESHOLD_DEGREES,
    NUM_HAND_LANDMARKS,
    FingerStateDetector,
    FingerStates,
    HandLandmark,
)

Point3D = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Helpers for building synthetic 21-point landmark sets.
# ---------------------------------------------------------------------------

def _default_landmarks() -> List[Point3D]:
    """Build 21 placeholder points that are never coincident with each other.

    `detect()` always computes an angle for every finger, so even
    landmarks a given test doesn't care about must form valid (non-zero,
    non-degenerate) vectors. Each placeholder point is offset by its own
    index so that no three of them can accidentally coincide.
    """
    return [(index * 10.0, index * 7.0, index * 3.0) for index in range(NUM_HAND_LANDMARKS)]


def _landmarks_with(overrides: Dict[HandLandmark, Point3D]) -> List[Point3D]:
    """Build a full 21-point landmark list, using placeholders except `overrides`.

    Only the landmarks actually read by the detector for a given finger
    matter for what a test asserts on; unrelated landmarks are filled
    with non-degenerate placeholders so their (irrelevant) angle
    computations don't raise.
    """
    landmarks = _default_landmarks()
    for index, point in overrides.items():
        landmarks[index] = point
    return landmarks


def _straight_chain(
    base: Point3D, step: Point3D, count: int
) -> List[Point3D]:
    """Return `count` collinear points starting at `base`, moving by `step`."""
    return [
        (base[0] + step[0] * i, base[1] + step[1] * i, base[2] + step[2] * i)
        for i in range(count)
    ]


def _bent_chain(mcp: Point3D, pip: Point3D) -> Point3D:
    """Return a DIP point that bends ~90 degrees at PIP relative to MCP.

    Given MCP -> PIP as one vector, returns a point perpendicular to it
    (in the XY plane) so that the MCP -> PIP -> DIP angle is ~90 degrees,
    simulating a folded finger.
    """
    dx = pip[0] - mcp[0]
    dy = pip[1] - mcp[1]
    # Perpendicular vector in-plane: (dy, -dx) rotated 90 degrees.
    perp = (-dy, dx, 0.0)
    return (pip[0] + perp[0], pip[1] + perp[1], pip[2] + perp[2])


def _extended_long_finger(
    mcp_index: HandLandmark,
    pip_index: HandLandmark,
    dip_index: HandLandmark,
    tip_index: HandLandmark,
    base: Point3D = (0.5, 0.5, 0.0),
    step: Point3D = (0.0, -0.1, 0.0),
) -> Dict[HandLandmark, Point3D]:
    """Build a collinear (fully extended) chain of MCP/PIP/DIP/TIP points."""
    mcp, pip, dip, tip = _straight_chain(base, step, 4)
    return {mcp_index: mcp, pip_index: pip, dip_index: dip, tip_index: tip}


def _folded_long_finger(
    mcp_index: HandLandmark,
    pip_index: HandLandmark,
    dip_index: HandLandmark,
    tip_index: HandLandmark,
    base: Point3D = (0.5, 0.5, 0.0),
    step: Point3D = (0.0, -0.1, 0.0),
) -> Dict[HandLandmark, Point3D]:
    """Build a chain where the finger bends sharply (~90 deg) at the PIP."""
    mcp = base
    pip = (base[0] + step[0], base[1] + step[1], base[2] + step[2])
    dip = _bent_chain(mcp, pip)
    tip = dip  # TIP position is irrelevant to the detector's calculation.
    return {mcp_index: mcp, pip_index: pip, dip_index: dip, tip_index: tip}


def _extended_thumb(
    base: Point3D = (0.2, 0.5, 0.0), step: Point3D = (-0.1, 0.0, 0.0)
) -> Dict[HandLandmark, Point3D]:
    """Build a collinear (fully extended) thumb CMC/MCP/IP/TIP chain."""
    cmc, mcp, ip, tip = _straight_chain(base, step, 4)
    return {
        HandLandmark.THUMB_CMC: cmc,
        HandLandmark.THUMB_MCP: mcp,
        HandLandmark.THUMB_IP: ip,
        HandLandmark.THUMB_TIP: tip,
    }


def _folded_thumb(
    base: Point3D = (0.2, 0.5, 0.0), step: Point3D = (-0.1, 0.0, 0.0)
) -> Dict[HandLandmark, Point3D]:
    """Build a thumb chain bent ~90 degrees at the IP joint."""
    mcp = base
    ip = (base[0] + step[0], base[1] + step[1], base[2] + step[2])
    tip = _bent_chain(mcp, ip)
    return {
        HandLandmark.THUMB_CMC: (base[0] - step[0], base[1] - step[1], base[2]),
        HandLandmark.THUMB_MCP: mcp,
        HandLandmark.THUMB_IP: ip,
        HandLandmark.THUMB_TIP: tip,
    }


def _angled_point(vertex: Point3D, reference: Point3D, degrees: float, length: float = 0.1) -> Point3D:
    """Return a point `length` away from `vertex`, at `degrees` from `reference`.

    `reference` defines the zero-angle direction (vertex -> reference).
    The returned point lies in the XY plane at the requested angle from
    that direction, used to build precise boundary-angle test cases.
    """
    ref_dx = reference[0] - vertex[0]
    ref_dy = reference[1] - vertex[1]
    ref_len = math.hypot(ref_dx, ref_dy)
    ref_angle = math.atan2(ref_dy, ref_dx)

    theta = ref_angle + math.radians(degrees)
    return (
        vertex[0] + length * math.cos(theta),
        vertex[1] + length * math.sin(theta),
        vertex[2],
    )


FULL_EXTENDED_HAND_ANGLE = 180.0
FULL_FOLDED_HAND_ANGLE = 90.0


def _all_extended_overrides() -> Dict[HandLandmark, Point3D]:
    overrides: Dict[HandLandmark, Point3D] = {}
    overrides.update(_extended_thumb())
    overrides.update(
        _extended_long_finger(
            HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP,
            HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP,
            base=(0.4, 0.5, 0.0),
        )
    )
    overrides.update(
        _extended_long_finger(
            HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP,
            HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP,
            base=(0.5, 0.5, 0.0),
        )
    )
    overrides.update(
        _extended_long_finger(
            HandLandmark.RING_MCP, HandLandmark.RING_PIP,
            HandLandmark.RING_DIP, HandLandmark.RING_TIP,
            base=(0.6, 0.5, 0.0),
        )
    )
    overrides.update(
        _extended_long_finger(
            HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP,
            base=(0.7, 0.5, 0.0),
        )
    )
    return overrides


def _all_folded_overrides() -> Dict[HandLandmark, Point3D]:
    overrides: Dict[HandLandmark, Point3D] = {}
    overrides.update(_folded_thumb())
    overrides.update(
        _folded_long_finger(
            HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP,
            HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP,
            base=(0.4, 0.5, 0.0),
        )
    )
    overrides.update(
        _folded_long_finger(
            HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP,
            HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP,
            base=(0.5, 0.5, 0.0),
        )
    )
    overrides.update(
        _folded_long_finger(
            HandLandmark.RING_MCP, HandLandmark.RING_PIP,
            HandLandmark.RING_DIP, HandLandmark.RING_TIP,
            base=(0.6, 0.5, 0.0),
        )
    )
    overrides.update(
        _folded_long_finger(
            HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP,
            base=(0.7, 0.5, 0.0),
        )
    )
    return overrides


# ---------------------------------------------------------------------------
# 1. Clearly extended fingers / 2. Clearly folded fingers
# ---------------------------------------------------------------------------

class TestClearlyExtendedAndFolded:
    def test_all_fingers_extended(self) -> None:
        detector = FingerStateDetector()
        landmarks = _landmarks_with(_all_extended_overrides())
        states = detector.detect(landmarks)
        assert states == FingerStates(
            thumb=True, index=True, middle=True, ring=True, pinky=True
        )

    def test_all_fingers_folded(self) -> None:
        detector = FingerStateDetector()
        landmarks = _landmarks_with(_all_folded_overrides())
        states = detector.detect(landmarks)
        assert states == FingerStates(
            thumb=False, index=False, middle=False, ring=False, pinky=False
        )


# ---------------------------------------------------------------------------
# 3. Mixed finger states
# ---------------------------------------------------------------------------

class TestMixedFingerStates:
    def test_index_and_middle_extended_ring_and_pinky_folded(self) -> None:
        detector = FingerStateDetector()
        overrides: Dict[HandLandmark, Point3D] = {}
        overrides.update(_folded_thumb())
        overrides.update(
            _extended_long_finger(
                HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP,
                HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP,
                base=(0.4, 0.5, 0.0),
            )
        )
        overrides.update(
            _extended_long_finger(
                HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP,
                HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP,
                base=(0.5, 0.5, 0.0),
            )
        )
        overrides.update(
            _folded_long_finger(
                HandLandmark.RING_MCP, HandLandmark.RING_PIP,
                HandLandmark.RING_DIP, HandLandmark.RING_TIP,
                base=(0.6, 0.5, 0.0),
            )
        )
        overrides.update(
            _folded_long_finger(
                HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
                HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP,
                base=(0.7, 0.5, 0.0),
            )
        )
        landmarks = _landmarks_with(overrides)
        states = detector.detect(landmarks)
        assert states == FingerStates(
            thumb=False, index=True, middle=True, ring=False, pinky=False
        )

    def test_only_pinky_extended(self) -> None:
        detector = FingerStateDetector()
        overrides: Dict[HandLandmark, Point3D] = {}
        overrides.update(_folded_thumb())
        for mcp, pip, dip, tip, base in [
            (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP, (0.4, 0.5, 0.0)),
            (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP, (0.5, 0.5, 0.0)),
            (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP, (0.6, 0.5, 0.0)),
        ]:
            overrides.update(_folded_long_finger(mcp, pip, dip, tip, base=base))
        overrides.update(
            _extended_long_finger(
                HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
                HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP,
                base=(0.7, 0.5, 0.0),
            )
        )
        landmarks = _landmarks_with(overrides)
        states = detector.detect(landmarks)
        assert states == FingerStates(
            thumb=False, index=False, middle=False, ring=False, pinky=True
        )


# ---------------------------------------------------------------------------
# 4. Different hand orientations (rotation / translation invariance)
# ---------------------------------------------------------------------------

class TestOrientationStability:
    @staticmethod
    def _rotate_z(point: Point3D, degrees: float, origin: Point3D = (0.0, 0.0, 0.0)) -> Point3D:
        theta = math.radians(degrees)
        x, y, z = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
        rx = x * math.cos(theta) - y * math.sin(theta)
        ry = x * math.sin(theta) + y * math.cos(theta)
        return (rx + origin[0], ry + origin[1], z + origin[2])

    @staticmethod
    def _translate(point: Point3D, offset: Point3D) -> Point3D:
        return (point[0] + offset[0], point[1] + offset[1], point[2] + offset[2])

    def test_extended_hand_stays_extended_when_rotated_and_translated(self) -> None:
        detector = FingerStateDetector()
        base_overrides = _all_extended_overrides()

        rotated_overrides = {
            index: self._translate(self._rotate_z(point, 37.0), (2.0, -1.0, 0.5))
            for index, point in base_overrides.items()
        }

        landmarks = _landmarks_with(rotated_overrides)
        states = detector.detect(landmarks)
        assert states == FingerStates(
            thumb=True, index=True, middle=True, ring=True, pinky=True
        )

    def test_folded_hand_stays_folded_when_rotated_and_translated(self) -> None:
        detector = FingerStateDetector()
        base_overrides = _all_folded_overrides()

        rotated_overrides = {
            index: self._translate(self._rotate_z(point, -63.0), (-5.0, 3.0, -2.0))
            for index, point in base_overrides.items()
        }

        landmarks = _landmarks_with(rotated_overrides)
        states = detector.detect(landmarks)
        assert states == FingerStates(
            thumb=False, index=False, middle=False, ring=False, pinky=False
        )


# ---------------------------------------------------------------------------
# 5. Invalid landmark count
# ---------------------------------------------------------------------------

class TestInvalidLandmarkCount:
    def test_too_few_landmarks_raises(self) -> None:
        detector = FingerStateDetector()
        landmarks = [(0.0, 0.0, 0.0)] * 20
        with pytest.raises(ValueError):
            detector.detect(landmarks)

    def test_too_many_landmarks_raises(self) -> None:
        detector = FingerStateDetector()
        landmarks = [(0.0, 0.0, 0.0)] * 22
        with pytest.raises(ValueError):
            detector.detect(landmarks)

    def test_empty_landmarks_raises(self) -> None:
        detector = FingerStateDetector()
        with pytest.raises(ValueError):
            detector.detect([])


# ---------------------------------------------------------------------------
# 6. Boundary behavior around the angle threshold
# ---------------------------------------------------------------------------

class TestThresholdBoundary:
    def test_index_angle_exactly_at_threshold_counts_as_extended(self) -> None:
        threshold = DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES
        detector = FingerStateDetector(finger_threshold_degrees=threshold)
        mcp = (0.5, 0.6, 0.0)
        pip = (0.5, 0.5, 0.0)
        # `degrees` is the resulting MCP-PIP-DIP angle itself (see
        # `_angled_point`), so passing the threshold directly produces
        # a finger angle exactly equal to it.
        dip = _angled_point(vertex=pip, reference=mcp, degrees=threshold)
        overrides = {
            HandLandmark.INDEX_MCP: mcp,
            HandLandmark.INDEX_PIP: pip,
            HandLandmark.INDEX_DIP: dip,
            HandLandmark.INDEX_TIP: dip,
        }
        landmarks = _landmarks_with(overrides)
        states = detector.detect(landmarks)
        assert states.index is True

    def test_index_angle_just_below_threshold_counts_as_folded(self) -> None:
        threshold = DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES
        detector = FingerStateDetector(finger_threshold_degrees=threshold)
        mcp = (0.5, 0.6, 0.0)
        pip = (0.5, 0.5, 0.0)
        dip = _angled_point(vertex=pip, reference=mcp, degrees=threshold - 1.0)
        overrides = {
            HandLandmark.INDEX_MCP: mcp,
            HandLandmark.INDEX_PIP: pip,
            HandLandmark.INDEX_DIP: dip,
            HandLandmark.INDEX_TIP: dip,
        }
        landmarks = _landmarks_with(overrides)
        states = detector.detect(landmarks)
        assert states.index is False

    def test_index_angle_just_above_threshold_counts_as_extended(self) -> None:
        threshold = DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES
        detector = FingerStateDetector(finger_threshold_degrees=threshold)
        mcp = (0.5, 0.6, 0.0)
        pip = (0.5, 0.5, 0.0)
        dip = _angled_point(
            vertex=pip, reference=mcp, degrees=min(threshold + 1.0, 180.0)
        )
        overrides = {
            HandLandmark.INDEX_MCP: mcp,
            HandLandmark.INDEX_PIP: pip,
            HandLandmark.INDEX_DIP: dip,
            HandLandmark.INDEX_TIP: dip,
        }
        landmarks = _landmarks_with(overrides)
        states = detector.detect(landmarks)
        assert states.index is True

    def test_custom_threshold_is_respected(self) -> None:
        # A finger bent to a 165-degree PIP angle should read as extended
        # under a lenient (160 degree) threshold, but folded under a
        # stricter (178 degree) threshold -- the same geometry, two
        # different conclusions, purely from the configured threshold.
        mcp = (0.5, 0.6, 0.0)
        pip = (0.5, 0.5, 0.0)
        dip = _angled_point(vertex=pip, reference=mcp, degrees=165.0)
        overrides = {
            HandLandmark.INDEX_MCP: mcp,
            HandLandmark.INDEX_PIP: pip,
            HandLandmark.INDEX_DIP: dip,
            HandLandmark.INDEX_TIP: dip,
        }
        landmarks = _landmarks_with(overrides)

        lenient_detector = FingerStateDetector(finger_threshold_degrees=160.0)
        strict_detector = FingerStateDetector(finger_threshold_degrees=178.0)
        assert lenient_detector.detect(landmarks).index is True
        assert strict_detector.detect(landmarks).index is False


# ---------------------------------------------------------------------------
# 7. Thumb behavior separately from the four long fingers
# ---------------------------------------------------------------------------

class TestThumbHandledSeparately:
    def test_thumb_extended_independent_of_other_fingers(self) -> None:
        detector = FingerStateDetector()
        overrides: Dict[HandLandmark, Point3D] = {}
        overrides.update(_extended_thumb())
        for mcp, pip, dip, tip, base in [
            (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP, (0.4, 0.5, 0.0)),
            (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP, (0.5, 0.5, 0.0)),
            (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP, (0.6, 0.5, 0.0)),
            (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP, (0.7, 0.5, 0.0)),
        ]:
            overrides.update(_folded_long_finger(mcp, pip, dip, tip, base=base))
        landmarks = _landmarks_with(overrides)
        states = detector.detect(landmarks)
        assert states.thumb is True
        assert states.index is False
        assert states.middle is False
        assert states.ring is False
        assert states.pinky is False

    def test_thumb_uses_its_own_lower_threshold_than_long_fingers(self) -> None:
        # Build a thumb bent so its IP angle sits strictly between the
        # thumb threshold and the (higher) long-finger threshold. Under
        # the thumb's own default threshold it should read as extended;
        # if the long-finger threshold were mistakenly applied to it,
        # it would read as folded. This demonstrates the two rules are
        # genuinely independent, not just a copy-pasted constant.
        assert (
            DEFAULT_THUMB_STRAIGHTNESS_THRESHOLD_DEGREES
            < DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES
        )
        mid_angle = (
            DEFAULT_THUMB_STRAIGHTNESS_THRESHOLD_DEGREES
            + DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES
        ) / 2.0

        mcp = (0.3, 0.5, 0.0)
        ip = (0.2, 0.5, 0.0)
        tip = _angled_point(vertex=ip, reference=mcp, degrees=mid_angle)

        overrides = {
            HandLandmark.THUMB_CMC: (0.4, 0.5, 0.0),
            HandLandmark.THUMB_MCP: mcp,
            HandLandmark.THUMB_IP: ip,
            HandLandmark.THUMB_TIP: tip,
        }
        landmarks = _landmarks_with(overrides)

        thumb_rule_detector = FingerStateDetector(
            thumb_threshold_degrees=DEFAULT_THUMB_STRAIGHTNESS_THRESHOLD_DEGREES
        )
        long_finger_rule_applied_to_thumb = FingerStateDetector(
            thumb_threshold_degrees=DEFAULT_FINGER_STRAIGHTNESS_THRESHOLD_DEGREES
        )

        assert thumb_rule_detector.detect(landmarks).thumb is True
        assert long_finger_rule_applied_to_thumb.detect(landmarks).thumb is False

    def test_thumb_folded(self) -> None:
        detector = FingerStateDetector()
        landmarks = _landmarks_with(_folded_thumb())
        assert detector.detect(landmarks).thumb is False