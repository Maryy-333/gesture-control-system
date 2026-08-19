"""Temporal stabilization of recognized gestures -- smoothing, not recognition.

This module sits between `GestureRecognizer` (which produces a raw
`Gesture` for a single frame) and `ActionMapper`/`GestureActionGate`
(which turn a confirmed gesture into an `Action` and decide when it
fires). Its only job is to answer one question per frame: given the
gesture recognized this frame, what is the one "stable" gesture we
should report downstream?

Why this matters (real webcam behavior):
    MediaPipe's per-frame landmark output is noisy at the joints, so a
    finger near a bent/straight threshold can flip the reported
    `Gesture` between two distinct gestures for 1-2 frames even though
    the user's hand has not actually changed pose (for example an
    isolated `FIST` flicker in the middle of a sustained `POINT`). Each
    *distinct* gesture that reaches `GestureActionGate` can fire its
    discrete `Action` (a click), so that raw flicker is exactly what
    produces the "I am only pointing and why did it click?" symptom.

What this does:
    A new gesture must be observed for `stability_frames` *consecutive*
    frames before it is allowed to *replace* the currently stable
    gesture. Until it does, the previously-stable gesture is returned.
    A gesture equal to the currently stable gesture reports immediately
    and clears any competing candidate, so a held gesture is never
    needlessly delayed. This is a simple consecutive-count state machine
    -- no windows, no voting, no timestamps.

What this deliberately does NOT do:
    - It does NOT import `Action`, `ActionMapper`, cursor coordinates,
      landmarks, or any per-frame input control. It only consumes
      `Gesture` values, keeping it a self-contained, dependency-free
      gesture-level component. (Only the project's `Gesture` enum and
      the Python standard library are used.)
    - It does NOT perform action mapping, debouncing, pause/resume, or
      move-cursor handling. Those remain `ActionMapper`'s and
      `GestureActionGate`'s separate responsibilities.
    - It does NOT smooth landmarks, FingerStates, coordinates, or frame
      counts unrelated to candidate confirmation.

Dependency direction (do not reverse):
    FingerStates -> GestureRecognizer -> Gesture -> GestureSmoothener ->
    Gesture -> ActionMapper -> Action -> GestureActionGate -> Controller
"""

from dataclasses import dataclass, field
from typing import Optional

from .recognizer import Gesture


@dataclass
class GestureSmoothener:
    """Stabilize gesture transitions across consecutive frames."""

    stability_frames: int = 3
    _stable_gesture: Optional[Gesture] = field(default=None, init=False)
    _candidate_gesture: Optional[Gesture] = field(default=None, init=False)
    _candidate_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stability_frames, int) or isinstance(
            self.stability_frames, bool
        ):
            raise ValueError(
                "stability_frames must be an int, got "
                f"{type(self.stability_frames).__name__}."
            )
        if self.stability_frames < 1:
            raise ValueError(
                f"stability_frames must be >= 1, got {self.stability_frames!r}."
            )

    @property
    def stable_gesture(self) -> Optional[Gesture]:
        """The currently stable gesture, or `None` before any is confirmed."""
        return self._stable_gesture

    def smooth(self, gesture: Gesture) -> Gesture:
        """Return the stable gesture to report for this frame."""
        if self._stable_gesture is None:
            self._stable_gesture = gesture
            self._candidate_gesture = None
            self._candidate_count = 0
            return gesture
        if gesture == self._stable_gesture:
            self._candidate_gesture = None
            self._candidate_count = 0
            return gesture
        if gesture == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = gesture
            self._candidate_count = 1
        if self._candidate_count >= self.stability_frames:
            self._stable_gesture = gesture
            self._candidate_gesture = None
            self._candidate_count = 0
            return gesture
        return self._stable_gesture

    def reset(self) -> None:
        """Clear all internal state."""
        self._stable_gesture = None
        self._candidate_gesture = None
        self._candidate_count = 0
