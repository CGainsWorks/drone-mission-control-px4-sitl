"""Geometric path elements.

A mission's waypoints become a continuous 3-D path made of straight segments and
circular arcs, parameterised by arc length. Timing is a separate concern and
lives in :mod:`mission_control.mission.trajectory`; keeping geometry and timing
apart is what lets the same path be flown at three different speeds in the
comparison without regenerating it.

Every element answers four questions at an arc length ``s``: where, which way,
how sharply it is turning, and which way to face. Curvature is the important
one: it is what bounds the speed profile, because a turn of radius ``R`` at
speed ``v`` needs a lateral acceleration of ``v^2 / R`` and the vehicle has only
so much tilt.
"""

from __future__ import annotations

import abc
import math

import numpy as np


class PathElement(abc.ABC):
    """One piece of a path, parameterised by arc length from its own start."""

    length = 0.0

    @abc.abstractmethod
    def point(self, s: float) -> np.ndarray:
        """Position in NED at arc length ``s``."""

    @abc.abstractmethod
    def tangent(self, s: float) -> np.ndarray:
        """Unit direction of travel at ``s``."""

    def curvature(self, s: float) -> float:
        """Reciprocal of the turn radius at ``s``, 1/m. Zero for a straight."""
        return 0.0

    def normal(self, s: float) -> np.ndarray:
        """Unit vector towards the centre of curvature. Zero when straight."""
        return np.zeros(3)


class Line(PathElement):
    """Straight segment between two NED points."""

    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = np.asarray(start, dtype=float).reshape(3)
        self.end = np.asarray(end, dtype=float).reshape(3)
        delta = self.end - self.start
        self.length = float(np.linalg.norm(delta))
        # A zero-length line has no defined direction. Callers filter these out,
        # but a degenerate one must not produce NaNs if it slips through.
        self._direction = delta / self.length if self.length > 1e-9 else np.array([1.0, 0.0, 0.0])

    def point(self, s: float) -> np.ndarray:
        return self.start + self._direction * float(np.clip(s, 0.0, self.length))

    def tangent(self, s: float) -> np.ndarray:
        return self._direction.copy()


class Arc(PathElement):
    """Circular arc in a horizontal plane, optionally climbing as it turns.

    Used for two things: rounding the corner between two straight legs, and
    orbiting a structure, which is the manoeuvre an inspection aircraft spends
    most of its time doing.
    """

    def __init__(self, centre: np.ndarray, radius: float, start_angle: float,
                 sweep: float, start_down: float = None, end_down: float = None):
        self.centre = np.asarray(centre, dtype=float).reshape(3)
        self.radius = float(radius)
        self.start_angle = float(start_angle)
        # Angles are measured from north towards east, so increasing angle turns
        # clockwise when viewed from above. A positive sweep is a right turn.
        self.sweep = float(sweep)
        if self.radius <= 1e-6:
            raise ValueError("arc radius must be positive")
        self.length = abs(self.sweep) * self.radius
        self._start_down = self.centre[2] if start_down is None else float(start_down)
        self._end_down = self._start_down if end_down is None else float(end_down)

    def _angle(self, s: float) -> float:
        fraction = 0.0 if self.length <= 1e-9 else float(np.clip(s, 0.0, self.length)) / self.length
        return self.start_angle + self.sweep * fraction

    def point(self, s: float) -> np.ndarray:
        angle = self._angle(s)
        fraction = 0.0 if self.length <= 1e-9 else float(np.clip(s, 0.0, self.length)) / self.length
        return np.array(
            [
                self.centre[0] + self.radius * math.cos(angle),
                self.centre[1] + self.radius * math.sin(angle),
                self._start_down + (self._end_down - self._start_down) * fraction,
            ]
        )

    def tangent(self, s: float) -> np.ndarray:
        angle = self._angle(s)
        sign = 1.0 if self.sweep >= 0.0 else -1.0
        horizontal = np.array([-math.sin(angle) * sign, math.cos(angle) * sign, 0.0])
        climb = (self._end_down - self._start_down) / self.length if self.length > 1e-9 else 0.0
        vector = np.array([horizontal[0], horizontal[1], climb])
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else horizontal

    def curvature(self, s: float) -> float:
        return 1.0 / self.radius

    def normal(self, s: float) -> np.ndarray:
        angle = self._angle(s)
        # Points from the path towards the centre.
        return np.array([-math.cos(angle), -math.sin(angle), 0.0])


class Path:
    """An ordered chain of elements, addressed by total arc length."""

    def __init__(self, elements=None):
        self.elements = []
        self._starts = []
        self.length = 0.0
        for element in elements or []:
            self.append(element)

    def append(self, element: PathElement) -> None:
        if element.length <= 1e-9:
            return  # degenerate, contributes nothing and breaks the lookup
        self._starts.append(self.length)
        self.elements.append(element)
        self.length += element.length

    def _locate(self, s: float) -> tuple:
        if not self.elements:
            raise ValueError("empty path")
        s = float(np.clip(s, 0.0, self.length))
        index = int(np.searchsorted(self._starts, s, side="right") - 1)
        index = max(0, min(index, len(self.elements) - 1))
        return self.elements[index], s - self._starts[index]

    def point(self, s: float) -> np.ndarray:
        element, local = self._locate(s)
        return element.point(local)

    def tangent(self, s: float) -> np.ndarray:
        element, local = self._locate(s)
        return element.tangent(local)

    def curvature(self, s: float) -> float:
        element, local = self._locate(s)
        return element.curvature(local)

    def normal(self, s: float) -> np.ndarray:
        element, local = self._locate(s)
        return element.normal(local)

    def sample(self, count: int = 400) -> np.ndarray:
        """Evenly spaced points, for plotting and geofence pre-checks."""
        if not self.elements:
            return np.zeros((0, 3))
        return np.array([self.point(s) for s in np.linspace(0.0, self.length, count)])


def round_corner(previous: np.ndarray, corner: np.ndarray, following: np.ndarray,
                 radius: float) -> tuple:
    """Fit an arc of ``radius`` into the corner at ``corner``.

    Returns ``(arc, entry_point, exit_point)``, or ``(None, corner, corner)`` if
    the corner is too shallow to need rounding, too sharp to fit, or the legs
    are too short to give up the length.

    The arc is horizontal: it is fitted in the north-east plane and the altitude
    is interpolated across it. Inspection missions change height on the straights
    and turn at a constant height, so this covers the real cases without needing
    a general 3-D blend.
    """
    previous = np.asarray(previous, dtype=float)
    corner = np.asarray(corner, dtype=float)
    following = np.asarray(following, dtype=float)

    incoming = corner[0:2] - previous[0:2]
    outgoing = following[0:2] - corner[0:2]
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length < 1e-6 or outgoing_length < 1e-6:
        return None, corner, corner

    incoming_unit = incoming / incoming_length
    outgoing_unit = outgoing / outgoing_length

    cos_turn = float(np.clip(np.dot(incoming_unit, outgoing_unit), -1.0, 1.0))
    turn = math.acos(cos_turn)
    if turn < math.radians(3.0):
        return None, corner, corner  # effectively straight through
    if turn > math.radians(170.0):
        return None, corner, corner  # a reversal; rounding it is meaningless

    # Tangent distance from the corner to where the arc touches each leg.
    tangent_distance = radius / math.tan((math.pi - turn) / 2.0)
    # Never eat more than 45% of a leg, or consecutive corners would overlap.
    limit = 0.45 * min(incoming_length, outgoing_length)
    if tangent_distance > limit:
        tangent_distance = limit
        radius = tangent_distance * math.tan((math.pi - turn) / 2.0)
        if radius < 0.5:
            return None, corner, corner

    entry = corner.copy()
    entry[0:2] = corner[0:2] - incoming_unit * tangent_distance
    exit_point = corner.copy()
    exit_point[0:2] = corner[0:2] + outgoing_unit * tangent_distance
    # Altitude is taken from the legs at the tangent points so the arc joins
    # them without a step.
    entry[2] = corner[2] + (previous[2] - corner[2]) * (tangent_distance / incoming_length)
    exit_point[2] = corner[2] + (following[2] - corner[2]) * (tangent_distance / outgoing_length)

    # Cross product's sign tells us which way the path turns.
    cross = incoming_unit[0] * outgoing_unit[1] - incoming_unit[1] * outgoing_unit[0]
    sign = 1.0 if cross > 0.0 else -1.0
    # Centre lies perpendicular to the incoming leg at the entry point.
    perpendicular = np.array([-incoming_unit[1], incoming_unit[0]]) * sign
    centre_ne = entry[0:2] + perpendicular * radius
    centre = np.array([centre_ne[0], centre_ne[1], entry[2]])

    start_angle = math.atan2(entry[1] - centre_ne[1], entry[0] - centre_ne[0])
    sweep = turn * sign
    arc = Arc(centre, radius, start_angle, sweep, start_down=entry[2], end_down=exit_point[2])
    return arc, entry, exit_point
