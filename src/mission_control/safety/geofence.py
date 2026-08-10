"""Geofence: a horizontal polygon plus an altitude band.

Two things distinguish a geofence that works from one that only looks like it
works.

**It has to be predictive.** By the time a 4 m/s aircraft has crossed the
boundary it is already outside, and an RTL that begins at the boundary will
overshoot it by several metres while it decelerates. So the check runs against
where the vehicle *will be*, using its current velocity over a lookahead that
covers the stopping distance.

**It has to be signed.** "Inside or outside" is a boolean and gives the mission
layer nothing to act on. Distance to the boundary lets it warn, slow down, and
only then trigger, which is the difference between a fence that contains a
vehicle and one that merely reports the breach afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeofenceStatus:
    """Result of one geofence evaluation."""

    inside: bool = True
    #: Signed distance to the nearest boundary, m. Positive inside.
    margin_m: float = 0.0
    #: Which boundary is nearest: ``polygon``, ``ceiling``, ``floor`` or ``none``.
    limiting: str = "none"
    #: True when the current position is already outside.
    breached: bool = False
    #: True when the projected position breaches within the lookahead.
    predicted_breach: bool = False
    #: Seconds until the projected breach, or ``inf``.
    time_to_breach_s: float = float("inf")
    #: True when inside but within the warning margin.
    warning: bool = False

    @property
    def is_clear(self) -> bool:
        return not (self.breached or self.predicted_breach)


class Geofence:
    """Polygonal horizontal fence with an altitude band."""

    def __init__(self, polygon_ned=None, min_altitude: float = 2.0,
                 max_altitude: float = 120.0, warning_margin: float = 5.0,
                 lookahead_s: float = 2.0):
        polygon = np.zeros((0, 2)) if polygon_ned is None else np.asarray(polygon_ned, dtype=float)
        if polygon.size and polygon.shape[1] != 2:
            raise ValueError("polygon must be an (N, 2) array of north/east vertices")
        self.polygon = polygon
        self.min_altitude = float(min_altitude)
        self.max_altitude = float(max_altitude)
        self.warning_margin = float(warning_margin)
        self.lookahead_s = float(lookahead_s)

    @classmethod
    def from_spec(cls, spec) -> "Geofence":
        return cls(
            polygon_ned=spec.polygon_ned,
            min_altitude=spec.min_altitude,
            max_altitude=spec.max_altitude,
            warning_margin=spec.warning_margin,
        )

    @property
    def is_defined(self) -> bool:
        return self.polygon.shape[0] >= 3

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def contains_horizontal(self, position) -> bool:
        """Crossing-number point-in-polygon test in the north-east plane.

        A vertical ray is cast from the point; an odd number of edge crossings
        means inside. The half-open edge convention (``>`` on one end, ``<=`` on
        the other) is what stops a point exactly level with a vertex from being
        counted twice.
        """
        if not self.is_defined:
            return True
        north, east = float(position[0]), float(position[1])
        vertices = self.polygon
        count = len(vertices)
        inside = False
        j = count - 1
        for i in range(count):
            ni, ei = vertices[i]
            nj, ej = vertices[j]
            if (ei > east) != (ej > east):
                crossing_north = ni + (east - ei) * (nj - ni) / (ej - ei)
                if north < crossing_north:
                    inside = not inside
            j = i
        return inside

    def distance_to_polygon(self, position) -> float:
        """Unsigned distance to the nearest polygon edge, m."""
        if not self.is_defined:
            return float("inf")
        point = np.asarray(position, dtype=float)[0:2]
        vertices = self.polygon
        starts = vertices
        ends = np.roll(vertices, -1, axis=0)
        edges = ends - starts
        lengths_squared = np.sum(edges * edges, axis=1)
        # Projection parameter of the point onto each edge, clamped to the
        # segment so the nearest point is a real point on the boundary.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.sum((point - starts) * edges, axis=1) / lengths_squared
        t = np.nan_to_num(np.clip(t, 0.0, 1.0))
        closest = starts + edges * t[:, None]
        return float(np.min(np.linalg.norm(point - closest, axis=1)))

    def signed_margin(self, position, enforce_floor: bool = True) -> tuple:
        """Signed distance to the nearest constraint and which one it is.

        ``enforce_floor`` exists because the altitude floor and the act of
        taking off are in direct contradiction: a vehicle on the ground is below
        any floor worth setting. Enforcing it unconditionally means every
        mission trips a geofence breach at ``t = 0`` and returns home from the
        ground, which is exactly what happened the first time this ran. The
        caller — which knows the flight phase — switches the floor off during
        takeoff and landing and leaves it on for everything in between.
        """
        altitude = -float(position[2])
        candidates = [(self.max_altitude - altitude, "ceiling")]
        if enforce_floor:
            candidates.append((altitude - self.min_altitude, "floor"))
        if self.is_defined:
            horizontal = self.distance_to_polygon(position)
            sign = 1.0 if self.contains_horizontal(position) else -1.0
            candidates.append((sign * horizontal, "polygon"))

        margin, limiting = min(candidates, key=lambda pair: pair[0])
        return float(margin), limiting

    # ------------------------------------------------------------------
    def check(self, position, velocity=None, enforce_floor: bool = True) -> GeofenceStatus:
        """Evaluate the fence at a position, optionally predicting ahead."""
        margin, limiting = self.signed_margin(position, enforce_floor)
        status = GeofenceStatus(
            inside=margin >= 0.0,
            margin_m=margin,
            limiting=limiting,
            breached=margin < 0.0,
            warning=0.0 <= margin < self.warning_margin,
        )
        if status.breached or velocity is None:
            return status

        velocity = np.asarray(velocity, dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed < 0.1:
            return status

        # March forward along the current velocity. A closed-form intersection
        # would be faster but would have to special-case the polygon, the
        # ceiling and the floor separately; at 10 samples this is not the
        # expensive part of the control loop.
        steps = 10
        for step in range(1, steps + 1):
            horizon = self.lookahead_s * step / steps
            projected = np.asarray(position, dtype=float) + velocity * horizon
            projected_margin, projected_limit = self.signed_margin(projected, enforce_floor)
            if projected_margin < 0.0:
                status.predicted_breach = True
                status.time_to_breach_s = horizon
                status.limiting = projected_limit
                break
        return status

    def clamp_to_fence(self, position, inset: float = 1.0) -> np.ndarray:
        """Nearest position that is comfortably inside, for RTL path planning.

        Only the altitude band is enforced exactly; horizontally the point is
        pulled towards the polygon centroid, which is crude but always produces
        a point inside a convex fence and a sensible one inside a concave fence.
        """
        clamped = np.asarray(position, dtype=float).copy()
        altitude = -clamped[2]
        altitude = float(
            np.clip(altitude, self.min_altitude + inset, self.max_altitude - inset)
        )
        clamped[2] = -altitude
        if self.is_defined and not self.contains_horizontal(clamped):
            centroid = self.polygon.mean(axis=0)
            direction = centroid - clamped[0:2]
            distance = float(np.linalg.norm(direction))
            if distance > 1e-6:
                step = min(distance, self.distance_to_polygon(clamped) + inset)
                clamped[0:2] = clamped[0:2] + direction / distance * step
        return clamped

    def describe(self) -> dict:
        return {
            "vertices": int(self.polygon.shape[0]),
            "min_altitude_m": self.min_altitude,
            "max_altitude_m": self.max_altitude,
            "warning_margin_m": self.warning_margin,
            "lookahead_s": self.lookahead_s,
            "area_m2": round(self.area(), 1),
        }

    def area(self) -> float:
        """Polygon area by the shoelace formula, m^2. Zero if undefined."""
        if not self.is_defined:
            return 0.0
        north = self.polygon[:, 0]
        east = self.polygon[:, 1]
        return float(0.5 * abs(np.dot(north, np.roll(east, -1)) - np.dot(east, np.roll(north, -1))))
