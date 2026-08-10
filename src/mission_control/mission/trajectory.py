"""Time parameterisation of a geometric path.

Given a :class:`~mission_control.mission.path.Path`, this produces the reference
both trackers follow: position, velocity, acceleration and yaw as functions of
time.

The speed profile
-----------------

Standard forward-backward pass over an arc-length grid, respecting three limits:

* a cruise speed,
* lateral acceleration, which bounds speed in a turn to ``sqrt(a_lat * R)``,
* tangential acceleration, which bounds how fast the speed itself can change.

The forward pass enforces "you cannot speed up faster than ``a_tan``"; the
backward pass enforces "you must already be slowing down for the corner ahead".
Running both gives the fastest profile that satisfies every limit, which is what
makes the reference demanding but *feasible* — an important property, because a
reference the vehicle physically cannot fly would make the comparison a measure
of infeasibility rather than of control.

``sharp`` corner mode deliberately breaks that property, and the comparison uses
it on purpose: see :class:`TrajectoryOptions`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..attitude import wrap_pi
from ..control.interface import ReferenceProvider
from ..types import TrajectorySetpoint
from .path import Path


def _segmented_grid(pinned, length: float, spacing: float) -> np.ndarray:
    """Arc-length grid containing every pinned point, with no tiny intervals.

    Each gap between consecutive pinned points is subdivided into whole steps of
    at most ``spacing``, so every interval is between half and one times the
    nominal spacing and every pinned point is a grid node exactly.
    """
    nodes = [value for value in pinned if 0.0 <= value <= length]
    if not nodes or nodes[0] > 0.0:
        nodes.insert(0, 0.0)
    if nodes[-1] < length:
        nodes.append(length)

    pieces = []
    for start, end in zip(nodes[:-1], nodes[1:]):
        gap = end - start
        if gap <= 1e-9:
            continue  # coincident pinned points; one node is enough
        steps = max(int(math.ceil(gap / spacing)), 1)
        pieces.append(np.linspace(start, end, steps + 1)[:-1])
    pieces.append(np.array([length]))
    grid = np.concatenate(pieces)
    return grid if grid.size >= 3 else np.linspace(0.0, length, 3)


def _horizontal_unit(tangent: np.ndarray):
    """Unit heading vector of a tangent, or ``None`` if it is purely vertical."""
    horizontal = np.asarray(tangent, dtype=float)[0:2]
    norm = float(np.linalg.norm(horizontal))
    return None if norm < 1e-6 else horizontal / norm


@dataclass
class TrajectoryOptions:
    """Limits and style for turning a path into a timed reference."""

    cruise_speed: float = 4.0
    max_tangential_acceleration: float = 2.5
    #: Lateral acceleration budget in a turn. 3.0 m/s^2 is about 17 degrees of
    #: bank, which keeps an inspection camera usable.
    max_lateral_acceleration: float = 3.0
    max_climb_rate: float = 2.5
    max_descent_rate: float = 1.5
    #: Arc-length resolution of the profile grid, m. Finer costs build time and
    #: buys nothing once it is well below the turn radii.
    grid_spacing: float = 0.10
    yaw_rate_limit: float = math.radians(90.0)
    #: When ``False`` (the default), the profile is pinned to zero speed
    #: wherever the path direction changes discontinuously, which guarantees the
    #: reference velocity really is the derivative of the reference position.
    #: ``sharp`` corner mode sets this ``True`` on purpose, producing a
    #: reference with a velocity step that no vehicle can follow — see
    #: :class:`~mission_control.mission.schema.MissionDefaults`.
    allow_velocity_discontinuity: bool = False
    #: Direction change above which a joint counts as a corner, rad.
    corner_threshold_rad: float = math.radians(2.0)


class PiecewiseTrajectory(ReferenceProvider):
    """A path plus a speed profile, sampled by time.

    Sampling outside ``[0, duration]`` holds the endpoints rather than
    extrapolating, so an MPC horizon can run past the end of the mission without
    the reference running away.
    """

    def __init__(self, path: Path, options: TrajectoryOptions = None,
                 stops=None, dwells=None, yaw_mode: str = "path",
                 fixed_yaw: float = 0.0, point_of_interest=None,
                 start_time_s: float = 0.0):
        self.path = path
        self.options = options or TrajectoryOptions()
        self.yaw_mode = yaw_mode
        self.fixed_yaw = float(fixed_yaw)
        self.point_of_interest = (
            None if point_of_interest is None
            else np.asarray(point_of_interest, dtype=float).reshape(3)
        )
        self.start_time_s = float(start_time_s)
        self._build(stops or [], dwells or {})

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _speed_limit_grid(self, s_grid: np.ndarray) -> np.ndarray:
        o = self.options
        limits = np.full(s_grid.size, o.cruise_speed, dtype=float)
        for i, s in enumerate(s_grid):
            curvature = self.path.curvature(float(s))
            if curvature > 1e-9:
                limits[i] = min(
                    limits[i], math.sqrt(o.max_lateral_acceleration / curvature)
                )
            # Climb and descent limits act on the vertical component of the
            # tangent, so a steep leg is speed-limited by the climb rate.
            tangent = self.path.tangent(float(s))
            vertical = tangent[2]
            if vertical < -1e-6:  # climbing (NED down is negative up)
                limits[i] = min(limits[i], o.max_climb_rate / abs(vertical))
            elif vertical > 1e-6:  # descending
                limits[i] = min(limits[i], o.max_descent_rate / vertical)
        return np.clip(limits, 0.05, None)

    def corner_arc_lengths(self) -> list:
        """Arc lengths where the path *heading* changes discontinuously.

        These are the joints corner rounding did not smooth: leaving an orbit
        onto a straight leg, entering one from inside it, or a corner too sharp
        or too tight to fit an arc into. Pinning the speed to zero at each keeps
        the guarantee that the horizontal reference velocity is the derivative
        of the horizontal reference position.

        The test is on **heading only**, deliberately. Where a climbing leg
        meets a level turn the vertical tangent has a small kink, and treating
        that as a corner would force a full stop at every rounded corner in any
        mission that changes height — which would defeat corner rounding
        entirely and make the reference far gentler than a real survey. The
        horizontal turn is the one that matters: a 90-degree corner at 5 m/s is
        a 7 m/s velocity step, whereas the vertical kink is bounded by the climb
        and descent rate limits and is worth a few tenths of a metre per second.
        ``vertical_velocity_steps`` reports those so the bound can be asserted
        rather than assumed.
        """
        corners = []
        threshold = math.cos(self.options.corner_threshold_rad)
        elements = self.path.elements
        for index in range(1, len(elements)):
            before = _horizontal_unit(elements[index - 1].tangent(elements[index - 1].length))
            after = _horizontal_unit(elements[index].tangent(0.0))
            if before is None or after is None:
                continue  # a purely vertical element has no heading to compare
            if float(np.dot(before, after)) < threshold:
                corners.append(float(self.path._starts[index]))
        return corners

    def vertical_velocity_steps(self) -> list:
        """Size of the vertical velocity step at each element joint, m/s.

        Non-zero wherever the altitude profile changes slope, which is any point
        a climb starts or stops. Bounded by the climb and descent rate limits by
        construction; ``tests/test_trajectory.py`` checks that bound holds.
        """
        steps = []
        for index in range(1, len(self.path.elements)):
            start = float(self.path._starts[index])
            _, speed, _ = self._state_at(float(np.interp(start, self._knot_s, self._knot_t)))
            before = self.path.elements[index - 1].tangent(self.path.elements[index - 1].length)
            after = self.path.elements[index].tangent(0.0)
            steps.append(abs(float(after[2] - before[2])) * speed)
        return steps

    def _build(self, stops, dwells) -> None:
        o = self.options
        length = self.path.length
        if length <= 1e-6:
            self._degenerate(dwells)
            return

        # Every stop, plus both ends, pins the speed to zero.
        pinned = [0.0, length] + [float(np.clip(s, 0.0, length)) for s in stops]
        if not o.allow_velocity_discontinuity:
            pinned.extend(self.corner_arc_lengths())
        pinned.extend(float(np.clip(s, 0.0, length)) for s in (dwells or {}))

        # The pinned arc lengths must land on the grid *exactly*, rather than
        # being rounded to the nearest uniform sample — snapping them was worth
        # up to half a grid spacing, which left a real velocity step of a few
        # tenths of a metre per second at every corner.
        #
        # The grid is therefore built segment by segment *between* pinned
        # points, each subdivided uniformly, rather than by merging the pinned
        # points into one uniform grid. Merging looks equivalent and is not: it
        # leaves intervals as short as the floating-point gap between a pinned
        # point and its neighbouring sample. Differentiating the yaw profile
        # across an interval that short produced yaw-rate feedforward spikes
        # that saturated at the rate limit and threw the vehicle off every
        # corner.
        s_grid = _segmented_grid(sorted(set(np.round(pinned, 9))), length, o.grid_spacing)
        count = s_grid.size
        spacing = np.diff(s_grid)

        limits = self._speed_limit_grid(s_grid)
        for value in pinned:
            index = int(np.argmin(np.abs(s_grid - value)))
            limits[index] = 0.0

        # Forward: cannot accelerate harder than a_tan.
        speeds = limits.copy()
        for i in range(1, count):
            reachable = math.sqrt(
                speeds[i - 1] ** 2 + 2.0 * o.max_tangential_acceleration * spacing[i - 1]
            )
            speeds[i] = min(speeds[i], reachable)
        # Backward: must already be braking for what is ahead.
        for i in range(count - 2, -1, -1):
            reachable = math.sqrt(
                speeds[i + 1] ** 2 + 2.0 * o.max_tangential_acceleration * spacing[i]
            )
            speeds[i] = min(speeds[i], reachable)

        # Arc length to time. Within a grid interval the speed goes from v[i] to
        # v[i+1] over a fixed distance, which is exactly constant tangential
        # acceleration, so the interval duration is 2*ds / (v[i] + v[i+1]). That
        # form stays finite at a stop, where 1/v would not.
        times = np.zeros(count)
        for i in range(1, count):
            mean_speed = 0.5 * (speeds[i - 1] + speeds[i])
            times[i] = times[i - 1] + (
                spacing[i - 1] / mean_speed if mean_speed > 1e-6 else 0.0
            )

        # Insert dwell time at each stop by duplicating its knot.
        knot_t, knot_s, knot_v = [], [], []
        offset = 0.0
        dwell_by_index = {}
        for s_stop, duration in (dwells or {}).items():
            index = int(np.argmin(np.abs(s_grid - float(np.clip(s_stop, 0.0, length)))))
            dwell_by_index[index] = float(duration)

        for i in range(count):
            knot_t.append(times[i] + offset)
            knot_s.append(s_grid[i])
            knot_v.append(speeds[i])
            dwell = dwell_by_index.get(i, 0.0)
            if dwell > 0.0:
                offset += dwell
                knot_t.append(times[i] + offset)
                knot_s.append(s_grid[i])
                knot_v.append(0.0)

        self._knot_t = np.asarray(knot_t)
        self._knot_s = np.asarray(knot_s)
        self._knot_v = np.asarray(knot_v)
        self.duration = float(self._knot_t[-1])
        self._ds = float(np.min(spacing)) if spacing.size else o.grid_spacing
        self._build_yaw()

    def _degenerate(self, dwells) -> None:
        """A path with no length: a pure hold at a single point."""
        hold = float(sum((dwells or {}).values())) or 1.0
        self._knot_t = np.array([0.0, hold])
        self._knot_s = np.array([0.0, 0.0])
        self._knot_v = np.array([0.0, 0.0])
        self.duration = hold
        self._ds = 1.0
        self._build_yaw()

    def _yaw_at_arc_length(self, s: float) -> float:
        if self.yaw_mode == "fixed":
            return self.fixed_yaw
        if self.yaw_mode == "poi" and self.point_of_interest is not None:
            point = self.path.point(s)
            delta = self.point_of_interest[0:2] - point[0:2]
            if float(np.linalg.norm(delta)) < 1e-3:
                return self.fixed_yaw
            return math.atan2(delta[1], delta[0])
        tangent = self.path.tangent(s)
        if float(np.linalg.norm(tangent[0:2])) < 1e-6:
            # Purely vertical travel has no heading of its own; hold the last one.
            return self.fixed_yaw
        return math.atan2(tangent[1], tangent[0])

    def _build_yaw(self) -> None:
        if self.path.length <= 1e-6:
            self._knot_yaw = np.full(self._knot_t.size, self.fixed_yaw)
            self._knot_yaw_rate = np.zeros(self._knot_t.size)
            return

        # Sample just off the knot so that at a stop the arriving and departing
        # duplicates pick up the incoming and outgoing headings respectively.
        # That makes the vehicle turn on the spot during a hold instead of
        # snapping round at the moment it starts moving again.
        yaw = np.zeros(self._knot_t.size)
        seen_stop = set()
        epsilon = 0.25 * self._ds
        for i, s in enumerate(self._knot_s):
            key = round(float(s), 6)
            if key in seen_stop:
                yaw[i] = self._yaw_at_arc_length(min(float(s) + epsilon, self.path.length))
            else:
                yaw[i] = self._yaw_at_arc_length(max(float(s) - epsilon, 0.0))
                seen_stop.add(key)

        # Unwrap so interpolation takes the short way round rather than
        # sweeping 350 degrees through a wrap at +-pi.
        self._knot_yaw = np.unwrap(yaw)
        rate = np.gradient(self._knot_yaw, self._knot_t)
        self._knot_yaw_rate = np.clip(
            np.nan_to_num(rate), -self.options.yaw_rate_limit, self.options.yaw_rate_limit
        )

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _state_at(self, t: float) -> tuple:
        """Arc length, speed and tangential acceleration at time ``t``.

        Interpolating arc length *linearly* between knots would be wrong near a
        stop, where one arc-length interval spans a long time and the true
        ``s(t)`` is strongly curved — it produced a 0.35 m/s discrepancy between
        the reference velocity and the derivative of the reference position at
        the start of every leg. Within an interval the speed profile has
        constant tangential acceleration by construction, so the quadratic can
        be inverted exactly instead, and the reference becomes consistent to
        round-off.
        """
        knots = self._knot_t
        index = int(np.clip(np.searchsorted(knots, t, side="right") - 1, 0, knots.size - 2))
        tau = t - knots[index]
        s0, s1 = self._knot_s[index], self._knot_s[index + 1]
        v0, v1 = self._knot_v[index], self._knot_v[index + 1]
        span = s1 - s0
        if span <= 1e-12:
            return float(s0), 0.0, 0.0  # a dwell: parked at this arc length
        acceleration = (v1 * v1 - v0 * v0) / (2.0 * span)
        speed = v0 + acceleration * tau
        s = s0 + v0 * tau + 0.5 * acceleration * tau * tau
        return (
            float(np.clip(s, min(s0, s1), max(s0, s1))),
            float(np.clip(speed, min(v0, v1), max(v0, v1))),
            float(acceleration),
        )

    def sample(self, time_s: float) -> TrajectorySetpoint:
        local = float(time_s) - self.start_time_s
        clamped = float(np.clip(local, 0.0, self.duration))
        # Past either end the reference is a stationary hold at the endpoint.
        beyond = clamped != local

        s, speed, tangential = self._state_at(clamped)
        if beyond:
            speed = 0.0
            tangential = 0.0
        yaw = float(np.interp(clamped, self._knot_t, self._knot_yaw))
        yaw_rate = 0.0 if beyond else float(
            np.interp(clamped, self._knot_t, self._knot_yaw_rate)
        )

        position = self.path.point(s)
        tangent = self.path.tangent(s)
        curvature = self.path.curvature(s)
        normal = self.path.normal(s)

        velocity = tangent * speed
        # Tangential term plus centripetal term. The centripetal part is what a
        # controller has to lean into before the corner, and it is exactly what
        # a preview horizon sees coming.
        acceleration = tangent * tangential + normal * (speed * speed * curvature)

        return TrajectorySetpoint(
            time_s=float(time_s),
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            yaw=wrap_pi(yaw),
            yaw_rate=yaw_rate,
        )

    @property
    def end_position(self) -> np.ndarray:
        return self.path.point(self.path.length)

    def summary(self) -> dict:
        speeds = self._knot_v
        return {
            "duration_s": round(self.duration, 3),
            "path_length_m": round(self.path.length, 3),
            "peak_speed_ms": round(float(np.max(speeds)), 3),
            "mean_speed_ms": round(
                float(self.path.length / self.duration) if self.duration > 0 else 0.0, 3
            ),
            "elements": len(self.path.elements),
        }
