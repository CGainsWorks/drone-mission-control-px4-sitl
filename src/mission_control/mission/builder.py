"""Turning mission items into flyable trajectories, one flight phase at a time.

Each phase gets its own trajectory rather than one trajectory for the whole
flight. That is not just tidiness: a failsafe can interrupt at any moment, and
an RTL has to be planned from wherever the vehicle actually is, not from where
a precomputed plan expected it to be.
"""

from __future__ import annotations

import math

import numpy as np

from .path import Arc, Line, Path, round_corner
from .schema import Mission, MissionDefaults
from .trajectory import PiecewiseTrajectory, TrajectoryOptions


def orbit_entry_angle(approach_ned, centre_ned, radius: float, sweep: float) -> float:
    """Angle on the orbit circle at which to join it tangentially.

    Flying at the circle and turning onto it at the nearest point means arriving
    radially, which is a 90-degree velocity step the vehicle cannot make. The
    tangent line from the approach point touches the circle at

    .. math:: \\theta = \\beta \\pm \\arccos(R / d)

    where ``beta`` is the bearing from the centre to the approach point and ``d``
    the distance between them. The sign picks which of the two tangent points
    leaves the vehicle already travelling in the orbit's direction: ``+`` for a
    clockwise orbit, ``-`` for anticlockwise.

    If the approach point is inside the circle there is no tangent line, so the
    nearest point is used and the corner-detection in the trajectory builder
    inserts a stop instead.
    """
    approach = np.asarray(approach_ned, dtype=float)
    centre = np.asarray(centre_ned, dtype=float)
    offset = approach[0:2] - centre[0:2]
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-6:
        return 0.0
    bearing = math.atan2(offset[1], offset[0])
    if distance <= radius * 1.001:
        return bearing
    half_angle = math.acos(float(np.clip(radius / distance, -1.0, 1.0)))
    return bearing + (half_angle if sweep >= 0.0 else -half_angle)


def options_from_defaults(defaults: MissionDefaults, cruise_speed: float = None,
                          **overrides) -> TrajectoryOptions:
    options = TrajectoryOptions(
        cruise_speed=float(cruise_speed if cruise_speed is not None else defaults.cruise_speed),
        max_tangential_acceleration=float(defaults.max_tangential_acceleration),
        max_lateral_acceleration=float(defaults.max_lateral_acceleration),
        max_climb_rate=float(defaults.max_climb_rate),
        max_descent_rate=float(defaults.max_descent_rate),
    )
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


def build_takeoff(start_ned, altitude: float, defaults: MissionDefaults,
                  start_time_s: float = 0.0) -> PiecewiseTrajectory:
    """Vertical climb from where the vehicle is to the takeoff altitude."""
    start = np.asarray(start_ned, dtype=float).copy()
    end = start.copy()
    end[2] = -abs(float(altitude))
    path = Path([Line(start, end)])
    options = options_from_defaults(defaults, cruise_speed=defaults.max_climb_rate)
    return PiecewiseTrajectory(
        path, options, yaw_mode="fixed", fixed_yaw=0.0, start_time_s=start_time_s
    )


def build_land(start_ned, defaults: MissionDefaults, yaw: float = 0.0,
               start_time_s: float = 0.0) -> PiecewiseTrajectory:
    """Vertical descent to the ground.

    The descent rate is deliberately the mission's, not something faster: the
    last two metres are where ground effect and a mis-estimated altitude do the
    damage.
    """
    start = np.asarray(start_ned, dtype=float).copy()
    end = start.copy()
    end[2] = 0.0
    path = Path([Line(start, end)])
    options = options_from_defaults(defaults, cruise_speed=defaults.max_descent_rate)
    return PiecewiseTrajectory(
        path, options, yaw_mode="fixed", fixed_yaw=yaw, start_time_s=start_time_s
    )


def build_rtl(start_ned, home_ned, rtl_altitude: float, defaults: MissionDefaults,
              yaw: float = 0.0, start_time_s: float = 0.0) -> PiecewiseTrajectory:
    """Climb to the RTL altitude, transit home, then descend to hover over it.

    Climbing first is the conservative order. The vehicle does not know what is
    between it and home, and the one thing that is certainly clear is the air it
    has just flown through.
    """
    start = np.asarray(start_ned, dtype=float).copy()
    home = np.asarray(home_ned, dtype=float).copy()
    altitude = -abs(float(rtl_altitude))

    climb_point = start.copy()
    climb_point[2] = min(start[2], altitude)
    transit_point = home.copy()
    transit_point[2] = climb_point[2]
    arrival = home.copy()
    arrival[2] = min(-2.0, altitude)

    path = Path()
    path.append(Line(start, climb_point))
    path.append(Line(climb_point, transit_point))
    path.append(Line(transit_point, arrival))

    options = options_from_defaults(defaults)
    heading = yaw
    horizontal = transit_point[0:2] - climb_point[0:2]
    if float(np.linalg.norm(horizontal)) > 1.0:
        heading = math.atan2(horizontal[1], horizontal[0])
    return PiecewiseTrajectory(
        path, options, yaw_mode="fixed", fixed_yaw=heading, start_time_s=start_time_s
    )


def build_hold(position_ned, duration_s: float, defaults: MissionDefaults,
               yaw: float = 0.0, start_time_s: float = 0.0) -> PiecewiseTrajectory:
    """Stay put. A zero-length path with a dwell."""
    position = np.asarray(position_ned, dtype=float).copy()
    path = Path([Line(position, position + np.array([1e-6, 0.0, 0.0]))])
    options = options_from_defaults(defaults)
    return PiecewiseTrajectory(
        path,
        options,
        dwells={0.0: max(float(duration_s), 0.1)},
        yaw_mode="fixed",
        fixed_yaw=yaw,
        start_time_s=start_time_s,
    )


def build_mission_path(start_ned, mission: Mission, start_time_s: float = 0.0) -> tuple:
    """Build the survey trajectory from the mission's positional items.

    Returns ``(trajectory, waypoint_arc_lengths)``. The arc lengths let the
    analysis mark where each waypoint fell on the flown path, which is how the
    per-leg breakdown in the results is produced.

    Corner handling follows ``defaults.corner_mode``:

    ``round``
        An arc of ``corner_radius`` is fitted into each corner, capped so it
        never eats more than 45% of either leg. The result is curvature-limited
        and therefore flyable, which is the realistic case.
    ``stop``
        The corner is kept sharp and the speed profile is pinned to zero there.
        Also flyable, and slow.
    ``sharp``
        The corner is kept sharp and the speed is not pinned, so the reference
        has a velocity step no vehicle can follow. Deliberately infeasible; the
        comparison uses it to see what each controller does when asked for the
        impossible.
    """
    defaults = mission.defaults
    positional = [item for item in mission.items if item.is_positional]

    points = [np.asarray(start_ned, dtype=float).copy()]
    holds = []  # (index into points, duration)
    orbits = {}  # index into points -> item
    for item in positional:
        if item.type == "orbit":
            orbits[len(points)] = item
        points.append(item.position_ned.copy())
        if item.hold_s > 0.0:
            holds.append((len(points) - 1, item.hold_s))

    path = Path()
    stops = []
    dwells = {}
    waypoint_lengths = []

    corner_mode = defaults.corner_mode
    index = 0
    current = points[0]
    while index < len(points) - 1:
        target_index = index + 1
        target = points[target_index]

        if target_index in orbits:
            item = orbits[target_index]
            centre = target
            sweep = math.radians(item.sweep_deg) * (1.0 if item.clockwise else -1.0)
            entry_angle = orbit_entry_angle(current, centre, item.radius, sweep)
            entry = np.array(
                [
                    centre[0] + item.radius * math.cos(entry_angle),
                    centre[1] + item.radius * math.sin(entry_angle),
                    centre[2],
                ]
            )
            path.append(Line(current, entry))
            waypoint_lengths.append(path.length)
            path.append(Arc(centre, item.radius, entry_angle, sweep,
                            start_down=centre[2], end_down=centre[2]))
            current = path.point(path.length)
            index = target_index
            continue

        following = points[target_index + 1] if target_index + 1 < len(points) else None
        needs_corner = (
            following is not None
            and corner_mode == "round"
            and target_index not in orbits
            and target_index + 1 not in orbits
            and not any(i == target_index for i, _ in holds)
        )

        if needs_corner:
            arc, entry, exit_point = round_corner(current, target, following,
                                                  defaults.corner_radius)
            if arc is not None:
                path.append(Line(current, entry))
                waypoint_lengths.append(path.length + 0.5 * arc.length)
                path.append(arc)
                current = exit_point
                index = target_index
                continue

        path.append(Line(current, target))
        waypoint_lengths.append(path.length)
        if corner_mode == "stop" and following is not None:
            stops.append(path.length)
        for hold_index, duration in holds:
            if hold_index == target_index:
                stops.append(path.length)
                dwells[path.length] = duration
        current = target
        index = target_index

    yaw_mode = defaults.yaw_mode
    point_of_interest = None
    if yaw_mode == "poi":
        # Face the first orbit centre, which for an inspection mission is the
        # structure being inspected.
        for item in positional:
            if item.type == "orbit":
                point_of_interest = item.position_ned
                break
        if point_of_interest is None:
            yaw_mode = "path"

    options = options_from_defaults(
        defaults, allow_velocity_discontinuity=(corner_mode == "sharp")
    )
    trajectory = PiecewiseTrajectory(
        path,
        options,
        stops=stops,
        dwells=dwells,
        yaw_mode=yaw_mode,
        point_of_interest=point_of_interest,
        start_time_s=start_time_s,
    )
    return trajectory, waypoint_lengths
