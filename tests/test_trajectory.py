"""Trajectory generation: geometry, the speed profile and its guarantees.

The central invariant checked here is that the reference is *self-consistent*:
the velocity it reports is the derivative of the position it reports. A
reference that violates that is asking the vehicle for something impossible, and
any tracking error measured against it is partly a measure of the reference
rather than the controller.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mission_control.mission.builder import (
    build_land,
    build_mission_path,
    build_rtl,
    build_takeoff,
    orbit_entry_angle,
)
from mission_control.mission.path import Arc, Line, Path, round_corner
from mission_control.mission.trajectory import PiecewiseTrajectory, TrajectoryOptions


class TestPathElements:
    def test_line_length_and_tangent(self):
        line = Line([0, 0, 0], [3, 4, 0])
        assert line.length == pytest.approx(5.0)
        assert np.allclose(line.tangent(0.0), [0.6, 0.8, 0.0])
        assert np.allclose(line.point(5.0), [3, 4, 0])

    def test_arc_geometry(self):
        arc = Arc(centre=np.zeros(3), radius=10.0, start_angle=0.0, sweep=math.pi / 2)
        assert arc.length == pytest.approx(10.0 * math.pi / 2)
        assert np.allclose(arc.point(0.0)[0:2], [10.0, 0.0])
        assert np.allclose(arc.point(arc.length)[0:2], [0.0, 10.0], atol=1e-9)
        assert arc.curvature(0.0) == pytest.approx(0.1)

    def test_arc_tangent_is_perpendicular_to_the_radius(self):
        arc = Arc(centre=np.zeros(3), radius=7.0, start_angle=0.3, sweep=-2.0)
        for s in np.linspace(0.0, arc.length, 25):
            radial = arc.point(s) - arc.centre
            radial[2] = 0.0
            assert float(np.dot(radial, arc.tangent(s))) == pytest.approx(0.0, abs=1e-9)

    def test_positive_sweep_turns_clockwise_seen_from_above(self):
        """North to east is a right turn; the sign convention must match."""
        arc = Arc(centre=np.zeros(3), radius=5.0, start_angle=0.0, sweep=0.5)
        start, end = arc.point(0.0), arc.point(arc.length)
        assert start[0] == pytest.approx(5.0)
        assert end[1] > 0.0  # moved towards east

    def test_path_lookup_is_continuous_across_elements(self):
        path = Path([Line([0, 0, 0], [10, 0, 0]), Line([10, 0, 0], [10, 10, 0])])
        assert path.length == pytest.approx(20.0)
        assert np.allclose(path.point(10.0), [10, 0, 0])
        assert np.allclose(path.point(0.0), [0, 0, 0])
        assert np.allclose(path.point(20.0), [10, 10, 0])

    def test_zero_length_elements_are_dropped(self):
        path = Path([Line([0, 0, 0], [0, 0, 0]), Line([0, 0, 0], [5, 0, 0])])
        assert len(path.elements) == 1


class TestCornerRounding:
    def test_arc_joins_both_legs_tangentially(self):
        previous = np.array([0.0, 0.0, -10.0])
        corner = np.array([20.0, 0.0, -10.0])
        following = np.array([20.0, 20.0, -10.0])
        arc, entry, exit_point = round_corner(previous, corner, following, 5.0)

        assert arc is not None
        incoming = (corner - previous) / np.linalg.norm(corner - previous)
        outgoing = (following - corner) / np.linalg.norm(following - corner)
        assert np.dot(arc.tangent(0.0), incoming) == pytest.approx(1.0, abs=1e-6)
        assert np.dot(arc.tangent(arc.length), outgoing) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(arc.point(0.0), entry, atol=1e-9)
        assert np.allclose(arc.point(arc.length), exit_point, atol=1e-6)

    def test_shallow_corners_are_left_alone(self):
        arc, entry, _ = round_corner([0, 0, 0], [10, 0, 0], [20, 0.2, 0], 5.0)
        assert arc is None
        assert np.allclose(entry, [10, 0, 0])

    def test_radius_is_capped_by_leg_length(self):
        """A generous radius must not eat more than a leg can spare."""
        arc, entry, exit_point = round_corner([0, 0, 0], [6, 0, 0], [6, 6, 0], 50.0)
        assert arc is not None
        assert arc.radius < 50.0
        assert np.linalg.norm(entry - np.array([0, 0, 0])) > 0.0
        assert entry[0] >= 0.45 * 6 - 1e-6


class TestOrbitEntry:
    def test_entry_is_tangential(self):
        """Approaching an orbit radially is a 90-degree velocity step.

        Regression: the first implementation joined the circle at its nearest
        point, producing exactly that step at the start of every orbit.
        """
        centre = np.zeros(3)
        radius = 12.0
        approach = np.array([0.0, 40.0, 0.0])
        for sweep in (1.0, -1.0):
            angle = orbit_entry_angle(approach, centre, radius, sweep)
            entry = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
            arc = Arc(centre, radius, angle, sweep * 2 * math.pi)
            direction = entry - approach
            direction /= np.linalg.norm(direction)
            assert np.dot(direction, arc.tangent(0.0)) == pytest.approx(1.0, abs=1e-6)

    def test_falls_back_when_inside_the_circle(self):
        angle = orbit_entry_angle(np.array([1.0, 0.0, 0.0]), np.zeros(3), 12.0, 1.0)
        assert np.isfinite(angle)


class TestSpeedProfile:
    def _square(self, **options):
        path = Path([
            Line([0, 0, -15], [40, 0, -15]),
            Line([40, 0, -15], [40, 40, -15]),
        ])
        return PiecewiseTrajectory(path, TrajectoryOptions(**options))

    def test_starts_and_ends_at_rest(self):
        trajectory = self._square(cruise_speed=5.0)
        assert np.linalg.norm(trajectory.sample(0.0).velocity) == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(
            trajectory.sample(trajectory.duration).velocity
        ) == pytest.approx(0.0, abs=1e-9)

    def test_respects_the_cruise_speed(self):
        trajectory = self._square(cruise_speed=4.0)
        speeds = [
            float(np.linalg.norm(trajectory.sample(t).velocity))
            for t in np.linspace(0, trajectory.duration, 500)
        ]
        assert max(speeds) <= 4.0 + 1e-6

    def test_respects_the_tangential_acceleration_limit(self):
        trajectory = self._square(cruise_speed=6.0, max_tangential_acceleration=2.0)
        times = np.linspace(0, trajectory.duration, 3000)
        speeds = np.array([float(np.linalg.norm(trajectory.sample(t).velocity)) for t in times])
        rate = np.abs(np.diff(speeds) / np.diff(times))
        assert np.max(rate) <= 2.0 * 1.10

    def test_curvature_limits_speed_in_a_turn(self):
        lateral = 3.0
        radius = 8.0
        path = Path([Arc(np.zeros(3), radius, 0.0, math.pi)])
        trajectory = PiecewiseTrajectory(
            path, TrajectoryOptions(cruise_speed=20.0, max_lateral_acceleration=lateral)
        )
        speeds = [
            float(np.linalg.norm(trajectory.sample(t).velocity))
            for t in np.linspace(0, trajectory.duration, 400)
        ]
        assert max(speeds) <= math.sqrt(lateral * radius) + 1e-3

    def test_sampling_outside_the_span_holds_the_endpoints(self):
        trajectory = self._square(cruise_speed=5.0)
        before = trajectory.sample(-3.0)
        after = trajectory.sample(trajectory.duration + 10.0)
        assert np.allclose(before.position, trajectory.path.point(0.0))
        assert np.allclose(after.position, trajectory.path.point(trajectory.path.length))
        assert np.allclose(before.velocity, 0.0)
        assert np.allclose(after.velocity, 0.0)


class TestReferenceConsistency:
    """The invariant: reference velocity is the derivative of reference position."""

    def _trajectory(self, mission, start=np.array([0.0, 0.0, -15.0])):
        trajectory, _ = build_mission_path(start, mission)
        return trajectory

    def test_horizontal_velocity_matches_position_derivative(self, simple_mission):
        trajectory = self._trajectory(simple_mission)
        dt = 1e-3
        times = np.arange(0.0, trajectory.duration - dt, 0.01)
        worst = 0.0
        for t in times:
            forward = trajectory.sample(t + dt).position
            backward = trajectory.sample(t).position
            velocity = trajectory.sample(t + dt / 2).velocity
            difference = (forward - backward) / dt - velocity
            worst = max(worst, float(np.max(np.abs(difference[0:2]))))
        assert worst < 0.05

    def test_vertical_velocity_steps_are_bounded(self, simple_mission):
        """Altitude-profile joints do step the vertical velocity — but only within
        the climb and descent limits, which is what makes them harmless."""
        trajectory = self._trajectory(simple_mission)
        steps = trajectory.vertical_velocity_steps()
        limit = max(trajectory.options.max_climb_rate, trajectory.options.max_descent_rate)
        assert all(step <= limit + 1e-6 for step in steps)

    def test_no_tiny_grid_intervals(self, simple_mission):
        """Regression: merging pinned points into the grid created intervals
        near the floating-point gap, and differentiating the yaw profile across
        them produced full-scale yaw-rate spikes."""
        trajectory = self._trajectory(simple_mission)
        spacing = np.diff(trajectory._knot_s)
        positive = spacing[spacing > 0]
        assert positive.min() > 0.4 * trajectory.options.grid_spacing

    def test_position_is_continuous(self, simple_mission):
        trajectory = self._trajectory(simple_mission)
        times = np.arange(0.0, trajectory.duration, 0.01)
        points = np.array([trajectory.sample(t).position for t in times])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        peak = max(
            float(np.linalg.norm(trajectory.sample(t).velocity)) for t in times
        )
        assert steps.max() <= peak * 0.01 * 1.2 + 1e-6

    def test_sharp_corner_mode_is_deliberately_inconsistent(self, simple_mission_dict):
        """``sharp`` exists to produce an infeasible reference. It must actually
        be infeasible, or the scenario it drives is measuring nothing."""
        from mission_control.mission.loader import mission_from_dict

        simple_mission_dict["defaults"]["corner_mode"] = "sharp"
        mission = mission_from_dict(simple_mission_dict)
        trajectory = self._trajectory(mission)

        times = np.arange(0.0, trajectory.duration, 0.005)
        speeds = np.array([float(np.linalg.norm(trajectory.sample(t).velocity)) for t in times])
        directions = []
        for t in times:
            velocity = trajectory.sample(t).velocity
            speed = float(np.linalg.norm(velocity))
            directions.append(velocity / speed if speed > 0.1 else np.zeros(3))
        directions = np.array(directions)
        turn = np.sum(directions[:-1] * directions[1:], axis=1)
        moving = (speeds[:-1] > 1.0) & (speeds[1:] > 1.0)
        # Somewhere the direction changes abruptly at speed: that is the point.
        assert np.any(turn[moving] < math.cos(math.radians(20.0)))


class TestPhaseTrajectories:
    def test_takeoff_climbs_to_the_requested_altitude(self, simple_mission):
        trajectory = build_takeoff(np.zeros(3), 18.0, simple_mission.defaults)
        end = trajectory.sample(trajectory.duration).position
        assert -end[2] == pytest.approx(18.0, abs=1e-6)
        assert np.allclose(end[0:2], 0.0, atol=1e-9)

    def test_land_descends_to_the_ground(self, simple_mission):
        trajectory = build_land(np.array([5.0, -3.0, -12.0]), simple_mission.defaults)
        end = trajectory.sample(trajectory.duration).position
        assert end[2] == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(end[0:2], [5.0, -3.0], atol=1e-9)

    def test_rtl_climbs_before_transiting(self, simple_mission):
        start = np.array([40.0, 30.0, -8.0])
        trajectory = build_rtl(start, np.zeros(3), 30.0, simple_mission.defaults)
        # Sample early in the flight: the vehicle should be gaining height while
        # still near where it started.
        early = trajectory.sample(trajectory.duration * 0.15).position
        assert -early[2] > 8.0
        assert np.linalg.norm(early[0:2] - start[0:2]) < 12.0
        end = trajectory.sample(trajectory.duration).position
        assert np.allclose(end[0:2], 0.0, atol=0.5)
