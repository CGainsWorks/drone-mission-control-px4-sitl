"""Geofence and the failsafe matrix.

The failsafe tests iterate the matrix itself rather than hard-coding a list of
rules, so a rule added without a test that fires it shows up as a failure here
rather than as a surprise in flight.
"""

from __future__ import annotations

import numpy as np
import pytest

from mission_control.mission.schema import FailsafeSpec, GeofenceSpec
from mission_control.safety import (
    FailsafeAction,
    FailsafeMatrix,
    Geofence,
    SystemStatus,
    build_default_rules,
)

SQUARE = np.array([[-50.0, -50.0], [-50.0, 50.0], [50.0, 50.0], [50.0, -50.0]])
CONCAVE = np.array([
    [0.0, 0.0], [0.0, 40.0], [40.0, 40.0], [40.0, 0.0], [25.0, 0.0],
    [25.0, 25.0], [15.0, 25.0], [15.0, 0.0],
])


class TestGeofenceGeometry:
    @pytest.mark.parametrize("point,inside", [
        ((0.0, 0.0), True), ((49.0, 49.0), True), ((-49.0, 0.0), True),
        ((51.0, 0.0), False), ((0.0, -60.0), False), ((100.0, 100.0), False),
    ])
    def test_point_in_polygon(self, point, inside):
        fence = Geofence(SQUARE)
        assert fence.contains_horizontal(np.array([point[0], point[1], -10.0])) is inside

    def test_concave_polygon(self):
        """A crossing-number test must handle a notch correctly."""
        fence = Geofence(CONCAVE)
        assert fence.contains_horizontal(np.array([5.0, 20.0, -10.0]))
        assert fence.contains_horizontal(np.array([35.0, 20.0, -10.0]))
        # Inside the notch, which is outside the polygon.
        assert not fence.contains_horizontal(np.array([20.0, 10.0, -10.0]))

    def test_distance_to_boundary(self):
        fence = Geofence(SQUARE)
        assert fence.distance_to_polygon(np.array([0.0, 0.0, -10.0])) == pytest.approx(50.0)
        assert fence.distance_to_polygon(np.array([45.0, 0.0, -10.0])) == pytest.approx(5.0)
        assert fence.distance_to_polygon(np.array([60.0, 0.0, -10.0])) == pytest.approx(10.0)

    def test_area(self):
        assert Geofence(SQUARE).area() == pytest.approx(100.0 * 100.0)

    def test_undefined_fence_contains_everything(self):
        fence = Geofence(None)
        assert not fence.is_defined
        assert fence.contains_horizontal(np.array([1e6, 1e6, -10.0]))


class TestGeofenceChecks:
    def test_altitude_band(self):
        fence = Geofence(SQUARE, min_altitude=5.0, max_altitude=40.0)
        assert fence.check(np.array([0.0, 0.0, -20.0])).inside
        assert fence.check(np.array([0.0, 0.0, -45.0])).breached
        assert fence.check(np.array([0.0, 0.0, -2.0])).breached

    def test_floor_can_be_suspended(self):
        """Takeoff and the floor are in direct contradiction.

        Regression: enforcing the floor unconditionally meant every mission
        breached its geofence at t = 0 and returned home from the ground.
        """
        fence = Geofence(SQUARE, min_altitude=5.0, max_altitude=40.0)
        on_ground = np.array([0.0, 0.0, 0.0])
        assert fence.check(on_ground, enforce_floor=True).breached
        assert not fence.check(on_ground, enforce_floor=False).breached
        # The ceiling stays armed either way.
        assert fence.check(np.array([0.0, 0.0, -45.0]), enforce_floor=False).breached

    def test_warning_margin_fires_before_the_breach(self):
        fence = Geofence(SQUARE, warning_margin=8.0)
        status = fence.check(np.array([45.0, 0.0, -20.0]))
        assert status.inside and status.warning and not status.breached
        assert fence.check(np.array([20.0, 0.0, -20.0])).warning is False

    def test_predictive_breach(self):
        """The reason the check is predictive rather than positional.

        A vehicle heading at the boundary at 5 m/s crosses it before a
        positional check can do anything useful about it.
        """
        fence = Geofence(SQUARE, lookahead_s=2.0)
        position = np.array([45.0, 0.0, -20.0])
        approaching = fence.check(position, np.array([5.0, 0.0, 0.0]))
        assert not approaching.breached
        assert approaching.predicted_breach
        assert 0.0 < approaching.time_to_breach_s <= 2.0

        receding = fence.check(position, np.array([-5.0, 0.0, 0.0]))
        assert not receding.predicted_breach

    def test_slow_vehicle_does_not_trigger_prediction(self):
        fence = Geofence(SQUARE, lookahead_s=2.0)
        status = fence.check(np.array([49.0, 0.0, -20.0]), np.array([0.02, 0.0, 0.0]))
        assert not status.predicted_breach

    def test_clamp_returns_a_point_inside(self):
        fence = Geofence(SQUARE, min_altitude=5.0, max_altitude=40.0)
        for point in (np.array([80.0, 0.0, -20.0]), np.array([0.0, 0.0, -60.0]),
                      np.array([0.0, 0.0, -1.0])):
            clamped = fence.clamp_to_fence(point)
            assert fence.check(clamped).inside

    def test_from_spec(self):
        spec = GeofenceSpec(polygon_ned=SQUARE, min_altitude=3.0, max_altitude=55.0,
                            warning_margin=7.0)
        fence = Geofence.from_spec(spec)
        assert fence.min_altitude == 3.0
        assert fence.max_altitude == 55.0
        assert fence.warning_margin == 7.0


def airborne(**overrides) -> SystemStatus:
    base = dict(time_s=100.0, in_air=True, armed=True)
    base.update(overrides)
    return SystemStatus(**base)


class TestFailsafeMatrix:
    def test_nothing_wrong_means_no_action(self):
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        assert matrix.update(airborne()) == FailsafeAction.NONE

    def test_every_rule_has_a_trigger_that_fires_it(self):
        """Iterating the matrix, not a hard-coded list.

        Adding a rule without a way to exercise it should fail here.
        """
        spec = FailsafeSpec()
        triggers = {
            "battery_critical": dict(battery_remaining=0.05),
            "battery_low": dict(battery_remaining=0.20),
            "gnss_loss": dict(gnss_usable=False),
            "estimator_uncertain": dict(estimator_healthy=False),
            "geofence_breach": dict(geofence_breached=True),
            "geofence_predicted": dict(geofence_predicted_breach=True),
            "link_loss_critical": dict(link_silence_s=60.0),
            "link_loss": dict(link_silence_s=5.0),
        }
        for rule in build_default_rules(spec):
            assert rule.name in triggers, f"rule {rule.name} has no test trigger"
            matrix = FailsafeMatrix(spec=spec)
            status = airborne(**triggers[rule.name])
            # Advance past the debounce.
            matrix.update(status)
            later = airborne(time_s=status.time_s + rule.debounce_s + 0.1,
                             **triggers[rule.name])
            action = matrix.update(later)
            assert action >= rule.action, (
                f"rule {rule.name} did not produce at least {rule.action.label}"
            )

    def test_debounce_suppresses_a_transient(self):
        spec = FailsafeSpec(low_battery_remaining=0.25)
        matrix = FailsafeMatrix(spec=spec)
        # The rule debounces for 2 s; one sample must not trip it.
        assert matrix.update(airborne(time_s=10.0, battery_remaining=0.2)) == FailsafeAction.NONE
        assert matrix.update(airborne(time_s=10.5, battery_remaining=0.2)) == FailsafeAction.NONE
        assert matrix.update(airborne(time_s=13.0, battery_remaining=0.2)) == FailsafeAction.RTL

    def test_condition_clearing_resets_the_debounce(self):
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        matrix.update(airborne(time_s=10.0, battery_remaining=0.2))
        matrix.update(airborne(time_s=11.0, battery_remaining=0.9))  # recovered
        assert matrix.update(airborne(time_s=12.5, battery_remaining=0.2)) == FailsafeAction.NONE

    def test_actions_latch_and_never_de_escalate(self):
        """A vehicle returning home does not resume because the voltage recovered.

        Battery voltage always recovers when the load drops, which is exactly
        what happens when a vehicle stops climbing to fly home. Un-latching on
        that reading would send it back out on a pack that cannot make it.
        """
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        for time_s in (10.0, 13.0):
            matrix.update(airborne(time_s=time_s, battery_remaining=0.2))
        assert matrix.action == FailsafeAction.RTL
        assert matrix.update(airborne(time_s=20.0, battery_remaining=0.95)) == FailsafeAction.RTL

    def test_escalation_is_allowed(self):
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        for time_s in (10.0, 13.0):
            matrix.update(airborne(time_s=time_s, battery_remaining=0.2))
        assert matrix.action == FailsafeAction.RTL
        for time_s in (20.0, 22.0):
            matrix.update(airborne(time_s=time_s, battery_remaining=0.05))
        assert matrix.action == FailsafeAction.LAND

    def test_priority_decides_when_several_fire(self):
        """Critical battery outranks a geofence breach: land now, do not fly home."""
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        status = airborne(time_s=10.0, battery_remaining=0.05, geofence_breached=True)
        matrix.update(status)
        action = matrix.update(airborne(time_s=12.0, battery_remaining=0.05,
                                        geofence_breached=True))
        assert action == FailsafeAction.LAND
        assert matrix.active_rule == "battery_critical"

    def test_nothing_fires_on_the_ground(self):
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        grounded = SystemStatus(time_s=100.0, in_air=False, battery_remaining=0.01,
                                gnss_usable=False, geofence_breached=True)
        assert matrix.update(grounded) == FailsafeAction.NONE

    def test_events_record_why(self):
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        matrix.update(airborne(time_s=10.0, gnss_usable=False))
        matrix.update(airborne(time_s=11.0, gnss_usable=False))
        assert matrix.events
        assert matrix.events[0].rule == "gnss_loss"
        assert matrix.events[0].action == FailsafeAction.LAND

    def test_a_broken_rule_does_not_take_the_vehicle_with_it(self):
        from mission_control.safety.failsafe import FailsafeRule

        def explode(status):
            raise RuntimeError("bad rule")

        rules = [
            FailsafeRule("explodes", "always raises", explode, FailsafeAction.LAND, 100),
            FailsafeRule("fine", "battery low", lambda s: s.battery_remaining < 0.2,
                         FailsafeAction.RTL, 50, debounce_s=0.0),
        ]
        matrix = FailsafeMatrix(rules=rules)
        assert matrix.update(airborne(battery_remaining=0.1)) == FailsafeAction.RTL

    def test_table_is_documentable(self):
        matrix = FailsafeMatrix(spec=FailsafeSpec())
        table = matrix.to_table()
        assert len(table) == len(matrix.rules)
        assert all(row["rationale"] for row in table)
        # Sorted by priority, descending, so the generated document reads in the
        # order the rules actually resolve.
        priorities = [row["priority"] for row in table]
        assert priorities == sorted(priorities, reverse=True)

    def test_thresholds_come_from_the_mission_spec(self):
        strict = FailsafeSpec(low_battery_remaining=0.5, low_battery_action="land")
        matrix = FailsafeMatrix(spec=strict)
        matrix.update(airborne(time_s=10.0, battery_remaining=0.45))
        assert matrix.update(airborne(time_s=13.0, battery_remaining=0.45)) == FailsafeAction.LAND
