"""End-to-end: the mission state machine, the runner and the failsafe responses.

These are the tests that would catch a subsystem working perfectly on its own
and wrong in company. They are the slowest in the suite because each flies a
whole mission, so the missions used here are deliberately short.
"""

from __future__ import annotations

import numpy as np
import pytest

from mission_control.logs.flight_log import STATE_CODES
from mission_control.mission import MissionState
from mission_control.runner import ScenarioConfig, run_scenario
from mission_control.sim import GnssParams, WindParams
from mission_control.types import GnssFixType


def short_mission_dict():
    """A minimal square, short enough to fly repeatedly in a test suite."""
    return {
        "name": "short_square",
        "origin": {"latitude": 51.5074, "longitude": -0.1278, "altitude": 10.0},
        "defaults": {"cruise_speed": 5.0, "corner_mode": "round", "corner_radius": 5.0},
        "geofence": {
            "polygon_ned": [[-60, -60], [-60, 60], [60, 60], [60, -60]],
            "min_altitude": 2.0,
            "max_altitude": 50.0,
        },
        "failsafe": {"rtl_altitude": 25.0},
        "items": [
            {"type": "takeoff", "altitude": 15.0},
            {"type": "waypoint", "north": 25, "east": 0, "altitude": 15},
            {"type": "waypoint", "north": 25, "east": 25, "altitude": 15},
            {"type": "waypoint", "north": 0, "east": 25, "altitude": 15},
            {"type": "land"},
        ],
    }


@pytest.fixture(scope="module")
def short_mission():
    """Module-scoped so the cached flights below actually hit.

    Nothing in the runner mutates a mission, so sharing one instance across the
    module is safe and saves flying the same nominal mission five times.
    """
    from mission_control.mission import mission_from_dict

    return mission_from_dict(short_mission_dict())


def flown_states(result):
    return {int(code) for code in np.unique(result.log["mission_state"])}


#: Flying a whole mission is the expensive part of this module, and several
#: tests want the same nominal flight. Cached so the suite flies it once.
_NOMINAL = {}


def nominal_flight(mission, tracker="pid", seed=1):
    key = (id(mission), tracker, seed)
    if key not in _NOMINAL:
        _NOMINAL[key] = run_scenario(
            ScenarioConfig(mission=mission, tracker=tracker, seed=seed,
                           max_duration_s=200.0)
        )
    return _NOMINAL[key]


class TestNominalFlight:
    @pytest.mark.parametrize("tracker", ["pid", "mpc"])
    def test_mission_completes(self, short_mission, tracker):
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker=tracker, seed=1,
                           max_duration_s=200.0)
        )
        assert not result.diverged
        assert result.completed
        assert result.final_state == MissionState.DISARMED.value

    def test_all_phases_are_flown_in_order(self, short_mission):
        result = nominal_flight(short_mission)
        states = flown_states(result)
        for phase in ("armed", "takeoff", "mission", "landing", "disarmed"):
            assert STATE_CODES[phase] in states, f"never entered {phase}"
        sequence = [transition["to"] for transition in result.transitions]
        assert sequence.index("takeoff") < sequence.index("mission")
        assert sequence.index("mission") < sequence.index("landing")

    def test_takeoff_reaches_the_commanded_altitude(self, short_mission):
        result = nominal_flight(short_mission)
        log = result.log
        mission_phase = log.mask_state("mission")
        altitude = -log["true_down"][mission_phase]
        assert altitude.min() > 14.0

    def test_lands_on_the_ground(self, short_mission):
        result = nominal_flight(short_mission)
        assert -result.log["true_down"][-1] < 0.3

    def test_no_failsafe_fires_on_a_clean_flight(self, short_mission):
        result = nominal_flight(short_mission)
        assert result.failsafe_events == []

    def test_stays_inside_the_geofence(self, short_mission):
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="mpc", seed=2,
                           wind=WindParams(mean_ned=(4.0, 2.0, 0.0), gust_sigma=1.5),
                           max_duration_s=200.0)
        )
        breached = result.log["geofence_breached"]
        assert not np.any(breached > 0.5)


class TestDeterminism:
    def test_identical_configs_give_identical_logs(self, short_mission):
        """Reproducibility is a property the whole results pipeline rests on."""
        first = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="mpc", seed=7,
                           wind=WindParams(mean_ned=(3.0, 1.0, 0.0), gust_sigma=1.0),
                           max_duration_s=200.0)
        )
        second = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="mpc", seed=7,
                           wind=WindParams(mean_ned=(3.0, 1.0, 0.0), gust_sigma=1.0),
                           max_duration_s=200.0)
        )
        assert len(first.log) == len(second.log)
        for column in ("true_north", "true_east", "true_down", "cmd_thrust"):
            assert np.array_equal(first.log[column], second.log[column])

    def test_controllers_meet_the_same_disturbances(self, short_mission):
        """The comparison is paired: swapping the controller must not change the
        wind, the IMU noise or the GNSS errors."""
        wind = WindParams(mean_ned=(3.0, 1.0, 0.0), gust_sigma=1.2)
        pid = run_scenario(ScenarioConfig(mission=short_mission, tracker="pid", seed=5,
                                          wind=wind, max_duration_s=200.0))
        mpc = run_scenario(ScenarioConfig(mission=short_mission, tracker="mpc", seed=5,
                                          wind=wind, max_duration_s=200.0))
        # The sensor and wind generators are seeded independently of the
        # controller, so the very first samples must agree exactly.
        assert pid.log["true_north"][0] == pytest.approx(mpc.log["true_north"][0], abs=1e-12)
        assert pid.log["ekf_position_std_h"][0] == pytest.approx(
            mpc.log["ekf_position_std_h"][0], abs=1e-12
        )

    def test_different_seeds_give_different_flights(self, short_mission):
        wind = WindParams(mean_ned=(3.0, 1.0, 0.0), gust_sigma=1.5)
        first = run_scenario(ScenarioConfig(mission=short_mission, tracker="pid", seed=1,
                                            wind=wind, max_duration_s=200.0))
        second = run_scenario(ScenarioConfig(mission=short_mission, tracker="pid", seed=2,
                                             wind=wind, max_duration_s=200.0))
        assert not np.array_equal(
            first.log["true_north"][: min(len(first.log), len(second.log))],
            second.log["true_north"][: min(len(first.log), len(second.log))],
        )


class TestFailsafeIntegration:
    """Each row of the matrix, driven end to end through a real flight."""

    def test_link_loss_holds_then_returns(self, short_mission):
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="pid", seed=1,
                           link_dropouts=((14.0, 1e6),), max_duration_s=250.0)
        )
        rules = {event["rule"] for event in result.failsafe_events}
        assert "link_loss" in rules
        assert "link_loss_critical" in rules
        assert STATE_CODES["rtl"] in flown_states(result)

    def test_brief_link_loss_does_not_return_home(self, short_mission):
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="pid", seed=1,
                           link_dropouts=((14.0, 19.0),), max_duration_s=250.0)
        )
        rules = {event["rule"] for event in result.failsafe_events}
        assert "link_loss" in rules
        assert "link_loss_critical" not in rules

    def test_low_battery_returns_home(self, short_mission):
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="pid", seed=1,
                           initial_battery=0.30, battery_drain_scale=90.0,
                           max_duration_s=250.0)
        )
        rules = {event["rule"] for event in result.failsafe_events}
        assert "battery_low" in rules
        assert STATE_CODES["rtl"] in flown_states(result)

    def test_gnss_loss_lands(self, short_mission):
        result = run_scenario(
            ScenarioConfig(
                mission=short_mission, tracker="pid", seed=1,
                gnss=GnssParams(
                    rate_hz=5.0, latency_s=0.08,
                    fix_schedule=[(0.0, GnssFixType.RTK_FIXED), (16.0, GnssFixType.NO_FIX)],
                ),
                max_duration_s=250.0,
            )
        )
        rules = {event["rule"] for event in result.failsafe_events}
        assert "gnss_loss" in rules
        assert STATE_CODES["landing"] in flown_states(result)

    def test_rtk_dropout_degrades_but_keeps_flying(self, short_mission):
        """Losing RTK is a degradation, not an emergency: the mission continues."""
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="pid", seed=1,
                           rtk_dropouts=((14.0, 1e6),), max_duration_s=250.0)
        )
        assert result.completed
        assert result.failsafe_events == []
        # The fix type must actually have degraded, or the test proved nothing.
        assert np.any(result.log["gnss_fix_type"] == int(GnssFixType.RTK_FLOAT))

    def test_geofence_breach_returns_home(self):
        """Shrink the fence so the survey cannot stay inside it."""
        from mission_control.mission import mission_from_dict

        data = short_mission_dict()
        data["geofence"]["polygon_ned"] = [[-18, -18], [-18, 18], [18, 18], [18, -18]]
        # The mission's own waypoints would now fail validation, so move them in
        # and rely on overshoot in wind to breach instead.
        data["items"][1]["north"] = 15
        data["items"][2] = {"type": "waypoint", "north": 15, "east": 15, "altitude": 15}
        data["items"][3] = {"type": "waypoint", "north": 0, "east": 15, "altitude": 15}
        data["defaults"]["cruise_speed"] = 9.0
        data["defaults"]["corner_radius"] = 2.0
        mission = mission_from_dict(data)

        result = run_scenario(
            ScenarioConfig(mission=mission, tracker="pid", seed=3,
                           wind=WindParams(mean_ned=(8.0, 6.0, 0.0), gust_sigma=2.5),
                           max_duration_s=250.0)
        )
        rules = {event["rule"] for event in result.failsafe_events}
        assert rules & {"geofence_breach", "geofence_predicted"}


class TestDivergenceGuard:
    def test_a_broken_controller_stops_the_run(self, short_mission):
        """A controller that loses the vehicle must abort, not fill the log."""
        from mission_control.control import TRACKERS
        from mission_control.control.interface import TrajectoryTracker
        from mission_control.types import AttitudeSetpoint

        class Useless(TrajectoryTracker):
            name = "useless"

            def reset(self, state):
                pass

            def update(self, state, reference, time_s, dt):
                return AttitudeSetpoint(time_s=time_s, thrust=1.0)

        TRACKERS["useless"] = Useless
        try:
            result = run_scenario(
                ScenarioConfig(mission=short_mission, tracker="useless", seed=1,
                               max_duration_s=250.0)
            )
        finally:
            del TRACKERS["useless"]
        assert result.diverged
        assert not result.completed


class TestTruthStateMode:
    def test_truth_mode_removes_estimation_error(self, short_mission):
        """The mode exists to attribute error to the controller, not the filter."""
        result = run_scenario(
            ScenarioConfig(mission=short_mission, tracker="pid", seed=1,
                           use_truth_state=True, max_duration_s=200.0)
        )
        assert np.max(np.abs(result.log.estimation_error())) < 1e-9
