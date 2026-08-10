"""Mission definition: validation, resolution and geodesy.

Validation is strict on purpose. A mission file is the one input that arrives
from outside the software, and a typo in it should stop the vehicle on the
ground rather than change what it does in the air.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mission_control.geodesy import GeodeticOrigin, projection_error_bound
from mission_control.mission import MissionValidationError, load_mission, mission_from_dict
from mission_control.mission.schema import validate_mission_dict


class TestGeodesy:
    def test_origin_round_trip(self):
        origin = GeodeticOrigin(51.5074, -0.1278, 12.0)
        for north, east, up in ((0, 0, 0), (100, -250, 30), (-1000, 2000, -5)):
            ned = np.array([north, east, -up], dtype=float)
            lat, lon, alt = origin.to_geodetic(ned)
            recovered = origin.to_ned(lat, lon, alt)
            assert np.allclose(recovered, ned, atol=1e-6)

    def test_origin_maps_to_zero(self):
        origin = GeodeticOrigin(51.5074, -0.1278, 12.0)
        assert np.allclose(origin.to_ned(51.5074, -0.1278, 12.0), np.zeros(3), atol=1e-9)

    def test_north_and_east_have_the_right_signs(self):
        origin = GeodeticOrigin(51.5, 0.0, 0.0)
        north = origin.to_ned(51.51, 0.0, 0.0)
        assert north[0] > 0 and abs(north[1]) < 1e-6
        east = origin.to_ned(51.5, 0.01, 0.0)
        assert east[1] > 0 and abs(east[0]) < 1e-6
        up = origin.to_ned(51.5, 0.0, 50.0)
        assert up[2] == pytest.approx(-50.0)

    def test_scale_is_about_right(self):
        """One degree of latitude is about 111 km."""
        origin = GeodeticOrigin(51.5, 0.0, 0.0)
        assert origin.to_ned(52.5, 0.0, 0.0)[0] == pytest.approx(111_000.0, rel=0.01)

    def test_flat_earth_error_is_negligible_at_mission_scale(self):
        """The linearisation is justified, not assumed.

        Inspection missions run to a few hundred metres, where the second-order
        curvature term is well under a millimetre - far below RTK noise.
        """
        assert projection_error_bound(500.0) < 1e-3
        assert projection_error_bound(5000.0) < 1.0

    def test_rejects_impossible_coordinates(self):
        with pytest.raises(ValueError):
            GeodeticOrigin(95.0, 0.0)
        with pytest.raises(ValueError):
            GeodeticOrigin(0.0, 200.0)


class TestValidation:
    def test_a_good_mission_has_no_problems(self, simple_mission_dict):
        assert validate_mission_dict(simple_mission_dict) == []

    def test_missing_required_keys_are_reported(self):
        problems = validate_mission_dict({"name": "x"})
        assert any("origin" in problem for problem in problems)
        assert any("items" in problem for problem in problems)

    def test_unknown_keys_are_rejected(self, simple_mission_dict):
        """A typo must fail loudly rather than silently taking a default.

        ``altitiude: 30`` should stop the mission on the ground, not fly it at
        whatever the default height happens to be.
        """
        simple_mission_dict["items"][0]["altitiude"] = 30.0
        problems = validate_mission_dict(simple_mission_dict)
        assert any("unknown key" in problem for problem in problems)
        assert any("altitude" in problem for problem in problems)  # suggests the fix

    def test_unknown_top_level_key(self, simple_mission_dict):
        simple_mission_dict["geofense"] = {}
        assert any("unknown key" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_all_problems_are_collected_not_just_the_first(self, simple_mission_dict):
        simple_mission_dict["origin"]["latitude"] = 200.0
        simple_mission_dict["defaults"]["cruise_speed"] = -4.0
        simple_mission_dict["items"][1]["altitude"] = -5.0
        problems = validate_mission_dict(simple_mission_dict)
        assert len(problems) >= 3

    def test_out_of_range_numbers(self, simple_mission_dict):
        simple_mission_dict["defaults"]["cruise_speed"] = 500.0
        assert any("cruise_speed" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_bad_enum(self, simple_mission_dict):
        simple_mission_dict["defaults"]["corner_mode"] = "swoopy"
        assert any("corner_mode" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_mission_must_start_with_takeoff(self, simple_mission_dict):
        simple_mission_dict["items"] = simple_mission_dict["items"][1:]
        assert any("takeoff" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_waypoint_needs_a_position(self, simple_mission_dict):
        simple_mission_dict["items"][1] = {"type": "waypoint", "altitude": 10.0}
        assert any("needs a position" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_mixed_coordinate_systems_are_rejected(self, simple_mission_dict):
        simple_mission_dict["items"][1]["latitude"] = 51.5
        simple_mission_dict["items"][1]["longitude"] = -0.13
        assert any("not both" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_geofence_altitude_band_must_be_ordered(self, simple_mission_dict):
        simple_mission_dict["geofence"]["min_altitude"] = 60.0
        simple_mission_dict["geofence"]["max_altitude"] = 50.0
        assert any("above min_altitude" in problem
                   for problem in validate_mission_dict(simple_mission_dict))

    def test_critical_battery_must_be_below_low_battery(self, simple_mission_dict):
        simple_mission_dict["failsafe"] = {
            "low_battery_remaining": 0.10,
            "critical_battery_remaining": 0.30,
        }
        problems = validate_mission_dict(simple_mission_dict)
        assert any("never fires" in problem for problem in problems)

    def test_polygon_needs_three_vertices(self, simple_mission_dict):
        simple_mission_dict["geofence"]["polygon_ned"] = [[0, 0], [1, 1]]
        assert any("3 vertices" in problem
                   for problem in validate_mission_dict(simple_mission_dict))


class TestResolution:
    def test_local_coordinates_are_preserved(self, simple_mission):
        waypoint = simple_mission.items[1]
        assert np.allclose(waypoint.position_ned, [30.0, 0.0, -15.0])

    def test_geodetic_coordinates_are_projected(self, simple_mission_dict):
        simple_mission_dict["items"][1] = {
            "type": "waypoint", "latitude": 51.5074, "longitude": -0.1278, "altitude": 20.0,
        }
        mission = mission_from_dict(simple_mission_dict)
        # Same point as the origin horizontally, 20 m above it.
        assert np.allclose(mission.items[1].position_ned[0:2], 0.0, atol=1e-6)
        assert mission.items[1].position_ned[2] == pytest.approx(-20.0, abs=1e-6)

    def test_defaults_are_applied(self, simple_mission):
        assert simple_mission.defaults.cruise_speed == 4.0
        assert simple_mission.defaults.corner_mode == "round"

    def test_summary_is_serialisable(self, simple_mission):
        import json

        json.dumps(simple_mission.summary())


class TestCrossFieldGeometry:
    """Checks that need the resolved geometry, caught before takeoff."""

    def test_waypoint_above_the_ceiling_is_rejected(self, simple_mission_dict):
        simple_mission_dict["items"][1]["altitude"] = 200.0
        simple_mission_dict["geofence"]["max_altitude"] = 50.0
        with pytest.raises(MissionValidationError, match="ceiling"):
            mission_from_dict(simple_mission_dict)

    def test_waypoint_below_the_floor_is_rejected(self, simple_mission_dict):
        simple_mission_dict["items"][1]["altitude"] = 1.0
        with pytest.raises(MissionValidationError, match="floor"):
            mission_from_dict(simple_mission_dict)

    def test_waypoint_outside_the_polygon_is_rejected(self, simple_mission_dict):
        simple_mission_dict["items"][1]["north"] = 500.0
        with pytest.raises(MissionValidationError, match="outside the geofence"):
            mission_from_dict(simple_mission_dict)

    def test_takeoff_above_the_ceiling_is_rejected(self, simple_mission_dict):
        simple_mission_dict["items"][0]["altitude"] = 90.0
        simple_mission_dict["geofence"]["max_altitude"] = 50.0
        with pytest.raises(MissionValidationError, match="ceiling"):
            mission_from_dict(simple_mission_dict)


class TestShippedMissions:
    """Every mission in ``missions/`` must load and validate."""

    def test_all_shipped_missions_are_valid(self, missions_dir):
        names = [name for name in os.listdir(missions_dir) if name.endswith(".yaml")]
        assert names, "no missions found"
        for name in names:
            mission = load_mission(os.path.join(missions_dir, name))
            assert mission.items
            assert mission.name

    def test_shipped_missions_stay_inside_their_own_geofence(self, missions_dir):
        """The generated path, not just the waypoints, must fit inside the fence.

        A path can leave a fence its waypoints stay inside: corner rounding cuts
        inside a turn, but an orbit bulges outward past its own centre point.
        """
        from mission_control.mission.builder import build_mission_path
        from mission_control.safety import Geofence

        for name in os.listdir(missions_dir):
            if not name.endswith(".yaml"):
                continue
            mission = load_mission(os.path.join(missions_dir, name))
            fence = Geofence.from_spec(mission.geofence)
            start = np.array([0.0, 0.0, -mission.takeoff_altitude])
            trajectory, _ = build_mission_path(start, mission)
            for point in trajectory.path.sample(600):
                assert fence.contains_horizontal(point), (
                    f"{name}: path leaves the geofence at {point}"
                )
                altitude = -point[2]
                assert mission.geofence.min_altitude - 1e-6 <= altitude
                assert altitude <= mission.geofence.max_altitude + 1e-6
