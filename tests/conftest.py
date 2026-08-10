"""Shared fixtures and path setup."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.mission.loader import mission_from_dict  # noqa: E402
from mission_control.sim import QuadrotorParams  # noqa: E402


@pytest.fixture(scope="session")
def params():
    return QuadrotorParams()


@pytest.fixture
def simple_mission_dict():
    """A short, valid mission. Deliberately small so tests using it stay fast."""
    return {
        "name": "unit_test_square",
        "origin": {"latitude": 51.5074, "longitude": -0.1278, "altitude": 10.0},
        "defaults": {
            "cruise_speed": 4.0,
            "corner_mode": "round",
            "corner_radius": 5.0,
            "yaw_mode": "path",
        },
        "geofence": {
            "polygon_ned": [[-60, -60], [-60, 60], [60, 60], [60, -60]],
            "min_altitude": 2.0,
            "max_altitude": 50.0,
        },
        "items": [
            {"type": "takeoff", "altitude": 15.0},
            {"type": "waypoint", "north": 30, "east": 0, "altitude": 15},
            {"type": "waypoint", "north": 30, "east": 30, "altitude": 18},
            {"type": "waypoint", "north": 0, "east": 30, "altitude": 18},
            {"type": "land"},
        ],
    }


@pytest.fixture
def simple_mission(simple_mission_dict):
    return mission_from_dict(simple_mission_dict)


@pytest.fixture(scope="session")
def missions_dir():
    return os.path.join(ROOT, "missions")
