#!/usr/bin/env python3
"""Validate every mission file and print what each one will actually fly.

Useful before a flight and useful in CI: a mission file is the one input that
comes from outside the software, so it gets its own check with a readable
failure rather than being buried in the test suite output.

Beyond schema validation this builds the trajectory each mission generates and
reports its length, duration and peak speed — because "valid" and "sensible"
are different questions, and a mission that validates but takes forty minutes to
fly is worth noticing on the ground.

Run:  python scripts/validate_missions.py [path ...]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.mission import MissionValidationError, load_mission  # noqa: E402
from mission_control.mission.builder import build_mission_path  # noqa: E402
from mission_control.safety import Geofence  # noqa: E402


def check(path: str) -> bool:
    name = os.path.relpath(path, ROOT)
    try:
        mission = load_mission(path)
    except MissionValidationError as error:
        print(f"FAIL  {name}")
        for problem in error.problems:
            print(f"        {problem}")
        return False
    except Exception as error:
        print(f"FAIL  {name}: {type(error).__name__}: {error}")
        return False

    start = np.array([0.0, 0.0, -mission.takeoff_altitude])
    trajectory, waypoints = build_mission_path(start, mission)
    summary = trajectory.summary()

    fence = Geofence.from_spec(mission.geofence)
    outside = 0
    if fence.is_defined:
        for point in trajectory.path.sample(500):
            if not fence.contains_horizontal(point):
                outside += 1

    status = "FAIL" if outside else "ok  "
    print(f"{status}  {name}")
    print(f"        {mission.name}: {len(mission.items)} items, "
          f"{len(waypoints)} survey waypoints")
    print(f"        path {summary['path_length_m']:.1f} m, "
          f"{summary['duration_s']:.1f} s, "
          f"peak {summary['peak_speed_ms']:.1f} m/s, "
          f"mean {summary['mean_speed_ms']:.1f} m/s")
    print(f"        geofence {fence.describe()['vertices']} vertices, "
          f"{fence.area():.0f} m^2, "
          f"altitude {mission.geofence.min_altitude:.0f}-"
          f"{mission.geofence.max_altitude:.0f} m")
    if outside:
        print(f"        {outside} sampled path points lie outside the geofence")
    return outside == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*",
                        help="mission files; defaults to every file in missions/")
    args = parser.parse_args()

    paths = args.paths or sorted(glob.glob(os.path.join(ROOT, "missions", "*.yaml")))
    if not paths:
        print("no mission files found")
        return 1

    failures = sum(0 if check(path) else 1 for path in paths)
    print()
    print(f"{len(paths) - failures}/{len(paths)} mission(s) valid")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
