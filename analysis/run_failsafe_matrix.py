#!/usr/bin/env python3
"""Fly every failsafe scenario and record what the vehicle actually did.

`docs/FAILSAFE_MATRIX.md` documents what each rule is *supposed* to do. This
produces the evidence that it does: each scenario injects one real fault into a
complete simulated flight — telemetry genuinely goes silent, the battery
genuinely drains, GNSS genuinely stops producing fixes — and records the rule
that fired, when, and what the vehicle did next.

Output is `results/failsafe_runs.csv`, which `analysis/make_results.py` folds
into the matrix document as a verification table.

Run:  python analysis/run_failsafe_matrix.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.parallel import limit_blas_threads  # noqa: E402

limit_blas_threads()

import numpy as np  # noqa: E402

from mission_control.analysis.scenarios import FAILSAFE_SCENARIOS  # noqa: E402
from mission_control.logs.flight_log import STATE_NAMES  # noqa: E402
from mission_control.mission import load_mission  # noqa: E402
from mission_control.runner import ScenarioConfig, run_scenario  # noqa: E402
from mission_control.sim import GnssParams, WindParams  # noqa: E402

COLUMNS = (
    "scenario", "description", "expected_action", "observed_action",
    "rules_fired", "first_trigger_s", "final_state", "phases_entered",
    "landed", "completed", "geofence_breached", "min_altitude_m", "duration_s",
    "matches_expectation",
)


def build_config(name: str, spec: dict, mission):
    """Turn a failsafe scenario definition into a runner configuration."""
    mission = copy.deepcopy(mission)
    if "shrink_geofence" in spec:
        # Injected *after* loading, so it bypasses the cross-field validation
        # that would otherwise reject a mission whose waypoints lie outside its
        # own fence. That rejection is real and is tested separately; here the
        # point is to fault-inject a fence the aircraft cannot honour.
        #
        # With ``shrink_waypoints`` the survey is pulled in to match and still
        # fits, so only the *predictive* check should fire. Without it the path
        # genuinely leaves the fence and a real breach is unavoidable.
        extent = float(spec["shrink_geofence"])
        mission.geofence.polygon_ned = np.array(
            [[-extent, -extent], [-extent, extent], [extent, extent], [extent, -extent]]
        )
        if "shrink_waypoints" in spec:
            scale = float(spec["shrink_waypoints"])
            for item in mission.items:
                if item.is_positional:
                    item.position_ned[0:2] *= scale
        mission.defaults.cruise_speed *= 1.6

    gnss = GnssParams(rate_hz=5.0, latency_s=0.08)
    if "gnss_schedule" in spec:
        gnss.fix_schedule = list(spec["gnss_schedule"])

    return ScenarioConfig(
        mission=mission,
        tracker="mpc",
        seed=1,
        label=f"failsafe-{name}",
        wind=spec.get("wind", WindParams(mean_ned=(3.0, 2.0, 0.0), gust_sigma=1.2)),
        gnss=gnss,
        link_dropouts=tuple(tuple(w) for w in spec.get("link_dropouts", ())),
        rtk_dropouts=tuple(tuple(w) for w in spec.get("rtk_dropouts", ())),
        initial_battery=float(spec.get("initial_battery", 1.0)),
        battery_drain_scale=float(spec.get("battery_drain_scale", 1.0)),
        max_duration_s=400.0,
    )


def run_one(name: str, spec: dict, mission) -> dict:
    result = run_scenario(build_config(name, spec, mission))
    log = result.log

    rules = [event["rule"] for event in result.failsafe_events]
    actions = [event["action"] for event in result.failsafe_events]
    observed = actions[-1] if actions else "none"
    first = result.failsafe_events[0]["time_s"] if result.failsafe_events else float("nan")

    phases = sorted({STATE_NAMES.get(int(code), "?") for code in np.unique(log["mission_state"])})
    altitude = -log["true_down"]

    expected = spec["expect_action"]
    # "none" means the fault must be survived without any action, which is as
    # much a requirement as firing correctly is.
    matches = observed == expected

    return {
        "scenario": name,
        "description": spec["description"],
        "expected_action": expected,
        "observed_action": observed,
        "rules_fired": ";".join(dict.fromkeys(rules)) or "-",
        "first_trigger_s": round(first, 2) if np.isfinite(first) else "",
        "final_state": result.final_state,
        "phases_entered": ";".join(phases),
        "landed": int(altitude[-1] < 0.3),
        "completed": int(result.completed),
        "geofence_breached": int(np.any(log["geofence_breached"] > 0.5)),
        "min_altitude_m": round(float(altitude[log.mask_state("mission")].min()), 2)
        if np.any(log.mask_state("mission")) else "",
        "duration_s": round(result.duration_s, 1),
        "matches_expectation": int(matches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission",
                        default=os.path.join(ROOT, "missions", "facade_inspection.yaml"))
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    mission = load_mission(args.mission)
    os.makedirs(args.output, exist_ok=True)

    print(f"{len(FAILSAFE_SCENARIOS)} failsafe scenarios on {mission.name}")
    rows = []
    for name, spec in FAILSAFE_SCENARIOS.items():
        row = run_one(name, spec, mission)
        rows.append(row)
        mark = "ok  " if row["matches_expectation"] else "FAIL"
        print(f"  {mark} {name:22s} expected {row['expected_action']:<6} "
              f"observed {row['observed_action']:<6} "
              f"({row['rules_fired']}) -> {row['final_state']}")

    path = os.path.join(args.output, "failsafe_runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if not row["matches_expectation"]]
    print()
    print(f"wrote {os.path.relpath(path, ROOT)}")
    print(f"{len(rows) - len(failures)}/{len(rows)} scenarios behaved as specified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
