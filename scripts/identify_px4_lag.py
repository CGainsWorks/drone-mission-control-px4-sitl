#!/usr/bin/env python3
"""Measure how PX4 SITL responds to an acceleration demand, from a flight log.

``scripts/identify_lag.py`` measures this for the reference simulation's
attitude loop: 0.13 s, a first-order lag. Against PX4 the path from a command
to its effect is longer. The setpoint crosses MAVLink and MAVSDK, PX4's
attitude and rate loops act on it and the state the next command is computed
from has come back through PX4's EKF and MAVLink again. Gains placed for 0.13 s
are not placed for that.

This fits ``delay + first-order lag`` from the commanded acceleration (the
flight script's loop log) to the thrust acceleration the vehicle actually had
(ground-truth velocity from PX4's ULog, plus SIH's linear drag, which is a
known force and not part of the response), over the survey phase of one
flight. The result is what ``sitl.PX4_SIH_COMMAND_DELAY_S`` and
``sitl.PX4_SIH_LAG_TAU_S`` hold and what ``scripts/tune_gains.py --airframe
sih`` tunes against.

Run:  python scripts/identify_px4_lag.py sitl/results/flights/facade_inspection-mpc
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control import sitl  # noqa: E402
from mission_control.logs.flight_log import STATE_CODES  # noqa: E402
from mission_control.mission import load_mission  # noqa: E402

GRID_DT = 0.02


def fit(flight_dir: str) -> dict:
    with open(os.path.join(flight_dir, "events.json"), encoding="utf-8") as handle:
        events = json.load(handle)
    mission = load_mission(os.path.join(ROOT, "missions", f"{events['mission']}.yaml"))
    loop = sitl.read_loop_log(os.path.join(flight_dir, "loop.csv.gz"))
    ulog = sitl.load_ulog(os.path.join(flight_dir, "px4.ulg.gz"), mission.origin)
    est = np.column_stack([loop["est_north"], loop["est_east"], loop["est_down"]])
    offset, scale, _ = sitl.align_clocks(loop["time_s"], est, ulog["est_t"], ulog["est_ned"])

    mask = loop["mission_state"] == STATE_CODES["mission"]
    t = loop["time_s"][mask]
    grid = np.arange(t[0], t[-1], GRID_DT)
    demand = np.column_stack([np.interp(grid, t, loop[f"accel_demand_{a}"][mask])
                              for a in ("n", "e")])
    velocity = np.column_stack([
        np.interp(offset + scale * grid, ulog["truth_vel_t"], ulog["truth_vel"][:, axis])
        for axis in range(2)
    ])
    drag = sitl.sih_quadx_params().drag_linear[0] / sitl.sih_quadx_params().mass
    achieved = np.gradient(velocity, grid, axis=0) + drag * velocity

    best = None
    for delay in np.arange(0.0, 0.40 + 1e-9, GRID_DT):
        shift = int(round(delay / GRID_DT))
        delayed = np.vstack([np.repeat(demand[:1], shift, axis=0), demand[:len(demand) - shift]])
        for tau in np.arange(0.02, 0.60 + 1e-9, GRID_DT):
            alpha = GRID_DT / (tau + GRID_DT)
            response = np.zeros_like(delayed)
            response[0] = delayed[0]
            for index in range(1, len(delayed)):
                response[index] = response[index - 1] + alpha * (delayed[index]
                                                                  - response[index - 1])
            error = float(np.sqrt(np.mean((response - achieved) ** 2)))
            if best is None or error < best[0]:
                best = (error, float(delay), float(tau))
    return {
        "flight": events["flight"],
        "delay_s": round(best[1], 2),
        "lag_tau_s": round(best[2], 2),
        "fit_rms_ms2": round(best[0], 3),
        "demand_rms_ms2": round(float(np.sqrt(np.mean(demand**2))), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flights", nargs="+", help="flight directories")
    args = parser.parse_args()
    for flight in args.flights:
        result = fit(flight)
        print(f"{result['flight']}: delay {result['delay_s']:.2f} s, "
              f"lag {result['lag_tau_s']:.2f} s (fit residual {result['fit_rms_ms2']:.2f} "
              f"of {result['demand_rms_ms2']:.2f} m/s^2 RMS demand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
