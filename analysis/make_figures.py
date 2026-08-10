#!/usr/bin/env python3
"""Regenerate every figure from ``results/runs.csv`` and the saved logs.

Run:  python analysis/make_figures.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from mission_control.analysis import figures  # noqa: E402
from mission_control.logs.flight_log import COLUMNS, FlightLog  # noqa: E402


def load_log(path: str) -> FlightLog:
    """Read a per-run CSV back into a :class:`FlightLog`."""
    table = np.genfromtxt(path, delimiter=",", names=True)
    log = FlightLog()
    log.data = {name: np.atleast_1d(table[name]) for name in COLUMNS}
    log.finalised = True
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    parser.add_argument("--output", default=os.path.join(ROOT, "analysis", "figures"))
    args = parser.parse_args()

    runs_csv = os.path.join(args.results, "runs.csv")
    if not os.path.exists(runs_csv):
        print(f"no results at {runs_csv}; run analysis/run_comparison.py first")
        return 1

    os.makedirs(args.output, exist_ok=True)
    written = [
        figures.rms_by_scenario(runs_csv, os.path.join(args.output, "rms_by_scenario.png")),
        figures.error_distribution(runs_csv, os.path.join(args.output, "error_split.png")),
        figures.compute_cost(runs_csv, os.path.join(args.output, "compute_cost.png")),
    ]

    # A matched pair of trajectories, from the hardest held-out scenario that
    # both controllers actually completed.
    log_dir = os.path.join(args.results, "logs")
    pairs = [
        ("corridor_survey_fast_gusty_round_s1", "corridor survey, fast, gusty"),
        ("corridor_survey_nominal_calm_round_s1", "corridor survey, nominal, calm"),
        ("facade_inspection_nominal_calm_round_s1", "facade inspection, nominal, calm"),
    ]
    for stem, title in pairs:
        pid_path = os.path.join(log_dir, f"{stem}_pid.csv")
        mpc_path = os.path.join(log_dir, f"{stem}_mpc.csv")
        if os.path.exists(pid_path) and os.path.exists(mpc_path):
            written.append(
                figures.trajectory_figure(
                    load_log(pid_path), load_log(mpc_path),
                    os.path.join(args.output, "trajectories.png"),
                    title=f"Matched pair: {title}",
                )
            )
            written.append(
                figures.estimator_figure(
                    load_log(mpc_path),
                    os.path.join(args.output, "estimator.png"),
                    title=f"Estimator performance: {title}",
                )
            )
            break

    for path in written:
        print(f"  wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
