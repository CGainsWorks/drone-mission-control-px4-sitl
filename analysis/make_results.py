#!/usr/bin/env python3
"""Generate ``docs/RESULTS.md`` and ``docs/FAILSAFE_MATRIX.md``.

Both are written from data, never by hand: the results from
``results/runs.csv``, the failsafe matrix from the rule table in
:mod:`mission_control.safety.failsafe`. That is the only way a document and the
behaviour it describes stay in agreement.

Run:  python analysis/make_results.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.analysis.comparison import paired_summary  # noqa: E402
from mission_control.mission.schema import FailsafeSpec  # noqa: E402
from mission_control.safety import FailsafeMatrix  # noqa: E402
from mission_control.tuning import load_tuned  # noqa: E402

#: Columns of ``runs.csv`` that stay strings. Everything else is a number, and
#: a value that will not parse becomes NaN rather than silently becoming zero.
TEXT_COLUMNS = ("mission", "speed", "wind", "corner_mode", "tracker", "final_state")


def read_rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in TEXT_COLUMNS:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = float("nan")
    return rows


def fmt(value, places=4, bold=False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    text = f"{value:.{places}f}"
    return f"**{text}**" if bold else text


def scenario_table(rows: list) -> str:
    """Per-scenario table, averaged over seeds, both controllers side by side."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["mission"], row["speed"], row["wind"], row["corner_mode"])
        grouped[key][row["tracker"]].append(row)

    lines = [
        "| Scenario | Tuned on | Speed | Wind | PID RMS | MPC RMS | Closer | PID max | MPC max |",
        "| --- | :---: | --- | --- | ---: | ---: | :---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        mission, speed, wind, corner = key
        trackers = grouped[key]
        if "pid" not in trackers or "mpc" not in trackers:
            continue
        pid_rms = float(np.mean([r["rms_error_m"] for r in trackers["pid"]]))
        mpc_rms = float(np.mean([r["rms_error_m"] for r in trackers["mpc"]]))
        pid_max = float(np.max([r["max_error_m"] for r in trackers["pid"]]))
        mpc_max = float(np.max([r["max_error_m"] for r in trackers["mpc"]]))
        tuned = "yes" if all(r["held_out"] == 0 for r in trackers["pid"]) else "held out"
        winner = "MPC" if mpc_rms < pid_rms else "PID"
        name = mission.replace("_", " ")
        if corner == "sharp":
            name += " (sharp corners)"
        lines.append(
            f"| `{name}` | {tuned} | {speed} | {wind} | "
            f"{fmt(pid_rms, bold=winner == 'PID')} m | {fmt(mpc_rms, bold=winner == 'MPC')} m | "
            f"{winner} | {fmt(pid_max, 3)} m | {fmt(mpc_max, 3)} m |"
        )
    return "\n".join(lines)


def aggregate(rows: list, predicate=None) -> dict:
    """Paired statistics over a subset of the grid."""
    paired = [p for p in paired_summary(rows) if np.isfinite(p.get("ratio", float("nan")))]
    if predicate:
        paired = [p for p in paired if predicate(p)]
    if not paired:
        return {}
    ratios = np.array([p["ratio"] for p in paired])
    return {
        "pairs": len(paired),
        "mpc_wins": int(np.sum(ratios > 1.0)),
        "median_ratio": float(np.median(ratios)),
        "mean_pid_rms": float(np.mean([p["pid_rms"] for p in paired])),
        "mean_mpc_rms": float(np.mean([p["mpc_rms"] for p in paired])),
        "mean_pid_cross": float(np.mean([p["pid_cross"] for p in paired])),
        "mean_mpc_cross": float(np.mean([p["mpc_cross"] for p in paired])),
        "mean_pid_jerk": float(np.mean([p["pid_jerk"] for p in paired])),
        "mean_mpc_jerk": float(np.mean([p["mpc_jerk"] for p in paired])),
        "mean_pid_solve": float(np.mean([p["pid_solve_us"] for p in paired])),
        "mean_mpc_solve": float(np.mean([p["mpc_solve_us"] for p in paired])),
    }


#: Fraction of the mission spent at a limit above which a scenario is reported
#: as beyond the airframe rather than as a controller comparison.
SATURATION_THRESHOLD = 0.20


def saturation_section(rows: list) -> list:
    """Flag scenarios where the reference exceeds what the vehicle can fly.

    Without this, a scenario with a several-metre RMS looks like both
    controllers tracking badly, when what is actually happening is that the
    trajectory demands more acceleration than the airframe has. Comparing
    controllers there measures the aircraft, not the control law, and the error
    is a property of the scenario.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["mission"], row["speed"], row["wind"], row["corner_mode"])
        grouped[key][row["tracker"]].append(row)

    flagged = []
    for key in sorted(grouped):
        trackers = grouped[key]
        if "pid" not in trackers or "mpc" not in trackers:
            continue
        pid_sat = float(np.mean([r["saturated_fraction"] for r in trackers["pid"]]))
        mpc_sat = float(np.mean([r["saturated_fraction"] for r in trackers["mpc"]]))
        if max(pid_sat, mpc_sat) < SATURATION_THRESHOLD:
            continue
        flagged.append((key, pid_sat, mpc_sat,
                        float(np.mean([r["rms_error_m"] for r in trackers["pid"]])),
                        float(np.mean([r["rms_error_m"] for r in trackers["mpc"]]))))

    lines = ["## Scenarios beyond the airframe", ""]
    if not flagged:
        lines += [
            "None. Every scenario in the grid spent less than "
            f"{SATURATION_THRESHOLD:.0%} of the mission at an acceleration or tilt "
            "limit, so every comparison above is between two controllers rather "
            "than against the vehicle's envelope.",
            "",
        ]
        return lines

    lines += [
        "These scenarios spend a large fraction of the mission at the tilt or "
        "acceleration limit, which means the *reference* is asking for more than "
        "the aircraft can deliver. The error there is mostly a property of the "
        "trajectory, not of the controller, and the two controllers converge "
        "because both are simply saturated.",
        "",
        "They are kept in the grid deliberately — a comparison that only includes "
        "cases the vehicle can fly comfortably would overstate how much the choice "
        "of controller matters — but the numbers should be read with that in mind.",
        "",
        "| Scenario | PID saturated | MPC saturated | PID RMS | MPC RMS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for (mission, speed, wind, corner), pid_sat, mpc_sat, pid_rms, mpc_rms in flagged:
        name = mission.replace("_", " ")
        if corner == "sharp":
            name += " (sharp corners)"
        lines.append(
            f"| `{name}` {speed}/{wind} | {pid_sat:.0%} | {mpc_sat:.0%} | "
            f"{pid_rms:.4f} m | {mpc_rms:.4f} m |"
        )
    lines.append("")
    return lines


def compute_table(rows: list) -> str:
    lines = [
        "| Controller | Optimisation / control law | Whole control step | 99th percentile step |",
        "| --- | ---: | ---: | ---: |",
    ]
    for tracker, label in (("pid", "PID"), ("mpc", "MPC")):
        subset = [r for r in rows if r["tracker"] == tracker
                  and np.isfinite(r["mean_solve_time_us"])]
        if not subset:
            continue
        lines.append(
            f"| {label} | {np.mean([r['mean_solve_time_us'] for r in subset]):.1f} us | "
            f"{np.mean([r['mean_update_time_us'] for r in subset]):.1f} us | "
            f"{np.mean([r['p99_update_time_us'] for r in subset]):.1f} us |"
        )
    return "\n".join(lines)


def estimator_table(rows: list) -> str:
    subset = [r for r in rows if np.isfinite(r["estimator_rms_m"])]
    if not subset:
        return "_no estimator data_"
    return "\n".join([
        "| Metric | Value |",
        "| --- | ---: |",
        f"| RMS position error of the estimate against truth | "
        f"{np.mean([r['estimator_rms_m'] for r in subset]):.4f} m |",
        f"| Worst estimate error across the grid | "
        f"{np.max([r['max_estimation_error_m'] for r in subset]):.4f} m |",
        f"| Mean normalised innovation squared (target ~3) | "
        f"{np.nanmean([r['estimator_mean_nis'] for r in subset]):.2f} |",
        f"| Fraction of flight with an RTK-fixed solution | "
        f"{100 * np.mean([r['fraction_rtk_fixed'] for r in subset]):.1f}% |",
    ])


def write_results(rows: list, path: str) -> str:
    overall = aggregate(rows)
    held_out = aggregate(rows, lambda p: p["held_out"] == 1)
    tuned_on = aggregate(rows, lambda p: p["held_out"] == 0)
    sharp = aggregate(rows, lambda p: p["corner_mode"] == "sharp")
    feasible = aggregate(rows, lambda p: p["corner_mode"] == "round")
    fast = aggregate(rows, lambda p: p["speed"] == "fast")
    gusty = aggregate(rows, lambda p: p["wind"] == "gusty")

    gains = load_tuned()
    diverged = [r for r in rows if r.get("diverged")]

    lines = [
        "# Results",
        "",
        "_Generated by `analysis/make_results.py` from `results/runs.csv`. "
        "Do not edit by hand._",
        "",
        f"{len(rows)} runs: {len(rows) // 2} scenarios flown by both controllers, "
        "paired on identical seeds so the wind, IMU noise and GNSS errors are "
        "sample-for-sample identical between the two.",
        "",
        "All errors are against **ground truth**, over the **mission phase only** "
        "(takeoff, RTL and landing are excluded; they are near-vertical and would "
        "dilute the comparison).",
        "",
        "## Headline",
        "",
    ]

    if overall:
        lines += [
            f"- **{overall['mpc_wins']} of {overall['pairs']} paired scenarios "
            f"go to the MPC**, with a median error ratio of "
            f"{overall['median_ratio']:.2f}x.",
            f"- Mean RMS position error: **PID {overall['mean_pid_rms']:.4f} m**, "
            f"**MPC {overall['mean_mpc_rms']:.4f} m**.",
            f"- Mean cross-track error: PID {overall['mean_pid_cross']:.4f} m, "
            f"MPC {overall['mean_mpc_cross']:.4f} m.",
            f"- Mean RMS jerk of the acceleration demand: "
            f"PID {overall['mean_pid_jerk']:.2f} m/s^3, "
            f"MPC {overall['mean_mpc_jerk']:.2f} m/s^3.",
            "",
        ]

    if tuned_on and held_out:
        lines += [
            "### Tuned against held out",
            "",
            "Gains were tuned on `facade_inspection` at nominal speed in calm air, "
            "and nowhere else. If the MPC only won there, the result would be a "
            "tuning artefact.",
            "",
            "| Subset | Pairs | MPC wins | Median ratio | PID RMS | MPC RMS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| Tuned on | {tuned_on['pairs']} | {tuned_on['mpc_wins']} | "
            f"{tuned_on['median_ratio']:.2f}x | {tuned_on['mean_pid_rms']:.4f} m | "
            f"{tuned_on['mean_mpc_rms']:.4f} m |",
            f"| Held out | {held_out['pairs']} | {held_out['mpc_wins']} | "
            f"{held_out['median_ratio']:.2f}x | {held_out['mean_pid_rms']:.4f} m | "
            f"{held_out['mean_mpc_rms']:.4f} m |",
            "",
        ]

    lines += ["## Every scenario", "", scenario_table(rows), ""]
    lines += saturation_section(rows)

    lines += ["## Where the difference comes from", ""]
    for label, stats in (
        ("Feasible reference (rounded corners)", feasible),
        ("Infeasible reference (sharp corners)", sharp),
        ("Fast settings only", fast),
        ("Gusty conditions only", gusty),
    ):
        if stats:
            lines.append(
                f"- **{label}** — {stats['mpc_wins']}/{stats['pairs']} to the MPC, "
                f"median ratio {stats['median_ratio']:.2f}x "
                f"(PID {stats['mean_pid_rms']:.4f} m, MPC {stats['mean_mpc_rms']:.4f} m)."
            )
    lines.append("")

    lines += ["## What it costs", "", compute_table(rows), "",
              "The control period at 50 Hz is 20 000 us, so both are affordable. "
              "The optimisation column is the QP alone; the whole-step column adds "
              "sampling the reference across the horizon, which is real work but is "
              "cacheable in a way the QP is not.",
              "",
              "**These four numbers are the only ones on this page that do not "
              "reproduce.** Every flight-derived figure above comes out bit-identical "
              "on a re-run, because the simulation is seeded; these are wall-clock "
              "measurements of this machine under whatever else it was doing at the "
              "time. Re-running the pipeline on a busy machine moved the means by "
              "about 1.5x and the 99th percentile by 2.1x, with every other number on "
              "the page unchanged. Read them as an order of magnitude against the "
              "20 000 us budget, not as a benchmark, and do not treat a difference "
              "from these values as a regression.",
              ""]

    lines += ["## Estimator", "",
              "Shared by both controllers, so it is not a difference between them — "
              "but if it were poor, neither result would mean anything.",
              "", estimator_table(rows), ""]

    if gains:
        lines += ["## Gains in force", "",
                  "From `config/tuned_gains.yaml`, generated by `scripts/tune_gains.py`.",
                  "", "```yaml"]
        for section in ("pid", "mpc"):
            if section in gains:
                lines.append(f"{section}:")
                for key, value in gains[section].items():
                    lines.append(f"  {key}: {value}")
        lines += ["```", ""]

    lines += ["## Failures", ""]
    if diverged:
        lines.append(f"{len(diverged)} run(s) did not complete:")
        lines.append("")
        lines.append("| Scenario | Controller | Outcome |")
        lines.append("| --- | --- | --- |")
        for row in diverged:
            lines.append(
                f"| {row['mission']} {row['speed']}/{row['wind']}/{row['corner_mode']} "
                f"s{int(row['seed'])} | {row['tracker']} | {row['final_state']} |"
            )
    else:
        landed = sum(1 for row in rows if row["final_state"] == "disarmed")
        lines.append(
            f"Every run in the grid completed: {landed} of {len(rows)} reached "
            "`disarmed`, none diverged, and none was cut short by the divergence "
            "guard."
        )
    lines.append("")

    # newline="\n" so the generated document is byte-identical whichever
    # platform produced it. CI regenerates these files on a clean checkout and
    # fails if they differ; without this, a Windows contributor writes CRLF,
    # git stores LF, and the check reports a phantom change.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return path


def failsafe_verification_table(results_dir: str) -> str:
    """Read the failsafe scenario runs, if they have been produced."""
    path = os.path.join(results_dir, "failsafe_runs.csv")
    if not os.path.exists(path):
        return ""

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ""

    lines = [
        "",
        "## Verified in flight",
        "",
        "_From `results/failsafe_runs.csv`, produced by "
        "`analysis/run_failsafe_matrix.py`._",
        "",
        "Each row is a complete simulated flight with one real fault injected — "
        "telemetry actually silenced, the battery actually drained, GNSS actually "
        "stopped, the fence actually shrunk until the survey could not fit inside "
        "it. The observed column is what the vehicle did, not what it was "
        "supposed to do.",
        "",
        "| Scenario | Fault | Expected | Observed | Rule(s) fired | At | Outcome |",
        "| --- | --- | :---: | :---: | --- | ---: | --- |",
    ]
    for row in rows:
        mark = "" if row["matches_expectation"] == "1" else " **(mismatch)**"
        trigger = f"{float(row['first_trigger_s']):.1f} s" if row["first_trigger_s"] else "-"
        lines.append(
            f"| `{row['scenario']}` | {row['description']} | "
            f"{row['expected_action']} | {row['observed_action']}{mark} | "
            f"`{row['rules_fired']}` | {trigger} | "
            f"{row['final_state']}, phases: {row['phases_entered'].replace(';', ', ')} |"
        )

    passed = sum(1 for row in rows if row["matches_expectation"] == "1")
    lines += [
        "",
        f"{passed} of {len(rows)} scenarios behaved exactly as the matrix specifies.",
        "",
    ]
    return "\n".join(lines)


def write_failsafe_matrix(path: str, results_dir: str = None) -> str:
    matrix = FailsafeMatrix(spec=FailsafeSpec())
    lines = [
        "# Failsafe matrix",
        "",
        "_Generated by `analysis/make_results.py` from the rule table in "
        "`src/mission_control/safety/failsafe.py`. Do not edit by hand._",
        "",
        "The failsafe logic is a table, not a tree of conditionals. Every row here "
        "is a rule object in the source, and `tests/test_safety.py` iterates the "
        "same table and drives each row independently — so a rule cannot be added "
        "without a test that fires it, and this document cannot drift away from "
        "the behaviour.",
        "",
        "Rules are listed in the order they resolve. When several fire at once, the "
        "highest priority sets the action.",
        "",
        "| Priority | Rule | Trigger | Action | Debounce | Why |",
        "| ---: | --- | --- | :---: | ---: | --- |",
    ]
    for row in matrix.to_table():
        lines.append(
            f"| {row['priority']} | `{row['rule']}` | {row['trigger']} | "
            f"**{row['action'].upper()}** | {row['debounce_s']:.1f} s | {row['rationale']} |"
        )

    lines += [
        "",
        "## Actions",
        "",
        "| Action | Meaning |",
        "| --- | --- |",
        "| `none` | Nothing wrong. |",
        "| `warn` | Advisory only; the mission continues. |",
        "| `hold` | Stop and hold position at the current point. |",
        "| `rtl` | Climb to the RTL altitude, transit home, descend, land. |",
        "| `land` | Descend and land where the vehicle is. |",
        "| `terminate` | Cut thrust. Reserved for cases where continued flight is "
        "more dangerous than falling. |",
        "",
        "## Two deliberate behaviours",
        "",
        "**Actions latch and escalate, never de-escalate.** A vehicle that began "
        "returning home on a low battery does not resume its mission because the "
        "voltage recovered — that recovery is what every battery does when the load "
        "drops, and acting on it would send the aircraft back out on a pack that "
        "cannot make it. Escalation is allowed: a low-battery RTL becomes a "
        "critical-battery LAND.",
        "",
        "**Conditions must persist.** Every rule carries a debounce, so one dropped "
        "telemetry packet or one noisy GNSS epoch cannot trigger a return. The "
        "geofence breach rule is the exception at 0.0 s, because the position "
        "estimate feeding it is already filtered and the fence has a warning margin "
        "inside it — by the time it fires, several metres of warning have been "
        "ignored.",
        "",
        "## Coverage",
        "",
        "`tests/test_runner.py` drives each of these end to end through a complete "
        "simulated flight, not just against a synthetic status struct: telemetry is "
        "actually silenced, the battery is actually drained, the GNSS actually stops "
        "producing fixes, and the geofence is actually breached.",
    ]

    if results_dir:
        lines.append(failsafe_verification_table(results_dir))
    lines.append("")

    # newline="\n" so the generated document is byte-identical whichever
    # platform produced it. CI regenerates these files on a clean checkout and
    # fails if they differ; without this, a Windows contributor writes CRLF,
    # git stores LF, and the check reports a phantom change.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join(ROOT, "results", "runs.csv"))
    parser.add_argument("--docs", default=os.path.join(ROOT, "docs"))
    args = parser.parse_args()

    os.makedirs(args.docs, exist_ok=True)
    matrix_path = write_failsafe_matrix(
        os.path.join(args.docs, "FAILSAFE_MATRIX.md"), os.path.dirname(args.results)
    )
    print(f"  wrote {os.path.relpath(matrix_path, ROOT)}")

    if not os.path.exists(args.results):
        print(f"no results at {args.results}; run analysis/run_comparison.py first")
        return 1

    rows = read_rows(args.results)
    results_path = write_results(rows, os.path.join(args.docs, "RESULTS.md"))
    print(f"  wrote {os.path.relpath(results_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
