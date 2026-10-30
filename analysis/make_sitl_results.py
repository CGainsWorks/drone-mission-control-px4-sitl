#!/usr/bin/env python3
"""Turn the PX4 SITL flights into the committed summary, document and figure.

Reads every flight directory under ``--flights`` (``loop.csv.gz``,
``events.json`` and PX4's ``px4.ulg.gz``) and writes:

    sitl/results/summary.json      every flight's metrics and outcome
    docs/SITL_RESULTS.md           the same, as a document
    analysis/figures/sitl_tracking.png

and, with ``--write-readme``, the generated SITL table in README.md.

``--check`` exits non-zero unless every tracking flight completed and every
failsafe flight did what its row of the matrix says. CI uses it after flying
the campaign, and ``--check-only`` applies the same gate to the one-flight smoke run
without writing anything that is committed.

All numbers are simulated: PX4 v1.17.0 SIH, not an aircraft.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control import sitl  # noqa: E402
from mission_control.mission import load_mission  # noqa: E402

FLIGHTS = os.path.join(ROOT, "sitl", "results", "flights")
#: Flights with the reference comparison's gains, kept as the evidence for why
#: the campaign uses gains tuned for PX4 SIH. Summarised, never gated.
TRANSFER = os.path.join(ROOT, "sitl", "results", "transfer")
SUMMARY = os.path.join(ROOT, "sitl", "results", "summary.json")
DOCUMENT = os.path.join(ROOT, "docs", "SITL_RESULTS.md")
FIGURE = os.path.join(ROOT, "analysis", "figures", "sitl_tracking.png")
README = os.path.join(ROOT, "README.md")


def mission_for(name: str):
    return load_mission(os.path.join(ROOT, "missions", f"{name}.yaml"))


def collect(flights_dir: str) -> list:
    flights = []
    for name in sorted(os.listdir(flights_dir)):
        path = os.path.join(flights_dir, name)
        if not os.path.isfile(os.path.join(path, "events.json")):
            continue
        with open(os.path.join(path, "events.json"), encoding="utf-8") as handle:
            mission_name = json.load(handle)["mission"]
        summary = sitl.summarise_flight(path, mission_for(mission_name))
        summary["matches_expectation"] = sitl.failsafe_matches(summary)
        flights.append(summary)
    return flights


def build_summary(flights: list, transfer: list = ()) -> dict:
    failsafes = []
    for flight in flights:
        spec = sitl.SITL_FAILSAFE_SCENARIOS.get(flight["scenario"])
        if spec is None:
            continue
        failsafes.append({
            "scenario": flight["scenario"],
            "description": spec["description"],
            "expected_action": spec["expect_action"],
            "expected_rule": spec["expect_rule"],
            "observed_action": flight["observed_action"],
            "rules_fired": flight["rules_fired"],
            "landed": flight.get("landed"),
            "px4_modes": flight.get("px4_modes", []),
            "matches_expectation": flight["matches_expectation"],
        })
    return {
        "note": "Simulated: PX4 SIH software-in-the-loop, not flight hardware.",
        "px4_version": sitl.PX4_VERSION,
        "airframe": "PX4 SIH quadrotor (airframe 10040, sihsim_quadx)",
        "flights": flights,
        "comparison": sitl.tracker_comparison(flights),
        "failsafes": failsafes,
        "gains_file": "config/tuned_gains_px4_sih.yaml",
        "transfer": [
            {key: flight.get(key) for key in (
                "flight", "tracker", "result", "rms_error_m", "max_error_m",
                "rms_estimation_error_m", "rms_jerk_ms3", "saturated_fraction")}
            for flight in transfer
        ],
    }


def gate(summary: dict) -> list:
    problems = []
    for flight in summary["flights"]:
        if flight.get("simulator_faults"):
            problems.append(f"{flight['flight']}: the simulator stalled "
                            f"({flight['simulator_faults'][0]}); re-fly it")
        if flight["scenario"] == "none":
            if flight["result"] != "completed":
                problems.append(f"{flight['flight']}: {flight['result']} ({flight['end_reason']})")
            elif flight.get("rms_error_m") is None:
                problems.append(f"{flight['flight']}: no tracking metrics")
            elif flight["rms_error_m"] > sitl.TRACKING_GATE_M:
                # Completing is not enough. A tracker that limit-cycles can
                # drag PX4's EKF away from the truth and then "complete" the
                # mission tens of metres from where it believes it is; this
                # happened, with the reference gains, before the SITL gains.
                problems.append(
                    f"{flight['flight']}: RMS error to truth {flight['rms_error_m']:.2f} m, "
                    f"above the {sitl.TRACKING_GATE_M:.1f} m gate")
        elif not flight["matches_expectation"]:
            expected = sitl.SITL_FAILSAFE_SCENARIOS[flight["scenario"]]["expect_action"]
            rules = ", ".join(flight["rules_fired"]) or "none"
            problems.append(
                f"{flight['flight']}: expected {flight['scenario']} to give {expected}, "
                f"observed {flight['observed_action']} ({rules})"
            )
    return problems


def metres(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f} m"


def render_document(summary: dict) -> str:
    lines = [
        "# PX4 SITL results",
        "",
        "_Generated by `analysis/make_sitl_results.py` from `sitl/results/flights/`. "
        "Do not edit by hand._",
        "",
        f"**Simulated.** Every number here comes from PX4 {summary['px4_version']} "
        "software-in-the-loop with PX4's own SIH physics model "
        f"({summary['airframe']}), driven over MAVLink by this repository's mission "
        "layer and trackers, with the gains tuned for this airframe in "
        f"`{summary['gains_file']}`. No aircraft was flown.",
        "",
        "Errors are against the simulator's ground truth from PX4's ULog "
        "(`vehicle_global_position_groundtruth`), over the mission phase only, using "
        "the same `tracking_metrics` as the reference comparison in "
        "[`RESULTS.md`](RESULTS.md). Each flight is one run in still air (SIH has no "
        "wind model), so these are single samples and not a statistical comparison.",
        "",
        "## Trackers",
        "",
        "| Mission | Tracker | RMS error | Max error | RMS cross-track | RMS jerk of demand "
        "| Saturated | Loop rate | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for flight in summary["flights"]:
        if flight["scenario"] != "none":
            continue
        saturated = flight.get("saturated_fraction")
        jerk = flight.get("rms_jerk_ms3")
        lines.append(
            f"| `{flight['mission'].replace('_', ' ')}` | {flight['tracker'].upper()} | "
            f"{metres(flight.get('rms_error_m'))} | {metres(flight.get('max_error_m'))} | "
            f"{metres(flight.get('rms_cross_track_m'))} | "
            f"{'n/a' if jerk is None else f'{jerk:.1f} m/s^3'} | "
            f"{'n/a' if saturated is None else f'{saturated:.0%}'} | "
            f"{flight.get('loop_rate_hz') or 0:.0f} Hz | {flight['result']} |"
        )
    lines += [
        "",
        "## Failsafe matrix",
        "",
        "Each row injects one real fault into PX4 or into the MAVLink path on the "
        f"first leg of `facade_inspection`, {sitl.INJECT_AFTER_MISSION_S:.0f} s into the "
        "survey, with the MPC flying. The matrix sits above the tracker, so the "
        "choice of tracker does not change which rule fires.",
        "",
        "| Fault | How it was injected | Expected | Observed | Rules fired "
        "| PX4 modes | Landed | As specified |",
        "| --- | --- | --- | --- | --- | --- | :---: | :---: |",
    ]
    for row in summary["failsafes"]:
        modes = " > ".join(dict.fromkeys(row["px4_modes"])) or "n/a"
        lines.append(
            f"| {row['scenario'].replace('_', ' ')} | {row['description']} | "
            f"{row['expected_action']} | {row['observed_action']} | "
            f"{', '.join(row['rules_fired']) or 'none'} | {modes} | "
            f"{'yes' if row['landed'] else 'no'} | "
            f"{'yes' if row['matches_expectation'] else '**no**'} |"
        )
    if summary.get("transfer"):
        lines += [
            "",
            "## The reference gains, unchanged",
            "",
            "Flown before the campaign, with `config/tuned_gains.yaml` (tuned for the "
            "reference airframe) instead of the gains tuned for PX4 SIH. Kept because "
            "they are why the campaign re-tunes: see [`sitl/README.md`](../sitl/README.md). "
            "Logs in `sitl/results/transfer/`.",
            "",
            "| Flight | RMS error | Max error | PX4 estimate error | RMS jerk of demand "
            "| Saturated | Result |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for flight in summary["transfer"]:
            jerk = flight.get("rms_jerk_ms3")
            saturated = flight.get("saturated_fraction")
            lines.append(
                f"| `{flight['flight']}` | {metres(flight.get('rms_error_m'))} | "
                f"{metres(flight.get('max_error_m'))} | "
                f"{metres(flight.get('rms_estimation_error_m'))} | "
                f"{'n/a' if jerk is None else f'{jerk:.1f} m/s^3'} | "
                f"{'n/a' if saturated is None else f'{saturated:.0%}'} | {flight['result']} |"
            )
    lines += [
        "",
        "## Flights",
        "",
        "| Flight | Duration of survey | Clock alignment residual | ULog (gzip) | Attempts "
        "| End |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for flight in summary["flights"]:
        duration = flight.get("duration_s")
        lines.append(
            f"| `{flight['flight']}` | "
            f"{'n/a' if duration is None else f'{duration:.0f} s'} | "
            f"{metres(flight.get('clock_alignment_residual_m'))} | "
            f"{(flight.get('ulog_bytes') or 0) / 1024:.0f} kB | {flight.get('attempts', 1)} | "
            f"{flight['end_reason']} |"
        )
    lines += [
        "",
        "The clock alignment residual is what is left after superimposing the "
        "position estimate the mission layer received over MAVLink on the one PX4 "
        "logged: MAVLink transport and sampling. It bounds how much of each error "
        "above could be an artefact of putting two clocks on one axis.",
        "",
        "PX4 runs on the wall clock (the nolockstep build), so a host that stops "
        "scheduling it for long enough makes PX4 report a simulated sensor as timed "
        "out, and its EKF degrades. A flight where PX4 logged that is a fault in the "
        "simulation rather than in anything under test, so the campaign re-flies it; "
        "the attempts column counts the flights, and `attempts.json` beside each "
        "flight records why.",
        "",
    ]
    return "\n".join(lines)


def make_figure(flights_dir: str, summary: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from mission_control.logs.flight_log import STATE_CODES

    rows = summary["comparison"]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.6 * len(rows)), squeeze=False)
    colours = {"pid": "#c2410c", "mpc": "#1d4ed8"}
    for axis, row in zip(axes[:, 0], rows):
        for tracker in sitl.TRACKERS:
            directory = os.path.join(flights_dir, f"{row['mission']}-{tracker}")
            loop = sitl.read_loop_log(os.path.join(directory, "loop.csv.gz"))
            mission = mission_for(row["mission"])
            ulog = sitl.load_ulog(os.path.join(directory, "px4.ulg.gz"), mission.origin)
            est = np.column_stack([loop["est_north"], loop["est_east"], loop["est_down"]])
            offset, scale, _ = sitl.align_clocks(loop["time_s"], est, ulog["est_t"],
                                                 ulog["est_ned"])
            log = sitl.build_flight_log(loop, ulog, offset, scale)
            mask = log["mission_state"] == STATE_CODES["mission"]
            t = log["time_s"][mask]
            error = np.linalg.norm(log.position_error()[mask], axis=1)
            axis.plot(t - t[0], error, color=colours[tracker], lw=1.0,
                      label=f"{tracker.upper()}  RMS {row[f'{tracker}_rms_m']:.2f} m")
        axis.set_title(f"{row['mission'].replace('_', ' ')} (PX4 SIH, simulated)", fontsize=10)
        axis.set_ylabel("error to truth (m)")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("time into survey (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flights", default=FLIGHTS)
    parser.add_argument("--check", action="store_true",
                        help="fail unless every flight behaved as specified")
    parser.add_argument("--check-only", action="store_true",
                        help="summarise and gate only; write nothing that is committed")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()

    flights = collect(args.flights)
    transfer = collect(TRANSFER) if (os.path.isdir(TRANSFER) and not args.check_only) else []
    if not flights:
        print(f"no flights under {os.path.relpath(args.flights, ROOT)}")
        return 1
    summary = build_summary(flights, transfer)

    for flight in flights:
        rms = flight.get("rms_error_m")
        print(f"  {flight['flight']:<32} {flight['result']:<9} "
              f"rms={'n/a' if rms is None else f'{rms:.3f} m'}  "
              f"action={flight['observed_action']}  rules={','.join(flight['rules_fired']) or '-'}")

    if not args.check_only:
        write_text(SUMMARY, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        write_text(DOCUMENT, render_document(summary))
        make_figure(args.flights, summary, FIGURE)
        print(f"wrote {os.path.relpath(SUMMARY, ROOT)}, {os.path.relpath(DOCUMENT, ROOT)}, "
              f"{os.path.relpath(FIGURE, ROOT)}")
        if args.write_readme:
            with open(README, encoding="utf-8") as handle:
                text = handle.read()
            write_text(README, sitl.replace_block(text, sitl.readme_sitl_block(summary)))
            print("updated the SITL table in README.md")

    if args.check or args.check_only:
        problems = gate(summary)
        if problems:
            print("\nSITL check FAILED:")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"\nSITL check passed: {len(flights)} flight(s) behaved as specified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
