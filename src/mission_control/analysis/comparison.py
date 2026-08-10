"""The MPC-versus-PID comparison harness.

This is the thing the repository exists to produce. It runs every scenario in
the grid with both controllers, computes the same metrics for both from the same
code path, and writes the raw results as CSV. Nothing downstream — the figures,
``docs/RESULTS.md``, the numbers in the README — is written by hand; all of it is
generated from ``results/runs.csv``.

The harness was built before either controller was tuned, on purpose. Building
the measurement after the thing being measured is how you end up with a
benchmark shaped to flatter whichever result you already had.
"""

from __future__ import annotations

import csv
import json
import os
import time

from ..runner import run_scenario
from .metrics import summarise
from .scenarios import ScenarioSpec, build_grid, make_config

TRACKERS = ("pid", "mpc")

#: Columns of ``results/runs.csv``. One row per (scenario, controller).
RESULT_COLUMNS = (
    "mission", "speed", "wind", "corner_mode", "seed", "tracker", "held_out",
    "completed", "diverged", "final_state", "flight_duration_s",
    "rms_error_m", "mean_error_m", "max_error_m", "p95_error_m",
    "rms_horizontal_m", "rms_vertical_m",
    "rms_cross_track_m", "max_cross_track_m",
    "rms_along_track_m", "mean_along_track_m",
    "rms_estimation_error_m", "max_estimation_error_m",
    "rms_accel_demand_ms2", "max_accel_demand_ms2", "rms_jerk_ms3",
    "saturated_fraction", "outside_fraction", "longest_excursion_s",
    "mean_solve_time_us", "p99_solve_time_us",
    "mean_update_time_us", "p99_update_time_us",
    "mean_speed_ms", "samples",
    "estimator_rms_m", "estimator_mean_nis", "fraction_rtk_fixed",
    "wall_time_s",
)


def run_one(spec: ScenarioSpec, tracker: str, log_dir: str = None) -> dict:
    """Run a single (scenario, controller) pair and return one result row."""
    config = make_config(spec, tracker)
    started = time.perf_counter()
    result = run_scenario(config)
    wall = time.perf_counter() - started

    metrics = summarise(result.log)
    mission = metrics["mission"]
    settling = metrics["settling"]
    estimator = metrics["estimator"]

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        stem = (f"{spec.mission_name}_{spec.speed}_{spec.wind}_{spec.corner_mode}"
                f"_s{spec.seed}_{tracker}")
        result.log.to_csv(os.path.join(log_dir, stem + ".csv"))
        result.log.to_ulog(
            os.path.join(log_dir, stem + ".ulg"),
            info={"mission": spec.mission_name, "tracker": tracker},
            parameters=_flatten_parameters(result.tracker_description),
        )

    row = {
        "mission": spec.mission_name,
        "speed": spec.speed,
        "wind": spec.wind,
        "corner_mode": spec.corner_mode,
        "seed": spec.seed,
        "tracker": tracker,
        "held_out": int(spec.is_held_out),
        "completed": int(result.completed),
        "diverged": int(result.diverged),
        "final_state": result.final_state,
        "flight_duration_s": round(result.duration_s, 3),
        "wall_time_s": round(wall, 2),
        "estimator_rms_m": estimator["rms_error_m"],
        "estimator_mean_nis": estimator["mean_position_nis"],
        "fraction_rtk_fixed": estimator["fraction_rtk_fixed"],
        "outside_fraction": settling["outside_fraction"],
        "longest_excursion_s": settling["longest_excursion_s"],
    }
    for key in (
        "rms_error_m", "mean_error_m", "max_error_m", "p95_error_m",
        "rms_horizontal_m", "rms_vertical_m", "rms_cross_track_m",
        "max_cross_track_m", "rms_along_track_m", "mean_along_track_m",
        "rms_estimation_error_m", "max_estimation_error_m",
        "rms_accel_demand_ms2", "max_accel_demand_ms2", "rms_jerk_ms3",
        "saturated_fraction", "mean_solve_time_us", "p99_solve_time_us",
        "mean_update_time_us", "p99_update_time_us", "mean_speed_ms", "samples",
    ):
        row[key] = mission.get(key, float("nan"))
    return row


def _flatten_parameters(description: dict) -> dict:
    """Flatten a tracker description into ULog-writable scalars."""
    flat = {}
    for key, value in description.items():
        if isinstance(value, (int, float, str, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, (int, float)):
                    flat[f"{key}_{index}"] = item
    return flat


def _run_job(job):
    """Worker entry point. Must be importable at module level for pickling."""
    spec, tracker, log_dir = job
    try:
        return run_one(spec, tracker, log_dir)
    except Exception as error:  # a single bad cell must not lose the whole grid
        return {
            "mission": spec.mission_name, "speed": spec.speed, "wind": spec.wind,
            "corner_mode": spec.corner_mode, "seed": spec.seed, "tracker": tracker,
            "held_out": int(spec.is_held_out), "completed": 0, "diverged": 1,
            "final_state": f"error: {type(error).__name__}: {error}",
            "flight_duration_s": 0.0, "wall_time_s": 0.0,
        }


def run_comparison(output_dir: str, quick: bool = False, workers: int = None,
                   save_logs: bool = True, progress=print) -> list:
    """Run the whole grid and write ``runs.csv``. Returns the result rows."""
    specs = build_grid(quick=quick)
    log_dir = os.path.join(output_dir, "logs") if save_logs else None
    jobs = [(spec, tracker, log_dir) for spec in specs for tracker in TRACKERS]

    os.makedirs(output_dir, exist_ok=True)
    if log_dir and os.path.isdir(log_dir):
        # Clear stale logs first. Without this a `--quick` run leaves the
        # previous full grid's logs in place, `runs.csv` and `results/logs`
        # describe different experiments, and the figures can silently plot a
        # trajectory from a configuration that no longer exists.
        removed = 0
        for name in os.listdir(log_dir):
            if name.endswith((".csv", ".ulg")):
                os.remove(os.path.join(log_dir, name))
                removed += 1
        if removed:
            progress(f"cleared {removed} stale log file(s) from {log_dir}")
    progress(f"{len(jobs)} runs ({len(specs)} scenarios x {len(TRACKERS)} controllers)")

    from ..parallel import limit_blas_threads, worker_count

    # Entry-point scripts call this before importing numpy, which is the only
    # way it can affect *this* process. Calling it again here costs nothing and
    # guarantees the spawned children inherit the limit even when the harness is
    # driven from a notebook or another script that forgot to.
    limit_blas_threads()
    workers = worker_count(workers)

    started = time.perf_counter()
    rows = []
    if workers > 1:
        import multiprocessing

        # 'spawn' is the only start method available on Windows and is the
        # default on macOS, so it is used everywhere rather than having the
        # harness behave differently by platform.
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers) as pool:
            for index, row in enumerate(pool.imap_unordered(_run_job, jobs), start=1):
                rows.append(row)
                progress(f"  [{index}/{len(jobs)}] {row['mission']} "
                         f"{row['speed']}/{row['wind']}/{row['corner_mode']} "
                         f"s{row['seed']} {row['tracker']}: "
                         f"RMS {row.get('rms_error_m', float('nan')):.4f} m")
    else:
        for index, job in enumerate(jobs, start=1):
            row = _run_job(job)
            rows.append(row)
            progress(f"  [{index}/{len(jobs)}] {row['mission']} {row['tracker']}: "
                     f"RMS {row.get('rms_error_m', float('nan')):.4f} m")

    # Sorted so the CSV is byte-identical between runs regardless of the order
    # the worker pool happened to finish in. Without this the "committed results
    # are reproducible" check in CI would fail on scheduling noise alone.
    rows.sort(key=lambda r: (r["mission"], r["speed"], r["wind"], r["corner_mode"],
                             r["seed"], r["tracker"]))

    path = os.path.join(output_dir, "runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.6g}" if isinstance(value, float) else value)
                for key, value in row.items()
            })

    elapsed = time.perf_counter() - started
    progress(f"wrote {path} ({len(rows)} rows) in {elapsed:.1f} s "
             f"on {workers} worker(s)")

    with open(os.path.join(output_dir, "comparison_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": len(rows),
                "scenarios": len(specs),
                "trackers": list(TRACKERS),
                "quick": quick,
                "workers": workers,
                "wall_time_s": round(elapsed, 1),
                "diverged": int(sum(row.get("diverged", 0) for row in rows)),
                "completed": int(sum(row.get("completed", 0) for row in rows)),
            },
            handle,
            indent=2,
        )
    return rows


def paired_summary(rows: list) -> list:
    """Pair the two controllers on identical scenarios and difference them.

    Pairing is what makes the comparison sensitive. Averaging each controller
    over the grid and subtracting would bury a consistent difference under the
    variance between scenarios, which is far larger than the variance between
    controllers on the same scenario.
    """
    by_key = {}
    for row in rows:
        key = (row["mission"], row["speed"], row["wind"], row["corner_mode"], row["seed"])
        by_key.setdefault(key, {})[row["tracker"]] = row

    paired = []
    for key, pair in sorted(by_key.items()):
        if not all(tracker in pair for tracker in TRACKERS):
            continue
        pid, mpc = pair["pid"], pair["mpc"]
        if not (pid.get("completed") and mpc.get("completed")):
            # A pair where either side failed is reported, but its ratio is not
            # meaningful, so it is flagged rather than silently averaged in.
            paired.append({
                "mission": key[0], "speed": key[1], "wind": key[2],
                "corner_mode": key[3], "seed": key[4],
                "pid_rms": pid.get("rms_error_m", float("nan")),
                "mpc_rms": mpc.get("rms_error_m", float("nan")),
                "ratio": float("nan"),
                "winner": "incomplete",
                "held_out": pid.get("held_out", 1),
            })
            continue
        ratio = pid["rms_error_m"] / mpc["rms_error_m"] if mpc["rms_error_m"] > 0 else float("inf")
        paired.append({
            "mission": key[0], "speed": key[1], "wind": key[2],
            "corner_mode": key[3], "seed": key[4],
            "pid_rms": pid["rms_error_m"],
            "mpc_rms": mpc["rms_error_m"],
            "pid_max": pid["max_error_m"],
            "mpc_max": mpc["max_error_m"],
            "pid_cross": pid["rms_cross_track_m"],
            "mpc_cross": mpc["rms_cross_track_m"],
            "pid_jerk": pid["rms_jerk_ms3"],
            "mpc_jerk": mpc["rms_jerk_ms3"],
            "pid_solve_us": pid["mean_solve_time_us"],
            "mpc_solve_us": mpc["mean_solve_time_us"],
            "ratio": ratio,
            "winner": "mpc" if ratio > 1.0 else "pid",
            "held_out": pid["held_out"],
        })
    return paired
