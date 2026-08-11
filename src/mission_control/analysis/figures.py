"""Figures, generated from ``results/runs.csv`` and the saved flight logs.

Every figure is regenerated from the raw results, never edited. Matplotlib only,
no seaborn, no style packages: one fewer dependency and one fewer thing to
break on a clean machine.
"""

from __future__ import annotations

import os

# Non-interactive backend, chosen before pyplot is imported. Without it, running
# in Docker or CI raises on a missing display.
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PID_COLOUR = "#4C6EF5"
MPC_COLOUR = "#E8590C"
REFERENCE_COLOUR = "#495057"
GRID_KWARGS = dict(alpha=0.25, linewidth=0.6)

DPI = 150


def _finish(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _load_rows(results_csv: str) -> list:
    import csv

    with open(results_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in ("mission", "speed", "wind", "corner_mode", "tracker", "final_state"):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = float("nan")
    return rows


def rms_by_scenario(results_csv: str, path: str) -> str:
    """Paired bar chart of RMS tracking error, one pair per scenario.

    The headline figure. Scenarios are ordered by how hard they are for the PID,
    so the reader can see whether the gap opens up with difficulty rather than
    being uniform — which is the actual finding.
    """
    rows = _load_rows(results_csv)
    grouped = {}
    for row in rows:
        key = (row["mission"], row["speed"], row["wind"], row["corner_mode"])
        entry = grouped.setdefault(key, {"held_out": bool(row["held_out"])})
        entry.setdefault(row["tracker"], []).append(row["rms_error_m"])

    labels, pid_values, mpc_values, held_out = [], [], [], []
    for key, entry in grouped.items():
        if "pid" not in entry or "mpc" not in entry:
            continue
        mission, speed, wind, corner = key
        labels.append(
            f"{mission.replace('_', ' ')}\n{speed}, {wind}"
            + ("\nsharp corners" if corner == "sharp" else "")
        )
        pid_values.append(float(np.mean(entry["pid"])))
        mpc_values.append(float(np.mean(entry["mpc"])))
        # Taken from the results, not re-derived here: duplicating the rule for
        # what counts as held out would let the figure and the scenario
        # definition drift apart.
        held_out.append(entry["held_out"])

    order = np.argsort(pid_values)
    labels = [labels[i] for i in order]
    pid_values = [pid_values[i] for i in order]
    mpc_values = [mpc_values[i] for i in order]
    held_out = [held_out[i] for i in order]

    positions = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9.0, 1.15 * len(labels)), 5.2))
    ax.bar(positions - width / 2, pid_values, width, label="PID", color=PID_COLOUR)
    ax.bar(positions + width / 2, mpc_values, width, label="MPC", color=MPC_COLOUR)

    for index, (pid, mpc) in enumerate(zip(pid_values, mpc_values)):
        ratio = pid / mpc if mpc > 0 else float("inf")
        ax.text(index, max(pid, mpc) * 1.04,
                f"{ratio:.2f}x" if np.isfinite(ratio) else "-",
                ha="center", va="bottom", fontsize=8, color="#212529")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMS position error (m)")
    ax.set_title("Tracking error by scenario, mission phase only, mean over seeds\n"
                 "annotation is PID error divided by MPC error")
    ax.grid(axis="y", **GRID_KWARGS)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    # Mark which scenarios the gains were tuned on, so a reader can check the
    # result is not just a tuning artefact.
    for index, flag in enumerate(held_out):
        if not flag:
            ax.get_xticklabels()[index].set_color("#0B7285")
            ax.get_xticklabels()[index].set_fontweight("bold")
    ax.text(0.005, 0.98, "teal bold = gains tuned on this scenario; all others held out",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#0B7285")
    return _finish(fig, path)


def trajectory_figure(pid_log, mpc_log, path: str, title: str = "") -> str:
    """Ground track and error-versus-time for one matched pair of runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    ax = axes[0]
    mask = pid_log.mask_state("mission")
    ax.plot(pid_log["ref_east"][mask], pid_log["ref_north"][mask],
            color=REFERENCE_COLOUR, linewidth=2.4, alpha=0.55, label="reference")
    ax.plot(pid_log["true_east"][mask], pid_log["true_north"][mask],
            color=PID_COLOUR, linewidth=1.1, label="PID")
    mask_mpc = mpc_log.mask_state("mission")
    ax.plot(mpc_log["true_east"][mask_mpc], mpc_log["true_north"][mask_mpc],
            color=MPC_COLOUR, linewidth=1.1, label="MPC")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(**GRID_KWARGS)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("ground track")

    ax = axes[1]
    for log, colour, label in ((pid_log, PID_COLOUR, "PID"), (mpc_log, MPC_COLOUR, "MPC")):
        phase = log.mask_state("mission")
        times = log["time_s"][phase]
        error = np.linalg.norm(log.position_error()[phase], axis=1)
        ax.plot(times - times[0], error, color=colour, linewidth=1.0, label=label)
        ax.axhline(float(np.sqrt(np.mean(error**2))), color=colour, linestyle="--",
                   linewidth=0.9, alpha=0.7)
    ax.set_xlabel("time into mission phase (s)")
    ax.set_ylabel("position error (m)")
    ax.grid(**GRID_KWARGS)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("tracking error, dashed lines are RMS")

    if title:
        fig.suptitle(title, fontsize=11)
        fig.subplots_adjust(top=0.9)
    return _finish(fig, path)


def error_distribution(results_csv: str, path: str) -> str:
    """Where the error goes: cross-track against along-track, per controller.

    RMS alone cannot distinguish a controller that sits half a metre behind the
    reference from one that swings half a metre to the side of it. The first is
    a timing offset and harmless; the second is what puts an aircraft into a
    structure. Splitting them is the whole point of this figure.
    """
    rows = _load_rows(results_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for ax, metric, label in (
        (axes[0], "rms_cross_track_m", "cross-track RMS (m)"),
        (axes[1], "rms_along_track_m", "along-track RMS (m)"),
    ):
        data = []
        for tracker in ("pid", "mpc"):
            values = [row[metric] for row in rows
                      if row["tracker"] == tracker and np.isfinite(row[metric])]
            data.append(values)
        # Tick labels are set afterwards rather than passed to boxplot. The
        # keyword for them was renamed from `labels` to `tick_labels` in
        # matplotlib 3.9 and the old name removed later, so passing either one
        # pins the repository to a matplotlib range. Setting them on the axis
        # works on every version, which matters because this is generated inside
        # a container whose matplotlib is whatever pip resolved that day.
        parts = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, colour in zip(parts["boxes"], (PID_COLOUR, MPC_COLOUR)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.65)
        for median in parts["medians"]:
            median.set_color("#212529")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["PID", "MPC"])
        ax.set_ylabel(label)
        ax.grid(axis="y", **GRID_KWARGS)
        ax.set_axisbelow(True)

    fig.suptitle("Error decomposed across every run in the grid", fontsize=11)
    return _finish(fig, path)


def compute_cost(results_csv: str, path: str) -> str:
    """What the MPC costs to run, next to what it buys."""
    rows = _load_rows(results_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    for tracker, colour in (("pid", PID_COLOUR), ("mpc", MPC_COLOUR)):
        solve = [row["mean_solve_time_us"] for row in rows
                 if row["tracker"] == tracker and np.isfinite(row["mean_solve_time_us"])]
        update = [row["mean_update_time_us"] for row in rows
                  if row["tracker"] == tracker and np.isfinite(row["mean_update_time_us"])]
        ax.scatter(solve, update, s=18, alpha=0.7, color=colour, label=tracker.upper())
    ax.set_xlabel("optimisation / control law (us per step)")
    ax.set_ylabel("whole control step (us)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both", **GRID_KWARGS)
    ax.legend(frameon=False)
    ax.set_title("compute per control step")
    # The control period is the number that actually matters: anything below it
    # is affordable, anything above it is not a controller.
    ax.axhline(20000.0, color="#C92A2A", linestyle=":", linewidth=1.2)
    ax.text(ax.get_xlim()[0] * 1.2, 22000.0, "50 Hz control period (20 ms)",
            fontsize=8, color="#C92A2A")

    ax = axes[1]
    for tracker, colour in (("pid", PID_COLOUR), ("mpc", MPC_COLOUR)):
        jerk = [row["rms_jerk_ms3"] for row in rows
                if row["tracker"] == tracker and np.isfinite(row["rms_jerk_ms3"])]
        error = [row["rms_error_m"] for row in rows
                 if row["tracker"] == tracker and np.isfinite(row["rms_jerk_ms3"])]
        ax.scatter(jerk, error, s=18, alpha=0.7, color=colour, label=tracker.upper())
    ax.set_xlabel("RMS jerk of the acceleration demand (m/s^3)")
    ax.set_ylabel("RMS position error (m)")
    ax.grid(**GRID_KWARGS)
    ax.legend(frameon=False)
    ax.set_title("accuracy against smoothness of demand")

    return _finish(fig, path)


def estimator_figure(log, path: str, title: str = "") -> str:
    """Estimator error and reported uncertainty over one flight."""
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True)

    times = log["time_s"]
    error = log.estimation_error()

    ax = axes[0]
    for index, (name, colour) in enumerate(
        (("north", "#1C7ED6"), ("east", "#37B24D"), ("down", "#F76707"))
    ):
        ax.plot(times, error[:, index], color=colour, linewidth=0.9, label=name)
    ax.plot(times, log["ekf_position_std_h"], color="#212529", linewidth=1.1,
            linestyle="--", label="reported 1-sigma (horizontal)")
    ax.plot(times, -log["ekf_position_std_h"], color="#212529", linewidth=1.1,
            linestyle="--")
    ax.set_ylabel("estimate minus truth (m)")
    ax.grid(**GRID_KWARGS)
    ax.legend(frameon=False, fontsize=8, ncol=4)

    ax = axes[1]
    ax.plot(times, log["gnss_fix_type"], color="#5F3DC4", linewidth=1.2, label="GNSS fix type")
    ax.set_ylabel("fix type")
    ax.set_yticks([0, 3, 5, 6])
    ax.set_yticklabels(["none", "3D", "RTK float", "RTK fixed"], fontsize=8)
    ax.set_xlabel("time (s)")
    ax.grid(**GRID_KWARGS)
    ax.legend(frameon=False, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
        fig.subplots_adjust(top=0.92)
    return _finish(fig, path)
