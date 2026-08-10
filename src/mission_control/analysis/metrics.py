"""Metrics computed from a flight log.

Every number quoted in the README comes from this module, applied to a log the
comparison harness produced. Nothing is computed by hand and nothing is
transcribed.

Two decisions worth stating up front, because they determine whether the
comparison means anything:

**Error is measured against ground truth, not against the estimate.** A
controller that flies exactly where its estimator thinks the reference is, while
the estimator is wrong, has not tracked anything. Truth is available here
because it is a simulation; on a real vehicle this would come from RTK
post-processing.

**Only the mission phase counts.** Takeoff, RTL and landing are flown by the
same trackers, but they are vertical or trivially straight and would dilute the
comparison with segments where any controller looks identical. They are recorded
and reported separately rather than folded in.
"""

from __future__ import annotations

import numpy as np

from ..logs.flight_log import FlightLog


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def cross_track_error(log: FlightLog, mask: np.ndarray) -> np.ndarray:
    """Perpendicular distance from the reference path, m.

    Decomposed from the full position error by projecting out the component
    along the reference velocity. This separates *being in the wrong place* from
    *being in the right place at the wrong time*, and the two have very
    different consequences: lateral error puts a vehicle into a structure, while
    along-track error only means the survey photo was taken a metre early.
    """
    error = log.position_error()[mask]
    velocity = np.column_stack(
        [log["ref_vn"][mask], log["ref_ve"][mask], log["ref_vd"][mask]]
    )
    speed = np.linalg.norm(velocity, axis=1)
    moving = speed > 0.2

    cross = np.linalg.norm(error, axis=1)
    if np.any(moving):
        direction = velocity[moving] / speed[moving, None]
        along = np.sum(error[moving] * direction, axis=1)
        lateral = error[moving] - along[:, None] * direction
        cross[moving] = np.linalg.norm(lateral, axis=1)
    # Where the reference is stationary there is no path direction, so the whole
    # error is reported as cross-track. That is the conservative reading.
    return cross


def along_track_error(log: FlightLog, mask: np.ndarray) -> np.ndarray:
    """Signed distance ahead of (positive) or behind the reference, m."""
    error = log.position_error()[mask]
    velocity = np.column_stack(
        [log["ref_vn"][mask], log["ref_ve"][mask], log["ref_vd"][mask]]
    )
    speed = np.linalg.norm(velocity, axis=1)
    along = np.zeros(error.shape[0])
    moving = speed > 0.2
    if np.any(moving):
        direction = velocity[moving] / speed[moving, None]
        along[moving] = np.sum(error[moving] * direction, axis=1)
    return along


def tracking_metrics(log: FlightLog, phase: str = "mission") -> dict:
    """Tracking, effort and compute metrics over one flight phase."""
    mask = log.mask_state(phase)
    if not np.any(mask):
        return {"samples": 0}

    error = log.position_error()[mask]
    norm = np.linalg.norm(error, axis=1)
    horizontal = np.linalg.norm(error[:, 0:2], axis=1)
    vertical = np.abs(error[:, 2])
    cross = cross_track_error(log, mask)
    along = along_track_error(log, mask)

    estimation = np.linalg.norm(log.estimation_error()[mask], axis=1)

    # Control effort: the acceleration the tracker demanded. Two views of it —
    # the RMS is how hard it worked, the jerk is how smoothly. A controller that
    # matches another's tracking error while demanding a rougher input is buying
    # that accuracy with actuator wear and battery.
    demand = np.column_stack(
        [
            log["accel_demand_n"][mask],
            log["accel_demand_e"][mask],
            log["accel_demand_d"][mask],
        ]
    )
    demand_norm = np.linalg.norm(demand, axis=1)
    times = log["time_s"][mask]
    if times.size > 2:
        # Differenced over the actual timestamps rather than a nominal control
        # period, so a phase the mask happens to split does not read as an
        # enormous jerk. A gap makes dt large and the jerk small, which is the
        # safe direction to be wrong in for a quantity used as a constraint.
        dt = np.diff(times)
        jerk = np.linalg.norm(np.diff(demand, axis=0), axis=1) / np.maximum(dt, 1e-6)
    else:
        jerk = np.zeros(1)

    solve = log["solve_time_s"][mask]
    update = log["update_time_s"][mask]

    return {
        "samples": int(mask.sum()),
        "duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
        "rms_error_m": _rms(norm),
        "mean_error_m": float(np.mean(norm)),
        "max_error_m": float(np.max(norm)),
        "p95_error_m": float(np.percentile(norm, 95)),
        "rms_horizontal_m": _rms(horizontal),
        "rms_vertical_m": _rms(vertical),
        "rms_cross_track_m": _rms(cross),
        "max_cross_track_m": float(np.max(cross)),
        "rms_along_track_m": _rms(along),
        "mean_along_track_m": float(np.mean(along)),
        "rms_estimation_error_m": _rms(estimation),
        "max_estimation_error_m": float(np.max(estimation)),
        "rms_accel_demand_ms2": _rms(demand_norm),
        "max_accel_demand_ms2": float(np.max(demand_norm)),
        "rms_jerk_ms3": _rms(jerk),
        "saturated_fraction": float(np.mean(log["saturated"][mask])),
        "mean_solve_time_us": float(np.mean(solve) * 1e6),
        "p99_solve_time_us": float(np.percentile(solve, 99) * 1e6),
        "mean_update_time_us": float(np.mean(update) * 1e6),
        "p99_update_time_us": float(np.percentile(update, 99) * 1e6),
        "mean_speed_ms": float(
            np.mean(
                np.linalg.norm(
                    np.column_stack(
                        [log["true_vn"][mask], log["true_ve"][mask], log["true_vd"][mask]]
                    ),
                    axis=1,
                )
            )
        ),
    }


def settling_metrics(log: FlightLog, band_m: float = 0.10) -> dict:
    """How long the vehicle spends outside a tolerance band around the path.

    RMS hides the shape of an error. Two controllers with the same RMS can
    differ entirely in whether they sit at a small offset the whole flight or
    track perfectly except for a large excursion at each corner. This reports
    the fraction of the mission spent outside ``band_m`` and the longest single
    excursion, which distinguishes those cases.
    """
    mask = log.mask_state("mission")
    if not np.any(mask):
        return {"outside_fraction": float("nan"), "longest_excursion_s": float("nan")}

    norm = np.linalg.norm(log.position_error()[mask], axis=1)
    times = log["time_s"][mask]
    outside = norm > band_m

    longest = 0.0
    current_start = None
    for index, flag in enumerate(outside):
        if flag and current_start is None:
            current_start = times[index]
        elif not flag and current_start is not None:
            longest = max(longest, float(times[index] - current_start))
            current_start = None
    if current_start is not None:
        longest = max(longest, float(times[-1] - current_start))

    return {
        "band_m": band_m,
        "outside_fraction": float(np.mean(outside)),
        "longest_excursion_s": longest,
    }


def estimator_metrics(log: FlightLog) -> dict:
    """Estimator health over the whole flight, not just the mission phase."""
    error = np.linalg.norm(log.estimation_error(), axis=1)
    horizontal = np.linalg.norm(log.estimation_error()[:, 0:2], axis=1)
    nis = log["ekf_nis"]
    usable = nis > 0.0
    return {
        "rms_error_m": _rms(error),
        "rms_horizontal_m": _rms(horizontal),
        "max_error_m": float(np.max(error)) if error.size else float("nan"),
        "mean_reported_std_m": float(np.mean(log["ekf_position_std_h"])),
        # A consistent filter averages a NIS of about 3 on a 3-DOF update.
        # Materially below that means it is discarding information by
        # over-stating its measurement noise; above means it is overconfident
        # and will start rejecting good fixes.
        "mean_position_nis": float(np.mean(nis[usable])) if np.any(usable) else float("nan"),
        "fraction_rtk_fixed": float(np.mean(log["gnss_fix_type"] >= 6)),
    }


def summarise(log: FlightLog) -> dict:
    """Everything, keyed by section. What the comparison harness records."""
    return {
        "mission": tracking_metrics(log, "mission"),
        "takeoff": tracking_metrics(log, "takeoff"),
        "settling": settling_metrics(log),
        "estimator": estimator_metrics(log),
    }
