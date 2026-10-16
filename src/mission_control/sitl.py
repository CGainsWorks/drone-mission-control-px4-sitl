"""The PX4 SITL campaign: airframe, scenarios, flight-log schema and analysis.

Everything here is pure Python with no MAVSDK and no PX4, so the campaign's
definitions and its analysis are unit-tested like the rest of the stack. The
parts that talk to an autopilot are ``scripts/run_px4_sitl.py`` (one flight)
and ``scripts/run_sitl_campaign.py`` (the containers around it).

What is measured, and against what
----------------------------------

The flight script logs, at every control step, the reference, the state PX4
reported over MAVLink and what the tracker commanded. PX4 logs its own ULog,
which in SIH includes the simulator's ground truth
(``vehicle_global_position_groundtruth``). The analysis puts the two clocks on
one axis by matching the position estimate recorded on both sides, then fills
the ground-truth columns of an ordinary :class:`FlightLog` and hands it to the
same :func:`tracking_metrics` the reference comparison uses. The SITL numbers
and the reference-simulation numbers are therefore the same metric, computed
by the same code, over the same mission phase.

Every number this produces is simulated: PX4's SIH model, not an aircraft.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os

import numpy as np

from .logs.flight_log import COLUMNS as FLIGHT_LOG_COLUMNS
from .logs.flight_log import STATE_CODES, FlightLog
from .sim.quadrotor import QuadrotorParams

#: The PX4 release the campaign is pinned to. Must match sitl/sih/Dockerfile.
PX4_VERSION = "v1.17.0"


# ---------------------------------------------------------------------------
# Airframes
# ---------------------------------------------------------------------------
def sih_quadx_params() -> QuadrotorParams:
    """PX4 SIH quadrotor (airframe 10040), from the release's own defaults.

    ``src/modules/simulation/simulator_sih/sih_params.c`` at the pinned tag:
    SIH_MASS 1.0 kg, SIH_IXX/IYY 0.025, SIH_IZZ 0.030 kg m^2, SIH_T_MAX 5.0 N
    per rotor, SIH_Q_MAX 0.1 N m, SIH_L_ROLL 0.2 m, SIH_T_TAU 0.05 s and a
    linear drag SIH_KDV of 1.0 N per m/s. Thrust in SIH is linear in the motor
    command, so PX4's THR_MDL_FAC stays at its default of 0 and a normalised
    thrust demand already means thrust.

    Thrust-to-weight is 2.04, close to the reference airframe's 2.17, so the
    tilt limit the trackers derive from it is similar.
    """
    return QuadrotorParams(
        mass=1.0,
        inertia=np.array([0.025, 0.025, 0.030]),
        arm_length=0.2 * math.sqrt(2.0),
        max_rotor_thrust=5.0,
        torque_per_thrust=0.1 / 5.0,
        motor_tau=0.05,
        drag_linear=np.array([1.0, 1.0, 1.0]),
        drag_quadratic=np.zeros(3),
    )


def gz_x500_params() -> QuadrotorParams:
    """PX4's Gazebo x500, for the optional visual run (``sitl/gazebo.Dockerfile``).

    From ``Tools/simulation/gz/models/x500_base/model.sdf``: a 2.0 kg body plus
    four 16.1 g rotors at (+/-0.174, +/-0.174) m, and motorConstant 8.54858e-06
    against maxRotVelocity 1000 rad/s, which is 8.55 N per rotor.
    """
    return QuadrotorParams(
        mass=2.0643,
        inertia=np.array([0.021667, 0.021667, 0.040000]),
        arm_length=0.246,
        max_rotor_thrust=8.549,
        torque_per_thrust=0.016,
    )


AIRFRAMES = {"sih": sih_quadx_params, "gz_x500": gz_x500_params}

#: How PX4 SIH responds to an acceleration demand from this stack, measured by
#: ``scripts/identify_px4_lag.py`` on the campaign's facade flight: a transport
#: delay (MAVLink, MAVSDK, PX4's estimator on the way back) and a first-order
#: lag (PX4's attitude and rate loops). The reference simulation's attitude
#: loop, by comparison, is a 0.13 s lag with no delay.
PX4_SIH_COMMAND_DELAY_S = 0.18
PX4_SIH_LAG_TAU_S = 0.18

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Gains for the SITL campaign, from ``scripts/tune_gains.py --airframe sih``:
#: both trackers tuned by the same procedure as the reference comparison, on a
#: surrogate with SIH's airframe and the delay above.
SIH_GAINS_PATH = os.path.join(_ROOT, "config", "tuned_gains_px4_sih.yaml")
AIRFRAME_GAINS = {"sih": SIH_GAINS_PATH, "gz_x500": None}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
#: Faults are injected this long after the survey phase starts, so the vehicle
#: is established on the first leg and moving at cruise speed.
INJECT_AFTER_MISSION_S = 20.0

#: A HOLD is a terminal outcome for a test (the matrix latches, and the state
#: machine holds indefinitely), so the flight is ended this long after the hold
#: begins, by commanding PX4 to land.
HOLD_OBSERVE_S = 12.0

#: One row per failsafe the matrix documents. Each is a real fault injected
#: into PX4 or into the MAVLink path, not a flag set inside the mission layer:
#:
#: * link loss: the UDP relay between PX4 and MAVSDK drops every datagram in
#:   both directions (``mavlink_bridge/link_proxy.py``).
#: * low battery: PX4's battery simulator is told it may drain below its usual
#:   50% floor (SIM_BAT_MIN_PCT), so the reported state of charge falls.
#: * GNSS loss: PX4's simulated receiver is told it sees no satellites
#:   (SIM_GPS_USED 0), so it reports no fix and EKF2 loses GPS.
#: * geofence breach: the mission fence is tightened in flight to a square the
#:   vehicle is already outside, the way an airspace restriction arriving
#:   mid-flight would. SIH has no wind model, so the reference simulation's
#:   way of forcing a breach (a wind the airframe cannot hold against) is not
#:   available here; see docs/LIMITATIONS.md.
SITL_FAILSAFE_SCENARIOS = {
    "link_loss": {
        "description": "MAVLink cut both ways for 6 s on the first leg",
        "inject": "link_drop",
        "duration_s": 6.0,
        "expect_action": "hold",
        "expect_rule": "link_loss",
    },
    "low_battery": {
        "description": "battery simulator allowed to drain to 15%",
        "inject": "param",
        "param": "SIM_BAT_MIN_PCT",
        "value": 15.0,
        "expect_action": "rtl",
        "expect_rule": "battery_low",
    },
    "gnss_loss": {
        "description": "simulated receiver loses every satellite",
        "inject": "param",
        "param": "SIM_GPS_USED",
        "value": 0,
        "expect_action": "land",
        "expect_rule": "gnss_loss",
    },
    "geofence_breach": {
        "description": "fence tightened in flight to 22 m around the structure",
        "inject": "fence",
        "half_width_m": 22.0,
        "expect_action": "rtl",
        "expect_rule": "geofence_breach",
    },
}

#: Tracking flights: every shipped survey mission, flown by both trackers.
TRACKING_MISSIONS = ("facade_inspection", "corridor_survey", "tower_orbit")
TRACKERS = ("pid", "mpc")

#: A tracking flight with an RMS error to ground truth above this has not
#: flown its mission, whatever its final state says. Several times the worst
#: SITL tracking flight with the SITL gains, and far below what a tracker that
#: has lost the vehicle or the estimator produces.
TRACKING_GATE_M = 3.0

#: The per-push smoke flight. Short, so the whole job fits in a few minutes.
SMOKE_MISSION = "sitl_smoke"


def campaign_flights(kind: str = "full") -> list:
    """The flights a campaign consists of, as dicts the orchestrator runs.

    ``full`` is every tracking mission by both trackers plus every failsafe
    row, flown with the MPC (the matrix sits above the tracker and does not
    depend on which one is flying). ``smoke`` is one short MPC flight.
    """
    if kind == "smoke":
        return [{"name": f"{SMOKE_MISSION}-mpc", "mission": SMOKE_MISSION,
                 "tracker": "mpc", "scenario": "none"}]
    if kind != "full":
        raise KeyError(f"unknown campaign {kind!r}; use 'full' or 'smoke'")
    flights = []
    for mission in TRACKING_MISSIONS:
        for tracker in TRACKERS:
            flights.append({"name": f"{mission}-{tracker}", "mission": mission,
                            "tracker": tracker, "scenario": "none"})
    for scenario in SITL_FAILSAFE_SCENARIOS:
        flights.append({"name": f"failsafe-{scenario}", "mission": "facade_inspection",
                        "tracker": "mpc", "scenario": scenario})
    return flights


def tightened_fence(spec, half_width_m: float):
    """A copy of a geofence spec, shrunk to a square about the origin."""
    import copy

    fence = copy.deepcopy(spec)
    h = float(half_width_m)
    fence.polygon_ned = np.array([[-h, -h], [-h, h], [h, h], [h, -h]])
    return fence


# ---------------------------------------------------------------------------
# The flight script's log
# ---------------------------------------------------------------------------
#: Columns of the per-step loop log. The FlightLog columns that only a
#: simulator with ground truth can fill (``true_*``) are left out: they come
#: from PX4's ULog afterwards. The rest are what the mission layer saw and did.
TRUTH_COLUMNS = tuple(name for name in FLIGHT_LOG_COLUMNS if name.startswith("true_"))
EXTRA_COLUMNS = ("px4_mode", "armed", "gnss_satellites", "link_dropping", "fault_injected")
LOOP_COLUMNS = tuple(
    name for name in FLIGHT_LOG_COLUMNS if name not in TRUTH_COLUMNS
) + EXTRA_COLUMNS

#: PX4 flight modes as MAVSDK names them, coded for the CSV.
PX4_MODES = (
    "unknown", "ready", "takeoff", "hold", "mission", "return_to_launch", "land",
    "offboard", "follow_me", "manual", "altctl", "posctl", "acro", "stabilized",
    "rattitude",
)
PX4_MODE_CODES = {name: index for index, name in enumerate(PX4_MODES)}


def _column_format(name: str) -> str:
    """Fixed decimals per column: as fine as the quantity means, no finer.

    Millimetres and millimetres per second for positions and velocities,
    1e-4 for the loop clock, angles and normalised thrust, microseconds for
    compute times. Fixed decimals compress far better than significant figures
    and lose nothing the analysis uses.
    """
    if name in ("update_time_s", "solve_time_s"):
        return ".6f"
    if name in ("time_s", "battery_remaining") or name.endswith("_yaw") \
            or name.startswith("cmd_"):
        return ".4f"
    return ".3f"


def _format(value, spec: str) -> str:
    value = float(value)
    if value != value:
        return "nan"
    if value == int(value) and abs(value) < 1e9:
        return str(int(value))
    return format(value, spec)


def write_loop_log(path: str, rows: list) -> str:
    """Write the loop log as gzip-compressed CSV, byte-stable for a given input.

    ``mtime=0`` and an empty ``filename`` keep the gzip header free of a
    timestamp and of the name it was written under, so rewriting the same
    rows produces the same bytes on any machine.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LOOP_COLUMNS)
    formats = [_column_format(name) for name in LOOP_COLUMNS]
    for row in rows:
        writer.writerow([_format(value, spec) for value, spec in zip(row, formats)])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as gz:
            gz.write(buffer.getvalue().encode("utf-8"))
    return path


def read_loop_log(path: str) -> dict:
    """Read a loop log back into float columns."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [list(map(float, row)) for row in reader]
    table = np.asarray(rows, dtype=float).reshape(-1, len(header))
    return {name: table[:, index] for index, name in enumerate(header)}


# ---------------------------------------------------------------------------
# PX4's ULog
# ---------------------------------------------------------------------------
#: PX4 vehicle_status.nav_state values that appear in these flights.
NAV_STATES = {
    0: "manual", 1: "altctl", 2: "posctl", 3: "mission", 4: "hold", 5: "return",
    10: "acro", 12: "descend", 13: "termination", 14: "offboard", 15: "stabilized",
    17: "takeoff", 18: "land", 19: "follow_target", 20: "precland", 21: "orbit",
}


def _topic(ulog, name: str):
    for dataset in ulog.data_list:
        if dataset.name == name and dataset.multi_id == 0:
            return dataset.data
    return None


def load_ulog(path: str, origin) -> dict:
    """Pull out of a PX4 ULog what the analysis needs, in the mission frame.

    ``origin`` is the mission's :class:`GeodeticOrigin`. Positions are
    converted from latitude, longitude and altitude, which is how the flight
    script converted the estimate it flew on, so both sides use one frame.
    """
    from pyulog import ULog

    wanted = [
        "vehicle_global_position", "vehicle_global_position_groundtruth",
        "vehicle_local_position_groundtruth", "vehicle_attitude_groundtruth",
        "vehicle_status", "battery_status", "sensor_gps",
    ]
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            ulog = ULog(io.BytesIO(handle.read()), wanted)
    else:
        ulog = ULog(path, wanted)

    def geodetic(data):
        t = data["timestamp"].astype(float) * 1e-6
        ned = np.array([
            origin.to_ned(float(lat), float(lon), float(alt))
            for lat, lon, alt in zip(data["lat"], data["lon"], data["alt"])
        ]).reshape(-1, 3)
        return t, ned

    out = {"start_s": ulog.start_timestamp * 1e-6}
    estimate = _topic(ulog, "vehicle_global_position")
    truth = _topic(ulog, "vehicle_global_position_groundtruth")
    truth_local = _topic(ulog, "vehicle_local_position_groundtruth")
    attitude = _topic(ulog, "vehicle_attitude_groundtruth")
    if estimate is None or truth is None:
        raise ValueError(f"{path}: no global position or no ground truth logged")
    out["est_t"], out["est_ned"] = geodetic(estimate)
    out["truth_t"], out["truth_ned"] = geodetic(truth)
    if truth_local is not None:
        out["truth_vel_t"] = truth_local["timestamp"].astype(float) * 1e-6
        out["truth_vel"] = np.column_stack([truth_local["vx"], truth_local["vy"],
                                            truth_local["vz"]]).astype(float)
    if attitude is not None:
        out["att_t"] = attitude["timestamp"].astype(float) * 1e-6
        out["att_q"] = np.column_stack([attitude[f"q[{i}]"] for i in range(4)]).astype(float)

    status = _topic(ulog, "vehicle_status")
    if status is not None:
        t = status["timestamp"].astype(float) * 1e-6
        nav = status["nav_state"].astype(int)
        changes = [0] + [i for i in range(1, len(nav)) if nav[i] != nav[i - 1]]
        out["nav_states"] = [(float(t[i]), NAV_STATES.get(int(nav[i]), str(int(nav[i]))))
                             for i in changes]
    battery = _topic(ulog, "battery_status")
    if battery is not None:
        out["battery_min"] = float(np.min(battery["remaining"]))
    gps = _topic(ulog, "sensor_gps")
    if gps is not None:
        out["gps_fix_min"] = int(np.min(gps["fix_type"]))
    out["duration_s"] = float(ulog.last_timestamp - ulog.start_timestamp) * 1e-6
    return out


def align_clocks(loop_t: np.ndarray, loop_est: np.ndarray, ulog_t: np.ndarray,
                 ulog_est: np.ndarray, window_s: float = 8.0) -> tuple:
    """Map loop time onto PX4 time: ``t_px4 = offset + scale * t_loop``.

    Both sides recorded PX4's own position estimate: the flight script as it
    arrived over MAVLink, PX4 in its ULog. Sliding one against the other and
    taking the shift that best superimposes them puts the two clocks on one
    axis without trusting either.

    A shift alone is not enough. PX4's clock in SITL is simulation time, and
    SIH advances it from a wall-clock timer that slips when the machine is
    busy: on the first flight measured here PX4 time ran about 4% slow, which
    is two seconds over a minute and several metres of apparent error at
    survey speed. So the shift is found in short windows and a straight line
    is fitted through them, which recovers the rate as well as the offset.

    Returns ``(offset_s, scale, rms_residual_m)``. The residual is what is
    left after alignment: MAVLink transport and sampling.
    """
    loop_t = np.asarray(loop_t, dtype=float)
    ulog_t = np.asarray(ulog_t, dtype=float)
    finite = np.all(np.isfinite(loop_est), axis=1)
    loop_t, loop_est = loop_t[finite], loop_est[finite]

    def residual(times, est, offset, scale=1.0):
        shifted = offset + scale * times
        inside = (shifted >= ulog_t[0]) & (shifted <= ulog_t[-1])
        if inside.sum() < max(10, 0.5 * len(shifted)):
            return math.inf
        interp = np.column_stack([
            np.interp(shifted[inside], ulog_t, ulog_est[:, axis]) for axis in range(3)
        ])
        return float(np.sqrt(np.mean(np.sum((interp - est[inside]) ** 2, axis=1))))

    def best_shift(times, est, lo, hi):
        best = min(np.arange(lo, hi, 0.25), key=lambda c: residual(times, est, c))
        for step, span in ((0.02, 0.5), (0.002, 0.05)):
            grid = np.arange(best - span, best + span + step / 2, step)
            best = min(grid, key=lambda c: residual(times, est, c))
        return float(best)

    lo = ulog_t[0] - loop_t[-1]
    hi = ulog_t[-1] - loop_t[0]
    global_shift = best_shift(loop_t, loop_est, lo, hi)

    # Windows that contain movement; a stationary vehicle pins nothing down.
    centres, shifts = [], []
    start = loop_t[0]
    while start < loop_t[-1]:
        window = (loop_t >= start) & (loop_t < start + window_s)
        if window.sum() > 50:
            travel = np.ptp(loop_est[window], axis=0).max()
            if travel > 2.0:
                shift = best_shift(loop_t[window], loop_est[window],
                                   global_shift - 4.0, global_shift + 4.0)
                if residual(loop_t[window], loop_est[window], shift) < 1.0:
                    centres.append(float(np.mean(loop_t[window])))
                    shifts.append(shift)
        start += window_s / 2

    if len(centres) >= 3:
        # shift(t) = t_px4 - t = offset + (scale - 1) t
        slope, intercept = np.polyfit(centres, shifts, 1)
        offset, scale = float(intercept), float(1.0 + slope)
    else:
        offset, scale = global_shift, 1.0
    return offset, scale, residual(loop_t, loop_est, offset, scale)


def build_flight_log(loop: dict, ulog: dict, offset_s: float,
                     scale: float = 1.0) -> FlightLog:
    """An ordinary FlightLog with ground truth from the ULog filled in."""
    from .attitude import quat_to_euler

    log = FlightLog()
    t = loop["time_s"]
    px4_t = offset_s + scale * t
    data = {}
    for name in FLIGHT_LOG_COLUMNS:
        if name in loop:
            data[name] = np.asarray(loop[name], dtype=float)
    for axis, name in enumerate(("true_north", "true_east", "true_down")):
        data[name] = np.interp(px4_t, ulog["truth_t"], ulog["truth_ned"][:, axis])
    if "truth_vel" in ulog:
        for axis, name in enumerate(("true_vn", "true_ve", "true_vd")):
            data[name] = np.interp(px4_t, ulog["truth_vel_t"], ulog["truth_vel"][:, axis])
    else:
        positions = ("true_north", "true_east", "true_down")
        for axis, name in enumerate(("true_vn", "true_ve", "true_vd")):
            data[name] = np.gradient(data[positions[axis]], t)
    if "att_q" in ulog:
        q = np.column_stack([np.interp(px4_t, ulog["att_t"], ulog["att_q"][:, i])
                             for i in range(4)])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        euler = np.array([quat_to_euler(row) for row in q]).reshape(-1, 3)
        for axis, name in enumerate(("true_roll", "true_pitch", "true_yaw")):
            data[name] = euler[:, axis]
    for name in FLIGHT_LOG_COLUMNS:
        data.setdefault(name, np.full(len(t), np.nan))
    log.data = data
    log.finalised = True
    return log


# ---------------------------------------------------------------------------
# One flight, summarised
# ---------------------------------------------------------------------------
def _round(value, digits: int = 4):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def summarise_flight(flight_dir: str, mission) -> dict:
    """Metrics and outcome for one flight directory.

    Expects ``loop.csv.gz`` and ``events.json`` from the flight script and
    ``px4.ulg.gz`` (or an uncompressed ``px4.ulg``) from the autopilot.
    """
    from .analysis.metrics import tracking_metrics

    with open(os.path.join(flight_dir, "events.json"), encoding="utf-8") as handle:
        events = json.load(handle)
    loop = read_loop_log(os.path.join(flight_dir, "loop.csv.gz"))
    summary = {
        "flight": events["flight"],
        "mission": events["mission"],
        "tracker": events["tracker"],
        "scenario": events["scenario"],
        "px4_version": events.get("px4_version", PX4_VERSION),
        "result": events["result"],
        "end_reason": events.get("end_reason", ""),
        "loop_rate_hz": _round(events.get("loop_rate_hz"), 1),
        "failsafe_events": events.get("failsafe_events", []),
        "transitions": events.get("transitions", []),
        "injected_at_s": events.get("injected_at_s"),
        "mode_changes": events.get("mode_changes", []),
    }
    actions = [event["action"] for event in summary["failsafe_events"]]
    summary["observed_action"] = actions[-1] if actions else "none"
    summary["rules_fired"] = list(dict.fromkeys(e["rule"] for e in summary["failsafe_events"]))

    # Error as the tracker saw it: the PX4 estimate against the reference. No
    # ground truth needed, so it is available even without a ULog.
    mission_mask = loop["mission_state"] == STATE_CODES["mission"]
    est_err = np.column_stack([
        loop["est_north"] - loop["ref_north"],
        loop["est_east"] - loop["ref_east"],
        loop["est_down"] - loop["ref_down"],
    ])[mission_mask]
    if est_err.size:
        summary["rms_error_vs_estimate_m"] = _round(
            np.sqrt(np.mean(np.sum(est_err**2, axis=1))))

    ulog_path = os.path.join(flight_dir, "px4.ulg.gz")
    if not os.path.exists(ulog_path):
        ulog_path = os.path.join(flight_dir, "px4.ulg")
    if os.path.exists(ulog_path):
        ulog = load_ulog(ulog_path, mission.origin)
        est = np.column_stack([loop["est_north"], loop["est_east"], loop["est_down"]])
        offset, scale, residual = align_clocks(loop["time_s"], est, ulog["est_t"],
                                               ulog["est_ned"])
        log = build_flight_log(loop, ulog, offset, scale)
        summary["clock_alignment_residual_m"] = _round(residual)
        summary["px4_clock_rate"] = _round(scale, 4)
        summary["ulog_bytes"] = os.path.getsize(ulog_path)
        summary["px4_modes"] = [name for _, name in ulog.get("nav_states", [])]
        summary["battery_min"] = _round(ulog.get("battery_min"), 3)
        summary["gps_fix_min"] = ulog.get("gps_fix_min")
        metrics = tracking_metrics(log, "mission")
        if metrics.get("samples"):
            for key in ("rms_error_m", "max_error_m", "rms_cross_track_m",
                        "rms_along_track_m", "rms_jerk_ms3", "saturated_fraction",
                        "mean_update_time_us", "mean_solve_time_us", "duration_s",
                        "mean_speed_ms", "rms_estimation_error_m"):
                summary[key] = _round(metrics[key])
        altitude = -log["true_down"]
        summary["landed"] = bool(altitude[-1] < 0.5)
        # Airborne only: on the ground after landing the vehicle is below the
        # fence floor by definition.
        airborne = np.isin(loop["mission_state"], [STATE_CODES[name] for name in
                                                   ("takeoff", "mission", "hold", "rtl")])
        summary["geofence_breached"] = bool(np.any((log["geofence_breached"] > 0.5) & airborne))
        summary["min_altitude_after_takeoff_m"] = _round(
            float(np.min(altitude[loop["mission_state"] >= STATE_CODES["mission"]]))
            if np.any(loop["mission_state"] >= STATE_CODES["mission"]) else float("nan"), 2)
    return summary


def failsafe_matches(summary: dict) -> bool:
    """Did a failsafe flight do what its row of the matrix says it should?"""
    spec = SITL_FAILSAFE_SCENARIOS.get(summary["scenario"])
    if spec is None:
        return summary["result"] == "completed"
    return (summary["observed_action"] == spec["expect_action"]
            and spec["expect_rule"] in summary["rules_fired"])


# ---------------------------------------------------------------------------
# The committed summary and the README table generated from it
# ---------------------------------------------------------------------------
def tracker_comparison(flights: list) -> list:
    """Pair the tracking flights by mission. One row per mission."""
    by_mission = {}
    for flight in flights:
        if flight["scenario"] != "none" or flight["mission"] == SMOKE_MISSION:
            continue
        by_mission.setdefault(flight["mission"], {})[flight["tracker"]] = flight
    rows = []
    for mission in TRACKING_MISSIONS:
        pair = by_mission.get(mission, {})
        if "pid" not in pair or "mpc" not in pair:
            continue
        pid, mpc = pair["pid"], pair["mpc"]
        rows.append({
            "mission": mission,
            "pid_rms_m": pid.get("rms_error_m"),
            "mpc_rms_m": mpc.get("rms_error_m"),
            "pid_cross_track_m": pid.get("rms_cross_track_m"),
            "mpc_cross_track_m": mpc.get("rms_cross_track_m"),
            "pid_max_m": pid.get("max_error_m"),
            "mpc_max_m": mpc.get("max_error_m"),
            "pid_jerk_ms3": pid.get("rms_jerk_ms3"),
            "mpc_jerk_ms3": mpc.get("rms_jerk_ms3"),
        })
    return rows


README_BEGIN = "<!-- sitl-table: begin (generated by analysis/make_sitl_results.py) -->"
README_END = "<!-- sitl-table: end -->"


def readme_sitl_block(summary: dict) -> str:
    """The README's SITL table, rendered from the committed summary."""

    def metres(value):
        return "n/a" if value is None else f"{value:.2f} m"

    lines = [README_BEGIN,
             "| Mission (PX4 SIH, simulated) | PID RMS | MPC RMS "
             "| PID cross-track | MPC cross-track |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["comparison"]:
        lines.append(
            f"| `{row['mission'].replace('_', ' ')}` | {metres(row['pid_rms_m'])} | "
            f"{metres(row['mpc_rms_m'])} | {metres(row['pid_cross_track_m'])} | "
            f"{metres(row['mpc_cross_track_m'])} |"
        )
    lines.append("")
    lines.append("| Failsafe (PX4 SIH, simulated) | Expected | Observed | Rule | Vehicle |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in summary["failsafes"]:
        vehicle = "landed" if row.get("landed") else "airborne at end"
        lines.append(
            f"| {row['scenario'].replace('_', ' ')} | {row['expected_action']} | "
            f"{row['observed_action']} | {', '.join(row['rules_fired']) or 'none'} | {vehicle} |"
        )
    lines.append(README_END)
    return "\n".join(lines)


def replace_block(text: str, block: str, begin: str = README_BEGIN, end: str = README_END) -> str:
    """Replace the generated block in a document. Raises if it is missing."""
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop < start:
        raise ValueError("generated block markers not found")
    return text[:start] + block + text[stop + len(end):]


def extract_block(text: str, begin: str = README_BEGIN, end: str = README_END) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop < start:
        return ""
    return text[start:stop + len(end)]
