"""In-memory flight log and its exports.

One row per control cycle, holding ground truth, the estimate, the reference and
every diagnostic needed to explain a number in the results. Kept as flat scalar
columns rather than nested objects because that is what CSV, ULog and pandas all
want, and because a flat schema is one that a notebook written a year later can
still read.

Ground truth is recorded alongside the estimate on purpose. Controllers only
ever see the estimate; metrics are only ever computed against truth. Keeping
both in the log is what lets the analysis separate a tracking error from an
estimation error instead of reporting their sum and calling it tracking.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from ..attitude import quat_to_euler

#: Column order. Also the CSV header, so it is defined exactly once.
COLUMNS = (
    "time_s",
    # ground truth
    "true_north", "true_east", "true_down",
    "true_vn", "true_ve", "true_vd",
    "true_roll", "true_pitch", "true_yaw",
    # estimate
    "est_north", "est_east", "est_down",
    "est_vn", "est_ve", "est_vd",
    "est_yaw",
    # reference
    "ref_north", "ref_east", "ref_down",
    "ref_vn", "ref_ve", "ref_vd",
    "ref_yaw",
    # controller output
    "cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw",
    "accel_demand_n", "accel_demand_e", "accel_demand_d",
    "update_time_s", "solve_time_s", "solver_iterations", "saturated",
    # mission and safety
    "mission_state", "mission_progress", "failsafe_action",
    "geofence_margin_m", "geofence_breached", "geofence_predicted",
    "battery_remaining", "battery_voltage_v", "battery_current_a",
    "gnss_fix_type", "ekf_position_std_h", "ekf_nis", "link_silence_s",
)

#: Numeric codes for the mission state, so the state is usable in a plot and in
#: a ULog field without carrying strings through the whole pipeline.
STATE_CODES = {
    "idle": 0, "armed": 1, "takeoff": 2, "mission": 3, "hold": 4,
    "rtl": 5, "landing": 6, "disarmed": 7, "terminated": 8,
}
STATE_NAMES = {code: name for name, code in STATE_CODES.items()}


class FlightLog:
    """Append-only log, finalised into numpy arrays."""

    def __init__(self):
        self._rows = []
        self.data = {}
        self.finalised = False

    def __len__(self) -> int:
        return len(self.data["time_s"]) if self.finalised else len(self._rows)

    # ------------------------------------------------------------------
    def append(self, time_s, truth, estimate, setpoint, attitude_setpoint,
               diagnostics, mission_state, failsafe_action, geofence, battery,
               estimator_status, link_silence_s, mission_progress) -> None:
        truth_euler = quat_to_euler(truth.orientation)
        command_euler = quat_to_euler(attitude_setpoint.orientation)

        self._rows.append(
            (
                float(time_s),
                truth.position[0], truth.position[1], truth.position[2],
                truth.velocity[0], truth.velocity[1], truth.velocity[2],
                truth_euler[0], truth_euler[1], truth_euler[2],
                estimate.position[0], estimate.position[1], estimate.position[2],
                estimate.velocity[0], estimate.velocity[1], estimate.velocity[2],
                quat_to_euler(estimate.orientation)[2],
                setpoint.position[0], setpoint.position[1], setpoint.position[2],
                setpoint.velocity[0], setpoint.velocity[1], setpoint.velocity[2],
                setpoint.yaw,
                attitude_setpoint.thrust,
                command_euler[0], command_euler[1], command_euler[2],
                diagnostics.acceleration_demand[0],
                diagnostics.acceleration_demand[1],
                diagnostics.acceleration_demand[2],
                diagnostics.update_time_s,
                diagnostics.solve_time_s,
                float(diagnostics.iterations),
                float(diagnostics.saturated),
                float(STATE_CODES.get(mission_state, -1)),
                float(mission_progress),
                float(failsafe_action),
                geofence.margin_m if np.isfinite(geofence.margin_m) else 1e6,
                float(geofence.breached),
                float(geofence.predicted_breach),
                battery.remaining, battery.voltage_v, battery.current_a,
                float(int(estimator_status.fix_type)),
                estimator_status.horizontal_position_std,
                estimator_status.last_position_nis,
                float(link_silence_s),
            )
        )

    def finalise(self) -> None:
        """Convert the accumulated rows into column arrays."""
        if self.finalised:
            return
        if not self._rows:
            self.data = {name: np.zeros(0) for name in COLUMNS}
        else:
            table = np.asarray(self._rows, dtype=float)
            if table.shape[1] != len(COLUMNS):
                raise RuntimeError(
                    f"log row width {table.shape[1]} does not match "
                    f"{len(COLUMNS)} declared columns"
                )
            self.data = {name: table[:, index] for index, name in enumerate(COLUMNS)}
        self._rows = []
        self.finalised = True

    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        if not self.finalised:
            self.finalise()
        return self.data[key]

    @property
    def time(self) -> np.ndarray:
        return self["time_s"]

    def position_error(self) -> np.ndarray:
        """Ground-truth position minus reference, per axis, ``(n, 3)``."""
        return np.column_stack(
            [
                self["true_north"] - self["ref_north"],
                self["true_east"] - self["ref_east"],
                self["true_down"] - self["ref_down"],
            ]
        )

    def estimation_error(self) -> np.ndarray:
        """Estimate minus ground truth, per axis, ``(n, 3)``."""
        return np.column_stack(
            [
                self["est_north"] - self["true_north"],
                self["est_east"] - self["true_east"],
                self["est_down"] - self["true_down"],
            ]
        )

    def mask_state(self, *states) -> np.ndarray:
        """Boolean mask selecting rows in the named mission states."""
        codes = [STATE_CODES[name] for name in states]
        return np.isin(self["mission_state"], codes)

    # ------------------------------------------------------------------
    def to_csv(self, path: str) -> str:
        if not self.finalised:
            self.finalise()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            rows = np.column_stack([self.data[name] for name in COLUMNS])
            for row in rows:
                writer.writerow([f"{value:.6g}" for value in row])
        return path

    def to_ulog(self, path: str, info: dict = None, parameters: dict = None) -> str:
        """Write the log as a ULog file that PlotJuggler and pyulog can open.

        Topics are named after their PX4 equivalents where one exists, so the
        usual plotting layouts work without remapping: ``vehicle_local_position``
        for the estimate, ``vehicle_local_position_setpoint`` for the reference,
        ``vehicle_attitude_setpoint`` for the controller output. Ground truth
        goes in ``vehicle_local_position_groundtruth``, which is what PX4 SITL
        itself publishes.
        """
        from .ulog import UlogWriter

        if not self.finalised:
            self.finalise()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        timestamps = np.round(self["time_s"] * 1e6).astype(np.uint64)
        with UlogWriter(path) as writer:
            writer.write_info("sys_name", "mission_control")
            writer.write_info("ver_sw", _version())
            for key, value in (info or {}).items():
                writer.write_info(str(key), value)
            for key, value in (parameters or {}).items():
                writer.write_parameter(str(key), value)

            for topic, fields, columns in _TOPIC_LAYOUT:
                writer.add_topic(topic, fields)
            for topic, fields, columns in _TOPIC_LAYOUT:
                writer.write_batch(
                    topic,
                    timestamps,
                    {field: self[column] for field, column in columns.items()},
                )
        return path


def _version() -> str:
    from .. import __version__

    return __version__


def _topic(name, mapping, types=None):
    """Build a (topic, fields, columns) entry from a field->column mapping."""
    types = types or {}
    fields = [("timestamp", "uint64_t")] + [
        (field, types.get(field, "float")) for field in mapping
    ]
    return name, fields, mapping


#: Which log columns go into which ULog topic.
_TOPIC_LAYOUT = [
    _topic(
        "vehicle_local_position",
        {"x": "est_north", "y": "est_east", "z": "est_down",
         "vx": "est_vn", "vy": "est_ve", "vz": "est_vd",
         "heading": "est_yaw", "eph": "ekf_position_std_h"},
    ),
    _topic(
        "vehicle_local_position_groundtruth",
        {"x": "true_north", "y": "true_east", "z": "true_down",
         "vx": "true_vn", "vy": "true_ve", "vz": "true_vd",
         "heading": "true_yaw"},
    ),
    _topic(
        "vehicle_local_position_setpoint",
        {"x": "ref_north", "y": "ref_east", "z": "ref_down",
         "vx": "ref_vn", "vy": "ref_ve", "vz": "ref_vd",
         "yaw": "ref_yaw",
         "acceleration_0": "accel_demand_n",
         "acceleration_1": "accel_demand_e",
         "acceleration_2": "accel_demand_d"},
    ),
    _topic(
        "vehicle_attitude_setpoint",
        {"roll_body": "cmd_roll", "pitch_body": "cmd_pitch", "yaw_body": "cmd_yaw",
         "thrust_body_2": "cmd_thrust"},
    ),
    _topic(
        "trajectory_tracker_status",
        {"update_time_s": "update_time_s", "solve_time_s": "solve_time_s",
         "iterations": "solver_iterations", "saturated": "saturated",
         "mission_progress": "mission_progress"},
    ),
    _topic(
        "vehicle_status",
        {"nav_state": "mission_state", "failsafe_action": "failsafe_action",
         "link_silence_s": "link_silence_s"},
    ),
    _topic(
        "geofence_result",
        {"margin_m": "geofence_margin_m", "breached": "geofence_breached",
         "predicted_breach": "geofence_predicted"},
    ),
    _topic(
        "battery_status",
        {"remaining": "battery_remaining", "voltage_v": "battery_voltage_v",
         "current_a": "battery_current_a"},
    ),
    _topic(
        "estimator_status",
        {"gnss_fix_type": "gnss_fix_type", "position_std_h": "ekf_position_std_h",
         "position_nis": "ekf_nis"},
    ),
]
