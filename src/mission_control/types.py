"""Value types passed between the estimator, the trackers and the backends.

These are plain dataclasses holding numpy arrays. They are the contract between
subsystems: anything crossing a package boundary is one of these, so a backend
swap (sim to PX4 SITL) cannot silently change units or frames.

Units are SI throughout: metres, metres per second, radians, radians per second,
seconds, volts, newtons. Frames are NED unless a field says ``_body``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from .attitude import quat_identity, quat_to_euler


def _vec3(values=None) -> np.ndarray:
    if values is None:
        return np.zeros(3)
    return np.asarray(values, dtype=float).reshape(3)


@dataclass
class VehicleState:
    """Full rigid-body state, in NED.

    Produced by the plant as ground truth and by the EKF as an estimate. The
    comparison harness holds both and never confuses them: controllers only ever
    see the estimate, metrics are only ever computed against truth.
    """

    time_s: float = 0.0
    position: np.ndarray = field(default_factory=lambda: _vec3())
    velocity: np.ndarray = field(default_factory=lambda: _vec3())
    orientation: np.ndarray = field(default_factory=quat_identity)
    angular_velocity: np.ndarray = field(default_factory=lambda: _vec3())

    @property
    def euler(self) -> np.ndarray:
        return quat_to_euler(self.orientation)

    @property
    def yaw(self) -> float:
        return float(self.euler[2])

    @property
    def altitude(self) -> float:
        """Height above the mission origin. Positive up, unlike ``position[2]``."""
        return -float(self.position[2])

    def copy(self) -> "VehicleState":
        return VehicleState(
            time_s=self.time_s,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


@dataclass
class ImuSample:
    """Raw IMU measurement in the body frame.

    ``acceleration`` is specific force, i.e. what an accelerometer actually
    measures: it reads ``+g`` upward when sitting on the ground, not zero.
    """

    time_s: float
    acceleration: np.ndarray
    angular_velocity: np.ndarray


class GnssFixType(enum.IntEnum):
    """GNSS solution quality. Values match the MAVLink ``GPS_FIX_TYPE`` enum."""

    NO_FIX = 0
    FIX_2D = 2
    FIX_3D = 3
    RTK_FLOAT = 5
    RTK_FIXED = 6

    @property
    def is_usable(self) -> bool:
        return self >= GnssFixType.FIX_3D

    @property
    def is_rtk(self) -> bool:
        return self >= GnssFixType.RTK_FLOAT


@dataclass
class GnssSample:
    """GNSS fix, already projected into the local NED frame by the receiver model.

    ``time_s`` is the epoch the fix *describes*, not when it arrived;
    ``received_time_s`` is when the autopilot saw it. The gap is the receiver's
    processing and transport latency, tens of milliseconds, which is metres of
    error at inspection speeds if the estimator fuses it as though it were
    current. Carrying both timestamps is what lets the estimator fuse at the
    right point in time.

    ``position_std`` and ``velocity_std`` are the receiver's own reported
    accuracy. The EKF uses them directly rather than a fixed constant, which is
    what makes an RTK dropout show up as a covariance change instead of a silent
    degradation.
    """

    time_s: float
    position: np.ndarray
    velocity: np.ndarray
    fix_type: GnssFixType
    position_std: np.ndarray
    velocity_std: np.ndarray
    satellites: int = 18
    received_time_s: float = 0.0

    @property
    def latency_s(self) -> float:
        return max(0.0, self.received_time_s - self.time_s)

    @property
    def is_usable(self) -> bool:
        return self.fix_type.is_usable


@dataclass
class TrajectorySetpoint:
    """What the mission layer asks the tracker to follow at this instant.

    Acceleration is a feedforward term, not a demand: a tracker is free to
    ignore it. ``yaw`` is absolute.
    """

    time_s: float = 0.0
    position: np.ndarray = field(default_factory=lambda: _vec3())
    velocity: np.ndarray = field(default_factory=lambda: _vec3())
    acceleration: np.ndarray = field(default_factory=lambda: _vec3())
    yaw: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class AttitudeSetpoint:
    """What a tracker emits and the vehicle consumes.

    This is deliberately the MAVLink ``SET_ATTITUDE_TARGET`` payload: an
    orientation, a body rate feedforward and a normalised collective thrust in
    ``[0, 1]``. Both trackers emit exactly this, so the SITL backend forwards it
    unchanged and the comparison stays like-for-like.
    """

    time_s: float = 0.0
    orientation: np.ndarray = field(default_factory=quat_identity)
    body_rate: np.ndarray = field(default_factory=lambda: _vec3())
    thrust: float = 0.0
    #: Acceleration the tracker was demanding before tilt/thrust conversion.
    #: Logged for diagnosis and for the control-effort metric.
    acceleration_demand: np.ndarray = field(default_factory=lambda: _vec3())


@dataclass
class BatteryState:
    """Battery telemetry, as the failsafe layer sees it."""

    voltage_v: float = 25.2
    remaining: float = 1.0  # fraction in [0, 1]
    current_a: float = 0.0

    @property
    def remaining_percent(self) -> float:
        return 100.0 * self.remaining


@dataclass
class LinkState:
    """Telemetry link health."""

    connected: bool = True
    #: Seconds since the last packet from the ground station.
    silence_s: float = 0.0
