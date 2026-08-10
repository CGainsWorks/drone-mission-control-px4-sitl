"""Error-state extended Kalman filter fusing IMU with RTK-GNSS.

Why an *error*-state filter rather than a direct one: attitude has three degrees
of freedom but a quaternion has four components and a unit-norm constraint. A
direct EKF has to carry a singular covariance and renormalise, which quietly
corrupts the uncertainty. The error-state formulation keeps a minimal 3-parameter
attitude error whose covariance is well defined, and injects it into the
quaternion once per step. This is the same structure as PX4's ``ekf2``, minus
the sensors this vehicle does not carry.

State layout
------------

The *nominal* state is propagated with the IMU and holds the quaternion. The
*error* state is what the filter actually estimates:

======  ======  ==============================================
index   size    error quantity
======  ======  ==============================================
0:3     3       position error, NED, m
3:6     3       velocity error, NED, m/s
6:9     3       attitude error, rotation vector, rad
9:12    3       accelerometer bias error, m/s^2
12:15   3       gyroscope bias error, rad/s
======  ======  ==============================================

Biases are in the filter because they are the difference between an estimator
that holds position for a whole mission and one that walks off it. A 0.05 m/s^2
uncorrected accelerometer bias is 2.5 m of position error after ten seconds of
dead reckoning, which is roughly what a GNSS dropout gives you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import (
    quat_from_rotation_vector,
    quat_multiply,
    quat_normalise,
    quat_to_rotation_matrix,
    skew,
)
from ..geodesy import GRAVITY
from ..types import GnssFixType, GnssSample, ImuSample, VehicleState

STATE_SIZE = 15
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_ATT = slice(6, 9)
IDX_ACCEL_BIAS = slice(9, 12)
IDX_GYRO_BIAS = slice(12, 15)

#: Gravity in NED. Down is positive, so this is +g on the third axis.
GRAVITY_NED = np.array([0.0, 0.0, GRAVITY])


@dataclass
class EkfParams:
    """Process and gating parameters.

    The process noise entries are continuous-time densities and are scaled by
    ``dt`` on propagation, so the filter behaves the same at 100 Hz and 400 Hz.
    They are set from the IMU model's own datasheet figures inflated by about
    3x, which is standard practice: a filter tuned to the datasheet is
    overconfident because the datasheet does not include vibration, temperature
    drift or scale-factor error.
    """

    accel_noise: float = 0.0066  # m/s^2 / sqrt(Hz)
    gyro_noise: float = 0.00045  # rad/s / sqrt(Hz)
    accel_bias_walk: float = 9.0e-4  # m/s^2 / sqrt(s)
    gyro_bias_walk: float = 4.5e-5  # rad/s / sqrt(s)

    initial_position_std: float = 1.0
    initial_velocity_std: float = 0.5
    initial_attitude_std: float = 0.05  # rad
    initial_accel_bias_std: float = 0.10
    initial_gyro_bias_std: float = 0.010

    #: Chi-square gate on the 3-DOF GNSS innovation. 16.27 is the 0.999 quantile;
    #: tight enough to reject a jump, loose enough not to reject honest noise.
    innovation_gate: float = 16.27
    #: Consecutive rejected fixes before the filter gives up and accepts one
    #: anyway. Without this a genuine position jump (a base station moved, a
    #: cold start) locks the filter out permanently.
    max_consecutive_rejections: int = 10
    #: Position uncertainty above which the estimate is declared unusable, m.
    #: Consumed by the GNSS-loss failsafe.
    position_std_limit: float = 5.0

    #: Extra measurement variance added on top of the receiver's reported
    #: accuracy, per axis. The receiver does not report its correlated multipath
    #: bias, so believing it exactly makes the filter overconfident.
    gnss_position_inflation: float = 1.5
    gnss_velocity_inflation: float = 1.5


@dataclass
class EkfStatus:
    """What the safety layer needs to know about estimator health."""

    initialised: bool = False
    fix_type: GnssFixType = GnssFixType.NO_FIX
    #: Seconds since the last accepted GNSS update.
    time_since_update_s: float = 0.0
    position_std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Normalised innovation squared of the last position update. A healthy
    #: filter averages about 3 here (three degrees of freedom).
    last_position_nis: float = 0.0
    rejected_updates: int = 0
    consecutive_rejections: int = 0

    @property
    def horizontal_position_std(self) -> float:
        return float(np.hypot(self.position_std[0], self.position_std[1]))

    @property
    def is_healthy(self) -> bool:
        return self.initialised and self.fix_type.is_usable


class ErrorStateEkf:
    """15-state error-state EKF for a GNSS/INS multirotor."""

    def __init__(self, params: EkfParams = None):
        self.params = params or EkfParams()
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.covariance = np.zeros((STATE_SIZE, STATE_SIZE))
        self.status = EkfStatus()
        self._time_s = 0.0
        self._last_update_time = 0.0
        self._reset_covariance()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _reset_covariance(self) -> None:
        p = self.params
        diag = np.concatenate(
            [
                np.full(3, p.initial_position_std**2),
                np.full(3, p.initial_velocity_std**2),
                np.full(3, p.initial_attitude_std**2),
                np.full(3, p.initial_accel_bias_std**2),
                np.full(3, p.initial_gyro_bias_std**2),
            ]
        )
        self.covariance = np.diag(diag)

    def initialise(self, position: np.ndarray, velocity: np.ndarray,
                   orientation: np.ndarray, time_s: float = 0.0) -> None:
        """Seed the filter. In flight this comes from the first usable fix."""
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.asarray(velocity, dtype=float).copy()
        self.orientation = quat_normalise(orientation)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self._reset_covariance()
        self._time_s = time_s
        self._last_update_time = time_s
        self.status = EkfStatus(initialised=True)
        self._refresh_status()

    def initialise_from_gnss(self, sample: GnssSample, yaw: float = 0.0) -> bool:
        """Cold-start from a GNSS fix. Returns ``False`` if the fix is unusable.

        Attitude cannot be observed from a single fix, so roll and pitch start
        level and yaw is taken on trust from the argument. On a real aircraft
        that comes from the magnetometer or a moving-baseline RTK heading; the
        filter then corrects it once the vehicle accelerates.
        """
        if not sample.is_usable:
            return False
        from ..attitude import euler_to_quat

        self.initialise(sample.position, sample.velocity, euler_to_quat(0.0, 0.0, yaw),
                        sample.time_s)
        self.covariance[IDX_POS, IDX_POS] = np.diag(sample.position_std**2)
        self.covariance[IDX_VEL, IDX_VEL] = np.diag(sample.velocity_std**2)
        self._refresh_status()
        return True

    # ------------------------------------------------------------------
    # propagation
    # ------------------------------------------------------------------
    def predict(self, imu: ImuSample, dt: float) -> None:
        """Propagate the nominal state and covariance with one IMU sample."""
        if dt <= 0.0:
            return
        p = self.params

        accel_body = imu.acceleration - self.accel_bias
        gyro_body = imu.angular_velocity - self.gyro_bias
        rot = quat_to_rotation_matrix(self.orientation)
        accel_ned = rot @ accel_body + GRAVITY_NED

        # Nominal state. Trapezoidal on position keeps the altitude error from
        # accumulating during sustained climbs.
        self.position = self.position + self.velocity * dt + 0.5 * accel_ned * dt * dt
        self.velocity = self.velocity + accel_ned * dt
        self.orientation = quat_normalise(
            quat_multiply(self.orientation, quat_from_rotation_vector(gyro_body * dt))
        )

        # Error-state transition. F = I + A*dt, with A the linearisation of the
        # error dynamics about the current nominal state.
        f = np.eye(STATE_SIZE)
        f[IDX_POS, IDX_VEL] = np.eye(3) * dt
        f[IDX_VEL, IDX_ATT] = -rot @ skew(accel_body) * dt
        f[IDX_VEL, IDX_ACCEL_BIAS] = -rot * dt
        f[IDX_ATT, IDX_ATT] = np.eye(3) - skew(gyro_body) * dt
        f[IDX_ATT, IDX_GYRO_BIAS] = -np.eye(3) * dt

        q = np.zeros((STATE_SIZE, STATE_SIZE))
        q[IDX_VEL, IDX_VEL] = np.eye(3) * (p.accel_noise**2) * dt
        q[IDX_ATT, IDX_ATT] = np.eye(3) * (p.gyro_noise**2) * dt
        q[IDX_ACCEL_BIAS, IDX_ACCEL_BIAS] = np.eye(3) * (p.accel_bias_walk**2) * dt
        q[IDX_GYRO_BIAS, IDX_GYRO_BIAS] = np.eye(3) * (p.gyro_bias_walk**2) * dt

        self.covariance = f @ self.covariance @ f.T + q
        self._symmetrise()

        self._time_s = imu.time_s
        self._refresh_status()

    # ------------------------------------------------------------------
    # correction
    # ------------------------------------------------------------------
    def update_gnss(self, sample: GnssSample) -> bool:
        """Fuse a GNSS fix. Returns ``True`` if it was accepted."""
        if not sample.is_usable:
            self.status.fix_type = sample.fix_type
            return False
        if not self.status.initialised:
            return self.initialise_from_gnss(sample)

        p = self.params
        self.status.fix_type = sample.fix_type

        position_var = (sample.position_std * p.gnss_position_inflation) ** 2
        innovation = sample.position - self.position
        h = np.zeros((3, STATE_SIZE))
        h[:, IDX_POS] = np.eye(3)
        accepted = self._apply_update(innovation, h, np.diag(position_var), gate=True)

        if accepted:
            velocity_var = (sample.velocity_std * p.gnss_velocity_inflation) ** 2
            h_vel = np.zeros((3, STATE_SIZE))
            h_vel[:, IDX_VEL] = np.eye(3)
            self._apply_update(sample.velocity - self.velocity, h_vel, np.diag(velocity_var),
                               gate=False)
            self._last_update_time = sample.time_s

        self._refresh_status()
        return accepted

    def _apply_update(self, innovation: np.ndarray, h: np.ndarray, r: np.ndarray,
                      gate: bool) -> bool:
        s = h @ self.covariance @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False

        nis = float(innovation @ s_inv @ innovation)
        if gate:
            self.status.last_position_nis = nis
            if nis > self.params.innovation_gate:
                self.status.rejected_updates += 1
                self.status.consecutive_rejections += 1
                if self.status.consecutive_rejections < self.params.max_consecutive_rejections:
                    return False
                # Persistent rejection means the filter, not the sensor, is
                # wrong. Reopen the gate by inflating position and velocity
                # covariance so the next fix pulls the estimate back.
                self.covariance[IDX_POS, IDX_POS] += np.eye(3) * 25.0
                self.covariance[IDX_VEL, IDX_VEL] += np.eye(3) * 4.0
                self.status.consecutive_rejections = 0
                s = h @ self.covariance @ h.T + r
                s_inv = np.linalg.inv(s)
            else:
                self.status.consecutive_rejections = 0

        gain = self.covariance @ h.T @ s_inv
        error = gain @ innovation

        # Joseph form. It costs one extra matrix product and guarantees the
        # covariance stays symmetric positive-definite, which the simple form
        # does not once the gain is large.
        eye_kh = np.eye(STATE_SIZE) - gain @ h
        self.covariance = eye_kh @ self.covariance @ eye_kh.T + gain @ r @ gain.T
        self._symmetrise()
        self._inject(error)
        return True

    def _inject(self, error: np.ndarray) -> None:
        """Fold the error state into the nominal state and reset it to zero."""
        self.position += error[IDX_POS]
        self.velocity += error[IDX_VEL]
        attitude_error = error[IDX_ATT]
        self.orientation = quat_normalise(
            quat_multiply(self.orientation, quat_from_rotation_vector(attitude_error))
        )
        self.accel_bias += error[IDX_ACCEL_BIAS]
        self.gyro_bias += error[IDX_GYRO_BIAS]

        # Covariance reset: the attitude error is now expressed about a rotated
        # frame, so its block must be rotated too. Skipping this is a common
        # shortcut and it makes the filter mildly inconsistent.
        g = np.eye(STATE_SIZE)
        g[IDX_ATT, IDX_ATT] = np.eye(3) - skew(0.5 * attitude_error)
        self.covariance = g @ self.covariance @ g.T
        self._symmetrise()

    def _symmetrise(self) -> None:
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    def _refresh_status(self) -> None:
        variances = np.clip(np.diag(self.covariance), 0.0, None)
        self.status.position_std = np.sqrt(variances[IDX_POS])
        self.status.velocity_std = np.sqrt(variances[IDX_VEL])
        self.status.time_since_update_s = max(0.0, self._time_s - self._last_update_time)

    def vehicle_state(self) -> VehicleState:
        return VehicleState(
            time_s=self._time_s,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=np.zeros(3),
        )

    @property
    def position_uncertain(self) -> bool:
        """True when the position estimate should no longer be trusted to fly on."""
        return self.status.horizontal_position_std > self.params.position_std_limit
