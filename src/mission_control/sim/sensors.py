"""IMU and GNSS receiver models.

These generate the measurements the EKF consumes. Both are seeded independently
of the wind so that a scenario can hold the disturbance fixed while varying
sensor quality, or the reverse.

The GNSS model is the interesting one: it reproduces the three behaviours that
actually matter to an RTK-equipped inspection aircraft.

* **Fix-dependent accuracy.** RTK fixed is centimetre-level; RTK float is
  decimetre-level; an ordinary 3D fix is metre-level. The receiver reports its
  own accuracy, and the EKF believes it.
* **Slowly correlated error.** Non-RTK error is not white. It wanders with
  multipath and ionospheric delay, which is why averaging does not remove it and
  why a naive position loop drifts with it.
* **Latency.** A fix describes where the vehicle *was*, typically 50-100 ms ago.
  Ignoring that is a classic source of phantom estimator lag.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..types import GnssFixType, GnssSample, ImuSample


@dataclass
class ImuParams:
    """Consumer-grade MEMS IMU, roughly an ICM-42688-P at 200 Hz.

    Noise densities are per root-hertz, as datasheets quote them; the model
    converts to a per-sample sigma. Bias random walks are per root-second.
    """

    rate_hz: float = 200.0
    accel_noise_density: float = 0.0022  # m/s^2 / sqrt(Hz)
    gyro_noise_density: float = 0.00015  # rad/s / sqrt(Hz)
    accel_bias_walk: float = 3.0e-4  # m/s^2 / sqrt(s)
    gyro_bias_walk: float = 1.5e-5  # rad/s / sqrt(s)
    initial_accel_bias_std: float = 0.05  # m/s^2
    initial_gyro_bias_std: float = 0.005  # rad/s


class ImuSensor:
    """Corrupts true specific force and body rates into an IMU measurement."""

    def __init__(self, params: ImuParams = None, seed: int = 0):
        self.params = params or ImuParams()
        self._rng = np.random.default_rng(seed)
        self.accel_bias = self._rng.normal(0.0, self.params.initial_accel_bias_std, 3)
        self.gyro_bias = self._rng.normal(0.0, self.params.initial_gyro_bias_std, 3)

    def sample(self, time_s: float, specific_force_body: np.ndarray,
               angular_velocity: np.ndarray, dt: float) -> ImuSample:
        p = self.params
        # Datasheet noise density to per-sample sigma.
        accel_sigma = p.accel_noise_density / np.sqrt(dt)
        gyro_sigma = p.gyro_noise_density / np.sqrt(dt)

        self.accel_bias += self._rng.normal(0.0, p.accel_bias_walk * np.sqrt(dt), 3)
        self.gyro_bias += self._rng.normal(0.0, p.gyro_bias_walk * np.sqrt(dt), 3)

        return ImuSample(
            time_s=time_s,
            acceleration=specific_force_body
            + self.accel_bias
            + self._rng.normal(0.0, accel_sigma, 3),
            angular_velocity=angular_velocity
            + self.gyro_bias
            + self._rng.normal(0.0, gyro_sigma, 3),
        )


#: Reported 1-sigma accuracy by fix type: (horizontal m, vertical m, velocity m/s).
FIX_ACCURACY = {
    GnssFixType.RTK_FIXED: (0.020, 0.030, 0.030),
    GnssFixType.RTK_FLOAT: (0.250, 0.400, 0.080),
    GnssFixType.FIX_3D: (1.200, 2.000, 0.250),
    GnssFixType.FIX_2D: (4.000, 20.000, 1.000),
}

#: Standard deviation and correlation time of the slowly wandering bias, by fix
#: type. RTK carrier-phase ambiguity resolution removes almost all of it, which
#: is the whole reason to carry a base station.
FIX_BIAS = {
    GnssFixType.RTK_FIXED: (0.010, 60.0),
    GnssFixType.RTK_FLOAT: (0.300, 60.0),
    GnssFixType.FIX_3D: (0.900, 120.0),
    GnssFixType.FIX_2D: (3.000, 120.0),
}


@dataclass
class GnssParams:
    """GNSS receiver configuration.

    ``fix_schedule`` is a list of ``(start_time_s, fix_type)`` pairs applied in
    order, which is how the GNSS-degradation and GNSS-loss failsafe scenarios
    are driven. It is data rather than a callback so a scenario is serialisable
    and appears verbatim in the results.
    """

    rate_hz: float = 5.0
    latency_s: float = 0.08
    fix_schedule: list = field(default_factory=lambda: [(0.0, GnssFixType.RTK_FIXED)])
    satellites: int = 22
    #: If set, RTK corrections older than this downgrade a fixed solution to
    #: float, as a real receiver does when the correction link stalls.
    rtk_correction_timeout_s: float = 5.0


class GnssReceiver:
    """Produces delayed, biased, fix-dependent GNSS samples."""

    def __init__(self, params: GnssParams = None, seed: int = 0):
        self.params = params or GnssParams()
        self._rng = np.random.default_rng(seed)
        self._bias = np.zeros(3)
        self._next_sample_time = 0.0
        self._history = deque(maxlen=4096)
        self._last_correction_time = 0.0
        self.corrections_available = True

    def fix_type_at(self, time_s: float) -> GnssFixType:
        """Resolve the scheduled fix type, then apply RTK correction ageing."""
        fix = GnssFixType.NO_FIX
        for start, value in self.params.fix_schedule:
            if time_s >= start:
                fix = GnssFixType(value)
        if fix == GnssFixType.RTK_FIXED and not self.corrections_available:
            age = time_s - self._last_correction_time
            if age > self.params.rtk_correction_timeout_s:
                fix = GnssFixType.RTK_FLOAT
        return fix

    def set_corrections_available(self, available: bool, time_s: float) -> None:
        """Model the RTCM correction stream going away and coming back."""
        if available:
            self._last_correction_time = time_s
        self.corrections_available = available

    def record_truth(self, time_s: float, position: np.ndarray, velocity: np.ndarray) -> None:
        """Push ground truth into the latency buffer. Call every plant step."""
        self._history.append((time_s, np.asarray(position, dtype=float).copy(),
                              np.asarray(velocity, dtype=float).copy()))

    def _delayed_truth(self, time_s: float):
        target = time_s - self.params.latency_s
        if not self._history:
            return None
        # Linear interpolation between the two straddling samples. The buffer is
        # short and ordered, so a reverse scan is cheaper than bisecting.
        previous = None
        for entry in self._history:
            if entry[0] >= target:
                if previous is None:
                    return entry
                span = entry[0] - previous[0]
                if span <= 0.0:
                    return entry
                alpha = (target - previous[0]) / span
                return (
                    target,
                    previous[1] + alpha * (entry[1] - previous[1]),
                    previous[2] + alpha * (entry[2] - previous[2]),
                )
            previous = entry
        return self._history[-1]

    def sample(self, time_s: float):
        """Return a :class:`GnssSample` if one is due at ``time_s``, else ``None``."""
        if time_s + 1e-9 < self._next_sample_time:
            return None
        interval = 1.0 / self.params.rate_hz
        self._next_sample_time = max(self._next_sample_time + interval, time_s)

        fix = self.fix_type_at(time_s)
        truth = self._delayed_truth(time_s)
        if truth is None or fix == GnssFixType.NO_FIX:
            return GnssSample(
                time_s=time_s,
                position=np.zeros(3),
                velocity=np.zeros(3),
                fix_type=GnssFixType.NO_FIX,
                position_std=np.full(3, 1e3),
                velocity_std=np.full(3, 1e3),
                satellites=0,
                received_time_s=time_s,
            )

        _, true_position, true_velocity = truth
        horizontal, vertical, velocity_sigma = FIX_ACCURACY[fix]
        bias_sigma, bias_tau = FIX_BIAS[fix]

        # Ornstein-Uhlenbeck bias, exactly discretised so its stationary
        # variance does not depend on the sample interval.
        decay = np.exp(-interval / bias_tau)
        target_sigma = np.array([bias_sigma, bias_sigma, bias_sigma * 1.6])
        self._bias = decay * self._bias + target_sigma * np.sqrt(
            1.0 - decay**2
        ) * self._rng.normal(0.0, 1.0, 3)

        position_std = np.array([horizontal, horizontal, vertical])
        velocity_std = np.full(3, velocity_sigma)

        return GnssSample(
            # Timestamped with the epoch the fix describes, as a real receiver
            # does, not with the moment the autopilot received it.
            time_s=time_s - self.params.latency_s,
            received_time_s=time_s,
            position=true_position + self._bias + self._rng.normal(0.0, position_std),
            velocity=true_velocity + self._rng.normal(0.0, velocity_std),
            fix_type=fix,
            # The receiver reports its nominal accuracy, which does not include
            # the correlated bias. Understating the error like this is exactly
            # what a real receiver does, and the EKF has to cope.
            position_std=position_std,
            velocity_std=velocity_std,
            satellites=self.params.satellites if fix.is_usable else 4,
        )
