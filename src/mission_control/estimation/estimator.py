"""GNSS/INS estimator: a delayed-horizon EKF plus a forward output predictor.

The problem this solves
-----------------------

A GNSS fix describes where the vehicle was 50-100 ms ago. Fusing it as though it
described *now* drags the estimate backwards along the flight path, and the size
of that error is speed times latency: at 4 m/s with 80 ms of latency, 0.32 m.
That is sixteen times the noise of an RTK-fixed solution, so an estimator that
ignores latency throws away most of what RTK is for. It also feeds the position
controller a lagged measurement, which shows up as a phase margin loss and, when
the gains are raised, as an oscillation nobody can tune out.

The architecture
----------------

Two clocks, which is how PX4's ``ekf2`` does it:

* The **fusion horizon** runs ``fusion_delay_s`` behind real time. IMU samples
  are buffered and only propagated into the EKF once they are older than that,
  so when a GNSS fix arrives, the filter state is already at the right epoch and
  the fix can be fused with no time skew.
* The **output predictor** takes the horizon state and integrates the still-
  buffered IMU samples forward to the present. Controllers consume this, so they
  see a current, low-latency state with the horizon's bias corrections applied.

The cost is memory and one extra integration pass; the benefit is that the
position estimate the controller sees is correct in time as well as in space.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..attitude import (
    quat_from_rotation_vector,
    quat_multiply,
    quat_normalise,
    quat_to_rotation_matrix,
)
from ..types import GnssSample, ImuSample, VehicleState
from .ekf import GRAVITY_NED, EkfParams, EkfStatus, ErrorStateEkf


class GnssInsEstimator:
    """Latency-compensated wrapper around :class:`ErrorStateEkf`.

    ``fusion_delay_s`` must be at least the worst-case GNSS latency. Making it
    larger costs output-predictor drift; making it smaller means fixes arrive
    already stale relative to the horizon and get fused at the wrong epoch.
    """

    def __init__(self, params: EkfParams = None, fusion_delay_s: float = 0.10):
        self.ekf = ErrorStateEkf(params)
        self.fusion_delay_s = fusion_delay_s
        self._buffer = deque()
        self._horizon_time = 0.0
        self._now = 0.0
        self._predicted = VehicleState()
        self._initialised = False
        #: The output predictor re-integrates the whole IMU buffer, so running
        #: it eagerly on every IMU sample costs O(buffer) per sample at 200 Hz —
        #: and the result is only ever read by the controller at 50 Hz. It is
        #: therefore computed lazily, on demand. Measured at roughly a fifth of
        #: total simulation time before this change.
        self._predicted_stale = True

    # ------------------------------------------------------------------
    @property
    def status(self) -> EkfStatus:
        return self.ekf.status

    @property
    def initialised(self) -> bool:
        return self._initialised

    @property
    def position_uncertain(self) -> bool:
        return self.ekf.position_uncertain

    @property
    def accel_bias(self) -> np.ndarray:
        return self.ekf.accel_bias

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.ekf.gyro_bias

    # ------------------------------------------------------------------
    def add_imu(self, sample: ImuSample) -> None:
        """Buffer an IMU sample and advance the fusion horizon."""
        self._now = sample.time_s
        self._buffer.append(sample)

        if not self._initialised:
            # Nothing to propagate into yet; keep the buffer bounded so a long
            # pre-arm wait does not grow it without limit.
            while self._buffer and self._now - self._buffer[0].time_s > 2.0 * self.fusion_delay_s:
                self._buffer.popleft()
            return

        horizon = self._now - self.fusion_delay_s
        while self._buffer and self._buffer[0].time_s <= horizon:
            older = self._buffer.popleft()
            dt = older.time_s - self._horizon_time
            if dt > 0.0:
                self.ekf.predict(older, dt)
            self._horizon_time = older.time_s

        self._predicted_stale = True

    def add_gnss(self, sample: GnssSample) -> bool:
        """Fuse a GNSS fix at the fusion horizon."""
        if not self._initialised:
            if not sample.is_usable:
                return False
            if not self.ekf.initialise_from_gnss(sample):
                return False
            self._initialised = True
            self._horizon_time = sample.time_s
            # Drop buffered samples that predate the fix: they belong to a state
            # the filter never had.
            while self._buffer and self._buffer[0].time_s < sample.time_s:
                self._buffer.popleft()
            self._predicted_stale = True
            return True

        accepted = self.ekf.update_gnss(sample)
        self._predicted_stale = True
        return accepted

    # ------------------------------------------------------------------
    def _ensure_predicted(self) -> None:
        if self._predicted_stale:
            self._run_output_predictor()
            self._predicted_stale = False

    # ------------------------------------------------------------------
    def _run_output_predictor(self) -> None:
        """Integrate the horizon state forward over the still-buffered IMU."""
        position = self.ekf.position.copy()
        velocity = self.ekf.velocity.copy()
        orientation = self.ekf.orientation.copy()
        accel_bias = self.ekf.accel_bias
        gyro_bias = self.ekf.gyro_bias
        angular_velocity = np.zeros(3)

        previous_time = self._horizon_time
        for sample in self._buffer:
            dt = sample.time_s - previous_time
            previous_time = sample.time_s
            if dt <= 0.0:
                continue
            gyro = sample.angular_velocity - gyro_bias
            accel_ned = (
                quat_to_rotation_matrix(orientation) @ (sample.acceleration - accel_bias)
                + GRAVITY_NED
            )
            position = position + velocity * dt + 0.5 * accel_ned * dt * dt
            velocity = velocity + accel_ned * dt
            orientation = quat_normalise(
                quat_multiply(orientation, quat_from_rotation_vector(gyro * dt))
            )
            angular_velocity = gyro

        self._predicted = VehicleState(
            time_s=self._now,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
        )

    # ------------------------------------------------------------------
    def vehicle_state(self) -> VehicleState:
        """Current best estimate, forward-predicted to the latest IMU sample."""
        self._ensure_predicted()
        return self._predicted.copy()

    @property
    def predicted_is_stale(self) -> bool:
        """True when the cached prediction has not been recomputed yet.

        Exposed for the tests, which check the laziness is a pure optimisation:
        the state returned must be identical whether it was computed eagerly or
        on demand.
        """
        return self._predicted_stale

    def horizon_state(self) -> VehicleState:
        """The EKF state at the fusion horizon, for log analysis and tests."""
        return self.ekf.vehicle_state()
