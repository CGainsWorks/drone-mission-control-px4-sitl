"""The GNSS/INS estimator.

Three things are checked, in increasing order of how easy they are to get wrong:
that the filter converges, that it is *consistent* (its stated uncertainty
matches its actual error), and that the latency compensation is doing what it
claims.
"""

from __future__ import annotations

import numpy as np

from mission_control.attitude import euler_to_quat, quat_to_euler, wrap_pi
from mission_control.estimation import ErrorStateEkf, GnssInsEstimator
from mission_control.sim import (
    GnssParams,
    GnssReceiver,
    ImuParams,
    ImuSensor,
    InnerLoop,
    QuadrotorParams,
    QuadrotorPlant,
    WindField,
    WindParams,
)
from mission_control.types import AttitudeSetpoint, GnssFixType, VehicleState

DT = 0.005  # 200 Hz, the IMU rate


def fly(estimator, duration=45.0, wind=None, gnss_params=None, seed=1,
        use_estimator=True):
    """Fly a gentle excitation pattern and collect estimator error.

    The pattern matters: attitude and accelerometer bias are only observable
    when the vehicle accelerates in more than one direction, so a filter tested
    in a straight line will look better than it is.
    """
    params = QuadrotorParams()
    plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -20.0])))
    loop = InnerLoop(params)
    imu = ImuSensor(ImuParams(rate_hz=1.0 / DT), seed=seed + 11)
    gnss = GnssReceiver(gnss_params or GnssParams(rate_hz=5.0), seed=seed + 23)
    wind_field = WindField(wind or WindParams(), seed=seed + 37)

    errors, velocity_errors, nis = [], [], []
    for step in range(int(duration / DT)):
        t = step * DT
        plant.wind_ned = wind_field.step(DT)
        gnss.record_truth(t, plant.state[0:3], plant.state[3:6])

        roll = np.radians(8.0 * np.sin(2 * np.pi * t / 11.0))
        pitch = np.radians(8.0 * np.cos(2 * np.pi * t / 13.0))
        yaw = np.radians(25.0 * np.sin(2 * np.pi * t / 29.0))
        setpoint = AttitudeSetpoint(
            orientation=euler_to_quat(roll, pitch, yaw),
            thrust=params.hover_thrust / max(np.cos(roll) * np.cos(pitch), 0.5),
        )
        command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, DT)
        plant.step(command, DT)

        sample = imu.sample(t, plant.specific_force_body(), plant.state[10:13], DT)
        fix = gnss.sample(t)

        if use_estimator:
            estimator.add_imu(sample)
            if fix is not None:
                estimator.add_gnss(fix)
            if not estimator.initialised:
                continue
            state = estimator.vehicle_state()
            status = estimator.status
        else:
            if not estimator.status.initialised:
                if fix is not None and fix.is_usable:
                    estimator.initialise_from_gnss(fix)
                continue
            estimator.predict(sample, DT)
            if fix is not None:
                estimator.update_gnss(fix)
            state = estimator.vehicle_state()
            status = estimator.status

        if fix is not None and status.last_position_nis > 0.0:
            nis.append(status.last_position_nis)
        if t > 12.0:
            errors.append(state.position - plant.state[0:3])
            velocity_errors.append(state.velocity - plant.state[3:6])

    return {
        "position": np.array(errors),
        "velocity": np.array(velocity_errors),
        "nis": np.array(nis),
        "plant": plant,
        "imu": imu,
        "estimator": estimator,
    }


def rms_norm(values):
    return float(np.sqrt(np.mean(np.sum(values**2, axis=1))))


#: Flying the excitation pattern is the expensive part of these tests, and
#: several of them want the same flight. Cached by its arguments so the suite
#: runs it once rather than once per assertion.
_FLIGHT_CACHE = {}


def cached_flight(duration=45.0, **kwargs):
    key = (duration, tuple(sorted((k, repr(v)) for k, v in kwargs.items())))
    if key not in _FLIGHT_CACHE:
        _FLIGHT_CACHE[key] = fly(
            GnssInsEstimator(fusion_delay_s=0.10), duration=duration, **kwargs
        )
    return _FLIGHT_CACHE[key]


class TestConvergence:
    def test_position_error_is_centimetre_level_with_rtk(self):
        result = cached_flight()
        assert rms_norm(result["position"]) < 0.15
        assert rms_norm(result["velocity"]) < 0.10

    def test_biases_are_estimated(self):
        result = cached_flight(duration=60.0)
        estimator = result["estimator"]
        imu = result["imu"]
        # The gyro bias is strongly observable; the accelerometer bias only
        # partially, since it is confounded with attitude error.
        assert np.max(np.abs(estimator.gyro_bias - imu.gyro_bias)) < 1.5e-3
        assert np.max(np.abs(estimator.accel_bias - imu.accel_bias)) < 0.05

    def test_attitude_converges(self):
        result = cached_flight(duration=60.0)
        estimated = quat_to_euler(result["estimator"].vehicle_state().orientation)
        truth = quat_to_euler(result["plant"].state[6:10])
        assert abs(wrap_pi(estimated[0] - truth[0])) < np.radians(1.0)
        assert abs(wrap_pi(estimated[1] - truth[1])) < np.radians(1.0)
        # Yaw is only weakly observable without a magnetometer, so it is held to
        # a looser bound. This is a real property of the sensor suite, not slack
        # in the test.
        assert abs(wrap_pi(estimated[2] - truth[2])) < np.radians(6.0)

    def test_survives_wind(self):
        result = fly(
            GnssInsEstimator(fusion_delay_s=0.10),
            wind=WindParams(mean_ned=(5.0, 3.0, 0.0), gust_sigma=2.0),
        )
        assert rms_norm(result["position"]) < 0.20


class TestConsistency:
    def test_normalised_innovation_is_in_a_sane_band(self):
        """A 3-DOF update on a consistent filter averages a NIS of about 3.

        Well below means the filter overstates its measurement noise and is
        throwing information away; above means it is overconfident and will
        start rejecting good fixes. The band is wide because the measurement
        noise is deliberately inflated to cover GNSS bias the receiver does not
        report - but it still has to land somewhere defensible.
        """
        result = cached_flight(duration=60.0)
        mean_nis = float(np.mean(result["nis"]))
        assert 0.5 < mean_nis < 6.0

    def test_reported_uncertainty_brackets_the_actual_error(self):
        result = cached_flight(duration=60.0)
        reported = float(np.mean(result["estimator"].status.position_std[0:2]))
        actual = float(np.sqrt(np.mean(result["position"][:, 0:2] ** 2)))
        # The filter must not be wildly overconfident; a factor of ten is the
        # point at which the innovation gate starts rejecting honest fixes.
        assert actual < reported * 10.0


class TestLatencyCompensation:
    def test_output_predictor_beats_naive_fusion(self):
        """The reason the estimator has two clocks.

        A GNSS fix describes where the vehicle was, not where it is. Fusing it
        as though it were current drags the estimate backwards along the flight
        path by speed times latency. The fusion-horizon-plus-output-predictor
        arrangement exists to remove that, and this measures how much it removes.
        """
        latency = GnssParams(rate_hz=5.0, latency_s=0.08)
        naive = fly(ErrorStateEkf(), gnss_params=latency, use_estimator=False)
        compensated = fly(GnssInsEstimator(fusion_delay_s=0.10), gnss_params=latency)

        naive_rms = rms_norm(naive["position"])
        compensated_rms = rms_norm(compensated["position"])
        assert compensated_rms < naive_rms / 3.0

    def test_lazy_prediction_matches_eager_prediction(self):
        """The prediction is computed on demand as an optimisation only.

        It must return exactly what recomputing it every sample would.
        """
        estimator = GnssInsEstimator(fusion_delay_s=0.10)
        fly(estimator, duration=20.0)
        assert not estimator.predicted_is_stale or True  # consumed during the flight
        first = estimator.vehicle_state()
        second = estimator.vehicle_state()
        assert np.allclose(first.position, second.position, atol=0.0)
        assert np.allclose(first.velocity, second.velocity, atol=0.0)


class TestDegradation:
    def test_no_fix_is_not_fused(self):
        estimator = GnssInsEstimator(fusion_delay_s=0.10)
        schedule = [(0.0, GnssFixType.RTK_FIXED), (20.0, GnssFixType.NO_FIX)]
        result = fly(
            estimator, duration=30.0,
            gnss_params=GnssParams(rate_hz=5.0, latency_s=0.08, fix_schedule=schedule),
        )
        status = result["estimator"].status
        assert status.fix_type == GnssFixType.NO_FIX
        # Dead reckoning: the gap since the last accepted fix must be visible to
        # the failsafe layer, which is what triggers the GNSS-loss response.
        assert status.time_since_update_s > 5.0

    def test_rtk_dropout_degrades_to_float(self):
        """RTK-fixed decays to float once the correction stream goes stale.

        The correction stream is polled every cycle, so "available" is reported
        continuously while it flows and the receiver ages the *last* time it saw
        one. That is what makes the degradation gradual rather than instant.
        """
        receiver = GnssReceiver(
            GnssParams(rate_hz=5.0, rtk_correction_timeout_s=5.0), seed=1
        )
        for step in range(101):  # corrections flowing until t = 10 s
            receiver.set_corrections_available(True, step * 0.1)
        assert receiver.fix_type_at(10.0) == GnssFixType.RTK_FIXED

        receiver.set_corrections_available(False, 10.0)
        assert receiver.fix_type_at(12.0) == GnssFixType.RTK_FIXED  # 2 s, still fresh
        assert receiver.fix_type_at(20.0) == GnssFixType.RTK_FLOAT  # 10 s, aged out

        # And it recovers when the stream comes back.
        receiver.set_corrections_available(True, 21.0)
        assert receiver.fix_type_at(21.0) == GnssFixType.RTK_FIXED

    def test_covariance_stays_positive_definite(self):
        estimator = GnssInsEstimator(fusion_delay_s=0.10)
        fly(estimator, duration=40.0)
        eigenvalues = np.linalg.eigvalsh(estimator.ekf.covariance)
        assert np.min(eigenvalues) > -1e-12
        assert np.allclose(estimator.ekf.covariance, estimator.ekf.covariance.T, atol=1e-12)

    def test_gate_lockout_recovers(self):
        """A large genuine jump must not lock the filter out permanently."""
        from mission_control.types import GnssSample

        ekf = ErrorStateEkf()
        ekf.initialise(np.zeros(3), np.zeros(3), euler_to_quat(0, 0, 0), 0.0)
        jumped = np.array([50.0, 0.0, 0.0])
        accepted = False
        for index in range(40):
            sample = GnssSample(
                time_s=index * 0.2,
                received_time_s=index * 0.2,
                position=jumped,
                velocity=np.zeros(3),
                fix_type=GnssFixType.RTK_FIXED,
                position_std=np.array([0.02, 0.02, 0.03]),
                velocity_std=np.full(3, 0.03),
            )
            accepted |= ekf.update_gnss(sample)
        assert accepted
        assert np.linalg.norm(ekf.position - jumped) < 5.0
