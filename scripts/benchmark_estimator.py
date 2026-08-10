#!/usr/bin/env python3
"""Measure what latency compensation is worth in the estimator.

A GNSS fix describes where the vehicle *was*. Fusing it as though it described
*now* drags the estimate backwards along the flight path by speed times latency.
The estimator answers that with two clocks — a delayed fusion horizon and a
forward output predictor — and this script measures the difference, so the
figures in `docs/ESTIMATOR.md` come from a command rather than from memory.

Both configurations fly the same excitation pattern with the same seeds, so the
comparison is paired: identical wind, identical IMU noise, identical GNSS
errors, identical trajectory. Only the fusion architecture differs.

Run:  python scripts/benchmark_estimator.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.parallel import limit_blas_threads  # noqa: E402

limit_blas_threads()

import numpy as np  # noqa: E402

from mission_control.attitude import euler_to_quat, quat_to_euler, wrap_pi  # noqa: E402
from mission_control.estimation import ErrorStateEkf, GnssInsEstimator  # noqa: E402
from mission_control.sim import (  # noqa: E402
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
from mission_control.types import AttitudeSetpoint, VehicleState  # noqa: E402

DT = 0.005  # 200 Hz, the IMU rate
DURATION_S = 60.0
SETTLE_S = 12.0


def fly(compensated: bool, latency_s: float, seed: int = 1, wind=None) -> dict:
    """Fly the excitation pattern and collect estimator error.

    The pattern excites two axes at incommensurate periods on purpose:
    attitude and accelerometer bias are only observable when the vehicle
    accelerates in more than one direction, so a filter measured in a straight
    line would look better than it is.
    """
    params = QuadrotorParams()
    plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -20.0])))
    loop = InnerLoop(params)
    imu = ImuSensor(ImuParams(rate_hz=1.0 / DT), seed=seed + 11)
    gnss = GnssReceiver(GnssParams(rate_hz=5.0, latency_s=latency_s), seed=seed + 23)
    field = WindField(wind or WindParams(), seed=seed + 37)

    estimator = (
        GnssInsEstimator(fusion_delay_s=latency_s + 0.02) if compensated else ErrorStateEkf()
    )

    position_errors, velocity_errors, speeds, nis = [], [], [], []
    for step in range(int(DURATION_S / DT)):
        t = step * DT
        plant.wind_ned = field.step(DT)
        gnss.record_truth(t, plant.state[0:3], plant.state[3:6])

        roll = np.radians(8.0 * np.sin(2 * np.pi * t / 11.0))
        pitch = np.radians(8.0 * np.cos(2 * np.pi * t / 13.0))
        yaw = np.radians(25.0 * np.sin(2 * np.pi * t / 29.0))
        setpoint = AttitudeSetpoint(
            orientation=euler_to_quat(roll, pitch, yaw),
            thrust=params.hover_thrust / max(np.cos(roll) * np.cos(pitch), 0.5),
        )
        plant.step(loop.update(plant.state[6:10], plant.state[10:13], setpoint, DT), DT)

        sample = imu.sample(t, plant.specific_force_body(), plant.state[10:13], DT)
        fix = gnss.sample(t)

        if compensated:
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
        if t > SETTLE_S:
            position_errors.append(state.position - plant.state[0:3])
            velocity_errors.append(state.velocity - plant.state[3:6])
            speeds.append(float(np.linalg.norm(plant.state[3:6])))

    position_errors = np.asarray(position_errors)
    velocity_errors = np.asarray(velocity_errors)
    attitude_error = quat_to_euler(
        (estimator.vehicle_state() if compensated else estimator.vehicle_state()).orientation
    ) - quat_to_euler(plant.state[6:10])

    return {
        "position_rms": float(np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))),
        "velocity_rms": float(np.sqrt(np.mean(np.sum(velocity_errors**2, axis=1)))),
        "position_max": float(np.max(np.linalg.norm(position_errors, axis=1))),
        "mean_nis": float(np.mean(nis)) if nis else float("nan"),
        "mean_speed": float(np.mean(speeds)),
        "roll_error_deg": float(np.degrees(wrap_pi(attitude_error[0]))),
        "pitch_error_deg": float(np.degrees(wrap_pi(attitude_error[1]))),
        "yaw_error_deg": float(np.degrees(wrap_pi(attitude_error[2]))),
        "estimator": estimator,
        "imu": imu,
    }


def main() -> int:
    latency = 0.08
    print("Estimator: RTK-GNSS and IMU, error-state EKF")
    print(f"  {DURATION_S:.0f} s of two-axis excitation, RTK-fixed throughout, "
          f"GNSS at 5 Hz with {latency * 1000:.0f} ms of latency")
    print(f"  error measured after {SETTLE_S:.0f} s of settling")
    print()

    naive = fly(compensated=False, latency_s=latency)
    compensated = fly(compensated=True, latency_s=latency)

    print(f"  mean ground speed over the run: {compensated['mean_speed']:.2f} m/s")
    print(f"  speed x latency               : "
          f"{compensated['mean_speed'] * latency:.3f} m  <- the error to be removed")
    print()
    print(f"  {'':44} {'position RMS':>13} {'velocity RMS':>14} {'worst':>8}")
    print(f"  {'fusing each fix as though it were current':44} "
          f"{naive['position_rms']:11.3f} m {naive['velocity_rms']:12.3f} m/s "
          f"{naive['position_max']:6.3f} m")
    print(f"  {'fusion horizon plus output predictor':44} "
          f"{compensated['position_rms']:11.3f} m {compensated['velocity_rms']:12.3f} m/s "
          f"{compensated['position_max']:6.3f} m")
    print()
    ratio = naive["position_rms"] / max(compensated["position_rms"], 1e-9)
    print(f"  improvement: {ratio:.1f}x in position, "
          f"{naive['velocity_rms'] / max(compensated['velocity_rms'], 1e-9):.1f}x in velocity")
    print()

    print("Filter consistency (a 3-DOF update on a consistent filter averages ~3):")
    print(f"  mean NIS  {compensated['mean_nis']:.2f}")
    print("  below 3 because the measurement noise is inflated 1.5x over what the")
    print("  receiver reports, which does not include its correlated bias; the cost")
    print("  is discarded information, the benefit is that the gate stays open")
    print()

    print("Attitude error at the end of the run:")
    print(f"  roll  {compensated['roll_error_deg']:+6.2f} deg")
    print(f"  pitch {compensated['pitch_error_deg']:+6.2f} deg")
    print(f"  yaw   {compensated['yaw_error_deg']:+6.2f} deg  <- weakly observable "
          "without a magnetometer")
    print()

    estimator = compensated["estimator"]
    imu = compensated["imu"]
    print("Sensor bias estimation (estimate minus truth):")
    print(f"  accelerometer  {np.max(np.abs(estimator.accel_bias - imu.accel_bias)):.4f} m/s^2")
    print(f"  gyroscope      {np.max(np.abs(estimator.gyro_bias - imu.gyro_bias)):.6f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
