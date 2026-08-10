#!/usr/bin/env python3
"""Measure the closed-loop acceleration lag the MPC model assumes.

The MPC's internal model says a commanded acceleration reaches the vehicle
through a first-order lag of time constant ``lag_tau_s``. That number stands in
for the whole attitude loop: the time between asking for a tilt and getting the
horizontal acceleration that tilt produces. Guessing it is easy and wrong —
guess too fast and the optimiser plans manoeuvres the vehicle cannot start in
time; too slow and it leads the trajectory and overshoots.

So it is measured. This script steps the acceleration demand into the real
attitude loop and plant, records the achieved acceleration, and fits

.. math:: a(t) = a_\\infty (1 - e^{-t/\\tau})

by least squares on the log of the residual, for several step sizes and both
axes. It prints the fitted time constants and the value to put in
:class:`~mission_control.control.mpc_tracker.MpcWeights`.

Run:  python scripts/identify_lag.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mission_control.control.common import acceleration_to_attitude  # noqa: E402
from mission_control.sim import InnerLoop, QuadrotorParams, QuadrotorPlant  # noqa: E402
from mission_control.types import VehicleState  # noqa: E402

SIM_DT = 0.002
CONTROL_DT = 0.02


def step_response(demand: np.ndarray, params: QuadrotorParams, duration: float = 2.0):
    """Apply a constant acceleration demand and record achieved acceleration."""
    plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -30.0])))
    loop = InnerLoop(params)
    setpoint = acceleration_to_attitude(demand, 0.0, params)

    times, achieved = [], []
    previous_velocity = plant.state[3:6].copy()
    next_control = 0.0
    for step in range(int(duration / SIM_DT)):
        t = step * SIM_DT
        if t + 1e-9 >= next_control:
            next_control += CONTROL_DT
            # Re-issue the same demand each control cycle, as a tracker would.
            setpoint = acceleration_to_attitude(demand, 0.0, params)
            velocity = plant.state[3:6].copy()
            if step > 0:
                times.append(t)
                achieved.append((velocity - previous_velocity) / CONTROL_DT)
            previous_velocity = velocity
        command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, SIM_DT)
        plant.step(command, SIM_DT)

    return np.asarray(times), np.asarray(achieved)


#: Length of the window the fit uses, s. Kept short on purpose: at a constant
#: acceleration demand the vehicle keeps speeding up, drag grows with it, and
#: the achieved acceleration then *decays* towards a drag-limited value. Fitting
#: over a long window measures that decay — about 1.2 s — instead of the
#: attitude loop, which is twenty times faster. The window is chosen so the
#: velocity gained is small enough for the drag change to stay under a few
#: percent of the demand.
FIT_WINDOW_S = 0.8


def fit_time_constant(times: np.ndarray, signal: np.ndarray) -> tuple:
    """Fit amplitude and tau in ``a(t) = A (1 - exp(-t/tau))``.

    Both are fitted, rather than taking the amplitude from the tail: the tail is
    contaminated by drag, and pinning the amplitude to a contaminated value
    biases tau by more than an order of magnitude.
    """
    from scipy.optimize import curve_fit

    window = times <= FIT_WINDOW_S
    if window.sum() < 6:
        return float("nan"), float("nan")

    t = times[window] - times[window][0]
    y = signal[window]

    def model(t, amplitude, tau):
        return amplitude * (1.0 - np.exp(-t / max(tau, 1e-4)))

    try:
        (amplitude, tau), _ = curve_fit(
            model, t, y, p0=[max(y[-1], 1e-3), 0.15],
            bounds=([-50.0, 1e-3], [50.0, 2.0]), maxfev=20000,
        )
    except Exception:
        return float("nan"), float("nan")
    return float(amplitude), float(tau)


def main() -> int:
    params = QuadrotorParams()
    print("Closed-loop acceleration lag identification")
    print(f"  vehicle: {params.mass} kg, thrust/weight {params.thrust_to_weight:.2f}, "
          f"motor tau {params.motor_tau * 1000:.0f} ms")
    print(f"  control rate {1 / CONTROL_DT:.0f} Hz, plant rate {1 / SIM_DT:.0f} Hz")
    print()
    print(f"  fit window: first {FIT_WINDOW_S:.1f} s of the step")
    print()
    print(f"  {'axis':6} {'demand':>8} {'fitted A':>10} {'tau (s)':>9}")

    taus = []
    for axis, name in ((0, "north"), (1, "east")):
        for magnitude in (1.0, 2.0, 4.0, 6.0):
            demand = np.zeros(3)
            demand[axis] = magnitude
            times, achieved = step_response(demand, params)
            amplitude, tau = fit_time_constant(times, achieved[:, axis])
            if np.isfinite(tau):
                taus.append(tau)
            print(f"  {name:6} {magnitude:8.1f} {amplitude:10.3f} {tau:9.4f}")

    if not taus:
        print("\nno usable fits")
        return 1

    print()
    print(f"  median tau  {np.median(taus):.4f} s")
    print(f"  spread      {np.min(taus):.4f} to {np.max(taus):.4f} s")
    print()
    print(f"  MpcWeights.lag_tau_s = {np.median(taus):.2f}")
    print()
    print("  A vertical fit is not reported: thrust acts on the vertical axis")
    print("  directly, with only the rotor time constant in the way, so it is an")
    print("  order of magnitude faster and is not what the horizontal model needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
