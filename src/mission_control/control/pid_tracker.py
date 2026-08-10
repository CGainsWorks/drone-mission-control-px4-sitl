"""Cascaded PID position controller.

The baseline. Structure is PX4's ``mc_pos_control``: a proportional outer
position loop producing a velocity setpoint, and a PID inner velocity loop
producing an acceleration demand. It is the controller almost every multirotor
in service is actually flying, which is what makes it the right thing to measure
MPC against.

Its one structural limitation, and the thing the comparison is designed to
expose: it is causal in the reference. It reacts to the error it has now. It
cannot lean into a corner before reaching it, because at the moment it would
need to start leaning, the error is still zero. Feedforward velocity and
acceleration recover part of that, and are enabled here, so the comparison is
against a *fairly* configured PID rather than a straw man.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import AttitudeSetpoint, VehicleState
from .common import (
    acceleration_to_attitude,
    control_max_tilt,
    lowpass_alpha,
    tilt_limit_acceleration,
)
from .interface import ReferenceProvider, TrackerDiagnostics, TrajectoryTracker


@dataclass
class PidGains:
    """Position and velocity loop gains.

    Chosen by placing the closed-loop poles of the (double-integrator) position
    dynamics rather than by hand-turning knobs, so the two axes have a stated
    bandwidth and damping instead of a folk tuning:

    * horizontal: natural frequency 2.0 rad/s, damping 0.8
    * vertical: 3.0 rad/s, damping 0.9, faster because thrust acts directly on
      it with no tilt dynamics in the way

    which gives ``vel_p = 2 zeta wn`` and ``pos_p = wn^2 / vel_p``.
    ``scripts/tune_gains.py`` re-derives them and shows the sweep behind the
    choice.
    """

    position_p: np.ndarray = field(default_factory=lambda: np.array([1.25, 1.25, 1.667]))
    velocity_p: np.ndarray = field(default_factory=lambda: np.array([3.20, 3.20, 5.40]))
    velocity_i: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.60, 1.00]))
    velocity_d: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.20, 0.10]))

    max_velocity_xy: float = 12.0
    max_velocity_up: float = 4.0
    max_velocity_down: float = 2.5
    max_acceleration_xy: float = 8.0
    max_acceleration_z: float = 5.0
    #: Integrator clamp, m/s^2. Sized to cancel a steady wind, not to paper over
    #: a modelling error.
    integral_limit: float = 3.0
    #: Cutoff of the low-pass on the velocity derivative term, Hz.
    derivative_cutoff_hz: float = 8.0
    #: Feedforward gains on the reference velocity and acceleration.
    velocity_feedforward: float = 1.0
    acceleration_feedforward: float = 1.0


class PidTracker(TrajectoryTracker):
    """Cascaded P-position, PID-velocity trajectory tracker."""

    name = "pid"

    def __init__(self, params, gains: PidGains = None, max_tilt: float = None):
        super().__init__(params)
        self.gains = gains or PidGains()
        self.max_tilt = control_max_tilt(params) if max_tilt is None else max_tilt
        self._integral = np.zeros(3)
        self._velocity_filtered = np.zeros(3)
        self._last_velocity = np.zeros(3)
        self._initialised = False

    def reset(self, state: VehicleState) -> None:
        self._integral[:] = 0.0
        self._velocity_filtered = np.asarray(state.velocity, dtype=float).copy()
        self._last_velocity = self._velocity_filtered.copy()
        self._initialised = True

    def update(self, state: VehicleState, reference: ReferenceProvider,
               time_s: float, dt: float) -> AttitudeSetpoint:
        import time as _time

        started = _time.perf_counter()
        if not self._initialised:
            self.reset(state)
        g = self.gains
        setpoint = reference.sample(time_s)

        # --- outer loop: position error to velocity setpoint ----------------
        position_error = setpoint.position - state.position
        velocity_setpoint = (
            g.velocity_feedforward * setpoint.velocity + g.position_p * position_error
        )
        horizontal = velocity_setpoint[0:2]
        speed = float(np.linalg.norm(horizontal))
        if speed > g.max_velocity_xy:
            velocity_setpoint[0:2] = horizontal * (g.max_velocity_xy / speed)
        velocity_setpoint[2] = float(
            np.clip(velocity_setpoint[2], -g.max_velocity_up, g.max_velocity_down)
        )

        # --- inner loop: velocity error to acceleration demand --------------
        velocity_error = velocity_setpoint - state.velocity

        # Derivative on the measurement, low-pass filtered. On the error it
        # would spike every time the mission sequences to a new leg.
        alpha = lowpass_alpha(g.derivative_cutoff_hz, dt)
        self._velocity_filtered += alpha * (state.velocity - self._velocity_filtered)
        velocity_derivative = (self._velocity_filtered - self._last_velocity) / max(dt, 1e-6)
        self._last_velocity = self._velocity_filtered.copy()

        acceleration = (
            g.acceleration_feedforward * setpoint.acceleration
            + g.velocity_p * velocity_error
            + self._integral
            - g.velocity_d * velocity_derivative
        )

        unclamped = acceleration.copy()
        acceleration[0:2] = np.clip(
            acceleration[0:2], -g.max_acceleration_xy, g.max_acceleration_xy
        )
        acceleration[2] = float(
            np.clip(acceleration[2], -g.max_acceleration_z, g.max_acceleration_z)
        )
        acceleration, tilt_saturated = tilt_limit_acceleration(acceleration, self.max_tilt)
        saturated = tilt_saturated or not np.allclose(unclamped, acceleration, atol=1e-9)

        # Conditional integration: stop winding up against a limit we cannot
        # push past. Integrating through saturation is the classic way to get a
        # controller that overshoots for seconds after a long climb.
        if not saturated:
            self._integral = np.clip(
                self._integral + g.velocity_i * velocity_error * dt,
                -g.integral_limit,
                g.integral_limit,
            )
        else:
            # Still allow the integrator to unwind in the direction that
            # relieves the saturation.
            relief = g.velocity_i * velocity_error * dt
            self._integral = np.clip(
                np.where(np.sign(relief) != np.sign(self._integral), self._integral + relief,
                         self._integral),
                -g.integral_limit,
                g.integral_limit,
            )

        elapsed = _time.perf_counter() - started
        self.diagnostics = TrackerDiagnostics(
            update_time_s=elapsed,
            # A PID has no optimisation step; its "solve" is the whole update.
            solve_time_s=elapsed,
            iterations=1,
            saturated=bool(saturated),
            acceleration_demand=acceleration.copy(),
            velocity_setpoint=velocity_setpoint.copy(),
        )
        return acceleration_to_attitude(acceleration, setpoint.yaw, self.params,
                                        yaw_rate=setpoint.yaw_rate)

    def describe(self) -> dict:
        g = self.gains
        return {
            "name": self.name,
            "position_p": g.position_p.tolist(),
            "velocity_p": g.velocity_p.tolist(),
            "velocity_i": g.velocity_i.tolist(),
            "velocity_d": g.velocity_d.tolist(),
            "max_velocity_xy": g.max_velocity_xy,
            "max_acceleration_xy": g.max_acceleration_xy,
            "max_tilt_deg": float(np.degrees(self.max_tilt)),
            "velocity_feedforward": g.velocity_feedforward,
            "acceleration_feedforward": g.acceleration_feedforward,
        }
