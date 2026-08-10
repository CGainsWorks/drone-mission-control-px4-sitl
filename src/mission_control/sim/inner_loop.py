"""Cascaded attitude and rate controller.

This is the loop that already exists on the flight controller. It is modelled
here so the trajectory trackers under test see a realistic actuator: an attitude
command does not become an attitude, it becomes a rate demand, then a torque
demand, then four saturating motors, each with lag.

Structure and gain scaling follow PX4's ``mc_att_control`` and
``mc_rate_control``: a proportional quaternion attitude law feeding a PID rate
law with a gyroscopic feedforward term. Neither tracker is allowed to touch it,
so both fly the same vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import (
    cross3,
    quat_conjugate,
    quat_from_rotation_vector,
    quat_multiply,
    quat_normalise,
    quat_to_rotation_matrix,
)
from ..control.common import lowpass_alpha
from ..types import AttitudeSetpoint


@dataclass
class InnerLoopGains:
    """Attitude and rate gains.

    Roll and pitch share gains; yaw is slacker because a multirotor's yaw
    authority comes from rotor drag torque and is an order of magnitude weaker.
    """

    attitude_p: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 3.0]))
    rate_p: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 12.0]))
    rate_i: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 3.0]))
    rate_d: np.ndarray = field(default_factory=lambda: np.array([0.40, 0.40, 0.05]))
    #: Rate demand limits, rad/s. 220 deg/s in roll and pitch, 120 in yaw.
    max_rate: np.ndarray = field(default_factory=lambda: np.radians([220.0, 220.0, 120.0]))
    #: Integrator clamp, rad/s^2 of demanded angular acceleration.
    integral_limit: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0, 8.0]))
    #: Cutoff of the first-order filter on the rate derivative term, Hz.
    derivative_cutoff_hz: float = 30.0


class InnerLoop:
    """Attitude setpoint in, rotor thrust commands out."""

    def __init__(self, params, gains: InnerLoopGains = None):
        from .quadrotor import Mixer

        self.params = params
        self.gains = gains or InnerLoopGains()
        self.mixer = Mixer(params)
        #: How much of the heading error to correct relative to the thrust-axis
        #: error. Taken from the gain ratio so there is one number to tune, not
        #: two that can disagree.
        self._yaw_weight = float(
            np.clip(self.gains.attitude_p[2] / max(self.gains.attitude_p[0], 1e-6), 0.0, 1.0)
        )
        self._integral = np.zeros(3)
        self._rate_filtered = np.zeros(3)
        self._last_rate = np.zeros(3)
        self.last_rate_setpoint = np.zeros(3)

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._rate_filtered[:] = 0.0
        self._last_rate[:] = 0.0
        self.last_rate_setpoint[:] = 0.0

    def _reduced_setpoint(self, q: np.ndarray, q_sp: np.ndarray) -> np.ndarray:
        """Blend the attitude setpoint so that thrust direction beats heading.

        Naively driving the full quaternion error treats a heading error and a
        thrust-direction error as equally urgent. They are not. Thrust direction
        is what holds the vehicle up and moves it; heading only points the
        camera. Worse, at a yaw error approaching 180 degrees the full-quaternion
        law is ill-conditioned — the scalar part passes through zero, the
        shortest-path sign flips, and the rate demand chatters between its
        limits. That is not theoretical: it is what made this simulation lose
        the vehicle on a leg flown nose-backwards, and it did so intermittently,
        which is the worst way to find a bug.

        The fix is PX4's ``AttitudeControl`` construction. Build a *reduced*
        setpoint that rotates the current attitude the short way onto the
        desired thrust axis and leaves heading alone; then rotate a fraction
        ``yaw_weight`` of the way from there towards the full setpoint. Thrust
        direction is always corrected at full gain, heading is corrected at a
        lower one, and nothing is ill-conditioned anywhere.
        """
        # Body z axis, current and desired, expressed in NED.
        e_z = quat_to_rotation_matrix(q)[:, 2]
        e_z_sp = quat_to_rotation_matrix(q_sp)[:, 2]

        # Shortest rotation taking e_z onto e_z_sp, as a quaternion.
        dot = float(np.clip(np.dot(e_z, e_z_sp), -1.0, 1.0))
        axis = cross3(e_z, e_z_sp)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            # Already aligned, or exactly inverted. Inverted means the vehicle
            # is upside down relative to the demand, which no blending can make
            # sensible, so fall through to the full setpoint.
            q_reduced = q.copy() if dot > 0.0 else q_sp.copy()
        else:
            angle = float(np.arctan2(axis_norm, dot))
            q_align = quat_from_rotation_vector(axis / axis_norm * angle)
            q_reduced = quat_normalise(quat_multiply(q_align, q))

        # What is left between the reduced setpoint and the full one is a pure
        # rotation about the body z axis: the heading correction.
        q_mix = quat_multiply(quat_conjugate(q_reduced), q_sp)
        if q_mix[0] < 0.0:
            q_mix = -q_mix
        cos_half = float(np.clip(q_mix[0], -1.0, 1.0))
        sin_half = float(np.clip(q_mix[3], -1.0, 1.0))
        weight = self._yaw_weight
        partial = np.array(
            [np.cos(weight * np.arccos(cos_half)), 0.0, 0.0,
             np.sin(weight * np.arcsin(sin_half))]
        )
        return quat_normalise(quat_multiply(q_reduced, partial))

    def _attitude_to_rate(self, quat: np.ndarray, setpoint: AttitudeSetpoint) -> np.ndarray:
        q = quat_normalise(quat)
        q_sp = self._reduced_setpoint(q, quat_normalise(setpoint.orientation))
        q_err = quat_multiply(quat_conjugate(q), q_sp)
        # Shortest path: the double cover means -q and q are the same attitude,
        # so without this the vehicle occasionally takes the long way round.
        if q_err[0] < 0.0:
            q_err = -q_err
        rate_sp = 2.0 * self.gains.attitude_p * q_err[1:4]
        rate_sp = rate_sp + setpoint.body_rate
        return np.clip(rate_sp, -self.gains.max_rate, self.gains.max_rate)

    def update(self, quat: np.ndarray, angular_velocity: np.ndarray,
               setpoint: AttitudeSetpoint, dt: float) -> np.ndarray:
        """Run one inner-loop iteration and return four rotor thrust commands, N."""
        g = self.gains
        rate_sp = self._attitude_to_rate(quat, setpoint)
        self.last_rate_setpoint = rate_sp

        error = rate_sp - angular_velocity

        # Derivative on the measurement, not on the error: a step in the rate
        # setpoint should not produce a torque spike.
        alpha = lowpass_alpha(g.derivative_cutoff_hz, dt)
        self._rate_filtered += alpha * (angular_velocity - self._rate_filtered)
        rate_derivative = (self._rate_filtered - self._last_rate) / max(dt, 1e-6)
        self._last_rate = self._rate_filtered.copy()

        self._integral = np.clip(
            self._integral + g.rate_i * error * dt, -g.integral_limit, g.integral_limit
        )

        angular_accel = g.rate_p * error + self._integral - g.rate_d * rate_derivative

        inertia = self.params.inertia
        torque = inertia * angular_accel + cross3(
            angular_velocity, inertia * angular_velocity
        )

        collective = float(np.clip(setpoint.thrust, 0.0, 1.0)) * self.params.max_thrust
        return self.mixer.allocate(collective, torque)
