"""Shared acceleration-to-attitude conversion.

Both trackers end at the same place: an acceleration demand in NED that has to
become a tilt and a collective thrust. That step is differential flatness, not
control design, so it lives here and both use it. If each tracker rolled its own
version, part of any measured difference between them would be this conversion
rather than the control law, and the comparison would be worthless.
"""

from __future__ import annotations

import numpy as np

from ..attitude import attitude_from_thrust_and_yaw
from ..geodesy import GRAVITY
from ..types import AttitudeSetpoint

GRAVITY_NED = np.array([0.0, 0.0, GRAVITY])


def lowpass_alpha(cutoff_hz: float, dt: float) -> float:
    """Blend coefficient of a one-pole low-pass at ``cutoff_hz``, sampled at ``dt``.

    ``alpha = dt / (tau + dt)`` with ``tau = 1 / (2 pi f_c)``. Worth having in
    one place with its own name, because the obvious shortcut is wrong in a way
    that hides:

    The shortcut ``alpha = 2 pi f_c dt``, clipped to 1, is the first-order
    expansion and is fine while ``2 pi f_c dt`` is small. At the rates actually
    used here it is not small — an 8 Hz cutoff sampled at 50 Hz gives 1.005,
    which clips to exactly 1.0, and a coefficient of 1.0 is not a filter at all
    but a pass-through. The PID's velocity derivative was therefore
    differentiating raw velocity across a single 20 ms step. With gentle gains
    that is merely noisy; with the tuned gains it sustained a limit cycle, and
    the vehicle sat 0.18 m off a step reference indefinitely instead of settling.

    This form is exact for a first-order hold and is bounded in ``(0, 1)`` for
    any positive cutoff and step, so it cannot degenerate.
    """
    if cutoff_hz <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(dt / (tau + dt))


def tilt_limit_acceleration(acceleration: np.ndarray, max_tilt_rad: float) -> tuple:
    """Clamp a NED acceleration demand to one the vehicle can actually tilt to.

    The limit is on the *angle between the required thrust vector and vertical*,
    not on horizontal acceleration alone, because climbing gives you more
    horizontal authority and descending gives you less. Returns the clamped
    acceleration and whether the clamp bit.

    Vertical demand is preserved and horizontal demand is scaled: refusing to
    hold altitude in order to accelerate sideways is how a multirotor ends up in
    the ground.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    # Specific force the rotors must produce, in NED. Pointing "up" is negative.
    thrust_ned = acceleration - GRAVITY_NED
    vertical = -thrust_ned[2]  # positive when thrust has an upward component
    horizontal = float(np.linalg.norm(thrust_ned[0:2]))

    if vertical <= 1e-3:
        # Demanding zero or downward thrust: nothing sensible to tilt towards,
        # so fall back to the maximum horizontal at a nominally level hover.
        max_horizontal = GRAVITY * np.tan(max_tilt_rad)
    else:
        max_horizontal = vertical * np.tan(max_tilt_rad)

    if horizontal <= max_horizontal or horizontal < 1e-9:
        return acceleration, False

    scale = max_horizontal / horizontal
    limited = acceleration.copy()
    limited[0:2] = thrust_ned[0:2] * scale + GRAVITY_NED[0:2]
    return limited, True


def acceleration_to_attitude(acceleration: np.ndarray, yaw: float, params,
                             yaw_rate: float = 0.0) -> AttitudeSetpoint:
    """Turn an acceleration demand into an attitude and normalised thrust.

    ``params`` is a :class:`~mission_control.sim.quadrotor.QuadrotorParams`; only
    ``mass``, ``max_thrust`` and the tilt limit are used, all of which a real
    airframe file also carries.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    thrust_ned = acceleration - GRAVITY_NED
    force = params.mass * float(np.linalg.norm(thrust_ned))
    thrust_normalised = float(np.clip(force / params.max_thrust, 0.0, 1.0))

    return AttitudeSetpoint(
        orientation=attitude_from_thrust_and_yaw(thrust_ned, yaw),
        body_rate=np.array([0.0, 0.0, yaw_rate]),
        thrust=thrust_normalised,
        acceleration_demand=acceleration.copy(),
    )


def control_max_tilt(params, margin: float = 0.75) -> float:
    """Tilt the trajectory trackers are allowed to command, rad.

    Deliberately below the airframe's physical limit. At the physical limit the
    vehicle has no thrust left to hold altitude, so a controller that plans to
    fly there will trade height for horizontal error without ever being asked
    to. 75% of the limit is about 47 degrees here, comfortably aggressive.
    """
    return float(params.max_tilt_for_hover * margin)
