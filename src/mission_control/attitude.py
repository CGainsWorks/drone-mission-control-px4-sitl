"""Quaternion and rotation utilities.

Conventions, fixed once here so nothing downstream has to guess:

* Quaternions are Hamilton, scalar-first ``[w, x, y, z]``, and rotate a vector
  **from body into NED**: ``v_ned = R(q) @ v_body``.
* Euler angles are the aerospace 3-2-1 sequence: yaw about down, then pitch
  about east, then roll about north.
* NED means x north, y east, z **down**, so level flight has ``z_body`` pointing
  down and rotor thrust acting along ``-z_body``.

PX4 uses the same conventions, which keeps the SITL backend free of frame
conversions beyond unit changes.
"""

from __future__ import annotations

import math

import numpy as np


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalise(q: np.ndarray) -> np.ndarray:
    """Scale to unit length. **Does not** canonicalise the sign.

    That restraint matters more than it looks. ``q`` and ``-q`` are the same
    attitude, so it is tempting to normalise the sign here — force the scalar
    part non-negative — and get tidier logs. Doing that is a serious bug,
    because this function is applied to the vehicle's *state* quaternion on
    every integration step.

    A vehicle heading due south has a scalar part of exactly zero. Sitting on
    that boundary, the sign flips on alternating steps, and each flip reverses
    the sense of the integration ``q_dot = ½ q ⊗ [0, ω]``. The attitude then
    pins itself to the boundary: the body rates read a healthy 200 deg/s while
    the orientation does not move at all. The symptom is a vehicle that flies
    the northbound legs of a mission perfectly and cannot turn a corner on the
    southbound ones, which reads as a control tuning problem and is not one.

    Shortest-path sign selection belongs where it is meaningful — on the
    *error* quaternion inside a controller — and both places that need it do it
    explicitly.
    """
    q = np.asarray(q, dtype=float)
    # math.sqrt(q @ q) rather than np.linalg.norm: the latter dispatches through
    # several layers of generality this does not need, and this is one of the
    # most-called functions in the simulation.
    norm = math.sqrt(float(q @ q))
    if norm < 1e-12:
        return quat_identity()
    return q / norm


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Body-to-NED rotation matrix."""
    w, x, y, z = quat_normalise(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """Shepperd's method: picks the largest component to avoid cancellation."""
    m = np.asarray(rot, dtype=float)
    trace = np.trace(m)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array(
            [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    return quat_normalise(q)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3-2-1 aerospace Euler angles (radians) to quaternion."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return quat_normalise(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion to ``[roll, pitch, yaw]`` in radians."""
    w, x, y, z = quat_normalise(q)
    sin_pitch = 2.0 * (w * y - z * x)
    # Clamp rather than let arcsin produce NaN at gimbal lock. Inspection
    # missions never go near +-90 degrees of pitch, but a diverged controller
    # can, and a NaN there poisons the whole log.
    sin_pitch = float(np.clip(sin_pitch, -1.0, 1.0))
    return np.array(
        [
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(sin_pitch),
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ]
    )


def quat_from_rotation_vector(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map: a rotation vector (axis * angle, radians) to a quaternion."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        # Second-order series. Avoids a 0/0 while staying accurate enough that
        # the EKF's error-state injection is not the dominant error term.
        return quat_normalise(np.concatenate(([1.0], 0.5 * rotvec)))
    axis = rotvec / angle
    return np.concatenate(([np.cos(angle * 0.5)], axis * np.sin(angle * 0.5)))


def quat_to_rotation_vector(q: np.ndarray) -> np.ndarray:
    """Logarithmic map, inverse of :func:`quat_from_rotation_vector`."""
    q = quat_normalise(q)
    vec_norm = float(np.linalg.norm(q[1:]))
    if vec_norm < 1e-9:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(vec_norm, q[0])
    return q[1:] / vec_norm * angle


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors.

    ``numpy.cross`` is general enough to handle broadcasting and axis
    arguments, and pays for it: on 3-vectors it is several times slower than
    writing the three components out. It appears four times per RK4 stage in
    the plant at 500 Hz, where that difference is measurable.
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``skew(a) @ b == cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def wrap_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def yaw_of(q: np.ndarray) -> float:
    return float(quat_to_euler(q)[2])


def attitude_from_thrust_and_yaw(thrust_ned: np.ndarray, yaw: float) -> np.ndarray:
    """Attitude whose body ``-z`` axis lies along ``thrust_ned``, at heading ``yaw``.

    The standard multirotor differential-flatness step: an acceleration demand
    plus a yaw demand determine the attitude uniquely. Both trackers share it,
    so the comparison measures trajectory logic rather than two different ways
    of turning acceleration into tilt.

    Solved through the Euler angles rather than by building an orthonormal
    frame from cross products. The cross-product construction is the more
    obvious one and it is subtly wrong for this codebase: the attitude it
    produces has a 3-2-1 Euler yaw that differs from the requested ``yaw`` once
    the vehicle is tilted — by nearly a degree at only 11 degrees of tilt, and
    more as the tilt grows. Since the trajectory generator, the estimator and
    every heading metric here all speak 3-2-1 Euler, that mismatch would show up
    as a tilt-dependent heading bias in all of them.

    Writing ``R = Rz(yaw) Ry(pitch) Rx(roll)`` and rotating the desired body-z
    axis into the yawed frame gives the two remaining angles in closed form,
    with the requested yaw reproduced exactly.
    """
    thrust_ned = np.asarray(thrust_ned, dtype=float)
    norm = float(np.linalg.norm(thrust_ned))
    if norm < 1e-6:
        # Degenerate free-fall demand: hold wings level at the requested yaw
        # rather than producing an arbitrary attitude.
        return euler_to_quat(0.0, 0.0, yaw)

    # Thrust acts along -z_body, so the body z axis is the opposite direction.
    z_body = -thrust_ned / norm

    # Express it in the frame already rotated by yaw.
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    u_x = cos_yaw * z_body[0] + sin_yaw * z_body[1]
    u_y = -sin_yaw * z_body[0] + cos_yaw * z_body[1]
    u_z = z_body[2]

    roll = float(np.arcsin(np.clip(-u_y, -1.0, 1.0)))
    pitch = float(np.arctan2(u_x, u_z))
    return euler_to_quat(roll, pitch, yaw)
