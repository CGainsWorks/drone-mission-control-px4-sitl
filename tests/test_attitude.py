"""Quaternion conventions and the rotation utilities built on them."""

from __future__ import annotations

import numpy as np
import pytest

from mission_control.attitude import (
    attitude_from_thrust_and_yaw,
    cross3,
    euler_to_quat,
    quat_conjugate,
    quat_from_rotation_vector,
    quat_multiply,
    quat_normalise,
    quat_to_euler,
    quat_to_rotation_matrix,
    quat_to_rotation_vector,
    rotation_matrix_to_quat,
    wrap_pi,
)
from mission_control.geodesy import GRAVITY

RNG = np.random.default_rng(20260810)


def random_quaternions(count):
    q = RNG.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_euler_quaternion_round_trip():
    for _ in range(200):
        roll = RNG.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1)
        pitch = RNG.uniform(-np.pi / 2 + 0.2, np.pi / 2 - 0.2)
        yaw = RNG.uniform(-np.pi, np.pi)
        recovered = quat_to_euler(euler_to_quat(roll, pitch, yaw))
        assert recovered[0] == pytest.approx(roll, abs=1e-9)
        assert recovered[1] == pytest.approx(pitch, abs=1e-9)
        assert wrap_pi(recovered[2] - yaw) == pytest.approx(0.0, abs=1e-9)


def test_rotation_matrix_round_trip():
    for q in random_quaternions(200):
        recovered = rotation_matrix_to_quat(quat_to_rotation_matrix(q))
        # q and -q are the same rotation, so compare the matrices.
        assert np.allclose(
            quat_to_rotation_matrix(recovered), quat_to_rotation_matrix(q), atol=1e-12
        )


def test_rotation_matrices_are_orthonormal():
    for q in random_quaternions(100):
        rot = quat_to_rotation_matrix(q)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_multiplication_matches_matrix_composition():
    for a, b in zip(random_quaternions(100), random_quaternions(100)):
        assert np.allclose(
            quat_to_rotation_matrix(quat_multiply(a, b)),
            quat_to_rotation_matrix(a) @ quat_to_rotation_matrix(b),
            atol=1e-12,
        )


def test_exponential_and_logarithm_are_inverse():
    for _ in range(200):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = RNG.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        rotvec = axis * angle
        assert np.allclose(quat_to_rotation_vector(quat_from_rotation_vector(rotvec)),
                           rotvec, atol=1e-9)


def test_small_rotation_vector_does_not_divide_by_zero():
    tiny = np.array([1e-14, -2e-14, 3e-14])
    q = quat_from_rotation_vector(tiny)
    assert np.all(np.isfinite(q))
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)


def test_normalise_does_not_canonicalise_sign():
    """Regression: normalising must not force the scalar part non-negative.

    ``quat_normalise`` is applied to the plant's state quaternion on every
    integration step. Flipping the sign there is not a cosmetic choice: a
    vehicle heading due south has a scalar part of exactly zero, so the flip
    triggers on alternating steps and reverses the sense of the integration.
    The attitude then pins itself to that heading while the body rates read
    normally, and the vehicle becomes unable to turn on southbound legs.
    """
    negative = np.array([-0.6, 0.8, 0.0, 0.0])
    result = quat_normalise(negative)
    assert result[0] < 0.0
    assert np.allclose(result, negative / np.linalg.norm(negative))

    # The specific case that broke: yaw = 180 degrees has w = 0 exactly.
    south = euler_to_quat(0.0, 0.0, np.pi)
    assert abs(south[0]) < 1e-9
    for _ in range(50):
        south = quat_normalise(south)
    assert quat_to_euler(south)[2] == pytest.approx(np.pi, abs=1e-9)


def test_integration_through_the_sign_boundary_is_continuous():
    """Integrating a constant yaw rate through 180 degrees must not stall."""
    q = euler_to_quat(0.0, 0.0, np.radians(170.0))
    omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s in yaw
    dt = 0.002
    for _ in range(200):  # 0.4 s, sweeping about 23 degrees through the boundary
        q = quat_normalise(q + 0.5 * quat_multiply(q, np.concatenate(([0.0], omega))) * dt)
    expected = wrap_pi(np.radians(170.0) + 1.0 * 0.4)
    assert wrap_pi(quat_to_euler(q)[2] - expected) == pytest.approx(0.0, abs=2e-3)


def test_cross3_matches_numpy():
    for _ in range(200):
        a, b = RNG.normal(size=3), RNG.normal(size=3)
        assert np.allclose(cross3(a, b), np.cross(a, b), atol=1e-15)


def test_conjugate_inverts():
    for q in random_quaternions(100):
        product = quat_multiply(q, quat_conjugate(q))
        assert np.allclose(product, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


class TestAttitudeFromThrust:
    """The differential-flatness step both trackers share."""

    def test_hover_is_level(self):
        thrust = np.array([0.0, 0.0, -GRAVITY])
        for yaw in (-np.pi, -1.0, 0.0, 1.0, np.pi - 1e-9):
            q = attitude_from_thrust_and_yaw(thrust, yaw)
            euler = quat_to_euler(q)
            assert euler[0] == pytest.approx(0.0, abs=1e-9)
            assert euler[1] == pytest.approx(0.0, abs=1e-9)
            assert wrap_pi(euler[2] - yaw) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("yaw", [0.0, np.pi / 2, np.pi, -np.pi / 2, 2.3])
    def test_body_z_aligns_with_thrust(self, yaw):
        """-z_body must point along the demanded thrust, at every heading."""
        for demand in ([2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [1.5, 2.5, -1.0]):
            thrust = np.asarray(demand, dtype=float) - np.array([0.0, 0.0, GRAVITY])
            q = attitude_from_thrust_and_yaw(thrust, yaw)
            body_z = quat_to_rotation_matrix(q)[:, 2]
            expected = thrust / np.linalg.norm(thrust)
            assert np.allclose(-body_z, expected, atol=1e-12)

    def test_yaw_is_respected(self):
        thrust = np.array([2.0, 0.0, -GRAVITY])
        for yaw in (-2.0, 0.0, 1.2, 3.0):
            q = attitude_from_thrust_and_yaw(thrust, yaw)
            assert wrap_pi(quat_to_euler(q)[2] - yaw) == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_thrust_gives_level_attitude(self):
        q = attitude_from_thrust_and_yaw(np.zeros(3), 0.7)
        euler = quat_to_euler(q)
        assert euler[0] == pytest.approx(0.0, abs=1e-9)
        assert euler[1] == pytest.approx(0.0, abs=1e-9)
        assert euler[2] == pytest.approx(0.7, abs=1e-9)


def test_quat_to_euler_clamps_at_gimbal_lock():
    """A diverged controller can demand 90 degrees of pitch; it must not NaN."""
    q = euler_to_quat(0.0, np.pi / 2, 0.0)
    euler = quat_to_euler(q)
    assert np.all(np.isfinite(euler))
    assert euler[1] == pytest.approx(np.pi / 2, abs=1e-6)
