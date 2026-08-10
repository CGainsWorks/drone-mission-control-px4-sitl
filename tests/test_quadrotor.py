"""The plant and its control allocation."""

from __future__ import annotations

import numpy as np
import pytest

from mission_control.attitude import euler_to_quat, quat_to_euler
from mission_control.geodesy import GRAVITY
from mission_control.sim import InnerLoop, Mixer, QuadrotorPlant
from mission_control.types import AttitudeSetpoint, VehicleState

DT = 0.002


def tilt_of(quat) -> float:
    euler = quat_to_euler(quat)
    return float(np.degrees(np.arccos(np.clip(np.cos(euler[0]) * np.cos(euler[1]), -1.0, 1.0))))


class TestMixer:
    def test_hover_allocation_is_even(self, params):
        mixer = Mixer(params)
        thrusts = mixer.allocate(params.weight, np.zeros(3))
        assert np.allclose(thrusts, params.weight / 4.0)
        assert float(np.sum(thrusts)) == pytest.approx(params.weight)

    def test_achieves_a_feasible_wrench_exactly(self, params):
        mixer = Mixer(params)
        for torque in ([0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.05],
                       [0.3, -0.2, 0.03]):
            achieved = mixer.achieved(params.weight, np.array(torque))
            assert achieved[0] == pytest.approx(params.weight, abs=1e-9)
            assert np.allclose(achieved[1:4], torque, atol=1e-9)

    def test_yaw_is_sacrificed_before_roll_and_pitch(self, params):
        """The priority that keeps the vehicle flying.

        Regression for a real failure: an earlier mixer scaled all three torque
        axes together when saturated, so the large yaw demand generated while
        turning a corner scaled roll and pitch down with it. The vehicle could
        not hold the turn and departed.
        """
        mixer = Mixer(params)
        roll, pitch = 0.6, 0.4
        baseline = mixer.achieved(params.weight, np.array([roll, pitch, 0.0]))

        for yaw_demand in (0.5, 2.0, 10.0, 100.0):
            achieved = mixer.achieved(params.weight, np.array([roll, pitch, yaw_demand]))
            assert achieved[1] == pytest.approx(roll, abs=1e-9)
            assert achieved[2] == pytest.approx(pitch, abs=1e-9)
            assert achieved[0] == pytest.approx(baseline[0], abs=1e-9)
            # Yaw gets whatever is left, and never more than asked for.
            assert 0.0 <= achieved[3] <= yaw_demand + 1e-9

    def test_collective_is_sacrificed_before_roll_and_pitch(self, params):
        mixer = Mixer(params)
        torque = np.array([2.0, 0.0, 0.0])
        for collective in (0.0, params.weight, params.max_thrust):
            achieved = mixer.achieved(collective, torque)
            assert achieved[1] == pytest.approx(2.0, abs=1e-9)

    def test_never_exceeds_rotor_limits(self, params):
        mixer = Mixer(params)
        rng = np.random.default_rng(7)
        for _ in range(500):
            collective = rng.uniform(-10.0, 60.0)
            torque = rng.normal(size=3) * 3.0
            thrusts = mixer.allocate(collective, torque)
            assert np.all(thrusts >= -1e-12)
            assert np.all(thrusts <= params.max_rotor_thrust + 1e-12)

    def test_plant_and_mixer_share_one_geometry(self, params):
        """The plant's torque must be the mixer's matrix, not a second opinion."""
        plant = QuadrotorPlant(params)
        mixer = Mixer(params)
        assert np.allclose(plant._torque_matrix, mixer.matrix[1:4, :])


class TestPlant:
    def test_hover_thrust_holds_altitude(self, params):
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -10.0])))
        loop = InnerLoop(params)
        setpoint = AttitudeSetpoint(orientation=euler_to_quat(0, 0, 0),
                                    thrust=params.hover_thrust)
        for _ in range(int(5.0 / DT)):
            command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, DT)
            plant.step(command, DT)
        state = plant.vehicle_state()
        assert state.altitude == pytest.approx(10.0, abs=0.01)
        assert np.linalg.norm(state.velocity) < 0.01

    def test_free_fall_starts_at_g(self, params):
        """From rest, with no thrust, the initial acceleration is exactly g.

        Only briefly: this airframe's vertical drag is not a small correction.
        At 5 m/s it is already about a third of the vehicle's weight, which is
        why the check is over 50 ms rather than the half second that first
        seemed reasonable.
        """
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -100.0])))
        plant.state[13:17] = 0.0
        for _ in range(int(0.05 / DT)):
            plant.step(np.zeros(4), DT)
        assert plant.state[5] == pytest.approx(GRAVITY * 0.05, rel=0.01)

    def test_terminal_velocity_matches_the_drag_model(self, params):
        """Falling long enough, weight must balance drag at the modelled speed."""
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -400.0])))
        plant.state[13:17] = 0.0
        for _ in range(int(20.0 / DT)):
            plant.step(np.zeros(4), DT)

        # m g = c_linear v + c_quadratic v^2, solved for v.
        linear = params.drag_linear[2]
        quadratic = params.drag_quadratic[2]
        expected = (-linear + np.sqrt(linear**2 + 4 * quadratic * params.weight)) / (
            2 * quadratic
        )
        assert plant.state[5] == pytest.approx(expected, rel=0.01)

    def test_does_not_fall_through_the_ground(self, params):
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -0.5])))
        plant.state[13:17] = 0.0
        for _ in range(int(3.0 / DT)):
            plant.step(np.zeros(4), DT)
        assert plant.state[2] <= 1e-9
        assert plant.on_ground

    def test_quaternion_stays_normalised(self, params):
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -20.0])))
        rng = np.random.default_rng(3)
        for _ in range(2000):
            plant.step(rng.uniform(0.0, params.max_rotor_thrust, 4), DT)
            assert np.linalg.norm(plant.state[6:10]) == pytest.approx(1.0, abs=1e-9)

    def test_specific_force_reads_g_in_hover(self, params):
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -10.0])))
        plant.state[13:17] = params.weight / 4.0
        force = plant.specific_force_body()
        # An accelerometer in hover reads +g upward, which is -g on body z.
        assert force[2] == pytest.approx(-GRAVITY, abs=1e-6)
        assert np.allclose(force[0:2], 0.0, atol=1e-9)

    def test_thrust_to_weight_matches_parameters(self, params):
        assert params.thrust_to_weight == pytest.approx(
            4 * params.max_rotor_thrust / (params.mass * GRAVITY)
        )
        assert params.hover_thrust == pytest.approx(1.0 / params.thrust_to_weight)


class TestInnerLoop:
    @pytest.mark.parametrize("roll_deg,pitch_deg", [(15, 0), (0, 15), (-25, 10), (30, -20)])
    def test_reaches_commanded_attitude(self, params, roll_deg, pitch_deg):
        plant = QuadrotorPlant(params, VehicleState(position=np.array([0.0, 0.0, -30.0])))
        loop = InnerLoop(params)
        target = euler_to_quat(np.radians(roll_deg), np.radians(pitch_deg), 0.0)
        setpoint = AttitudeSetpoint(orientation=target, thrust=0.5)
        for _ in range(int(3.0 / DT)):
            command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, DT)
            plant.step(command, DT)
        euler = np.degrees(quat_to_euler(plant.state[6:10]))
        assert euler[0] == pytest.approx(roll_deg, abs=0.5)
        assert euler[1] == pytest.approx(pitch_deg, abs=0.5)

    @pytest.mark.parametrize("yaw_deg", [0, 90, 179, -90, 180])
    def test_holds_tilt_at_any_heading(self, params, yaw_deg):
        """Regression: heading must not affect the ability to tilt.

        The quaternion sign-canonicalisation bug made this fail at exactly
        180 degrees, where the scalar part of the attitude quaternion is zero.
        """
        from mission_control.attitude import attitude_from_thrust_and_yaw

        yaw = np.radians(yaw_deg)
        plant = QuadrotorPlant(
            params,
            VehicleState(position=np.array([0.0, 0.0, -30.0]),
                         orientation=euler_to_quat(0, 0, yaw)),
        )
        loop = InnerLoop(params)
        demand = np.array([5.0, 0.0, 0.0]) - np.array([0.0, 0.0, GRAVITY])
        setpoint = AttitudeSetpoint(
            orientation=attitude_from_thrust_and_yaw(demand, yaw), thrust=0.55
        )
        for _ in range(int(3.0 / DT)):
            command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, DT)
            plant.step(command, DT)

        expected = tilt_of(setpoint.orientation)
        assert tilt_of(plant.state[6:10]) == pytest.approx(expected, abs=1.0)

    def test_thrust_axis_is_corrected_before_heading(self, params):
        """The reduced-attitude blend: a huge yaw error must not stall the tilt."""
        from mission_control.attitude import attitude_from_thrust_and_yaw

        plant = QuadrotorPlant(
            params,
            VehicleState(position=np.array([0.0, 0.0, -30.0]),
                         orientation=euler_to_quat(0, 0, 0.0)),
        )
        loop = InnerLoop(params)
        demand = np.array([5.0, 0.0, 0.0]) - np.array([0.0, 0.0, GRAVITY])
        # Demand the tilt at a heading 170 degrees away from where we are.
        target_yaw = np.radians(170.0)
        setpoint = AttitudeSetpoint(
            orientation=attitude_from_thrust_and_yaw(demand, target_yaw), thrust=0.55
        )
        tilts = []
        for step in range(int(1.5 / DT)):
            command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, DT)
            plant.step(command, DT)
            if step * DT > 1.0:
                tilts.append(tilt_of(plant.state[6:10]))
        # Tilt should be well under way within a second even though the heading
        # is nowhere near settled.
        assert np.mean(tilts) > 10.0
