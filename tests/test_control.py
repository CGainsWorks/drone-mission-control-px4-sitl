"""The trajectory trackers, tested against the real plant and inner loop.

Deliberately closed-loop rather than unit-testing the control law in isolation:
a controller that produces the right number and then cannot fly is not working.
These are the properties that must hold for the comparison to mean anything —
notably, that both controllers are given exactly the same interface and the same
information.
"""

from __future__ import annotations

import numpy as np
import pytest

from mission_control.control import (
    TRACKERS,
    MpcTracker,
    MpcWeights,
    PidTracker,
    make_tracker,
)
from mission_control.control.common import acceleration_to_attitude, tilt_limit_acceleration
from mission_control.control.interface import ReferenceProvider
from mission_control.geodesy import GRAVITY
from mission_control.sim import InnerLoop, QuadrotorPlant
from mission_control.types import TrajectorySetpoint, VehicleState

SIM_DT = 0.002
CONTROL_DT = 0.02


class StepReference(ReferenceProvider):
    def __init__(self, target, start=np.array([0.0, 0.0, -15.0]), yaw=0.0, at_s=1.0):
        self.target = np.asarray(target, dtype=float)
        self.start = np.asarray(start, dtype=float)
        self.yaw = yaw
        self.at_s = at_s

    def sample(self, time_s):
        position = self.target if time_s >= self.at_s else self.start
        return TrajectorySetpoint(time_s=time_s, position=position.copy(), yaw=self.yaw)


class RampReference(ReferenceProvider):
    """Constant velocity along north. Requires a steady tilt against drag."""

    def __init__(self, speed=5.0, yaw=0.0, altitude=-15.0):
        self.speed = speed
        self.yaw = yaw
        self.altitude = altitude

    def sample(self, time_s):
        return TrajectorySetpoint(
            time_s=time_s,
            position=np.array([self.speed * time_s, 0.0, self.altitude]),
            velocity=np.array([self.speed, 0.0, 0.0]),
            yaw=self.yaw,
        )


class CircleReference(ReferenceProvider):
    """Constant-speed circle: a sustained, exactly known lateral acceleration."""

    def __init__(self, radius=20.0, speed=5.0, altitude=-15.0):
        self.radius = radius
        self.speed = speed
        self.altitude = altitude

    def sample(self, time_s):
        rate = self.speed / self.radius
        angle = rate * time_s
        return TrajectorySetpoint(
            time_s=time_s,
            position=np.array([self.radius * np.cos(angle), self.radius * np.sin(angle),
                               self.altitude]),
            velocity=np.array([-self.speed * np.sin(angle), self.speed * np.cos(angle), 0.0]),
            acceleration=np.array([-self.speed * rate * np.cos(angle),
                                   -self.speed * rate * np.sin(angle), 0.0]),
            yaw=angle + np.pi / 2,
        )


def closed_loop(tracker, reference, params, duration=14.0, start=None):
    """Fly a tracker against the plant and return the error time series."""
    initial = VehicleState(position=(start if start is not None
                                     else reference.sample(0.0).position.copy()))
    plant = QuadrotorPlant(params, initial)
    loop = InnerLoop(params)
    tracker.reset(plant.vehicle_state())

    setpoint = None
    next_control = 0.0
    time_s = 0.0
    times, errors, solve_times = [], [], []
    for _ in range(int(duration / SIM_DT)):
        if time_s + 1e-9 >= next_control:
            next_control += CONTROL_DT
            setpoint = tracker.update(plant.vehicle_state(time_s), reference, time_s, CONTROL_DT)
            times.append(time_s)
            errors.append(plant.state[0:3] - reference.sample(time_s).position)
            solve_times.append(tracker.diagnostics.solve_time_s)
        command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, SIM_DT)
        plant.step(command, SIM_DT)
        time_s += SIM_DT

    times = np.asarray(times)
    errors = np.asarray(errors)
    tail = times > duration * 0.65
    return {
        "times": times,
        "errors": errors,
        "norm": np.linalg.norm(errors, axis=1),
        "steady": float(np.mean(np.linalg.norm(errors[tail], axis=1))),
        "peak": float(np.max(np.linalg.norm(errors, axis=1))),
        "solve_us": float(np.mean(solve_times)) * 1e6,
        "plant": plant,
    }


class TestSharedConversion:
    """Both trackers must go through the same acceleration-to-attitude step."""

    def test_hover_demand_gives_hover_thrust(self, params):
        setpoint = acceleration_to_attitude(np.zeros(3), 0.0, params)
        assert setpoint.thrust == pytest.approx(params.hover_thrust, rel=1e-9)

    def test_thrust_scales_with_demand(self, params):
        climbing = acceleration_to_attitude(np.array([0.0, 0.0, -3.0]), 0.0, params)
        assert climbing.thrust > params.hover_thrust

    def test_tilt_limit_preserves_vertical_demand(self, params):
        demand = np.array([50.0, 0.0, -2.0])
        limited, saturated = tilt_limit_acceleration(demand, np.radians(40.0))
        assert saturated
        assert limited[2] == pytest.approx(demand[2])
        # The limit is on the angle of the required thrust vector from vertical,
        # so the horizontal component must sit exactly on the cone.
        horizontal = float(np.linalg.norm(limited[0:2]))
        vertical = -(limited[2] - GRAVITY)
        assert horizontal == pytest.approx(vertical * np.tan(np.radians(40.0)), rel=1e-6)

    def test_tilt_limit_leaves_feasible_demands_alone(self):
        demand = np.array([1.0, 0.5, -0.5])
        limited, saturated = tilt_limit_acceleration(demand, np.radians(40.0))
        assert not saturated
        assert np.allclose(limited, demand)


class TestRegistry:
    def test_both_trackers_are_registered(self):
        assert set(TRACKERS) == {"pid", "mpc"}

    def test_unknown_tracker_fails_loudly(self, params):
        with pytest.raises(KeyError, match="unknown tracker"):
            make_tracker("lqr", params)

    def test_descriptions_are_serialisable(self, params):
        import json

        for name in TRACKERS:
            json.dumps(make_tracker(name, params).describe())


@pytest.mark.parametrize("name", ["pid", "mpc"])
class TestBothTrackers:
    """Properties both controllers must satisfy, or the comparison is invalid."""

    def test_holds_a_hover(self, params, name):
        tracker = make_tracker(name, params)
        result = closed_loop(tracker, StepReference([0.0, 0.0, -15.0], at_s=1e9), params,
                             duration=8.0)
        assert result["steady"] < 0.05

    def test_settles_after_a_horizontal_step(self, params, name):
        tracker = make_tracker(name, params)
        result = closed_loop(tracker, StepReference([6.0, 0.0, -15.0]), params)
        assert result["steady"] < 0.10

    def test_settles_after_a_climb_step(self, params, name):
        tracker = make_tracker(name, params)
        result = closed_loop(tracker, StepReference([0.0, 0.0, -21.0]), params)
        assert result["steady"] < 0.10

    def test_tracks_a_constant_velocity_ramp(self, params, name):
        tracker = make_tracker(name, params)
        result = closed_loop(tracker, RampReference(5.0), params)
        assert result["steady"] < 0.30

    @pytest.mark.parametrize("yaw_deg", [0, 90, 180, -90])
    def test_heading_does_not_change_tracking(self, params, name, yaw_deg):
        """Regression for the quaternion sign bug: south must fly like north."""
        tracker = make_tracker(name, params)
        result = closed_loop(tracker, RampReference(5.0, yaw=np.radians(yaw_deg)), params)
        assert result["steady"] < 0.35

    def test_tracks_a_circle(self, params, name):
        tracker = make_tracker(name, params)
        reference = CircleReference(radius=20.0, speed=5.0)
        result = closed_loop(tracker, reference, params, duration=20.0)
        assert result["steady"] < 0.60

    def test_affordable_at_the_control_rate(self, params, name):
        """A controller that cannot run inside its period is not a controller."""
        tracker = make_tracker(name, params)
        result = closed_loop(tracker, CircleReference(), params, duration=8.0)
        assert result["solve_us"] < CONTROL_DT * 1e6 * 0.5

    def test_reset_clears_internal_state(self, params, name):
        tracker = make_tracker(name, params)
        closed_loop(tracker, StepReference([8.0, 0.0, -15.0]), params, duration=6.0)
        tracker.reset(VehicleState(position=np.array([0.0, 0.0, -15.0])))
        first = tracker.update(
            VehicleState(position=np.array([0.0, 0.0, -15.0])),
            StepReference([0.0, 0.0, -15.0], at_s=1e9), 0.0, CONTROL_DT,
        )
        assert first.thrust == pytest.approx(params.hover_thrust, abs=0.02)


class TestMpcSpecifics:
    def test_disturbance_estimate_removes_the_steady_offset(self, params):
        """Without it, the MPC settles where position cost balances input cost.

        The vehicle's drag is unmodelled by design — the MPC's internal model is
        a triple integrator with a lag — so at cruise there is a steady force it
        does not know about. The estimator is what closes that gap.
        """
        weights = MpcWeights()
        weights.disturbance_gain = 0.0
        without = closed_loop(MpcTracker(params, weights), RampReference(5.0), params)

        with_estimate = closed_loop(MpcTracker(params), RampReference(5.0), params)
        assert without["steady"] > 0.3
        assert with_estimate["steady"] < without["steady"] / 5.0

    def test_preview_beats_no_preview(self, params):
        """A one-step horizon is an expensive PID; the horizon is the point."""
        short = MpcWeights(horizon=1, step_s=0.08)
        long = MpcWeights()
        reference = CircleReference(radius=12.0, speed=6.0)
        myopic = closed_loop(MpcTracker(params, short), reference, params, duration=16.0)
        previewing = closed_loop(MpcTracker(params, long), reference, params, duration=16.0)
        assert previewing["steady"] < myopic["steady"]

    def test_respects_its_acceleration_bounds(self, params):
        weights = MpcWeights()
        weights.max_acceleration_xy = 3.0
        tracker = MpcTracker(params, weights)
        closed_loop(tracker, StepReference([40.0, 0.0, -15.0]), params, duration=6.0)
        assert np.max(np.abs(tracker.diagnostics.acceleration_demand[0:2])) <= 3.0 + 1e-6

    def test_predicted_trajectory_starts_at_the_vehicle(self, params):
        tracker = MpcTracker(params)
        state = VehicleState(position=np.array([1.0, 2.0, -15.0]))
        tracker.reset(state)
        tracker.update(state, StepReference([10.0, 2.0, -15.0], at_s=0.0), 0.0, CONTROL_DT)
        predicted = tracker.predicted_trajectory(state)
        assert predicted.shape == (tracker.weights.horizon, 3)
        assert np.linalg.norm(predicted[0] - state.position) < 1.0
        # And it should head towards the target.
        assert predicted[-1][0] > predicted[0][0]


class TestPidSpecifics:
    def test_integrator_does_not_wind_up_against_saturation(self, params):
        """A long saturated run must not leave an overshoot behind it."""
        tracker = PidTracker(params)
        result = closed_loop(tracker, StepReference([120.0, 0.0, -15.0]), params,
                             duration=10.0)
        assert np.all(np.abs(tracker._integral) <= tracker.gains.integral_limit + 1e-9)
        assert np.all(np.isfinite(result["norm"]))

    def test_velocity_setpoint_is_speed_limited(self, params):
        tracker = PidTracker(params)
        state = VehicleState(position=np.zeros(3))
        tracker.reset(state)
        tracker.update(state, StepReference([1000.0, 0.0, 0.0], at_s=0.0), 0.0, CONTROL_DT)
        horizontal = np.linalg.norm(tracker.diagnostics.velocity_setpoint[0:2])
        assert horizontal <= tracker.gains.max_velocity_xy + 1e-9
