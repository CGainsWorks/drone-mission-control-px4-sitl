"""Closed-loop simulation executive.

This is the rig every result in the repository comes out of. It runs one mission
end to end and records everything, at four rates that mirror a real vehicle:

============  ========  ==================================================
subsystem     rate      why
============  ========  ==================================================
plant         500 Hz    rigid-body integration and the attitude/rate loop
IMU           200 Hz    typical MEMS output rate feeding the estimator
tracker       50 Hz     position control, the rate PX4 runs offboard at
GNSS          5 Hz      typical RTK receiver output
============  ========  ==================================================

Rate separation is not decoration. Running the tracker at the plant rate would
hide exactly the delay that an MPC's preview is supposed to overcome, and would
flatter both controllers.

Determinism
-----------

Given the same :class:`ScenarioConfig`, a run reproduces bit for bit. Wind, IMU
noise and GNSS noise each own a generator seeded from the scenario seed and a
fixed per-stream offset, so changing the controller cannot change the
disturbance sequence. That is what makes the MPC-versus-PID comparison *paired*:
the two controllers fly through the same gust at the same instant, and the
difference between them is the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .control import make_tracker
from .control.interface import TrajectoryTracker
from .estimation import EkfParams, GnssInsEstimator
from .logs import FlightLog
from .mission.schema import Mission
from .mission.state_machine import MissionExecutive, MissionState
from .safety import FailsafeMatrix, Geofence, SystemStatus
from .sim.battery import BatteryModel, BatteryParams
from .sim.inner_loop import InnerLoop, InnerLoopGains
from .sim.quadrotor import QuadrotorParams, QuadrotorPlant
from .sim.sensors import GnssParams, GnssReceiver, ImuParams, ImuSensor
from .sim.wind import WindField, WindParams
from .types import AttitudeSetpoint, VehicleState

#: Per-stream seed offsets. Fixed constants rather than sequential draws so that
#: adding a new noise source later cannot shift the existing ones.
SEED_WIND = 1009
SEED_IMU = 2003
SEED_GNSS = 3001


@dataclass
class ScenarioConfig:
    """Everything that defines one run.

    Two runs differing only in ``tracker`` are a matched pair: same wind, same
    sensor noise, same faults, same mission.
    """

    mission: Mission
    tracker: str = "pid"
    seed: int = 0
    label: str = ""

    wind: WindParams = field(default_factory=WindParams)
    imu: ImuParams = field(default_factory=ImuParams)
    gnss: GnssParams = field(default_factory=GnssParams)
    battery: BatteryParams = field(default_factory=BatteryParams)
    vehicle: QuadrotorParams = field(default_factory=QuadrotorParams)
    inner_loop: InnerLoopGains = field(default_factory=InnerLoopGains)
    ekf: EkfParams = field(default_factory=EkfParams)

    sim_dt: float = 0.002
    control_rate_hz: float = 50.0
    imu_rate_hz: float = 200.0
    #: Hard cap on run length. A diverged controller must not run forever.
    max_duration_s: float = 600.0

    #: ``(start_s, end_s)`` windows during which telemetry is silent.
    link_dropouts: tuple = ()
    #: ``(start_s, end_s)`` windows during which RTK corrections stop arriving.
    rtk_dropouts: tuple = ()
    #: Multiplier applied to the battery's starting charge, for failsafe tests.
    initial_battery: float = 1.0
    #: Speeds the battery drain up so a low-battery failsafe can be reached
    #: inside a short mission without waiting for a real discharge.
    battery_drain_scale: float = 1.0

    #: Transport delay between the tracker and the attitude loop, s. Zero for
    #: the reference comparison. Non-zero for the PX4 surrogate used to tune
    #: gains for SITL, where MAVLink, MAVSDK and PX4's own estimator put a
    #: measured delay between a command and its effect (see docs/SITL.md).
    command_delay_s: float = 0.0

    #: Abort the run if position error exceeds this, m. A controller that has
    #: lost the vehicle should stop the run rather than fill the log.
    divergence_limit_m: float = 60.0

    #: Feed the controller ground truth instead of the estimate. Never used for
    #: a headline result — it is not a flyable configuration — but it is the
    #: only way to attribute a tracking error to the controller rather than to
    #: the estimator, so the comparison reports both.
    use_truth_state: bool = False

    def describe(self) -> dict:
        return {
            "label": self.label or f"{self.mission.name}-{self.tracker}",
            "mission": self.mission.name,
            "tracker": self.tracker,
            "seed": self.seed,
            "wind_mean_ms": list(self.wind.mean_ned),
            "wind_gust_sigma_ms": self.wind.gust_sigma,
            "gnss_rate_hz": self.gnss.rate_hz,
            "gnss_latency_s": self.gnss.latency_s,
            "control_rate_hz": self.control_rate_hz,
            "link_dropouts": [list(window) for window in self.link_dropouts],
            "rtk_dropouts": [list(window) for window in self.rtk_dropouts],
            "initial_battery": self.initial_battery,
            "battery_drain_scale": self.battery_drain_scale,
        }


@dataclass
class RunResult:
    """Outcome of one scenario."""

    config: ScenarioConfig
    log: FlightLog
    completed: bool
    diverged: bool
    final_state: str
    duration_s: float
    failsafe_events: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    tracker_description: dict = field(default_factory=dict)


def _in_window(time_s: float, windows) -> bool:
    return any(start <= time_s < end for start, end in windows)


class SimulationRunner:
    """Runs one :class:`ScenarioConfig` to completion."""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        vehicle = config.vehicle

        self.plant = QuadrotorPlant(vehicle)
        self.inner_loop = InnerLoop(vehicle, config.inner_loop)
        self.wind = WindField(config.wind, seed=config.seed + SEED_WIND)
        self.imu = ImuSensor(config.imu, seed=config.seed + SEED_IMU)
        self.gnss = GnssReceiver(config.gnss, seed=config.seed + SEED_GNSS)
        self.estimator = GnssInsEstimator(
            config.ekf, fusion_delay_s=max(config.gnss.latency_s + 0.02, 0.05)
        )

        battery_params = BatteryParams(**vars(config.battery))
        battery_params.initial_remaining = config.initial_battery
        self.battery = BatteryModel(battery_params)

        self.tracker: TrajectoryTracker = make_tracker(config.tracker, vehicle)
        self.executive = MissionExecutive(config.mission)
        self.geofence = Geofence.from_spec(config.mission.geofence)
        self.failsafe = FailsafeMatrix(spec=config.mission.failsafe)

        self.log = FlightLog()
        self._link_silence_s = 0.0
        self._attitude_setpoint = AttitudeSetpoint(thrust=vehicle.hover_thrust)
        # Commands in flight to the attitude loop, oldest first, when a
        # command delay is configured: (time it takes effect, setpoint).
        self._command_queue = []
        self._applied_setpoint = self._attitude_setpoint
        self._last_queued = None

    # ------------------------------------------------------------------
    def _system_status(self, time_s: float, estimate: VehicleState,
                       geofence_status) -> SystemStatus:
        status = self.estimator.status
        battery = self.battery.state()
        return SystemStatus(
            time_s=time_s,
            battery_remaining=battery.remaining,
            battery_voltage_v=battery.voltage_v,
            link_silence_s=self._link_silence_s,
            link_connected=self._link_silence_s < 0.5,
            gnss_usable=status.fix_type.is_usable,
            gnss_is_rtk=status.fix_type.is_rtk,
            time_since_gnss_update_s=status.time_since_update_s,
            position_uncertainty_m=status.horizontal_position_std,
            estimator_healthy=not self.estimator.position_uncertain,
            geofence_breached=geofence_status.breached,
            geofence_predicted_breach=geofence_status.predicted_breach,
            geofence_margin_m=geofence_status.margin_m,
            in_air=self.executive.in_air,
            armed=self.executive.state != MissionState.IDLE,
        )

    def run(self) -> RunResult:
        config = self.config
        dt = config.sim_dt
        control_interval = 1.0 / config.control_rate_hz
        imu_interval = 1.0 / config.imu_rate_hz

        next_control = 0.0
        next_imu = 0.0
        time_s = 0.0
        diverged = False

        # Sit on the ground until the estimator has a fix, then arm and go. The
        # vehicle does not take off on a dead-reckoned position.
        armed = False
        started = False

        max_steps = int(config.max_duration_s / dt)
        for _ in range(max_steps):
            self.plant.wind_ned = self.wind.step(dt)
            truth = self.plant.vehicle_state(time_s)
            self.gnss.record_truth(time_s, truth.position, truth.velocity)

            # --- sensors ------------------------------------------------
            if time_s + 1e-9 >= next_imu:
                next_imu += imu_interval
                sample = self.imu.sample(
                    time_s, self.plant.specific_force_body(),
                    self.plant.state[10:13], imu_interval
                )
                self.estimator.add_imu(sample)

            self.gnss.set_corrections_available(
                not _in_window(time_s, config.rtk_dropouts), time_s
            )
            fix = self.gnss.sample(time_s)
            if fix is not None:
                self.estimator.add_gnss(fix)

            if _in_window(time_s, config.link_dropouts):
                self._link_silence_s += dt
            else:
                self._link_silence_s = 0.0

            # --- control and mission logic ------------------------------
            if time_s + 1e-9 >= next_control:
                next_control += control_interval
                if config.use_truth_state:
                    estimate = truth.copy()
                elif self.estimator.initialised:
                    estimate = self.estimator.vehicle_state()
                else:
                    estimate = truth.copy()

                # The altitude floor is suspended while deliberately near the
                # ground. Everything else — the polygon and the ceiling — stays
                # armed for the whole flight.
                enforce_floor = self.executive.state not in (
                    MissionState.IDLE,
                    MissionState.ARMED,
                    MissionState.TAKEOFF,
                    MissionState.LANDING,
                )
                geofence_status = self.geofence.check(
                    estimate.position, estimate.velocity, enforce_floor=enforce_floor
                )
                status = self._system_status(time_s, estimate, geofence_status)
                action = self.failsafe.update(status)

                if not armed and self.estimator.initialised:
                    self.executive.arm(time_s, estimate)
                    armed = True
                elif armed and not started:
                    self.executive.start(time_s, estimate)
                    self.tracker.reset(estimate)
                    started = True

                reference = self.executive.update(time_s, estimate, action)

                if self.executive.state == MissionState.TERMINATED:
                    self._attitude_setpoint = AttitudeSetpoint(time_s=time_s, thrust=0.0)
                elif self.executive.state in (MissionState.IDLE, MissionState.DISARMED):
                    self._attitude_setpoint = AttitudeSetpoint(time_s=time_s, thrust=0.0)
                elif self.executive.state == MissionState.ARMED:
                    # Idle at just under hover so the vehicle stays put but the
                    # motors are spinning, as a real armed aircraft does.
                    self._attitude_setpoint = AttitudeSetpoint(
                        time_s=time_s, thrust=0.35 * config.vehicle.hover_thrust
                    )
                else:
                    self._attitude_setpoint = self.tracker.update(
                        estimate, reference, time_s, control_interval
                    )

                setpoint = reference.sample(time_s)
                self.log.append(
                    time_s=time_s,
                    truth=truth,
                    estimate=estimate,
                    setpoint=setpoint,
                    attitude_setpoint=self._attitude_setpoint,
                    diagnostics=self.tracker.diagnostics,
                    mission_state=self.executive.state.value,
                    failsafe_action=int(action),
                    geofence=geofence_status,
                    battery=self.battery.state(),
                    estimator_status=self.estimator.status,
                    link_silence_s=self._link_silence_s,
                    mission_progress=self.executive.mission_progress,
                )

                error = float(np.linalg.norm(truth.position - setpoint.position))
                if error > config.divergence_limit_m:
                    diverged = True
                    break

                if self.executive.state.is_finished:
                    break

            # --- plant --------------------------------------------------
            if config.command_delay_s > 0.0:
                if self._attitude_setpoint is not self._last_queued:
                    self._last_queued = self._attitude_setpoint
                    self._command_queue.append(
                        (time_s + config.command_delay_s, self._attitude_setpoint))
                while self._command_queue and self._command_queue[0][0] <= time_s + 1e-9:
                    self._applied_setpoint = self._command_queue.pop(0)[1]
            else:
                self._applied_setpoint = self._attitude_setpoint
            command = self.inner_loop.update(
                self.plant.state[6:10], self.plant.state[10:13], self._applied_setpoint, dt
            )
            self.plant.step(command, dt)
            self.battery.step(
                self._attitude_setpoint.thrust,
                config.vehicle.hover_thrust,
                dt * config.battery_drain_scale,
            )
            time_s += dt

        self.log.finalise()
        return RunResult(
            config=config,
            log=self.log,
            completed=self.executive.state == MissionState.DISARMED and not diverged,
            diverged=diverged,
            final_state=self.executive.state.value,
            duration_s=time_s,
            failsafe_events=[
                {
                    "time_s": round(event.time_s, 3),
                    "rule": event.rule,
                    "action": event.action.label,
                    "detail": event.detail,
                }
                for event in self.failsafe.events
            ],
            transitions=[
                {"time_s": round(t, 3), "from": a, "to": b, "reason": r}
                for t, a, b, r in self.executive.transitions
            ],
            tracker_description=self.tracker.describe(),
        )


def run_scenario(config: ScenarioConfig) -> RunResult:
    """Convenience wrapper: build a runner and run it."""
    return SimulationRunner(config).run()
