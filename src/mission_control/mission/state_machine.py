"""Mission state machine.

Sequences the flight phases and owns the one thing the trackers consume: the
active reference. Everything else — which waypoint we are on, how far through
the mission we are, why we stopped flying it — is derived state exposed for
logging.

The important property is that **a failsafe action can arrive at any instant and
is always honoured**. There is no phase that ignores it and no point in a
transition where it is dropped. Failsafe transitions replan from the vehicle's
current position, because that is the only position it is actually at.

The matrix latches, so once a failsafe has taken the aircraft the mission is
over. Resuming a survey after a low-battery return is not a decision a state
machine should be making on its own.
"""

from __future__ import annotations

import enum

import numpy as np

from ..control.interface import ReferenceProvider
from ..safety.failsafe import FailsafeAction
from ..types import TrajectorySetpoint, VehicleState
from .builder import build_hold, build_land, build_mission_path, build_rtl, build_takeoff
from .schema import Mission


class MissionState(enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    TAKEOFF = "takeoff"
    MISSION = "mission"
    HOLD = "hold"
    RTL = "rtl"
    LANDING = "landing"
    DISARMED = "disarmed"
    TERMINATED = "terminated"

    @property
    def is_flying(self) -> bool:
        return self in (
            MissionState.TAKEOFF,
            MissionState.MISSION,
            MissionState.HOLD,
            MissionState.RTL,
            MissionState.LANDING,
        )

    @property
    def is_finished(self) -> bool:
        return self in (MissionState.DISARMED, MissionState.TERMINATED)


class _HoldReference(ReferenceProvider):
    """Static reference at a fixed point. Used before takeoff and after landing."""

    def __init__(self, position, yaw: float = 0.0):
        self.position = np.asarray(position, dtype=float).copy()
        self.yaw = float(yaw)

    def sample(self, time_s: float) -> TrajectorySetpoint:
        return TrajectorySetpoint(time_s=time_s, position=self.position.copy(), yaw=self.yaw)


class MissionExecutive:
    """Runs a mission, phase by phase."""

    #: Vertical tolerance for calling a climb or descent complete, m.
    ALTITUDE_TOLERANCE = 0.6

    def __init__(self, mission: Mission):
        self.mission = mission
        self.state = MissionState.IDLE
        self.reference = _HoldReference(np.zeros(3))
        self.failsafe_action = FailsafeAction.NONE
        self.transitions = []  # (time_s, from, to, reason)

        self._phase_started_s = 0.0
        self._waypoint_lengths = []
        self._mission_trajectory = None
        self._land_requested = False
        self._terminal_items = [
            item.type for item in mission.items if item.type in ("rtl", "land")
        ]

    # ------------------------------------------------------------------
    @property
    def in_air(self) -> bool:
        return self.state.is_flying

    @property
    def mission_progress(self) -> float:
        """Fraction of the survey trajectory completed, ``0.0`` to ``1.0``."""
        if self._mission_trajectory is None or self.state != MissionState.MISSION:
            return 1.0 if self.state.is_finished else 0.0
        elapsed = self._elapsed
        duration = max(self._mission_trajectory.duration, 1e-6)
        return float(np.clip(elapsed / duration, 0.0, 1.0))

    @property
    def active_waypoint(self) -> int:
        """Index of the survey waypoint being flown towards, or ``-1``.

        Resolved by arc length along the path rather than by proximity: two
        passes of a lawnmower survey come within a lane spacing of each other,
        so "nearest waypoint" would flip between them.
        """
        if self._mission_trajectory is None or not self._waypoint_lengths:
            return -1
        flown = self._flown_length
        for index, length in enumerate(self._waypoint_lengths):
            if length >= flown - 1e-6:
                return index
        return len(self._waypoint_lengths) - 1

    @property
    def _elapsed(self) -> float:
        return max(0.0, self._time - self._phase_started_s)

    @property
    def _flown_length(self) -> float:
        if self._mission_trajectory is None:
            return 0.0
        return self._mission_trajectory._state_at(
            float(np.clip(self._elapsed, 0.0, self._mission_trajectory.duration))
        )[0]

    # ------------------------------------------------------------------
    def _transition(self, new_state: MissionState, time_s: float, reason: str) -> None:
        if new_state == self.state:
            return
        self.transitions.append((time_s, self.state.value, new_state.value, reason))
        self.state = new_state
        self._phase_started_s = time_s

    def arm(self, time_s: float, state: VehicleState) -> None:
        """Accept an arm command. Refused unless idle and on the ground."""
        if self.state != MissionState.IDLE:
            return
        self._time = time_s
        self.reference = _HoldReference(state.position, state.yaw)
        self._transition(MissionState.ARMED, time_s, "arm command")

    def start(self, time_s: float, state: VehicleState) -> None:
        """Begin the mission: climb to the takeoff altitude."""
        if self.state != MissionState.ARMED:
            return
        self._time = time_s
        altitude = self.mission.takeoff_altitude
        self.reference = build_takeoff(state.position, altitude, self.mission.defaults,
                                       start_time_s=time_s)
        self._transition(MissionState.TAKEOFF, time_s, "mission start")

    # ------------------------------------------------------------------
    def update(self, time_s: float, state: VehicleState,
               failsafe_action: FailsafeAction = FailsafeAction.NONE) -> ReferenceProvider:
        """Advance the state machine and return the reference now in force."""
        self._time = time_s

        if failsafe_action > self.failsafe_action:
            self._apply_failsafe(time_s, state, failsafe_action)
        elif self.state.is_flying:
            self._advance_phase(time_s, state)

        return self.reference

    def _apply_failsafe(self, time_s: float, state: VehicleState,
                        action: FailsafeAction) -> None:
        self.failsafe_action = action
        reason = f"failsafe {action.label}"

        if action == FailsafeAction.TERMINATE:
            self.reference = _HoldReference(state.position, state.yaw)
            self._transition(MissionState.TERMINATED, time_s, reason)
            return
        if not self.state.is_flying:
            # On the ground a failsafe means "do not take off", not "land".
            if action >= FailsafeAction.HOLD and self.state == MissionState.ARMED:
                self._transition(MissionState.IDLE, time_s, reason)
            return

        if action == FailsafeAction.HOLD:
            self.reference = build_hold(state.position, 1e6, self.mission.defaults,
                                        yaw=state.yaw, start_time_s=time_s)
            self._transition(MissionState.HOLD, time_s, reason)
        elif action == FailsafeAction.RTL:
            self.reference = build_rtl(
                state.position,
                self.mission.home_ned,
                self.mission.failsafe.rtl_altitude,
                self.mission.defaults,
                yaw=state.yaw,
                start_time_s=time_s,
            )
            self._transition(MissionState.RTL, time_s, reason)
        elif action == FailsafeAction.LAND:
            self.reference = build_land(state.position, self.mission.defaults,
                                        yaw=state.yaw, start_time_s=time_s)
            self._transition(MissionState.LANDING, time_s, reason)
        elif action == FailsafeAction.WARN:
            pass  # advisory only; the mission continues

    def _advance_phase(self, time_s: float, state: VehicleState) -> None:
        if self.state == MissionState.TAKEOFF:
            target = self.mission.takeoff_altitude
            reached = abs(state.altitude - target) < self.ALTITUDE_TOLERANCE
            if reached and self._elapsed >= self.reference.duration:
                trajectory, lengths = build_mission_path(state.position, self.mission,
                                                         start_time_s=time_s)
                self._mission_trajectory = trajectory
                self._waypoint_lengths = lengths
                self.reference = trajectory
                self._transition(MissionState.MISSION, time_s, "takeoff altitude reached")

        elif self.state == MissionState.MISSION:
            finished = self._elapsed >= self.reference.duration
            at_end = float(np.linalg.norm(state.position - self.reference.end_position))
            if finished and at_end < max(self.mission.defaults.acceptance_radius, 1.0):
                self._begin_terminal_phase(time_s, state)

        elif self.state == MissionState.RTL:
            finished = self._elapsed >= self.reference.duration
            horizontal = float(
                np.linalg.norm(state.position[0:2] - self.mission.home_ned[0:2])
            )
            if finished and horizontal < max(self.mission.defaults.acceptance_radius, 1.5):
                self.reference = build_land(state.position, self.mission.defaults,
                                            yaw=state.yaw, start_time_s=time_s)
                self._transition(MissionState.LANDING, time_s, "arrived home")

        elif self.state == MissionState.LANDING:
            if state.altitude < 0.15 and abs(state.velocity[2]) < 0.3:
                self._transition(MissionState.DISARMED, time_s, "touchdown")

    def _begin_terminal_phase(self, time_s: float, state: VehicleState) -> None:
        """What happens after the last survey waypoint, per the mission file."""
        if "rtl" in self._terminal_items:
            self.reference = build_rtl(
                state.position,
                self.mission.home_ned,
                self.mission.failsafe.rtl_altitude,
                self.mission.defaults,
                yaw=state.yaw,
                start_time_s=time_s,
            )
            self._transition(MissionState.RTL, time_s, "mission complete")
        elif "land" in self._terminal_items:
            self.reference = build_land(state.position, self.mission.defaults,
                                        yaw=state.yaw, start_time_s=time_s)
            self._transition(MissionState.LANDING, time_s, "mission complete")
        else:
            self.reference = build_hold(state.position, 1e6, self.mission.defaults,
                                        yaw=state.yaw, start_time_s=time_s)
            self._transition(MissionState.HOLD, time_s, "mission complete, no terminal item")

    # ------------------------------------------------------------------
    _time = 0.0

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "failsafe_action": self.failsafe_action.label,
            "transitions": [
                {"time_s": round(t, 3), "from": a, "to": b, "reason": r}
                for t, a, b, r in self.transitions
            ],
            "mission_duration_s": (
                round(self._mission_trajectory.duration, 3)
                if self._mission_trajectory is not None else 0.0
            ),
        }
