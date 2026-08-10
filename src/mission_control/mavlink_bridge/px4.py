"""PX4 backend over MAVSDK.

Flies the same mission-control stack against a real PX4 autopilot — SITL or, in
principle, hardware. MAVSDK is imported lazily and is **not** a dependency of
anything else in this repository, so the comparison and the whole test suite run
without it.

What this is and is not
-----------------------

It is a working offboard-control bridge: it connects, waits for a position fix
and a healthy estimator, arms, engages offboard mode, streams attitude setpoints
from whichever tracker is running, and lands. ``scripts/run_px4_sitl.py`` drives
it, and ``sitl/`` brings up PX4 with Gazebo.

It is **not** where the headline numbers come from, and the README says so. The
MPC-versus-PID result is generated against the reference simulation because that
runs deterministically in seconds on any machine, and a result nobody can
reproduce is not a result. This backend is what demonstrates the architecture is
real: the mission layer, the trackers and the safety layer are unchanged between
the two, and only this file knows the difference.

Things PX4 will do that catch people out, and which are handled here:

* **Offboard mode will not engage** until setpoints are already streaming. The
  implementation primes the stream before requesting the mode change.
* **Offboard mode drops out** if setpoints stop for more than about half a
  second. Callers get a clear error rather than a vehicle silently reverting to
  hold.
* **Thrust sign.** MAVSDK's attitude thrust is positive-up in ``[0, 1]``, while
  the body-z axis points down. The conversion happens here, once.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..attitude import quat_to_euler
from ..types import AttitudeSetpoint, BatteryState, VehicleState
from .backend import BackendError, VehicleBackend

#: PX4 will fall out of offboard mode if setpoints stop for longer than this.
OFFBOARD_TIMEOUT_S = 0.5
#: Number of setpoints to prime the stream with before requesting offboard.
OFFBOARD_PRIME_COUNT = 20


def _require_mavsdk() -> None:
    """Fail with a message that says what to install, not an ImportError."""
    try:
        import mavsdk  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on environment
        raise BackendError(
            "MAVSDK is not installed. It is an optional dependency, needed only "
            "for the PX4 SITL backend: pip install mavsdk. The controller "
            "comparison does not require it."
        ) from error


class Px4SitlBackend(VehicleBackend):
    """Offboard attitude control of a PX4 vehicle via MAVSDK.

    MAVSDK's API is asyncio-based; this wraps it in a private event loop so the
    synchronous control loop above does not have to be rewritten around it. That
    is a deliberate trade: one thread and a little latency, against forcing an
    async colouring onto the estimator, the trackers and the state machine, none
    of which do I/O.
    """

    name = "px4-sitl"

    def __init__(self, system_address: str = "udp://:14540"):
        self.system_address = system_address
        self._drone = None
        self._loop = None
        self._state = VehicleState()
        self._battery = BatteryState()
        self._connected = False
        self._last_telemetry_s = 0.0
        self._offboard_active = False

    # ------------------------------------------------------------------
    def _run(self, coroutine):
        """Run one coroutine to completion on the private loop."""
        return self._loop.run_until_complete(coroutine)

    def connect(self, timeout_s: float = 30.0) -> None:
        import asyncio

        _require_mavsdk()
        from mavsdk import System

        self._loop = asyncio.new_event_loop()
        self._drone = System()
        self._run(self._drone.connect(system_address=self.system_address))

        async def wait_for_connection():
            deadline = time.monotonic() + timeout_s
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    return True
                if time.monotonic() > deadline:
                    return False
            return False

        if not self._run(wait_for_connection()):
            raise BackendError(
                f"no PX4 system on {self.system_address} within {timeout_s:.0f} s. "
                "Is SITL running? See sitl/README.md."
            )

        # A global position and a home position are both required before PX4
        # will arm. Waiting here turns an inscrutable arming rejection into a
        # clear timeout with a reason.
        async def wait_for_position():
            deadline = time.monotonic() + timeout_s
            async for health in self._drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    return True
                if time.monotonic() > deadline:
                    return False
            return False

        if not self._run(wait_for_position()):
            raise BackendError(
                "PX4 connected but never reported a usable global position. "
                "In SITL this usually means the simulated GPS has not locked yet."
            )

        self._connected = True
        self._last_telemetry_s = time.monotonic()

    def close(self) -> None:
        if self._loop is None:
            return
        try:
            if self._offboard_active:
                self.stop_offboard()
        except BackendError:
            pass
        self._loop.close()
        self._loop = None
        self._drone = None
        self._connected = False

    # ------------------------------------------------------------------
    def arm(self) -> None:
        self._require_connection()
        try:
            self._run(self._drone.action.arm())
        except Exception as error:
            raise BackendError(f"PX4 refused to arm: {error}") from error

    def disarm(self) -> None:
        self._require_connection()
        try:
            self._run(self._drone.action.disarm())
        except Exception as error:
            raise BackendError(f"PX4 refused to disarm: {error}") from error

    def start_offboard(self) -> None:
        self._require_connection()
        from mavsdk.offboard import Attitude, OffboardError

        # Prime the stream. PX4 rejects the mode change outright if no setpoint
        # has arrived recently, and the resulting error message says nothing
        # about why.
        level = Attitude(0.0, 0.0, 0.0, 0.0)
        for _ in range(OFFBOARD_PRIME_COUNT):
            self._run(self._drone.offboard.set_attitude(level))
            time.sleep(0.02)

        try:
            self._run(self._drone.offboard.start())
        except OffboardError as error:
            raise BackendError(
                f"PX4 refused offboard mode: {error._result.result}. "
                "The usual cause is a setpoint stream that stalled before the "
                "mode change."
            ) from error
        self._offboard_active = True

    def stop_offboard(self) -> None:
        if not self._offboard_active:
            return
        from mavsdk.offboard import OffboardError

        try:
            self._run(self._drone.offboard.stop())
        except OffboardError as error:
            raise BackendError(f"could not leave offboard mode: {error}") from error
        finally:
            self._offboard_active = False

    def send_attitude_setpoint(self, setpoint: AttitudeSetpoint) -> None:
        self._require_connection()
        from mavsdk.offboard import Attitude

        roll, pitch, yaw = np.degrees(quat_to_euler(setpoint.orientation))
        # MAVSDK takes normalised thrust positive-up, which is what the tracker
        # already produces; the body-z sign convention is handled inside PX4.
        thrust = float(np.clip(setpoint.thrust, 0.0, 1.0))
        self._run(
            self._drone.offboard.set_attitude(
                Attitude(float(roll), float(pitch), float(yaw), thrust)
            )
        )

    def land(self) -> None:
        """Hand the landing back to PX4's own land mode.

        Deliberate: PX4's land detector uses information this layer does not
        have — thrust, vibration and landing-gear state — and is better at
        deciding when the vehicle is actually down.
        """
        self._require_connection()
        self.stop_offboard()
        self._run(self._drone.action.land())

    # ------------------------------------------------------------------
    def poll(self) -> None:
        """Pull one round of telemetry into the cached state.

        MAVSDK exposes telemetry as async generators. Rather than hold a
        subscription open per field, this takes one sample of each per call,
        which suits a synchronous control loop running at a fixed rate.
        """
        self._require_connection()

        async def first(generator):
            async for item in generator:
                return item
            return None

        position = self._run(first(self._drone.telemetry.position_velocity_ned()))
        attitude = self._run(first(self._drone.telemetry.attitude_quaternion()))
        rates = self._run(first(self._drone.telemetry.attitude_angular_velocity_body()))
        battery = self._run(first(self._drone.telemetry.battery()))

        if position is not None:
            self._state.position = np.array(
                [position.position.north_m, position.position.east_m, position.position.down_m]
            )
            self._state.velocity = np.array(
                [position.velocity.north_m_s, position.velocity.east_m_s,
                 position.velocity.down_m_s]
            )
        if attitude is not None:
            self._state.orientation = np.array([attitude.w, attitude.x, attitude.y, attitude.z])
        if rates is not None:
            self._state.angular_velocity = np.array(
                [rates.roll_rad_s, rates.pitch_rad_s, rates.yaw_rad_s]
            )
        if battery is not None:
            self._battery = BatteryState(
                voltage_v=battery.voltage_v,
                remaining=battery.remaining_percent / 100.0,
                current_a=0.0,
            )
        self._last_telemetry_s = time.monotonic()
        self._state.time_s = time.monotonic()

    def state(self) -> VehicleState:
        return self._state.copy()

    def battery(self) -> BatteryState:
        return self._battery

    def is_connected(self) -> bool:
        return self._connected

    @property
    def link_silence_s(self) -> float:
        if not self._connected:
            return math.inf
        return max(0.0, time.monotonic() - self._last_telemetry_s)

    def describe(self) -> dict:
        return {"backend": self.name, "system_address": self.system_address}

    def _require_connection(self) -> None:
        if not self._connected or self._drone is None:
            raise BackendError("backend is not connected; call connect() first")
