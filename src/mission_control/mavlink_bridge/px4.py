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
it, and ``sitl/`` brings up PX4 (SIH, or Gazebo for a visual run).

The 84-run headline comparison is generated against the reference simulation
because that runs deterministically on any machine. The SITL campaign
(``scripts/run_sitl_campaign.py``) flies both trackers and the failsafe matrix
through this backend against PX4 SIH and commits the logs, so the comparison is
also checked against a real autopilot. The mission layer, the trackers and the
safety layer are unchanged between the two; only this file knows the
difference.

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
#: Rate to request for the streams the tracker closes its loop around, Hz.
TELEMETRY_RATE_HZ = 50.0
#: Global position is where the position in the mission frame comes from when
#: the mission declares an origin, so it is the position the tracker closes
#: its loop around and is requested at the control rate. At 10 Hz it was up to
#: 100 ms stale, which at survey speed is 0.4 m of error that is not tracking.
POSITION_RATE_HZ = 50.0
#: Battery state changes on a timescale of minutes.
BATTERY_RATE_HZ = 1.0
#: How long to wait for the first sample of every stream before giving up.
TELEMETRY_FIRST_SAMPLE_TIMEOUT_S = 10.0


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

    def __init__(self, system_address: str = "udpin://0.0.0.0:14540", origin=None):
        """``origin`` is the mission's :class:`GeodeticOrigin`, or None.

        This argument is not optional in any meaningful sense — it decides what
        the coordinates in the mission file *mean*.

        PX4's local NED frame is anchored wherever the EKF happened to
        initialise, which is the launch point. A mission file, by contrast,
        declares its own ``origin:`` in latitude and longitude, and every
        waypoint in it is relative to that. The two frames coincide only when
        the vehicle takes off from the mission origin.

        Anchoring to the launch point instead is silently wrong rather than
        obviously wrong: the vehicle flies a correctly shaped survey around the
        wrong point, offset by however far the launch pad is from the origin.
        Passing the origin here converts PX4's global position into the frame
        the mission is written in, so the survey lands on the structure it is
        meant to inspect.
        """
        import threading

        self.system_address = system_address
        self.origin = origin
        self._drone = None
        self._loop = None
        self._thread = None
        self._telemetry_tasks = []
        self._expected_streams = set()
        self._seen_streams = set()
        self._lock = threading.Lock()
        self._state = VehicleState()
        self._battery = BatteryState()
        self._connected = False
        self._last_telemetry_s = 0.0
        self._offboard_active = False
        self._last_attitude = None
        self.setpoints_failed = 0
        # What the autopilot says about itself, for the failsafe matrix and
        # the flight log: GNSS fix, flight mode, armed and in-air flags.
        self._gnss_fix_type = 0
        self._gnss_satellites = 0
        self._last_gnss_s = 0.0
        self._flight_mode = "unknown"
        self._armed = False
        self._in_air = False

    # ------------------------------------------------------------------
    def _run(self, coroutine, timeout_s: float = 30.0):
        """Run one coroutine on the background loop and wait for the result."""
        import asyncio

        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout_s)

    def _wait(self, coroutine, timeout_s: float) -> bool:
        """Run a wait-for-condition coroutine; False if it never came true."""
        import concurrent.futures

        try:
            return bool(self._run(coroutine, timeout_s=timeout_s + 5.0))
        except concurrent.futures.TimeoutError:
            return False

    def connect(self, timeout_s: float = 30.0) -> None:
        import asyncio
        import threading

        _require_mavsdk()
        from mavsdk import System

        # The loop runs on its own thread for the lifetime of the connection,
        # rather than being spun up per call. Telemetry arrives on it
        # continuously (see _start_telemetry) instead of being fetched, which
        # is the difference between a control loop that closes at 50 Hz and one
        # that closes at 0.5 Hz.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mavsdk-telemetry", daemon=True
        )
        self._thread.start()
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

        # Each wait is bounded twice: by its own deadline, checked when a
        # message arrives, and by the future's timeout, for when none does.
        # The second must be longer than the first. It used to be the 30 s
        # default whatever timeout_s was, so a cold PX4 on a CI runner, whose
        # EKF takes longer than that to report a global position, failed with a
        # bare TimeoutError while the stated limit was 120 s.
        if not self._wait(wait_for_connection(), timeout_s):
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

        if not self._wait(wait_for_position(), timeout_s):
            raise BackendError(
                "PX4 connected but never reported a usable global position. "
                "In SITL this usually means the simulated GPS has not locked yet."
            )

        self._connected = True
        self._last_telemetry_s = time.monotonic()
        self._start_telemetry()

    def _start_telemetry(self) -> None:
        """Subscribe once to each stream and keep the subscriptions open.

        This is the whole reason the loop lives on a thread. MAVSDK exposes
        telemetry as async generators, and it is tempting — it is what this
        file used to do — to open one, take a single item and drop it, once per
        control iteration. That reads like a synchronous getter and behaves
        like nothing of the sort: each fresh subscription waits for the *next*
        message to arrive, so one iteration paid that wait four times over.
        Measured against PX4 SITL it cost 2.0 seconds per iteration.

        A position loop closed at 0.5 Hz around state that is up to two seconds
        stale does not track anything. It oscillates, with a period of about
        ten seconds and an amplitude of tens of metres, and it does so whatever
        the tracker is and whatever its gains are — which is exactly what it
        looked like: PID and MPC failing identically and neither responding to
        tuning. Nothing above this file was wrong.

        Held open, the same streams cost nothing to read and the loop runs at
        the rate it asks for.
        """
        import asyncio

        async def follow(name, subscribe, apply):
            while True:
                try:
                    async for item in subscribe():
                        with self._lock:
                            apply(item)
                            self._seen_streams.add(name)
                            self._last_telemetry_s = time.monotonic()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A stream can drop without the link dropping. Re-subscribe
                    # rather than leaving one field frozen for the rest of the
                    # flight; link loss is the failsafe matrix's job, and it
                    # watches _last_telemetry_s.
                    await asyncio.sleep(0.5)

        def apply_position(sample) -> None:
            self._state.velocity = np.array(
                [sample.velocity.north_m_s, sample.velocity.east_m_s,
                 sample.velocity.down_m_s]
            )
            if self.origin is None:
                self._state.position = np.array(
                    [sample.position.north_m, sample.position.east_m,
                     sample.position.down_m]
                )

        def apply_global(sample) -> None:
            if self.origin is not None:
                self._state.position = self.origin.to_ned(
                    sample.latitude_deg, sample.longitude_deg,
                    sample.absolute_altitude_m,
                )

        def apply_attitude(sample) -> None:
            self._state.orientation = np.array(
                [sample.w, sample.x, sample.y, sample.z]
            )

        def apply_rates(sample) -> None:
            self._state.angular_velocity = np.array(
                [sample.roll_rad_s, sample.pitch_rad_s, sample.yaw_rad_s]
            )

        def apply_gps(sample) -> None:
            # MAVSDK's FixType: NO_GPS, NO_FIX, FIX_2D, FIX_3D, FIX_DGPS,
            # RTK_FLOAT, RTK_FIXED, in that order, which is MAVLink's
            # GPS_FIX_TYPE numbering. Kept as the integer.
            self._gnss_fix_type = int(sample.fix_type.value)
            self._gnss_satellites = int(sample.num_satellites)
            if self._gnss_fix_type >= 3:
                self._last_gnss_s = time.monotonic()

        def apply_mode(sample) -> None:
            self._flight_mode = str(sample.name).lower()

        def apply_armed(sample) -> None:
            self._armed = bool(sample)

        def apply_in_air(sample) -> None:
            self._in_air = bool(sample)

        def apply_battery(sample) -> None:
            self._battery = BatteryState(
                voltage_v=sample.voltage_v,
                remaining=sample.remaining_percent / 100.0,
                current_a=0.0,
            )

        async def launch():
            telemetry = self._drone.telemetry
            # Ask for rates that suit a control loop. PX4's defaults are set
            # for a ground station watching a map, not for closing a loop.
            for setter, rate in (
                (telemetry.set_rate_position_velocity_ned, TELEMETRY_RATE_HZ),
                (telemetry.set_rate_attitude_quaternion, TELEMETRY_RATE_HZ),
                (telemetry.set_rate_position, POSITION_RATE_HZ),
                (telemetry.set_rate_battery, BATTERY_RATE_HZ),
            ):
                try:
                    await setter(rate)
                except Exception:
                    # Not fatal: the stream still arrives, just at PX4's
                    # default rate. Worth continuing rather than refusing to
                    # fly over a rate request.
                    pass

            streams = [
                ("position_velocity_ned", telemetry.position_velocity_ned, apply_position),
                ("attitude_quaternion", telemetry.attitude_quaternion, apply_attitude),
                ("angular_velocity", telemetry.attitude_angular_velocity_body, apply_rates),
                ("battery", telemetry.battery, apply_battery),
                ("gps_info", telemetry.gps_info, apply_gps),
                ("flight_mode", telemetry.flight_mode, apply_mode),
                ("armed", telemetry.armed, apply_armed),
                ("in_air", telemetry.in_air, apply_in_air),
            ]
            if self.origin is not None:
                streams.append(("position", telemetry.position, apply_global))
            self._expected_streams = {name for name, _, _ in streams}
            return [
                asyncio.ensure_future(follow(name, sub, app))
                for name, sub, app in streams
            ]

        self._telemetry_tasks = self._run(launch())

        # Do not hand a caller a state full of zeros. Waiting for the first
        # sample of every stream here means `state()` is meaningful from the
        # first call, and a stream that never arrives is caught before arming
        # rather than halfway through a climb.
        deadline = time.monotonic() + TELEMETRY_FIRST_SAMPLE_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._lock:
                missing = self._expected_streams - self._seen_streams
            if not missing:
                return
            time.sleep(0.02)
        raise BackendError(
            "PX4 never sent these telemetry streams: "
            + ", ".join(sorted(missing))
            + ". The link is up but the vehicle is not reporting state."
        )

    def close(self) -> None:
        if self._loop is None:
            return
        try:
            if self._offboard_active:
                self.stop_offboard()
        except BackendError:
            pass
        # Cancel *and wait*. Stopping the loop the moment cancel() is called
        # leaves the tasks mid-teardown and asyncio complains on garbage
        # collection; the cancellation has to be processed by the loop that
        # owns them, which means awaiting it on that loop.
        import asyncio

        async def shutdown(tasks):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._telemetry_tasks:
            try:
                self._run(shutdown(self._telemetry_tasks), timeout_s=5.0)
            except Exception:
                pass
        self._telemetry_tasks = []
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._loop.close()
        self._loop = None
        self._drone = None
        self._connected = False

    # ------------------------------------------------------------------
    def arm(self, timeout_s: float = 20.0) -> None:
        """Arm, retrying while PX4 is still refusing for a transient reason.

        A global position and a home position are not the whole of PX4's
        preflight checks; on a cold start the estimator can report both and
        still be refused for a few seconds more (COMMAND_DENIED). A refusal
        that outlasts ``timeout_s`` is a real one and is raised.
        """
        self._require_connection()
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self._run(self._drone.action.arm())
                return
            except Exception as error:
                if time.monotonic() > deadline:
                    raise BackendError(f"PX4 refused to arm: {error}") from error
                time.sleep(1.0)

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
        # Prime with the last command sent, if there was one. On the ground
        # that is nothing, and level at zero thrust is right; when control is
        # being taken back in flight after PX4 fell out of offboard, a
        # zero-thrust setpoint is the last thing the vehicle should receive.
        prime = self._last_attitude or Attitude(0.0, 0.0, 0.0, 0.0)
        for _ in range(OFFBOARD_PRIME_COUNT):
            self._run(self._drone.offboard.set_attitude(prime))
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
        command = Attitude(float(roll), float(pitch), float(yaw), thrust)
        self._last_attitude = command
        try:
            self._run(self._drone.offboard.set_attitude(command), timeout_s=1.0)
        except Exception:
            # MAVSDK reports an error while it considers the system
            # disconnected. A lost setpoint is what a link loss is; it is the
            # failsafe matrix's job to react to that, not this call's to stop
            # the control loop by raising.
            self.setpoints_failed += 1

    def ensure_offboard(self) -> bool:
        """Re-engage offboard mode if PX4 has left it. Returns True if it did.

        PX4 leaves offboard on its own when the setpoint stream stops, which is
        exactly what a link loss does. When the link comes back the mission
        layer has already decided what to do about it (the matrix latches), so
        it takes control back and does that, rather than leaving the vehicle
        in whatever mode PX4 chose as its own fallback.
        """
        with self._lock:
            mode = self._flight_mode
        if mode == "offboard":
            return False
        self._offboard_active = False
        self.start_offboard()
        return True

    def set_param(self, name: str, value) -> None:
        """Set a PX4 parameter over MAVLink. Used to inject faults in SITL."""
        self._require_connection()
        if isinstance(value, int) and not isinstance(value, bool):
            self._run(self._drone.param.set_param_int(name, int(value)), timeout_s=10.0)
        else:
            self._run(self._drone.param.set_param_float(name, float(value)), timeout_s=10.0)

    def land(self) -> None:
        """Hand the landing back to PX4's own land mode.

        Deliberate: PX4's land detector uses information this layer does not
        have — thrust, vibration and landing-gear state — and is better at
        deciding when the vehicle is actually down.
        """
        self._require_connection()
        try:
            self.stop_offboard()
        except BackendError:
            # Leaving offboard first is tidy, not necessary: a land command is
            # accepted from offboard too. Failing to leave it, which happens if
            # the link is down at that moment, must not stop the landing.
            pass
        self._run(self._drone.action.land())

    # ------------------------------------------------------------------
    def poll(self) -> None:
        """Bring the cached state up to date.

        Telemetry arrives continuously on the background loop, so there is
        nothing to fetch: this only stamps the state with the time it is being
        read at. It is kept because the control loop reads as it should —
        poll, then act — and because the backend interface is shared with the
        reference simulation, where a poll does do work.
        """
        self._require_connection()
        with self._lock:
            self._state.time_s = time.monotonic()

    def state(self) -> VehicleState:
        with self._lock:
            return self._state.copy()

    def battery(self) -> BatteryState:
        with self._lock:
            return self._battery

    def is_connected(self) -> bool:
        return self._connected

    def gnss(self) -> dict:
        """Latest fix type and satellite count, and how long GNSS had been bad.

        The age is measured to the last telemetry received, not to now. While
        the link is down nothing is known about the receiver, and treating
        that silence as a GNSS outage made the GNSS rule (land) fire 2.5 s
        into a link loss, ahead of the link-loss rule it should have left to
        decide. Link silence is the link-loss rule's input; this is GNSS's.
        """
        with self._lock:
            if not self._last_gnss_s:
                age = math.inf
            else:
                age = max(0.0, self._last_telemetry_s - self._last_gnss_s)
            return {
                "fix_type": self._gnss_fix_type,
                "satellites": self._gnss_satellites,
                "age_s": age,
            }

    @property
    def flight_mode(self) -> str:
        with self._lock:
            return self._flight_mode

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def in_air(self) -> bool:
        with self._lock:
            return self._in_air

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
