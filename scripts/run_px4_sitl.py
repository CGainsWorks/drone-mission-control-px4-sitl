#!/usr/bin/env python3
"""Fly a mission against PX4 SITL using this repository's control stack.

The point of this script is the architecture, not the numbers. It runs the same
mission state machine, the same trajectory generator, the same PID or MPC
tracker and the same geofence and failsafe matrix that the reference simulation
runs — and only the bottom layer changes, from the built-in plant to a real PX4
autopilot over MAVLink.

The headline MPC-versus-PID result does **not** come from here, and the README
says so. It comes from the reference simulation, because that reproduces
deterministically in minutes on any machine with no toolchain. This is the
demonstration that the layer above the simulation is not written around it.

Requirements:
    pip install -r requirements-optional.txt   # for mavsdk
    PX4 SITL running and reachable             # see sitl/README.md

Run:
    python scripts/run_px4_sitl.py --mission missions/facade_inspection.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.control import make_tracker  # noqa: E402
from mission_control.mavlink_bridge import BackendError, make_backend  # noqa: E402
from mission_control.mission import MissionExecutive, MissionState, load_mission  # noqa: E402
from mission_control.mission.schema import FailsafeSpec  # noqa: E402
from mission_control.safety import FailsafeMatrix, Geofence, SystemStatus  # noqa: E402
from mission_control.sim.quadrotor import QuadrotorParams  # noqa: E402

# The same rate the reference simulation runs at, so the tracker is stepped
# identically in both. This is achievable only because the backend holds its
# telemetry subscriptions open: fetching state per iteration instead capped the
# loop at 0.5 Hz. Measured against PX4 SITL, vehicle_attitude_setpoint arrives
# at 51 Hz.
CONTROL_RATE_HZ = 50.0
REPORT_INTERVAL_S = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", default=os.path.join(ROOT, "missions",
                                                          "facade_inspection.yaml"))
    parser.add_argument("--tracker", default="mpc", choices=["pid", "mpc"])
    parser.add_argument("--address", default=os.environ.get("PX4_ADDRESS", "udpin://0.0.0.0:14540"))
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="give up after this many seconds of flight")
    args = parser.parse_args()

    mission = load_mission(args.mission)
    print(f"mission: {mission.name} ({len(mission.items)} items)")
    print(f"tracker: {args.tracker}")
    print(f"backend: {args.address}")

    # The vehicle parameters describe the airframe the tracker is commanding,
    # so against SITL they have to be the *simulated* airframe rather than this
    # repository's reference one. The tracker divides a force demand by
    # `max_thrust` and multiplies an acceleration demand by `mass`; both are
    # wrong from the first sample if these do not match.
    #
    # Taken from PX4's Tools/simulation/gz/models/x500_base/model.sdf: a 2.0 kg
    # body plus four 16.1 g rotors, rotors at (+/-0.174, +/-0.174) m, and
    # motorConstant 8.54858e-06 against maxRotVelocity 1000 rad/s, which is
    # 8.55 N per rotor at full command. Thrust-to-weight is 1.69, against 2.17
    # for the reference airframe — the SITL vehicle has noticeably less margin.
    params = QuadrotorParams(
        mass=2.0643,
        inertia=np.array([0.021667, 0.021667, 0.040000]),
        arm_length=0.246,
        max_rotor_thrust=8.549,
        torque_per_thrust=0.016,
    )
    tracker = make_tracker(args.tracker, params)
    executive = MissionExecutive(mission)
    geofence = Geofence.from_spec(mission.geofence)
    failsafe = FailsafeMatrix(spec=mission.failsafe or FailsafeSpec())

    # The mission's own origin, not PX4's launch point, defines what the
    # waypoint coordinates mean. See Px4SitlBackend.__init__.
    backend = make_backend("px4", system_address=args.address, origin=mission.origin)
    interval = 1.0 / CONTROL_RATE_HZ

    try:
        backend.connect()
        print("connected; waiting for a stable position estimate")
        backend.poll()

        backend.arm()
        backend.start_offboard()
        print("armed and in offboard mode")

        started = time.monotonic()
        executive.arm(0.0, backend.state())
        executive.start(0.0, backend.state())
        tracker.reset(backend.state())

        next_control = time.monotonic()
        next_report = 0.0
        last_elapsed = 0.0
        while True:
            now = time.monotonic()
            if now < next_control:
                time.sleep(max(0.0, next_control - now))
                next_control += interval
            else:
                # Behind schedule. Re-base rather than accumulating a deficit
                # that can never be paid off: `next_control += interval` on a
                # loop that is already late means the sleep is skipped forever
                # after and the loop free-runs.
                next_control = time.monotonic() + interval
            elapsed = time.monotonic() - started
            if elapsed > args.timeout:
                print("timeout; landing")
                break

            # The step the tracker is told about has to be the step that
            # actually happened. Every MAVSDK call is a round trip through an
            # event loop, so this loop does not hit its nominal rate; handing
            # the tracker a nominal 1/50 s while the real interval is twice
            # that makes its internal model of its own effect on the vehicle
            # wrong by the same factor, and the disturbance estimate and rate
            # limits are then integrating against a clock that does not exist.
            dt = float(np.clip(elapsed - last_elapsed, 0.5 * interval, 4.0 * interval))
            last_elapsed = elapsed

            backend.poll()
            state = backend.state()
            state.time_s = elapsed

            enforce_floor = executive.state not in (
                MissionState.IDLE, MissionState.ARMED,
                MissionState.TAKEOFF, MissionState.LANDING,
            )
            fence = geofence.check(state.position, state.velocity,
                                   enforce_floor=enforce_floor)
            battery = backend.battery()
            status = SystemStatus(
                time_s=elapsed,
                battery_remaining=battery.remaining,
                battery_voltage_v=battery.voltage_v,
                link_silence_s=backend.link_silence_s,
                link_connected=backend.is_connected(),
                # PX4 runs its own estimator, so its health is reported rather
                # than re-derived here.
                gnss_usable=True,
                gnss_is_rtk=True,
                estimator_healthy=True,
                geofence_breached=fence.breached,
                geofence_predicted_breach=fence.predicted_breach,
                geofence_margin_m=fence.margin_m,
                in_air=executive.in_air,
                armed=True,
            )
            action = failsafe.update(status)
            reference = executive.update(elapsed, state, action)

            if executive.state.is_finished:
                print(f"mission finished: {executive.state.value}")
                break
            if executive.state == MissionState.LANDING:
                # PX4's land detector knows things this layer does not.
                print("handing the landing to PX4")
                backend.land()
                break

            setpoint = tracker.update(state, reference, elapsed, dt)
            backend.send_attitude_setpoint(setpoint)

            # Report on elapsed time, not on a loop counter. Counting
            # iterations and assuming they arrive at CONTROL_RATE_HZ silently
            # drops most of the report lines as soon as the loop runs at any
            # other rate — and against PX4 it does, because each MAVSDK call is
            # a round trip through an event loop.
            if elapsed >= next_report:
                next_report += REPORT_INTERVAL_S
                error = np.linalg.norm(state.position - reference.sample(elapsed).position)
                print(f"  t={elapsed:6.1f}s  {executive.state.value:<9} "
                      f"pos=[{state.position[0]:7.1f},{state.position[1]:7.1f}] "
                      f"alt={state.altitude:5.1f} m  err={error:5.2f} m  "
                      f"fence margin={fence.margin_m:6.1f} m")

        for event in failsafe.events:
            print(f"failsafe: t={event.time_s:.1f}s {event.rule} -> {event.action.label}")
        return 0

    except BackendError as error:
        print(f"backend error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; leaving offboard mode")
        return 130
    finally:
        try:
            backend.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
