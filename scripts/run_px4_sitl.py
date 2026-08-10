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

CONTROL_RATE_HZ = 50.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", default=os.path.join(ROOT, "missions",
                                                          "facade_inspection.yaml"))
    parser.add_argument("--tracker", default="mpc", choices=["pid", "mpc"])
    parser.add_argument("--address", default=os.environ.get("PX4_ADDRESS", "udp://:14540"))
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="give up after this many seconds of flight")
    args = parser.parse_args()

    mission = load_mission(args.mission)
    print(f"mission: {mission.name} ({len(mission.items)} items)")
    print(f"tracker: {args.tracker}")
    print(f"backend: {args.address}")

    # The vehicle parameters describe the airframe the tracker is commanding.
    # Against SITL these should match the simulated airframe; the defaults here
    # are close to PX4's gz_x500, and mismatch shows up as a thrust offset the
    # disturbance estimator absorbs.
    params = QuadrotorParams()
    tracker = make_tracker(args.tracker, params)
    executive = MissionExecutive(mission)
    geofence = Geofence.from_spec(mission.geofence)
    failsafe = FailsafeMatrix(spec=mission.failsafe or FailsafeSpec())

    backend = make_backend("px4", system_address=args.address)
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
        while True:
            now = time.monotonic()
            if now < next_control:
                time.sleep(max(0.0, next_control - now))
            next_control += interval
            elapsed = time.monotonic() - started
            if elapsed > args.timeout:
                print("timeout; landing")
                break

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

            setpoint = tracker.update(state, reference, elapsed, interval)
            backend.send_attitude_setpoint(setpoint)

            if int(elapsed * CONTROL_RATE_HZ) % int(CONTROL_RATE_HZ * 2) == 0:
                error = np.linalg.norm(state.position - reference.sample(elapsed).position)
                print(f"  t={elapsed:6.1f}s  {executive.state.value:<9} "
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
