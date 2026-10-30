#!/usr/bin/env python3
"""Fly one mission against PX4 SITL with this repository's control stack.

It runs the same mission state machine, trajectory generator, PID or MPC
tracker, geofence and failsafe matrix as the reference simulation, and only
the bottom layer changes: the built-in plant is replaced by a PX4 autopilot
over MAVLink. With ``--scenario`` it injects one fault from the failsafe
matrix into PX4 or into the MAVLink path, at a fixed point on the first leg.

Outputs, in ``--output``:

    loop.csv.gz   every control step: reference, PX4's estimate, the command
    events.json   failsafe events, phase transitions, PX4 mode changes, outcome

PX4 writes its own ULog; ``scripts/run_sitl_campaign.py`` collects it beside
these. Every number derived from them is simulated (PX4 SIH), not flown.

Requirements:
    pip install -r requirements-optional.txt   # mavsdk
    PX4 SITL running in the same network namespace (see sitl/README.md)

Run:
    python scripts/run_px4_sitl.py --mission facade_inspection --tracker mpc
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control import sitl  # noqa: E402
from mission_control.attitude import quat_to_euler  # noqa: E402
from mission_control.control import make_tracker  # noqa: E402
from mission_control.logs.flight_log import STATE_CODES  # noqa: E402
from mission_control.mavlink_bridge import BackendError, make_backend  # noqa: E402
from mission_control.mavlink_bridge.link_proxy import UdpLinkProxy  # noqa: E402
from mission_control.mission import MissionExecutive, MissionState, load_mission  # noqa: E402
from mission_control.mission.builder import build_hold  # noqa: E402
from mission_control.mission.schema import FailsafeSpec  # noqa: E402
from mission_control.safety import FailsafeMatrix, Geofence, SystemStatus  # noqa: E402

# The rate the reference simulation runs its trackers at, so the tracker is
# stepped the same way in both. Achievable because the backend holds its
# telemetry subscriptions open rather than fetching per iteration.
CONTROL_RATE_HZ = 50.0
REPORT_INTERVAL_S = 5.0
#: After handing the landing to PX4, how long to wait for it to disarm.
LANDING_TIMEOUT_S = 120.0


def resolve_mission(name_or_path: str) -> str:
    if os.path.exists(name_or_path):
        return name_or_path
    candidate = os.path.join(ROOT, "missions", f"{name_or_path}.yaml")
    if os.path.exists(candidate):
        return candidate
    raise SystemExit(f"no mission {name_or_path!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", default="facade_inspection")
    parser.add_argument("--tracker", default="mpc", choices=["pid", "mpc"])
    parser.add_argument("--scenario", default="none",
                        choices=["none", *sitl.SITL_FAILSAFE_SCENARIOS])
    parser.add_argument("--airframe", default="sih", choices=sorted(sitl.AIRFRAMES))
    parser.add_argument("--gains", default="airframe", choices=["airframe", "reference"],
                        help="airframe: gains tuned for the simulated airframe (the "
                             "campaign); reference: the reference comparison's gains")
    parser.add_argument("--name", default="", help="flight name for the log")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "sitl"))
    parser.add_argument("--address", default=os.environ.get("PX4_ADDRESS",
                                                            "udpin://0.0.0.0:14541"))
    parser.add_argument("--no-relay", action="store_true",
                        help="connect MAVSDK straight to PX4 (no link-loss injection)")
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="give up after this many seconds of flight")
    args = parser.parse_args()

    mission = load_mission(resolve_mission(args.mission))
    name = args.name or f"{mission.name}-{args.tracker}"
    scenario = sitl.SITL_FAILSAFE_SCENARIOS.get(args.scenario)
    if scenario and args.no_relay and scenario["inject"] == "link_drop":
        raise SystemExit("the link-loss scenario needs the relay")

    print(f"flight:   {name}")
    print(f"mission:  {mission.name} ({len(mission.items)} items)")
    print(f"tracker:  {args.tracker}   airframe: {args.airframe}   scenario: {args.scenario}")

    params = sitl.AIRFRAMES[args.airframe]()
    gains_path = sitl.AIRFRAME_GAINS.get(args.airframe) if args.gains == "airframe" else None
    tracker = make_tracker(args.tracker, params, gains_path=gains_path)
    executive = MissionExecutive(mission)
    geofence = Geofence.from_spec(mission.geofence)
    failsafe = FailsafeMatrix(spec=mission.failsafe or FailsafeSpec())

    relay = None
    address = args.address
    if args.no_relay:
        address = os.environ.get("PX4_ADDRESS", "udpin://0.0.0.0:14540")
    else:
        relay = UdpLinkProxy(listen_port=14540, client_port=14541).start()
    backend = make_backend("px4", system_address=address, origin=mission.origin)

    rows = []
    events = {
        "flight": name,
        "mission": mission.name,
        "tracker": args.tracker,
        "scenario": args.scenario,
        "airframe": args.airframe,
        "px4_version": sitl.PX4_VERSION,
        "tracker_config": tracker.describe(),
        "gains_file": (os.path.relpath(gains_path, ROOT).replace(os.sep, "/")
                       if gains_path else "config/tuned_gains.yaml"),
        "result": "error",
        "end_reason": "",
        "injected_at_s": None,
        "mode_changes": [],
        "offboard_reengaged": [],
    }
    interval = 1.0 / CONTROL_RATE_HZ
    exit_code = 1

    try:
        backend.connect(timeout_s=args.connect_timeout)
        print("connected; position estimate is good")
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
        mission_started_s = None
        hold_started_s = None
        handed_over_s = None
        injected = False
        last_mode = None
        # Only a mode change *away from* offboard means PX4 took control. The
        # first mode report after connecting predates the switch to offboard,
        # so it is not evidence of anything.
        offboard_seen = False
        silent_since = None
        steps = 0

        while True:
            now = time.monotonic()
            if now < next_control:
                time.sleep(max(0.0, next_control - now))
                next_control += interval
            else:
                # Behind schedule: re-base rather than accumulate a deficit
                # that would make the loop free-run for the rest of the flight.
                next_control = time.monotonic() + interval
            elapsed = time.monotonic() - started
            dt = float(np.clip(elapsed - last_elapsed, 0.5 * interval, 4.0 * interval))
            last_elapsed = elapsed
            steps += 1
            if elapsed > args.timeout:
                events["result"] = "timeout"
                events["end_reason"] = f"no outcome within {args.timeout:.0f} s"
                break

            backend.poll()
            state = backend.state()
            state.time_s = elapsed
            gnss = backend.gnss()
            battery = backend.battery()
            mode = backend.flight_mode
            if mode != last_mode:
                events["mode_changes"].append([round(elapsed, 2), mode])
                last_mode = mode
            offboard_seen = offboard_seen or mode == "offboard"

            enforce_floor = executive.state not in (
                MissionState.IDLE, MissionState.ARMED,
                MissionState.TAKEOFF, MissionState.LANDING,
            )
            fence = geofence.check(state.position, state.velocity, enforce_floor=enforce_floor)
            status = SystemStatus(
                time_s=elapsed,
                battery_remaining=battery.remaining,
                battery_voltage_v=battery.voltage_v,
                link_silence_s=backend.link_silence_s,
                link_connected=backend.is_connected(),
                # Reported by PX4 over MAVLink: GPS_RAW_INT's fix type, and how
                # long since it last said 3D or better.
                gnss_usable=gnss["fix_type"] >= 3,
                gnss_is_rtk=gnss["fix_type"] >= 6,
                time_since_gnss_update_s=min(gnss["age_s"], 1e6),
                # PX4 runs its own estimator and gates its own outputs; the
                # covariance rule belongs to the reference simulation's EKF.
                estimator_healthy=True,
                geofence_breached=fence.breached,
                geofence_predicted_breach=fence.predicted_breach,
                geofence_margin_m=fence.margin_m,
                in_air=executive.in_air,
                armed=True,
            )
            if handed_over_s is None:
                action = failsafe.update(status)
            else:
                # PX4 is landing the vehicle and the matrix's job is done.
                # Evaluating it further would read the descent through the
                # fence floor as a breach, since the state machine is still in
                # whatever phase it handed over from.
                action = failsafe.action
            reference = executive.update(elapsed, state, action)

            if executive.state == MissionState.MISSION and mission_started_s is None:
                mission_started_s = elapsed

            # --- fault injection -------------------------------------------
            if (scenario and not injected and mission_started_s is not None
                    and elapsed - mission_started_s >= sitl.INJECT_AFTER_MISSION_S):
                injected = True
                events["injected_at_s"] = round(elapsed, 2)
                kind = scenario["inject"]
                if kind == "link_drop":
                    relay.drop_for(scenario["duration_s"])
                elif kind == "param":
                    backend.set_param(scenario["param"], scenario["value"])
                elif kind == "fence":
                    geofence = Geofence.from_spec(
                        sitl.tightened_fence(mission.geofence, scenario["half_width_m"]))
                print(f"  t={elapsed:6.1f}s  injected {args.scenario}: {scenario['description']}")

            # --- ending the flight -----------------------------------------
            if executive.state == MissionState.HOLD:
                if hold_started_s is None:
                    hold_started_s = elapsed
                if handed_over_s is None and elapsed - hold_started_s >= sitl.HOLD_OBSERVE_S:
                    print(f"  t={elapsed:6.1f}s  held for {sitl.HOLD_OBSERVE_S:.0f} s; "
                          "ending the test with a PX4 landing")
                    events["end_reason"] = "hold observed, landing commanded to end the test"
                    backend.land()
                    handed_over_s = elapsed
            if executive.state == MissionState.LANDING and handed_over_s is None:
                # PX4's land detector knows things this layer does not.
                print(f"  t={elapsed:6.1f}s  handing the landing to PX4")
                backend.land()
                handed_over_s = elapsed
            if executive.state.is_finished and handed_over_s is None:
                events["end_reason"] = events["end_reason"] or executive.state.value
                break
            if handed_over_s is not None:
                # Wait for PX4 to disarm, not just for this layer to see the
                # vehicle on the ground: the ULog closes on disarm.
                if not backend.armed and elapsed - handed_over_s > 1.0:
                    events["end_reason"] = events["end_reason"] or "landed and disarmed by PX4"
                    break
                if elapsed - handed_over_s > LANDING_TIMEOUT_S:
                    events["end_reason"] = "PX4 did not disarm after landing"
                    events["result"] = "timeout"
                    break

            # --- control ---------------------------------------------------
            sending = handed_over_s is None
            link_up = backend.link_silence_s < 0.5 and not (relay and relay.dropping)
            if not link_up:
                silent_since = elapsed if silent_since is None else silent_since
            elif silent_since is not None:
                went_silent, silent_since = silent_since, None
                if (executive.state == MissionState.HOLD
                        and executive.transitions[-1][0] >= went_silent):
                    # The hold was decided while the link was down, so it was
                    # anchored to the last position received before the
                    # silence. PX4 has since stopped the vehicle somewhere
                    # further on with its own offboard-loss hold. Holding means
                    # holding where the vehicle is, so re-anchor there.
                    executive.reference = build_hold(
                        state.position, 1e6, mission.defaults, yaw=state.yaw,
                        start_time_s=elapsed)
                    tracker.reset(state)
                    events["hold_reanchored_s"] = round(elapsed, 2)
            if (sending and executive.in_air and link_up and offboard_seen
                    and mode not in ("offboard", "unknown")):
                # PX4 dropped out of offboard (its own reaction to the setpoint
                # stream stopping). The link is back, so take control again and
                # fly whatever the matrix has decided.
                try:
                    if backend.ensure_offboard():
                        offboard_seen = False
                        tracker.reset(state)
                        events["offboard_reengaged"].append(round(elapsed, 2))
                        print(f"  t={elapsed:6.1f}s  link back; offboard re-engaged "
                              f"in {executive.state.value}")
                except BackendError as error:
                    print(f"  t={elapsed:6.1f}s  could not re-engage offboard: {error}")

            if sending:
                setpoint = tracker.update(state, reference, elapsed, dt)
                backend.send_attitude_setpoint(setpoint)
                command = quat_to_euler(setpoint.orientation)
                thrust = setpoint.thrust
                diagnostics = tracker.diagnostics
                accel = diagnostics.acceleration_demand
                timing = (diagnostics.update_time_s, diagnostics.solve_time_s,
                          float(diagnostics.iterations), float(diagnostics.saturated))
            else:
                command = (math.nan,) * 3
                thrust = math.nan
                accel = (math.nan,) * 3
                timing = (math.nan,) * 4

            ref = reference.sample(elapsed)
            row = {
                "time_s": elapsed,
                "est_north": state.position[0], "est_east": state.position[1],
                "est_down": state.position[2],
                "est_vn": state.velocity[0], "est_ve": state.velocity[1],
                "est_vd": state.velocity[2],
                "est_yaw": quat_to_euler(state.orientation)[2],
                "ref_north": ref.position[0], "ref_east": ref.position[1],
                "ref_down": ref.position[2],
                "ref_vn": ref.velocity[0], "ref_ve": ref.velocity[1], "ref_vd": ref.velocity[2],
                "ref_yaw": ref.yaw,
                "cmd_thrust": thrust, "cmd_roll": command[0], "cmd_pitch": command[1],
                "cmd_yaw": command[2],
                "accel_demand_n": accel[0], "accel_demand_e": accel[1],
                "accel_demand_d": accel[2],
                "update_time_s": timing[0], "solve_time_s": timing[1],
                "solver_iterations": timing[2], "saturated": timing[3],
                "mission_state": STATE_CODES.get(executive.state.value, -1),
                "mission_progress": executive.mission_progress,
                "failsafe_action": int(action),
                "geofence_margin_m": fence.margin_m if math.isfinite(fence.margin_m) else 1e6,
                "geofence_breached": float(fence.breached),
                "geofence_predicted": float(fence.predicted_breach),
                "battery_remaining": battery.remaining,
                "battery_voltage_v": battery.voltage_v,
                "battery_current_a": battery.current_a,
                "gnss_fix_type": gnss["fix_type"],
                "ekf_position_std_h": math.nan,
                "ekf_nis": math.nan,
                "link_silence_s": min(backend.link_silence_s, 1e6),
                "px4_mode": sitl.PX4_MODE_CODES.get(mode, 0),
                "armed": float(backend.armed),
                "gnss_satellites": gnss["satellites"],
                "link_dropping": float(bool(relay and relay.dropping)),
                "fault_injected": float(injected),
            }
            rows.append([row[column] for column in sitl.LOOP_COLUMNS])

            if elapsed >= next_report:
                next_report += REPORT_INTERVAL_S
                error = np.linalg.norm(state.position - ref.position)
                print(f"  t={elapsed:6.1f}s  {executive.state.value:<8} px4={mode:<16} "
                      f"pos=[{state.position[0]:6.1f},{state.position[1]:6.1f}] "
                      f"alt={-state.position[2]:5.1f} m  err={error:5.2f} m  "
                      f"batt={battery.remaining:4.0%}  sats={gnss['satellites']}")

        if events["result"] == "error":
            events["result"] = "failsafe" if failsafe.events else "completed"
        events["loop_rate_hz"] = steps / max(last_elapsed, 1e-6)
        exit_code = 0 if events["result"] in ("completed", "failsafe") else 1

    except BackendError as error:
        events["end_reason"] = f"backend error: {error}"
        print(f"backend error: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        events["end_reason"] = "interrupted"
        exit_code = 130
    finally:
        try:
            backend.close()
        except Exception:
            pass
        if relay is not None:
            events["relay_counters"] = dict(relay.counters)
            relay.close()

        events["failsafe_events"] = [
            {"time_s": round(e.time_s, 2), "rule": e.rule, "action": e.action.label}
            for e in failsafe.events
        ]
        events["transitions"] = executive.summary()["transitions"]
        os.makedirs(args.output, exist_ok=True)
        sitl.write_loop_log(os.path.join(args.output, "loop.csv.gz"), rows)
        with open(os.path.join(args.output, "events.json"), "w", encoding="utf-8",
                  newline="\n") as handle:
            json.dump(events, handle, indent=2, sort_keys=True)
            handle.write("\n")

    for event in events["failsafe_events"]:
        print(f"failsafe: t={event['time_s']:.1f}s {event['rule']} -> {event['action']}")
    print(f"result: {events['result']} ({events['end_reason']})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
