# PX4 SITL

This directory brings up PX4 software-in-the-loop with Gazebo, so the mission
control stack can be flown against a real autopilot rather than the reference
simulation in `src/mission_control/sim/`.

## What this is for, and what it is not

The **headline MPC-versus-PID result does not come from here.** It comes from
the reference simulation, for one reason: that runs deterministically, in
minutes, on any machine, with no toolchain — and a result nobody can reproduce
is not a result. See [`../docs/LIMITATIONS.md`](../docs/LIMITATIONS.md) for what
that simplification costs.

What this *is* for is demonstrating that the layer above the simulation was not
written around it. The mission state machine, the trajectory generator, both
trackers, the geofence and the failsafe matrix are all unchanged between the
two; only `src/mission_control/mavlink_bridge/px4.py` knows the difference.

## Running it

With Docker, which is the least painful route:

```bash
docker compose --profile sitl up px4
```

Then, in another terminal:

```bash
docker compose --profile sitl run --rm sitl-mission
```

Without Docker, if you already have a PX4 source tree:

```bash
cd /path/to/PX4-Autopilot
make px4_sitl gz_x500
```

and then, from this repository:

```bash
pip install -r requirements-optional.txt
python scripts/run_px4_sitl.py --mission missions/facade_inspection.yaml --tracker mpc
```

## Things that will catch you out

**Offboard mode will not engage until setpoints are already streaming.** PX4
rejects the mode change if it has not seen a recent setpoint, and the rejection
message does not explain why. The backend primes the stream with twenty
setpoints before requesting the mode; if you write your own, do the same.

**Offboard mode drops out if the stream stalls** for more than about half a
second. On a laptop also running Gazebo, a garbage collection pause is enough.
The vehicle reverts to hold, which is safe and confusing.

**The simulated airframe is not this repository's airframe.** `gz_x500` is
2.06 kg with a thrust-to-weight ratio of 1.69, against 1.6 kg and 2.17 for the
reference airframe. The tracker turns an acceleration demand into a normalised
collective through exactly those two numbers, so it is wrong from the first
sample unless told. `scripts/run_px4_sitl.py` passes the x500's own parameters,
read out of PX4's `x500_base/model.sdf`. This is the calibration step a real
deployment needs, and it is why the reproducible result uses a plant whose
parameters are known exactly.

**PX4's throttle is not linear in thrust, and by default it does not pretend to
be.** A rotor's thrust goes as the square of its speed, and the gz x500 is
exactly quadratic. Hover is 0.59 of full thrust but 0.77 of full *command*. A
controller that computes a normalised thrust — which is what a controller
should compute — asks for 0.59, gets 0.35 of the available force, and sits on
the ground at what looks like a sensible throttle while tilting towards its
first waypoint. `params/offboard.params` sets `THR_MDL_FAC 1.0`, which makes
PX4 invert the curve so a thrust demand means thrust. Nothing flies without it.

**Do not fetch telemetry per control iteration.** MAVSDK exposes telemetry as
async generators, and opening one, taking a single item and dropping it reads
exactly like a synchronous getter. It is not: each new subscription waits for
the *next* message. Measured here, that cost 2.0 seconds per iteration — a
0.5 Hz control loop closed around two-second-old state, which oscillates with a
ten-second period and tens of metres of amplitude regardless of which tracker
is running or how it is tuned. The backend holds its subscriptions open on a
background event loop; the same loop then closes at 50 Hz.

**Arming needs a home position.** In SITL the simulated GPS takes a few seconds
to lock. The backend waits for `is_global_position_ok` and
`is_home_position_ok` rather than arming into a rejection.

**A crashed vehicle stays unarmable.** PX4's failure detector latches
`fd_critical_failure` when the vehicle tumbles, and it is not cleared by
disarming — only by rebooting the autopilot. After a run that ends badly,
restart the container rather than wondering why arming is refused.

## What has actually been flown

`missions/facade_inspection.yaml` has been flown end to end against PX4 v1.15
with Gazebo Garden, headless, in the image this directory builds, by both
trackers: takeoff to 18 m, the four facade legs at 18 m and 24 m, the 18 m
orbit at 20 m, RTL at 35 m, and the landing handed back to PX4. No failsafe
triggered in either run. Reported tracking error against the reference:

| Tracker | Range over the run |
| --- | --- |
| MPC | 0.01–0.33 m |
| PID | 0.23–4.15 m |

**Those two numbers are not a result and must not be quoted as one.** They are
one run each, in still air, on an unloaded machine, with per-sample error read
off a progress line rather than integrated — and the gains both trackers are
using were tuned against the reference airframe, not this one. They are here to
show the stack flies a real autopilot and that neither controller falls over
doing it. The measured comparison, with paired scenarios, held-out missions and
a significance test, is in [`../docs/RESULTS.md`](../docs/RESULTS.md).

## Files

| File | What it is |
| --- | --- |
| `Dockerfile` | PX4 v1.15 with Gazebo, headless, exposing MAVLink on UDP 14540 |
| `worlds/inspection.sdf` | A Gazebo world with a structure to inspect, matching `missions/facade_inspection.yaml` |
| `params/offboard.params` | PX4 parameters that make offboard control behave predictably; installed as `px4-rc.params`, which is the only name PX4's `rcS` sources |

## Building it

The image takes about five minutes and lands around 4 GB. Two things in the
Dockerfile are load-bearing and both are commented there:

- `--shallow-submodules` on the clone. `--depth 1` alone applies only to the
  superproject, and PX4 has 36 submodules; without it the clone was still
  running after nine hours, against 98 seconds with it.
- A tag-only fetch in the NuttX submodule afterwards. PX4's version-header
  generator indexes the last element of a list of NuttX tags, and a shallow
  submodule has none, so the build stops at step 52 of 985 with an `IndexError`
  that says nothing about submodules.
