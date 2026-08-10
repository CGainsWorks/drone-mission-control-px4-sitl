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

**The simulated airframe is not this repository's airframe.** `gz_x500` has a
different mass and thrust-to-weight ratio from the `QuadrotorParams` the tracker
assumes, so the thrust mapping is off by a constant factor. The MPC's
disturbance estimator absorbs it; the PID's integrator absorbs it more slowly.
For a like-for-like comparison against SITL you would want to measure the
airframe first — which is exactly the calibration step a real deployment needs,
and exactly why the reproducible result uses a plant whose parameters are known.

**Arming needs a home position.** In SITL the simulated GPS takes a few seconds
to lock. The backend waits for `is_global_position_ok` and
`is_home_position_ok` rather than arming into a rejection.

## Files

| File | What it is |
| --- | --- |
| `Dockerfile` | PX4 v1.15 with Gazebo, headless, exposing MAVLink on UDP 14540 |
| `worlds/inspection.sdf` | A Gazebo world with a structure to inspect, matching `missions/facade_inspection.yaml` |
| `params/offboard.params` | PX4 parameters that make offboard control behave predictably |
