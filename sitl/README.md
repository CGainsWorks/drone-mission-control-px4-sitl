# PX4 SITL

This directory brings up PX4 software-in-the-loop so the mission-control stack
can be flown against a real autopilot rather than the reference simulation in
`src/mission_control/sim/`. Everything flown here is **simulated**: PX4's own
SIH physics model, not an aircraft.

| Path | What it is |
| --- | --- |
| `sih/Dockerfile` | PX4 **v1.17.0**, built from the release tag in the PX4 project's own build container (`px4io/px4-dev:v1.17.0`), `px4_sitl_nolockstep` target, SIH quadrotor (airframe 10040). About 200 MB. Used by the campaign and by CI. |
| `sih/10040_sihsim_quadx.post` | The PX4 parameters this project flies with and why each one is there |
| `sih/logger_topics.txt` | A short ULog topic list, so a whole flight's log is a few hundred kilobytes |
| `gazebo.Dockerfile` | PX4 v1.15 with Gazebo and the `gz_x500`, for watching a flight. Not used for any committed result and not run in CI. |
| `worlds/inspection.sdf`, `params/offboard.params` | The Gazebo world and parameter file for that image |
| `results/` | The committed campaign: one directory per flight and `summary.json` |

## What is flown

`scripts/run_sitl_campaign.py` starts a fresh PX4 container per flight, with its
home set to the mission's declared origin and runs `scripts/run_px4_sitl.py` in
a second container that shares PX4's network namespace. That script is the same
mission state machine, trajectory generator, trackers, geofence and failsafe
matrix as the reference simulation; only `mavlink_bridge/px4.py` differs. It
streams attitude setpoints over MAVLink (MAVSDK 3.10.2) at 50 Hz in offboard
mode and hands the final landing to PX4.

The full campaign is every survey mission (`facade_inspection`,
`corridor_survey`, `tower_orbit`) flown by both trackers, plus one flight per
failsafe row with a real fault injected on the first leg:

| Fault | How it is injected |
| --- | --- |
| link loss | a UDP relay in the MAVLink path (`mavlink_bridge/link_proxy.py`) drops every datagram, both ways, for 6 s |
| low battery | PX4's battery simulator is allowed to drain to 15% (`SIM_BAT_MIN_PCT`) |
| GNSS loss | PX4's simulated receiver is told it sees no satellites (`SIM_GPS_USED 0`) |
| geofence breach | the mission fence is tightened in flight to a square the vehicle is already outside |

PX4's own ULog for each flight, gzip-compressed and the flight script's 50 Hz
loop log are committed under `results/flights/`.
`analysis/make_sitl_results.py` turns them into `results/summary.json`,
[`../docs/SITL_RESULTS.md`](../docs/SITL_RESULTS.md), the figure
`analysis/figures/sitl_tracking.png` and the SITL table in the README. Errors
are measured against SIH's ground truth from the ULog, with the same
`tracking_metrics` the reference comparison uses.

## Running it

```bash
make sitl-images      # builds both images; the PX4 build takes a few minutes
make sitl-smoke       # one short flight, exactly as CI flies it on every push
make sitl-campaign    # the full campaign, about 25 minutes in real time
make sitl-results     # summary, document, figure and README table
```

or by hand, one flight at a time:

```bash
docker compose --profile sitl up -d px4
docker compose --profile sitl run --rm sitl-mission
docker compose --profile sitl down
```

## Isolation

PX4 SITL uses fixed UDP ports (MAVLink 14540, 14550 and 18570). The PX4
container runs with `--network none`, so it has a loopback interface and
nothing else and the flight container joins that namespace. Nothing on the
Docker host can reach those ports and they cannot meet another simulator's. The
Gazebo profile also pins gz-transport to loopback and its own partition
(`GZ_IP=127.0.0.1`, `GZ_PARTITION=px4mc`).

## Gains for this airframe

The reference comparison's gains were tuned for the reference airframe and its
attitude loop, which responds to an acceleration demand as a 0.13 s first-order
lag with no delay. Against PX4 the path is longer: `scripts/identify_px4_lag.py`
fits a **0.16 s delay and a 0.18 s lag** from a facade flight on a
GitHub-hosted runner (MAVLink and MAVSDK, PX4's attitude loop and PX4's
estimator on the way back) and other clean flights give 0.12 to 0.16 s and
0.16 to 0.22 s.

With the reference gains unchanged (`results/transfer/`, flown by the weekly
job), both trackers fly the facade survey but neither stays inside the
25 m/s³ jerk budget both were tuned to. The PID's demand runs at 41.5 m/s³
RMS and sits at an acceleration limit for 39% of the survey, a limit cycle
between its bounds and the MPC's at 36.2 m/s³. Their position errors
(0.73 m and 0.42 m RMS) are not a comparison worth quoting, because one
of the two constraints that make the tuning objective well posed is broken for
both.

So the campaign does not use those gains. `scripts/tune_gains.py --airframe sih`
re-tunes **both** trackers with the reference comparison's own procedure,
objective and constraints, on a surrogate of PX4 SIH: the reference simulation
with SIH's airframe (`mission_control.sitl.sih_quadx_params`) and the measured
delay on every attitude command. The result is
`config/tuned_gains_px4_sih.yaml` and it is what every committed campaign
flight used. The MPC's internal lag constant is set to the measured delay plus
lag, 0.34 s, the same way the reference MPC's 0.13 s was measured rather than
guessed.

## Simulator stalls

The nolockstep build runs SIH on the wall clock, so if the host stops
scheduling PX4 for long enough, PX4's sensor monitor reports a simulated
accelerometer as timed out (`[sensors] Accel #0 fail: TIMEOUT!`) and EKF2
degrades. On the shared development machine this happened in most flights
and produced what looked like estimator divergence, including an earlier
reading of the reference-gain PID flight as having dragged PX4's EKF tens of
metres off. On a GitHub-hosted runner it did not happen at all. The campaign
reads PX4's own error messages from every ULog, re-flies a stalled flight up to
three times, records each attempt in `attempts.json` and the results gate
rejects any flight that still shows one. Every committed flight was flown on a
GitHub-hosted runner by the weekly campaign job, run on manual dispatch, and
none needed a second attempt.

## Things that will catch you out

**Lockstep.** PX4's default SITL build runs on simulation time, which SIH
advances in fixed steps and then sleeps. The sleeps overshoot and simulation
time ran at 0.944 of wall-clock time here. A controller in another process on
wall-clock time then disagrees with the vehicle about how long a second is by
over 5%. The image builds `px4_sitl_nolockstep`, which runs on the real clock.

**Parameters equal to their default.** PX4 v1.17 has a `PX4_PARAM_<name>`
environment hook but `rcS` later calls `param set-default` on several
parameters and a value set through the hook that equals the compiled default
does not survive it: `COM_LOW_BAT_ACT` ended up at 3 instead of 0. The
parameters therefore live in the airframe's `.post` file, which runs after
PX4's defaults.

**Offboard mode will not engage until setpoints are already streaming**, and
drops out if the stream stalls for `COM_OF_LOSS_T`. The backend primes the
stream and after a link loss re-engages offboard priming with the last
command rather than a zero-thrust one.

**Do not fetch telemetry per control iteration.** MAVSDK's telemetry is async
generators; opening one per read waits for the *next* message. The backend
holds every subscription open on a background event loop.

**Global position at the control rate.** The mission frame comes from PX4's
global position, so that stream is requested at 50 Hz. At its default rate it
was up to 100 ms stale, which at survey speed is 0.4 m that is not tracking.

**SIH is not Gazebo.** No wind model, a linear drag of 1 N per m/s, a 1 kg
quadrotor with a thrust-to-weight ratio of 2.04. Its numbers are not the
reference simulation's and neither set is an aircraft's.
