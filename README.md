# drone-mission-control-px4-sitl

> ### Clean-room reimplementation
> This repository is an **independent reimplementation of a problem class**, built from
> scratch on personal time using public or synthetic data and open-source simulation.
> It is **not** derived from, and contains no material belonging to, any employer or
> client: no employer source code, schematics, calibration data, client data or internal
> documentation. It exists to demonstrate the architecture and engineering approach only.

A mission-control layer for an autonomous inspection multirotor, developed against PX4
software-in-the-loop. Compares MPC and PID trajectory tracking on identical missions,
with an RTK-GPS and IMU EKF state estimator, geofencing and a tested failsafe matrix.

The point of this repository is not that the aircraft flies. It is a **measured,
like-for-like comparison of model-predictive against PID trajectory tracking**: same
airframe, same missions, same wind, same sensor noise, same estimator, same inner loop,
with position error quantified against ground truth — and an honest account of when the
extra complexity earns its place and when it does not.

![Tracking error by scenario](analysis/figures/rms_by_scenario.png)

## The result

Across 84 runs — 42 scenarios (3 missions × 2 speeds × 2 wind conditions × 3 seeds, plus
an infeasible-reference case) flown by both controllers on identical seeds:

| | PID | MPC |
| --- | ---: | ---: |
| Mean RMS position error | 0.907 m | **0.710 m** |
| Mean cross-track error | 0.292 m | **0.230 m** |
| Mean RMS jerk of the acceleration demand | 25.1 m/s³ | **17.1 m/s³** |
| Optimisation per control step | 284 µs | 736 µs |
| Whole control step | 284 µs | 3624 µs |

**40 of 42 paired scenarios go to the MPC**, at a median error ratio of 1.23×. Full
numbers, including every scenario and the speed and wind breakdowns:
**[docs/RESULTS.md](docs/RESULTS.md)** — generated from the raw CSV, not written by hand.

**The advantage is no smaller on the held-out scenarios than on the one it was tuned on**
(1.23× against 1.21×), which is the answer to the obvious objection.

### Where it comes from, and where it doesn't

- **Preview is the whole mechanism, and you can see it arrive.** On a feasible reference
  with rounded corners the gap is 1.21×. On an *infeasible* one — a lawnmower survey
  whose turnarounds are a velocity step no vehicle can follow — it is **4.80×**. The
  reason is visible in the saturation figures: on that case the PID spends **24% of the
  mission at a limit and the MPC 5%**. The PID chases the impossible reference into
  saturation; the MPC, which can see the step coming and knows its own acceleration
  bounds, declines to.
- **On long straight legs it very nearly doesn't matter.** The facade survey at nominal
  speed in calm air is 0.140 m against 0.116 m. One scenario — facade, fast, gusty —
  goes to the PID outright. If your missions look like that, the MPC is not worth the
  QP solver.
- **The MPC is smoother as well as tighter**, at 17.1 m/s³ of demand jerk against 25.1.
  Both were tuned against the same 25 m/s³ budget; the PID needed all of it and the MPC
  did not.
- **Some scenarios are beyond the airframe, not beyond the controller.** Where the
  vehicle is saturated 20-50% of the mission, both controllers converge and the error is
  a property of the trajectory. `docs/RESULTS.md` flags those explicitly rather than
  letting them read as a controller comparison.

So: **MPC where the reference is aggressive relative to the airframe** — tight corners,
continuous curvature, plans that were never quite flyable — **and PID where it is not**,
which on a gentle survey is most of the time.

![Trajectories and tracking error](analysis/figures/trajectories.png)

## Reproducing it

```bash
docker compose up
```

That builds a small Python image and runs the whole pipeline: the test suite, the full
scenario grid with both controllers, every failsafe scenario, all the figures, and
`docs/RESULTS.md`. It then **verifies** that each artifact exists and is non-empty, that
every scenario was flown by both controllers, and that there is one log per run. No ROS,
no Gazebo, no GPU, no network.

Every number in this README comes out of that command. It ends with `END-TO-END PASSED`,
or a non-zero exit.

```bash
docker compose run --rm quick        # one seed, calm air, about a minute
docker compose run --rm comparison   # just the controller comparison
docker compose run --rm test         # just the test suite
docker compose run --rm tune         # re-tune both controllers from scratch
```

Without Docker:

```bash
pip install -r requirements.txt
PYTHONPATH=src bash scripts/run_all.sh
```

The comparison is deterministic. Every generator is seeded, results are sorted before
writing, and CI regenerates `docs/RESULTS.md`, `docs/FAILSAFE_MATRIX.md` and the figures
on a clean checkout — failing if they differ from what is committed.

## How the comparison is kept fair

This is the part that decides whether any of the numbers mean anything.

**Paired.** Both controllers fly every scenario with the same seed. Wind, IMU noise and
GNSS error are generated from streams seeded independently of the controller, so the two
aircraft meet the same gust at the same instant. The difference within a pair is the
controller and nothing else. `tests/test_runner.py` asserts it.

**Same everything below the control law.** Both emit the same
`AttitudeSetpoint` — deliberately the MAVLink `SET_ATTITUDE_TARGET` payload — and share
the acceleration-to-attitude conversion, the tilt limit, the inner loop, the estimator
and the reference object. The MPC samples its whole horizon and the PID samples only the
present, but that is a choice each controller makes, not something the harness did for
one of them.

**Both get integral action.** Giving one controller a way to reject a steady disturbance
and not the other would decide the result before it started.

**Tuned by the same procedure, on the same data, against the same objective.** Both are
tuned on `facade_inspection` at nominal speed in calm air, and nowhere else.

The objective took three attempts to get right, and the two failures are worth stating
because they are the ones a comparison like this usually hides. Minimising RMS error
alone is not well posed — it drives any controller to the edge of stability, and the
search duly ran to the edge of every range it was given, twice, buying a slightly better
number each time with a rougher input. Adding a **jerk budget** derived from the attitude
loop's rate limit fixed that. Then the winning PID turned out to track the smooth
training mission beautifully and settle into a permanent **bang-bang limit cycle** on a
step reference, because the training set contains no discontinuities. Adding a
**step-settling requirement** fixed that, at a cost of 0.0006 m in training RMS.

Both controllers are held to both constraints. Neither is allowed to buy accuracy with
roughness or instability the other is denied.

**Held out.** The corridor survey, the tower orbit, the fast settings, the gusty
settings and the infeasible-reference case are never seen by the tuner.
[`docs/RESULTS.md`](docs/RESULTS.md) reports tuned-on and held-out separately.

## What is in here

```
src/mission_control/
  mission/          schema, strict validation, path geometry, speed profile, state machine
  estimation/       15-state error-state EKF, fusion horizon and output predictor
  control/          shared interface, PID tracker, MPC tracker, box-QP solver
  safety/           predictive geofence, table-driven failsafe matrix
  sim/              6-DOF plant, PX4-style inner loop, IMU/GNSS/wind/battery models
  logs/             flight log, self-contained ULog writer and reader
  mavlink_bridge/   backend interface, PX4 over MAVSDK
  analysis/         metrics, scenario grid, comparison harness, figures
missions/           three example inspection missions
analysis/           entry points and the ULog notebooks
sitl/               PX4 + Gazebo, behind a compose profile
```

Everything above `mavlink_bridge` is vehicle-agnostic: swapping the reference simulation
for a real PX4 autopilot changes exactly one file.
[`scripts/run_px4_sitl.py`](scripts/run_px4_sitl.py) flies the same stack against SITL.

## Things worth looking at

A few pieces where the interesting decision is not the obvious one:

**[The estimator has two clocks.](docs/ESTIMATOR.md)** A GNSS fix describes where the
vehicle *was*. Fusing it as though it were current drags the estimate backwards along
the flight path by speed times latency, which at inspection speeds is an order of
magnitude more than the noise of an RTK-fixed solution. The filter runs at a delayed
fusion horizon and forward-predicts to the present: **0.222 m to 0.050 m position RMS**
on the identical flight, measured by
[`scripts/benchmark_estimator.py`](scripts/benchmark_estimator.py).

**[The QP solver is projected Newton, not ADMM.](src/mission_control/control/qp.py)**
ADMM was tried first; a condensed MPC Hessian is ill-conditioned and one scalar penalty
parameter cannot serve both ends of it — 4e-4 KKT residual after 60 iterations, about a
millisecond per axis. Projected Newton identifies the active set in a few iterations and
terminates exactly: **21.5 µs per axis warm-started, KKT residual 8.5e-14**, measured by
[`scripts/benchmark_qp.py`](scripts/benchmark_qp.py).

**[The mixer sacrifices yaw before roll and pitch.](src/mission_control/sim/quadrotor.py)**
An earlier version scaled all three torque axes together when saturated, so the yaw
demand generated while turning a corner scaled roll and pitch down with it. The aircraft
stopped being able to hold the turn and departed. The symptom looked like a controller
tuning problem and was not.

**[The failsafe matrix is a table, not a tree of conditionals.](docs/FAILSAFE_MATRIX.md)**
Which means [`docs/FAILSAFE_MATRIX.md`](docs/FAILSAFE_MATRIX.md) is generated from it and
cannot drift, `tests/test_safety.py` iterates it so a rule without a test trigger fails
the suite, and `analysis/run_failsafe_matrix.py` flies each row with the real fault
injected.

**[Mission validation is strict.](docs/MISSION_SCHEMA.md)** Unknown keys are an error
with a suggested correction, and cross-field geometry is checked before takeoff: a
mission whose waypoints sit outside its own geofence, or above its own ceiling, is
rejected on the ground.

## Documentation

| | |
| --- | --- |
| [`docs/RESULTS.md`](docs/RESULTS.md) | The full comparison. Generated from `results/runs.csv`. |
| [`docs/FAILSAFE_MATRIX.md`](docs/FAILSAFE_MATRIX.md) | Every rule, its rationale, and what it did in flight. Generated. |
| [`docs/CONTROLLERS.md`](docs/CONTROLLERS.md) | Both trackers, what they share, where the difference comes from. |
| [`docs/ESTIMATOR.md`](docs/ESTIMATOR.md) | The RTK-GNSS/IMU filter and its latency compensation. |
| [`docs/SIMULATION.md`](docs/SIMULATION.md) | The reference plant, inner loop and sensor models. |
| [`docs/MISSION_SCHEMA.md`](docs/MISSION_SCHEMA.md) | The mission file format. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, rate structure, and why the boundaries sit where they do. |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | What this does not model, and what that costs. |
| [`docs/CAA_NOTES.md`](docs/CAA_NOTES.md) | What real UK flight would require beyond what is here. |
| [`analysis/notebooks/`](analysis/notebooks/) | ULog flight analysis, and the comparison worked through from the raw results. |

## Limitations

The full list is in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and it is worth reading
before drawing conclusions. The three that matter most:

**The headline numbers come from the reference simulation, not from PX4 SITL.** The SITL
path exists, works, and flies the same stack — but the comparison runs against the
built-in plant because that reproduces deterministically in minutes on any machine, and
a result nobody can check is not a result. The plant models rigid-body dynamics, rotor
lag, quadratic drag and wind; it does not model rotor interaction, ground effect, vortex
ring state, airframe vibration or ESC nonlinearity. The *ranking* is robust because the
mechanism is preview, which does not depend on any of those. The *magnitudes* are
optimistic for both controllers by an unquantified amount.

**The estimator has no magnetometer.** GNSS and IMU only, so yaw is observable solely
through acceleration and converges to about 6 degrees where roll and pitch reach 1.

**Neither controller is exhaustively tuned.** The search covers tens of candidates, not
thousands. Both were tuned by the same procedure with the same budget, which makes the
comparison fair without making either one optimal.

## Licence

MIT. See [`LICENSE`](LICENSE).
