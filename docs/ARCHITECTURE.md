# Architecture

## The shape of it

```
                       missions/*.yaml
                             |
                     [ schema + loader ]        strict validation, geodetic -> NED
                             |
                          Mission
                             |
              +--------------+--------------+
              |                             |
       [ MissionExecutive ]          [ FailsafeMatrix ]  <-- SystemStatus
              |   state machine              |   table-driven, latching
              |                              |
     [ trajectory builder ]                  |
              |   path -> speed profile      |
              v                              v
        ReferenceProvider  ------------> action (none/hold/rtl/land/...)
              |
              |   sample(t) and sample_horizon(t, dt, N)
              v
     +------------------+
     | TrajectoryTracker|   PidTracker | MpcTracker
     +------------------+
              |   AttitudeSetpoint: quaternion + body rate + normalised thrust
              v
     +------------------+
     |  VehicleBackend  |   reference simulation | PX4 over MAVLink
     +------------------+
              |
        VehicleState  <---- [ GnssInsEstimator ]  <---- IMU, GNSS
```

Everything above `VehicleBackend` is vehicle-agnostic. Swapping the reference
simulation for a real PX4 autopilot changes exactly one file.

## Why the boundaries are where they are

**The tracker interface is `(state, reference, time, dt) -> AttitudeSetpoint`.**
Not `-> motor commands`, because that would put control allocation inside the
comparison and make it a comparison of two mixers. Not `-> velocity`, because
that would leave the interesting part — turning a demand into a tilt — outside
the controller. `AttitudeSetpoint` is deliberately the MAVLink
`SET_ATTITUDE_TARGET` payload, so the PX4 backend forwards it unchanged.

**Both trackers receive the whole reference object, not a pre-sliced setpoint.**
The MPC samples its full horizon; the PID samples only the present. Giving them
the same object is what makes preview an honest property of the controller
rather than a difference in what the harness handed them.

**The acceleration-to-attitude conversion is shared.** It lives in
`control/common.py` and both trackers call it. If each rolled its own, part of
any measured difference would be that conversion rather than the control law.

**Ground truth never reaches a controller.** The runner passes the estimate;
truth goes only into the log, for metrics. The one exception is
`use_truth_state`, which exists so that a tracking error can be attributed to
the controller rather than the estimator — and it is never used for a headline
number.

## Module map

| Module | What it owns |
| --- | --- |
| `attitude.py` | Quaternion conventions. Hamilton, scalar-first, body-to-NED. |
| `geodesy.py` | WGS-84 to local NED, and the error bound that justifies it. |
| `types.py` | The value types that cross module boundaries. |
| `mission/schema.py` | Mission definition and strict validation. |
| `mission/path.py` | Geometry: lines, arcs, corner rounding, orbit entry. |
| `mission/trajectory.py` | Timing: speed profile, and the consistency guarantee. |
| `mission/builder.py` | Mission items to flyable trajectories, per phase. |
| `mission/state_machine.py` | Phase sequencing and failsafe replanning. |
| `estimation/ekf.py` | 15-state error-state EKF. |
| `estimation/estimator.py` | Fusion horizon and output predictor. |
| `control/pid_tracker.py` | Cascaded P-position, PID-velocity. |
| `control/mpc_tracker.py` | Condensed linear MPC with preview. |
| `control/qp.py` | Projected-Newton box QP. |
| `safety/geofence.py` | Polygon + altitude band, predictive. |
| `safety/failsafe.py` | The rule table. |
| `sim/` | Reference plant, inner loop, sensors, wind, battery. |
| `logs/` | Flight log, ULog writer and reader. |
| `mavlink_bridge/` | Backend interface and the PX4 implementation. |
| `analysis/` | Metrics, scenario grid, comparison harness, figures. |

## Rate structure

| Subsystem | Rate | Why that rate |
| --- | ---: | --- |
| Plant integration (RK4) | 500 Hz | Motor time constant is 35 ms; this resolves it. |
| Attitude and rate control | 500 Hz | Same loop as the plant, as on a real FCU. |
| IMU | 200 Hz | Typical MEMS output rate. |
| Trajectory tracker | 50 Hz | What PX4 runs offboard control at. |
| GNSS | 5 Hz | Typical RTK receiver output. |

Rate separation is not decoration. Running the tracker at the plant rate would
hide the delay that an MPC's preview exists to overcome, and would flatter both
controllers.

## Determinism

Three independent generators, seeded from the scenario seed plus a fixed
per-stream offset:

```python
SEED_WIND = 1009
SEED_IMU  = 2003
SEED_GNSS = 3001
```

Fixed constants rather than sequential draws, so adding a fourth noise source
later cannot shift the existing three. The consequence is that swapping the
controller cannot change the disturbance sequence, which is what makes the
comparison **paired**: the two controllers fly through the same gust at the same
instant.

`tests/test_runner.py` asserts this directly.

## The results pipeline

```
scripts/tune_gains.py  ->  config/tuned_gains.yaml   (committed)
                                  |
analysis/run_comparison.py  ->  results/runs.csv          (not committed)
analysis/run_failsafe_matrix.py -> results/failsafe_runs.csv
                                  |
        +-------------------------+-------------------------+
        |                                                   |
analysis/make_figures.py                        analysis/make_results.py
        |                                                   |
analysis/figures/*.png  (committed)      docs/RESULTS.md, docs/FAILSAFE_MATRIX.md
                                                     (committed)
```

`runs.csv` is not committed because it is regenerable; the documents and figures
are, because they are what a reader looks at. CI regenerates them on a clean
checkout and fails if they differ, which is what keeps the committed artifacts
honest.

## Extending it

**Adding a controller:** implement `TrajectoryTracker`, register it in
`control/__init__.py::TRACKERS`. The comparison harness, the tuning script and
every test parameterised over trackers pick it up automatically.

**Adding a mission:** drop a YAML file in `missions/`. It is validated by
`scripts/validate_missions.py` and by the test suite, including a check that the
generated path stays inside its own geofence.

**Adding a failsafe rule:** add a `FailsafeRule` to `build_default_rules`.
`docs/FAILSAFE_MATRIX.md` regenerates from it, and `tests/test_safety.py`
iterates the table — so a rule without a test trigger fails the suite.

**Adding a backend:** implement `VehicleBackend`. Nothing above it changes.
