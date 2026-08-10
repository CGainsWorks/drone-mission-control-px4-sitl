# Documentation

| Document | What it covers |
| --- | --- |
| [`RESULTS.md`](RESULTS.md) | The MPC-versus-PID comparison. Generated from `results/runs.csv`. |
| [`FAILSAFE_MATRIX.md`](FAILSAFE_MATRIX.md) | Every failsafe rule, its trigger, action and rationale, plus what each one actually did in flight. Generated from the rule table. |
| [`CONTROLLERS.md`](CONTROLLERS.md) | How both trackers work, what they share, and where the difference comes from. |
| [`ESTIMATOR.md`](ESTIMATOR.md) | The RTK-GNSS/IMU error-state EKF, and the two-clock latency compensation. |
| [`SIMULATION.md`](SIMULATION.md) | The reference plant: vehicle, inner loop, sensors, wind, battery. |
| [`MISSION_SCHEMA.md`](MISSION_SCHEMA.md) | The mission file format and what validation catches. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Module map, rate structure, and why the boundaries are where they are. |
| [`LIMITATIONS.md`](LIMITATIONS.md) | What this does not model, and what that costs. Read this one. |
| [`CAA_NOTES.md`](CAA_NOTES.md) | What real UK flight would require beyond what is here. |

Two of these — `RESULTS.md` and `FAILSAFE_MATRIX.md` — are **generated**, never
edited by hand. CI regenerates them on a clean checkout and fails if they
differ, which is what keeps them honest.
