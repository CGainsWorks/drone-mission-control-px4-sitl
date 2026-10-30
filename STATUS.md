# Status

_As of 2026-10-30._

Everything here runs on a normal PC and in CI. Every number is simulated: the
reference comparison comes from this repository's own 6-DOF simulation and the
PX4 numbers from PX4 v1.17.0 software-in-the-loop with its SIH model. Nothing
has flown on hardware.

## Complete

| Item | Where | Evidence |
| --- | --- | --- |
| Mission schema, strict validation and loader | [`mission/schema.py`](src/mission_control/mission/schema.py), [`docs/MISSION_SCHEMA.md`](docs/MISSION_SCHEMA.md) | unit tests; every shipped mission validated in CI |
| Mission state machine | [`mission/state_machine.py`](src/mission_control/mission/state_machine.py) | unit tests; every reference and SITL flight |
| 15-state error-state EKF with RTK-GNSS and IMU, delayed fusion horizon | [`estimation/`](src/mission_control/estimation/), [`docs/ESTIMATOR.md`](docs/ESTIMATOR.md) | unit tests; `scripts/benchmark_estimator.py` |
| PID and MPC trajectory trackers, projected-Newton QP | [`control/`](src/mission_control/control/), [`docs/CONTROLLERS.md`](docs/CONTROLLERS.md) | unit tests; `scripts/benchmark_qp.py` |
| Reference comparison: 84 paired runs, tuned and held-out scenarios | [`docs/RESULTS.md`](docs/RESULTS.md) | regenerated weekly in CI and compared with the committed document |
| Predictive geofence and table-driven failsafe matrix | [`safety/`](src/mission_control/safety/), [`docs/FAILSAFE_MATRIX.md`](docs/FAILSAFE_MATRIX.md) | every row flown with its fault in the reference simulation |
| PX4 SITL, headless, from a pinned release | [`sitl/sih/`](sitl/sih/): PX4 v1.17.0, `px4_sitl_nolockstep`, SIH quadrotor | built and flown in CI on every push |
| MAVLink offboard control: arm, takeoff, survey, hand the landing to PX4 | [`mavlink_bridge/px4.py`](src/mission_control/mavlink_bridge/px4.py), MAVSDK 3.10.2 | per-push smoke flight; weekly campaign |
| PX4 SITL campaign: every survey mission by both trackers | [`docs/SITL_RESULTS.md`](docs/SITL_RESULTS.md), [`sitl/results/`](sitl/results/) | ULogs and loop logs committed; the suite recomputes the summary from them |
| Failsafe rows flown against PX4: link loss, low battery, GNSS loss, geofence breach | [`sitl.py`](src/mission_control/sitl.py), [`mavlink_bridge/link_proxy.py`](src/mission_control/mavlink_bridge/link_proxy.py) | each produces the action its row specifies, gated weekly in CI |
| Gains for PX4 SIH, tuned by the reference procedure on a delayed surrogate | [`config/tuned_gains_px4_sih.yaml`](config/tuned_gains_px4_sih.yaml), `scripts/tune_gains.py --airframe sih` | command delay and lag measured from SITL by `scripts/identify_px4_lag.py` |
| README numbers checked against their sources | [`scripts/check_readme.py`](scripts/check_readme.py) | every push |
| ULog writer and analysis notebooks | [`logs/`](src/mission_control/logs/), [`analysis/notebooks/`](analysis/notebooks/) | unit tests; notebooks executed weekly in CI |

**Tests:** 249, all passing locally (`python -m pytest`), run on every push.

## What CI runs

| When | Job | What it proves |
| --- | --- | --- |
| every push | Tests, lint, missions and README numbers | the suite, ruff, mission validation, README against `docs/RESULTS.md` and `sitl/results/summary.json` |
| every push | Docker image | the reproduction image builds and its quick pipeline passes |
| every push | PX4 SITL smoke flight | PX4 v1.17.0 SIH flies `sitl_smoke` with the MPC over MAVLink, end to end |
| weekly and on dispatch | Full reference pipeline | the 84 runs regenerate `docs/RESULTS.md` and `docs/FAILSAFE_MATRIX.md` unchanged |
| weekly and on dispatch | Full PX4 SITL campaign | every flight completes within the tracking gate and every fault gives its specified action |

## In progress

Nothing.

## Not started

- Payload and gimbal: triggering, image geotagging, an inspection deliverable.
- Magnetometer fusion in the reference estimator.
- The permanent link loss, critical battery, degraded RTK and predicted-breach
  rows against PX4 (they are flown in the reference simulation only).

## Limitations

- **No hardware.** Real flight would need an airframe, a flight controller and
  the steps in [`docs/CAA_NOTES.md`](docs/CAA_NOTES.md). Nothing here claims
  otherwise.
- **The PX4 comparison is thin.** One flight per mission and tracker, in still
  air (SIH has no wind), with gains tuned on a surrogate of PX4 rather than on
  PX4 itself. The statistical comparison is the reference one.
- **SITL is not bit reproducible.** It runs on the wall clock. The weekly job
  gates on behaviour and the committed numbers are from one campaign run on a
  GitHub-hosted runner.
- **The geofence breach against PX4 is injected by moving the fence**, not by
  blowing the vehicle across it, because SIH has no wind.
- **The Gazebo image is not exercised.** `sitl/gazebo.Dockerfile` (PX4 v1.15)
  is kept for watching a flight; no committed number comes from it.
- The full list, with what each costs, is in
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
