# Limitations

What this repository does not do, does not model, or models in a way that would
not survive contact with a real aircraft. Written plainly, because a portfolio
project that only lists its strengths is not evidence of engineering judgement.

## Everything is simulated

Nothing in this repository has flown. There are two simulations and every
number comes from one of them.

**The headline comparison** (84 runs, `docs/RESULTS.md`) runs against the
reference plant in `src/mission_control/sim/`. It is deterministic, runs the
whole grid on any machine and needs no toolchain, which is why it carries the
statistics. The plant is a 6-DOF rigid body with rotor lag, quadratic drag and
a wind model. It does not model aerodynamic interaction between rotors, ground
effect, vortex ring state in fast descents, structural flexibility, airframe
vibration coupling into the IMU or ESC nonlinearity.

**The PX4 campaign** (`docs/SITL_RESULTS.md`) flies the same stack over MAVLink
against PX4 v1.17.0 with PX4's SIH model. That adds a real autopilot: PX4's
attitude and rate loops, its EKF2, its failsafe state machine and a real
MAVLink link with real latency. SIH itself is simpler than the reference
plant: a rigid body with linear drag and **no wind model**, so every SITL
flight is in still air.

**What that means for the numbers:** the *ranking* in the reference comparison
is robust, because the mechanism by which the MPC wins is preview, which does
not depend on any of the above. The *magnitudes* are optimistic for both
controllers, by an unquantified amount. Do not read either set of RMS figures
as what a controller would achieve on an aircraft.

## The PX4 campaign is thin

- **One flight per mission and tracker.** SITL runs in real time and is not
  bit reproducible, so each number is a single sample. The statistical
  comparison is the reference one.
- **Gains tuned on a surrogate.** Both trackers' SITL gains come from the
  reference tuning procedure run on the reference simulation with SIH's
  airframe and PX4's measured command delay (`scripts/tune_gains.py --airframe
  sih`), not from tuning against PX4 itself. A first-order lag plus a delay is
  a coarse model of PX4's attitude loop, MAVLink and EKF2 together.
- **The geofence breach is injected by moving the fence.** The reference
  simulation forces a real breach with a wind the airframe cannot hold
  against; SIH has no wind and the predictive fence stops the vehicle before
  any boundary it can see coming. So the SITL row tightens the fence in flight
  to a square the vehicle is already outside, which is how an airspace
  restriction arriving mid-flight would look and tests the breach rule and the
  return. It does not test a vehicle being blown across a boundary.
- **Not every row is flown against PX4.** Link loss, low battery, GNSS loss and
  geofence breach are. The permanent link loss, critical battery, degraded RTK
  and predicted-breach rows are flown only in the reference simulation.
- **The simulation runs on the wall clock.** PX4 is built nolockstep so that
  an external real-time controller and the simulated vehicle agree about time.
  The price is that a busy host can stall the simulator, which PX4 reports as a
  sensor timeout and which degrades its EKF. Stalled flights are detected from
  the ULog and re-flown and the committed flights were all flown on a
  GitHub-hosted runner, where none stalled; on the shared development machine
  most flights did.
- **The Gazebo image is not exercised.** `sitl/gazebo.Dockerfile` (PX4 v1.15
  with Gazebo) is kept for watching a flight. No committed number comes from
  it and CI does not build it.

## The estimator

- **No magnetometer, no vision, no optical flow, no barometer.** GNSS and IMU
  only. That makes yaw weakly observable — it converges only through
  acceleration and the tests hold it to 6 degrees rather than the 1 degree that
  roll and pitch achieve. A real inspection aircraft would carry at least a
  magnetometer and near a steel structure would have to cope with it being
  wrong.
- **GNSS is modelled, not replayed.** The receiver model reproduces the three
  behaviours that matter (fix-dependent accuracy, slowly correlated error,
  latency) but it is a model. Real multipath near a building is worse and
  nastier than an Ornstein-Uhlenbeck process.
- **The filter is deliberately conservative.** Measurement noise is inflated by
  1.5x over what the receiver reports, because the receiver does not report its
  correlated bias. The consequence is a mean NIS below the theoretical 3: the
  filter discards some information in exchange for not rejecting good fixes.
  That trade is defensible and it is a trade.

## The controllers

- **The MPC's axes are decoupled.** Its constraint set is a box on per-axis
  acceleration but the vehicle's real constraint is a cone on total tilt. The
  tilt limit is applied afterwards, shared with the PID, so neither controller
  is advantaged — but the MPC is solving a slightly wrong problem and could do
  better with a proper second-order cone constraint.
- **The MPC's model has no drag.** It is a triple integrator with a first-order
  input lag. Drag is handled by the disturbance estimator, which is an
  integrator and therefore slower than modelling it would be.
- **Neither controller is exhaustively tuned.** The search in
  `scripts/tune_gains.py` covers tens of candidates, not thousands. A determined
  practitioner could find better gains for either. The defence is that both were
  tuned by the same procedure with the same budget, so the comparison is fair
  even if neither is optimal.
- **Gains are fixed for the whole flight.** No gain scheduling with speed or
  altitude, which a production controller would have.

## The safety layer

- **The geofence is a single polygon plus an altitude band.** No exclusion
  zones, no multiple polygons, no terrain following. A real inspection site
  typically needs all three.
- **RTL does not avoid anything.** It climbs, transits straight home and
  descends. It has no map and no sensor with which to notice the structure it
  has just been inspecting. This is the same behaviour PX4 ships with and it is
  the reason RTL altitudes are set conservatively in practice.
- **The failsafe matrix trusts its inputs.** A rule fires on the state it is
  given; there is no cross-checking of one sensor against another. A GNSS
  receiver reporting a confident but wrong position would not be caught.

## The simulation harness

- **The vehicle cannot crash.** Ground contact clamps the vehicle at zero
  altitude and zeroes its velocity. There is no impact model, so a run that
  would have destroyed the aircraft is scored as a landing. The divergence guard
  catches gross failures; a hard arrival is not distinguished from a soft one.
- **Wind is uniform in space.** It varies in time but is the same everywhere,
  so there is no gradient, no wake behind the structure and no gust front —
  which is precisely the environment inspection flying takes place in.
- **The battery model is crude.** State of charge is coulomb-counted and the
  terminal voltage sags with an internal resistance. It has no temperature
  dependence, no ageing and no per-cell behaviour.

## Reproducibility

- Reference results are deterministic given a seed. The weekly CI run checks
  that regenerating `docs/RESULTS.md` and `docs/FAILSAFE_MATRIX.md` on a clean
  checkout leaves them unchanged; the per-push run checks that every number the
  README quotes matches those committed documents.
- PX4 SITL results are not bit reproducible: SITL runs against the wall clock
  and MAVLink timing varies. The weekly CI run re-flies the campaign and gates
  on behaviour (every flight completes within the tracking gate, every fault
  produces its specified action), not on the committed numbers.
- The figures depend on matplotlib's default styling, so a major matplotlib
  version change may alter them cosmetically without changing any number.
- Timing figures (`mean_solve_time_us` and friends) are the one thing that is
  **not** reproducible across machines — they measure the machine as much as the
  code. They are reported because the relative cost of MPC against PID is the
  point, not the absolute microseconds.

## Not addressed at all

- Certification, airworthiness, or any assurance argument. See
  [`CAA_NOTES.md`](CAA_NOTES.md) for what real flight would require.
- Multi-vehicle coordination.
- Payload or gimbal control, or anything to do with actually capturing
  inspection imagery.
- State estimation of the *structure* being inspected.
- Cybersecurity of the MAVLink link, which is unauthenticated by default.
