# Limitations

What this repository does not do, does not model, or models in a way that would
not survive contact with a real aircraft. Written plainly, because a portfolio
project that only lists its strengths is not evidence of engineering judgement.

## The headline result comes from a simulation, not from PX4 SITL

The MPC-versus-PID comparison runs against the reference plant in
`src/mission_control/sim/`, not against PX4 with Gazebo. That is a deliberate
trade and it is the most important caveat here.

**Why:** the reference simulation is deterministic, runs the full scenario grid
in a few minutes on any machine, and needs no toolchain. A benchmark that
requires a two-gigabyte PX4 build and produces a different answer every run is
not one anybody will check.

**What it costs:** the plant is a 6-DOF rigid body with rotor lag, quadratic
drag and a wind model. It does not model aerodynamic interaction between rotors,
ground effect, vortex ring state in fast descents, structural flexibility,
airframe vibration coupling into the IMU, or ESC nonlinearity. Gazebo models
some of these; a real aircraft has all of them.

**What that means for the numbers:** the *ranking* of the two controllers is
robust — the mechanism by which the MPC wins is preview, which does not depend
on any of the above. The *magnitudes* are optimistic for both controllers, and
by an unquantified amount. Do not read the absolute RMS figures as what either
controller would achieve on a real aircraft.

`sitl/` and `scripts/run_px4_sitl.py` exist and work; they demonstrate the
architecture is backend-agnostic. They are not the source of the numbers.

## The estimator

- **No magnetometer, no vision, no optical flow, no barometer.** GNSS and IMU
  only. That makes yaw weakly observable — it converges only through
  acceleration, and the tests hold it to 6 degrees rather than the 1 degree that
  roll and pitch achieve. A real inspection aircraft would carry at least a
  magnetometer, and near a steel structure would have to cope with it being
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
  acceleration, but the vehicle's real constraint is a cone on total tilt. The
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
  has just been inspecting. This is the same behaviour PX4 ships with, and it is
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

- Results are deterministic given a seed, and CI checks that regenerating them
  on a clean checkout leaves the tree unchanged.
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
