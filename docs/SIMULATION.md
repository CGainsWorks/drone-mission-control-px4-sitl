# The reference simulation

Every headline number in this repository comes out of this plant, so it is worth
being explicit about what it models and what it does not.

Read [`LIMITATIONS.md`](LIMITATIONS.md) alongside this. This page describes what
is there; that one describes what is missing.

## Why not Gazebo

PX4 SITL with Gazebo exists in this repository (`sitl/`) and works. It is not
where the numbers come from, for one reason: **a result nobody can reproduce is
not a result.** The reference simulation runs the whole scenario grid in a few
minutes, deterministically, on any machine, with no toolchain. Gazebo needs a
two-gigabyte build and does not give the same answer twice.

The trade is faithfulness. See the limitations page for what that costs.

## The vehicle

A ~1.6 kg inspection quadrotor in X configuration, roughly a 450-class airframe
with 9-inch props carrying a gimbal. Representative, not measured from any
specific vehicle.

| Parameter | Value | Note |
| --- | ---: | --- |
| Mass | 1.6 kg | |
| Inertia (xx, yy, zz) | 0.017, 0.017, 0.031 kg·m² | |
| Arm length | 0.25 m | |
| Max thrust per rotor | 8.5 N | Thrust-to-weight 2.17 |
| Rotor torque per newton | 0.016 m | Sets yaw authority |
| Motor time constant | 35 ms | |
| Linear drag (body) | 0.20, 0.20, 0.10 N per m/s | Dominated by rotor drag |
| Quadratic drag (body) | 0.10, 0.10, 0.20 N per (m/s)² | Airframe form drag |

Two consequences of those drag figures worth knowing, because both surprised the
tests before they were understood:

- **Vertical drag is not a small correction.** At 5 m/s of descent it is already
  about a third of the vehicle's weight, and terminal velocity is 8.6 m/s.
- **Cruise drag is what the MPC's disturbance estimator is there for.** At 5 m/s
  it costs 2.2 m/s² of acceleration, which the MPC's internal model knows
  nothing about.

## State and integration

Seventeen states, integrated with RK4 at 500 Hz:

| Slice | Size | Quantity |
| --- | ---: | --- |
| 0:3 | 3 | position, NED |
| 3:6 | 3 | velocity, NED |
| 6:10 | 4 | orientation quaternion, body to NED |
| 10:13 | 3 | body angular velocity |
| 13:17 | 4 | rotor thrusts |

Rotor thrusts are **state**, not an instantaneous function of the command,
because motor lag is the dominant unmodelled delay in a multirotor and it is
exactly what punishes an aggressive controller. Leaving it out would flatter the
MPC.

## The inner loop

The loop that already exists on a flight controller, modelled so the trajectory
trackers see a realistic actuator: an attitude command does not become an
attitude, it becomes a rate demand, then a torque demand, then four saturating
motors each with lag.

Structure follows PX4's `mc_att_control` and `mc_rate_control`: a proportional
quaternion attitude law feeding a PID rate law with a gyroscopic feedforward
term. **Neither tracker is allowed to touch it**, so both fly the same vehicle.

### Reduced attitude control

The attitude law does not drive the full quaternion error. It builds a *reduced*
setpoint that rotates the current attitude the short way onto the desired thrust
axis, then rotates a fraction of the way from there towards the full setpoint —
PX4's construction, and it matters for two reasons.

The first is priority: thrust direction holds the vehicle up and moves it,
heading only points the camera. The second is conditioning. At a yaw error
approaching 180 degrees the full-quaternion law is ill-posed — the scalar part
passes through zero, the shortest-path sign flips, and the rate demand chatters
between its limits. That is not theoretical; it is what lost the vehicle on a
leg flown nose-backwards during development, intermittently, which is the worst
way to find a bug.

### Control allocation

When the demand does not fit inside the rotors' range, the order things give in
is the whole design:

1. **Roll and pitch** are satisfied first — they keep the vehicle the right way
   up and point the thrust vector.
2. **Collective** next; losing a little climb rate is survivable.
3. **Yaw** last, with whatever is left.

Yaw goes last because on a multirotor it is by far the weakest axis — it comes
from rotor drag torque, not thrust differential across an arm — and because
heading is the one thing a multirotor can be wrong about for a second without
consequence.

Getting this order wrong is not subtle. An earlier version scaled all three
torque axes together, so the large yaw demand generated while turning a corner
scaled roll and pitch down with it. The vehicle stopped being able to hold the
turn, flew wide, demanded more correction, and departed. The symptom looked like
a controller tuning problem and was not.

`tests/test_quadrotor.py::TestMixer` asserts the priority directly.

## Sensors

### IMU, 200 Hz

Roughly an ICM-42688-P. Noise densities are per root-hertz as datasheets quote
them, converted to a per-sample sigma; bias random walks are per root-second.
Initial biases are drawn per run, so the filter has something real to estimate.

### GNSS, 5 Hz

The receiver model reproduces the three behaviours that actually matter to an
RTK-equipped aircraft:

- **Fix-dependent accuracy**, and the receiver reports its own figure, so an RTK
  dropout shows up as a covariance change rather than a silent degradation.
- **Slowly correlated error**, wandering on a one- to two-minute timescale.
  Averaging does not remove it, which is why a naive position loop drifts with
  it.
- **Latency.** A fix describes where the vehicle *was*. See
  [`ESTIMATOR.md`](ESTIMATOR.md) for what ignoring that costs.

Fix type follows a schedule, which is how the GNSS-degradation and GNSS-loss
scenarios are driven — data rather than a callback, so a scenario is
serialisable and appears verbatim in the results.

## Wind

A steady vector plus a first-order Gauss-Markov gust per axis: the standard
low-order stand-in for the Dryden spectrum. It has the right variance and the
right correlation time, which is what determines whether a preview horizon
helps, and it is exactly reproducible from a seed.

Exactly discretised, so the stationary variance does not depend on the step
size, and started from the stationary distribution rather than from zero, so a
short run is not dominated by spin-up.

## Battery

Deliberately simple, because the failsafe layer needs only two things from a
battery: a state of charge that falls faster under load, and a terminal voltage
that **sags under load and recovers when the load drops**.

That second property is the whole reason the low-battery failsafe is debounced.
A voltage-based estimate dips during a climb and recovers in the cruise that
follows, and a naive threshold on it triggers a return the vehicle did not need.
Modelling the sag is what makes the debounce testable instead of decorative.

Current scales with thrust to the 3/2 power, as a propeller in hover does: a
climb at 1.4x hover thrust draws about 1.65x hover current, not 1.4x.

## Determinism

Three generators — wind, IMU, GNSS — seeded from the scenario seed plus a fixed
per-stream offset. Fixed constants rather than sequential draws, so adding a
fourth noise source later cannot shift the existing three.

The consequence is that **swapping the controller cannot change the disturbance
sequence**, which is what makes the comparison paired rather than merely
averaged. `tests/test_runner.py::TestDeterminism` asserts it.

## Speed

About 3x real time for a PID run on one core, 2.5x for MPC. The full scenario
grid runs in parallel across cores; on a 16-core machine it is a few minutes.

Two optimisations account for most of that, and both were found by profiling
rather than guessed:

- **The output predictor is computed lazily.** It re-integrates the IMU buffer,
  which is O(buffer) per call, and it was being called at the IMU rate (200 Hz)
  when its result is only ever read at the control rate (50 Hz).
- **The plant's rotor torque is a matrix product**, not a loop of cross
  products — and it reuses the mixer's own matrix, so the plant and the control
  allocator cannot disagree about the vehicle's geometry.

One thing to know before running the grid in parallel: numpy's BLAS sizes its
thread pool to the core count, so fifteen worker processes would each spawn
sixteen threads. Every entry point calls
`mission_control.parallel.limit_blas_threads()` before importing numpy to cap it
at one per process. The matrices here are 15x15 and 25x25, far below where BLAS
threading pays for itself, so nothing is lost by it.
