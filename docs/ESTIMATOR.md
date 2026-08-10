# The RTK-GNSS / IMU estimator

Both controllers fly on this filter, so it is not a difference between them —
but if it were poor, neither result would mean anything. This is what it does
and why it is built the way it is.

## Why an error-state filter

Attitude has three degrees of freedom; a quaternion has four components and a
unit-norm constraint. A direct EKF on the quaternion has to carry a covariance
that is singular in one direction and renormalise the state every step, which
quietly corrupts the uncertainty it is supposed to be tracking.

The error-state formulation splits the problem. A **nominal** state carries the
quaternion and is propagated with the IMU. An **error** state carries a minimal
three-parameter attitude perturbation whose covariance is well defined, and is
injected into the nominal state once per update and reset to zero. This is the
same structure as PX4's `ekf2`, minus the sensors this vehicle does not carry.

The 15 error states:

| Index | Quantity | Why it is in the filter |
| --- | --- | --- |
| 0:3 | position error, NED | The thing being estimated. |
| 3:6 | velocity error, NED | Observable from GNSS, and needed by the controller. |
| 6:9 | attitude error, rotation vector | Determines how the IMU projects into NED. |
| 9:12 | accelerometer bias | 0.05 m/s² uncorrected is 2.5 m after ten seconds of dead reckoning. |
| 12:15 | gyroscope bias | Feeds straight into attitude drift. |

## The two clocks

This is the part worth understanding, because it is worth a factor of five in
position accuracy and it is invisible until you look for it.

A GNSS fix describes where the vehicle **was**, typically 50 to 100 ms ago.
Fusing it as though it described *now* drags the estimate backwards along the
flight path by speed times latency. At 4 m/s with 80 ms of latency that is
0.32 m — sixteen times the noise of an RTK-fixed solution. An estimator that
ignores latency throws away most of what RTK is for, and feeds the position
controller a lagged measurement that costs phase margin and shows up as an
oscillation nobody can tune out.

So there are two clocks:

- The **fusion horizon** runs `fusion_delay_s` behind real time. IMU samples are
  buffered and only propagated into the EKF once they are older than that, so
  when a fix arrives the filter state is already at the right epoch and the fix
  fuses with no time skew.
- The **output predictor** takes the horizon state and integrates the still-
  buffered IMU forward to the present. Controllers consume this, so they see a
  current, low-latency state carrying the horizon's bias corrections.

Measured by `scripts/benchmark_estimator.py`: 60 s of two-axis excitation,
RTK-fixed throughout, 5 Hz fixes with 80 ms of latency, both configurations
flying the identical trajectory with identical noise seeds. Mean ground speed
2.26 m/s, so speed times latency is 0.18 m — that is the error being removed.

| | Position RMS | Velocity RMS | Worst |
| --- | ---: | ---: | ---: |
| Fusing each fix as though it were current | 0.222 m | 0.141 m/s | 0.369 m |
| Fusion horizon plus output predictor | 0.050 m | 0.033 m/s | 0.101 m |

A 4.4x improvement in position and 4.3x in velocity, for one buffer and one
extra integration pass.

`tests/test_estimation.py::TestLatencyCompensation` asserts the compensated
filter is at least three times better, so this cannot regress silently.

## Consistency, and why the NIS is below 3

A consistent filter's normalised innovation squared averages the number of
degrees of freedom in the update — 3, for a position fix. This filter runs
below that, and the reason is deliberate.

The receiver reports its nominal accuracy: 2 cm horizontal for an RTK-fixed
solution. It does **not** report its slowly-wandering multipath and ionospheric
error, because it cannot observe it. Believing the reported figure exactly makes
the filter overconfident, and an overconfident filter starts rejecting honest
fixes at its innovation gate — at which point it dead-reckons, drifts, and
rejects even harder. That failure mode is a sawtooth in the position error and
it is unpleasant to diagnose.

So the measurement noise is inflated by 1.5x over what the receiver claims. The
cost is a NIS below 3, meaning some information is discarded. The benefit is
that the gate stays open. That is a trade, it is stated here, and
`tests/test_estimation.py` holds the NIS to a band rather than a point so the
trade is visible rather than tuned away.

## Fix types and what they mean

| Fix | Horizontal 1-sigma | Vertical | Velocity | Correlated bias |
| --- | ---: | ---: | ---: | ---: |
| RTK fixed | 0.02 m | 0.03 m | 0.03 m/s | 0.01 m |
| RTK float | 0.25 m | 0.40 m | 0.08 m/s | 0.30 m |
| 3D | 1.20 m | 2.00 m | 0.25 m/s | 0.90 m |

The correlated bias column is why averaging does not rescue a non-RTK solution:
the error wanders on a one- to two-minute timescale rather than being white, so
it looks like real motion to a position loop.

RTK-fixed decays to float when the correction stream goes stale, which is what a
real receiver does when the RTCM link stalls. That degradation is **not** a
failsafe event — see [`FAILSAFE_MATRIX.md`](FAILSAFE_MATRIX.md). Losing
centimetre accuracy is a nuisance; losing position entirely is an emergency, and
conflating them would ground the aircraft every time a mobile network hiccuped.

## What is weakly observable

**Yaw.** There is no magnetometer and no moving-baseline heading, so yaw is only
observable through acceleration: the filter learns its heading by comparing
where it thought it was going with where GNSS says it went. On a gentle
trajectory that takes a while, and the tests hold yaw to 6 degrees where roll
and pitch are held to 1. This is a property of the sensor suite, not slack in
the tests, and it is called out in [`LIMITATIONS.md`](LIMITATIONS.md).

**Accelerometer bias** is partially confounded with attitude error — a small
tilt and a small bias produce the same horizontal acceleration — so it converges
more slowly and less completely than gyro bias does. Exciting the vehicle in
more than one direction is what separates them, which is why the estimator tests
fly a two-axis pattern rather than a straight line.

## Failure handling

- **Innovation gating.** A fix whose NIS exceeds the 0.999 chi-square quantile
  is rejected. Tight enough to reject a jump, loose enough not to reject honest
  noise.
- **Gate lockout recovery.** Ten consecutive rejections means the *filter* is
  wrong, not the sensor. Position and velocity covariance are inflated so the
  next fix pulls the estimate back. Without this, a genuine position jump — a
  base station moved, a cold start — locks the filter out permanently.
- **Joseph-form covariance update.** One extra matrix product, in exchange for
  the covariance provably staying symmetric positive-definite when the gain is
  large. `tests/test_estimation.py` checks the eigenvalues after a full flight.
- **Covariance reset on injection.** After the attitude error is folded into the
  quaternion, its covariance is rotated to match the new frame. Skipping this is
  a common shortcut and it makes the filter mildly inconsistent.
