# The two trajectory trackers

Both consume a state estimate and a reference, and both emit the same
`AttitudeSetpoint`: an orientation, a body-rate feedforward and a normalised
collective thrust. That is deliberately the MAVLink `SET_ATTITUDE_TARGET`
payload, so the PX4 backend forwards it unchanged and the comparison is
like-for-like at the interface.

## What they share

Everything below the control law:

- **The acceleration-to-attitude conversion.** An acceleration demand plus a
  yaw demand determine an attitude uniquely; that step is differential flatness,
  not control design. It lives in `control/common.py` and both call it. If each
  rolled its own, part of any measured difference would be that conversion.
- **The tilt limit.** 75% of the airframe's physical maximum, about 47 degrees.
  Below the physical limit on purpose: at the limit there is no thrust left to
  hold altitude, so a controller that plans to fly there trades height for
  horizontal error without ever being asked to.
- **The reference object.** Both are handed the whole trajectory, not a
  pre-sliced setpoint. The MPC samples its full horizon; the PID samples only
  the present. That is a choice each controller makes, not something the harness
  did for it.
- **Integral authority.** Both have it, at comparable strength. Giving one a way
  to reject a steady disturbance and not the other would decide the comparison
  before it started.
- **The tuning objective, and its constraint.** Both are tuned to minimise RMS
  position error *subject to the same jerk budget on the acceleration demand*.
  The constraint is what makes the problem well posed: minimising error alone
  drives any controller to the edge of stability, and the search ran to the edge
  of every range it was given until the budget was added. The budget is derived
  from the attitude loop's 220 deg/s rate limit, which caps how fast a tilt —
  and therefore a horizontal acceleration — can change.

## PID

Structure is PX4's `mc_pos_control`: a proportional outer position loop
producing a velocity setpoint, and a PID inner velocity loop producing an
acceleration demand.

Gains are **placed, not hand-turned**. The search in `scripts/tune_gains.py` is
over a closed-loop natural frequency and damping ratio per axis, which gives

```
velocity_p = 2 * zeta * wn
position_p = wn^2 / velocity_p
```

so the tuning is two interpretable numbers rather than six opaque ones, and the
result can be stated as a bandwidth rather than as a list.

The search finds a genuine interior optimum rather than running out of grid:
pushing the horizontal bandwidth one step beyond the winner both raises the
tracking error and blows through the jerk budget, because the loop is by then
fighting the actuator lag rather than the trajectory. Vertical bandwidth turns
out to be an insensitive parameter — the whole vertical range changes the result
by well under one percent — which is what you would expect when thrust acts on
that axis directly with no tilt dynamics in the way.

Details that matter more than they look:

- **Derivative on the measurement, not the error.** A step in the velocity
  setpoint — which happens every time the mission sequences to a new leg — must
  not produce a torque spike.
- **Conditional integration.** The integrator freezes while the demand is
  saturated, but is still allowed to unwind in the direction that relieves the
  saturation. Freezing it outright means it can never recover; not freezing it
  means seconds of overshoot after a long climb.
- **Feedforward is enabled.** Reference velocity and acceleration are fed
  forward at unity gain. Without this the PID would be a straw man, and the
  comparison would be measuring the feedforward rather than the controller.

Its one structural limitation, and the thing the comparison is designed to
expose: **it is causal in the reference.** It reacts to the error it has now. It
cannot lean into a corner before reaching it, because at the moment it would
need to start leaning, the error is still zero.

## MPC

Per axis, a triple state `[position, velocity, acceleration]` where the
commanded acceleration reaches the vehicle through a first-order lag:

```
p+ = p + v dt + a dt^2 / 2
v+ = v + a dt
a+ = alpha a + (1 - alpha) u,     alpha = exp(-dt / tau)
```

`tau` stands in for the whole attitude loop — the time between asking for a tilt
and getting the horizontal acceleration that tilt produces. It is **measured**,
not guessed: `scripts/identify_lag.py` fits it from the closed-loop step
response at **0.127 s**, consistently across step sizes from 1 to 6 m/s².

### Condensed, not sparse

The states are eliminated, leaving the `N` acceleration commands as the only
decision variables. The dense Hessian that results is small and, critically,
**constant** — it depends only on the model and the weights, not on the current
state — so it is assembled once at construction. Each control step then only has
to build a gradient vector and solve.

(The solver factorises the *free block* of that Hessian per iteration, since
which variables are free changes as constraints activate. At this size that
costs microseconds; the saving is in never rebuilding the Hessian itself.)

### The solver

Projected Newton with an Armijo line search on the projected arc (Bertsekas,
1982), in `control/qp.py`.

ADMM was tried first and is a poor fit. A condensed MPC Hessian is
ill-conditioned — position is integrated twice from acceleration, so the cheap
and expensive directions differ by orders of magnitude — and one scalar penalty
parameter cannot serve both ends of that spread. It was still at 4e-4 KKT
residual after 60 iterations and about a millisecond per axis.

Projected Newton identifies the active set in a few iterations and then
terminates finitely at the exact solution. Measured by
`scripts/benchmark_qp.py` on the shipped configuration:

| | Time per axis | Iterations |
| --- | ---: | ---: |
| warm-started from the previous step | 21.5 us | 1 |
| cold | 589.8 us | 7 |

with a KKT residual of 8.5e-14 at the solution, and worst-case 5.6e-15 against
`scipy.optimize` across forty random problems. Three axes per control step is
64 us of a 20 000 us budget.

One detail that is easy to get wrong: the convergence tolerance is **relative to
the problem scale**. The driver is gradient magnitude rather than conditioning —
the position weights are in the hundreds and the reference is in tens of metres,
so the linear term runs to six figures, and an absolute 1e-8 on that is below
what float64 can resolve. The solver would grind to its iteration cap every
step.

### Offset-free tracking

An MPC has no integrator. Left alone it settles wherever the position cost
balances the input cost, which against a steady force — drag at cruise, or a
headwind — is a standing offset. Measured at **0.73 m** on a 5 m/s straight leg
before this was added.

The fix is not to bolt a correction onto the output but to tell the model the
truth: a disturbance term is estimated and fed into the *prediction*, so the
optimiser plans the whole horizon knowing the force is there.

Two forms were tried and discarded, and both failures are instructive:

- **Comparing achieved against model-predicted acceleration.** Cannot
  distinguish a real disturbance from a mismatch between the model's actuator
  lag and the vehicle's, and feeds that confusion back through the optimiser.
- **The wrong sign.** Falling behind the reference means the position error
  `ref - p` is positive and the disturbance responsible is pushing the *other*
  way. Getting this backwards is not a subtle degradation: the estimate runs to
  its clamp and stays there, and the tracking error settles at two metres
  instead of two centimetres.

What it does now is integrate position error, with the sign that makes the
estimate converge to the true disturbance, frozen while saturated exactly as the
PID's integrator is.

### Horizon length

In the tuning search and worth stating: **1.6 s and 2.0 s of preview score
identically to four decimal places** on these missions. The vehicle's settling
time is well under a second and nothing useful happens beyond that. The horizon
is in the search so that this is measured rather than assumed — and it means the
MPC's advantage comes from having *any* preview, not from having a long one.

## Where the difference comes from

The MPC's structural advantage is preview: it samples the reference across its
whole horizon, so it starts leaning into a corner before the corner has produced
any error at all.

That predicts the gap should be largest where the reference changes direction
sharply and at speed, and smallest on long straight legs where there is nothing
to preview. [`RESULTS.md`](RESULTS.md) reports the split by scenario so that
prediction can be checked rather than asserted.

The cost is real and is reported alongside: roughly an order of magnitude more
compute per control step, a QP solver that has to exist on the vehicle, and more
parameters to get wrong.
