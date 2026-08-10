"""Linear model predictive trajectory tracker.

The model
---------

Per axis, a triple state ``[position, velocity, acceleration]`` where the
commanded acceleration reaches the vehicle through a first-order lag:

.. code::

    p+ = p + v dt + a dt^2 / 2
    v+ = v + a dt
    a+ = alpha a + (1 - alpha) u,     alpha = exp(-dt / tau)

``tau`` stands in for the whole attitude loop: the time it takes a tilt command
to become a real horizontal acceleration. Including it is the difference between
an MPC that knows it is driving a lagged plant and one that is surprised by it,
and it is measured from the closed-loop step response by
``scripts/identify_lag.py`` rather than guessed.

Axes are treated as independent, which is exact for the vertical channel and an
approximation horizontally: the true coupling is through the tilt limit, and
that is applied afterwards, shared with the PID controller. The cost is that the
MPC's own constraint set is a box rather than the true cone. See
``docs/LIMITATIONS.md``.

The formulation
---------------

Condensed (dense) rather than sparse: the states are eliminated, leaving the
``N`` acceleration commands as the only decision variables. For a horizon of a
few dozen steps the dense Hessian is small and, critically, **constant** — it
depends only on the model and the weights, not on the current state — so it is
assembled once at construction. Each control step then only has to build a
gradient vector and call the solver in :mod:`mission_control.control.qp`.

What this buys over PID is preview. The MPC samples the reference across the
whole horizon, so it starts leaning into a corner before the corner produces any
error at all. That is the effect the comparison is built to measure.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np

from ..types import AttitudeSetpoint, VehicleState
from .common import acceleration_to_attitude, control_max_tilt, tilt_limit_acceleration
from .interface import ReferenceProvider, TrackerDiagnostics, TrajectoryTracker
from .qp import BoxQpSolver


@dataclass
class MpcWeights:
    """Stage and terminal costs, plus horizon and limits.

    Weights are relative, so only their ratios matter. Position dominates,
    velocity is damped an order of magnitude more weakly, and the rate-of-change
    penalty ``input_rate`` is what stops the solver commanding a sawtooth that
    the attitude loop could never follow.
    """

    horizon: int = 25
    #: Prediction step, s. Horizon length is ``horizon * step_s`` = 2.0 s by
    #: default, comfortably longer than the vehicle's settling time.
    step_s: float = 0.08
    #: Attitude-loop lag standing between a command and a real acceleration, s.
    #: Measured, not guessed: ``scripts/identify_lag.py`` fits it from the
    #: closed-loop step response at 0.127 s, consistently across step sizes.
    lag_tau_s: float = 0.13

    position_weight: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 12.0]))
    velocity_weight: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 1.2]))
    acceleration_weight: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    input_weight: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    input_rate_weight: np.ndarray = field(default_factory=lambda: np.array([0.50, 0.50, 0.50]))
    #: Terminal cost as a multiple of the stage cost. A crude stand-in for the
    #: infinite-horizon tail; enough to stop the solver parking short of the
    #: waypoint because the horizon ends there.
    terminal_scale: float = 6.0

    #: Input bounds, matching the PID's acceleration clamps so neither
    #: controller is given more authority than the other. There is deliberately
    #: no velocity constraint: the MPC bounds its *input*, and speed is limited
    #: by the reference trajectory, which both controllers follow.
    max_acceleration_xy: float = 8.0
    max_acceleration_up: float = 5.0
    max_acceleration_down: float = 3.5

    solver_iterations: int = 50
    solver_tolerance: float = 1e-8

    #: Integral gain of the disturbance estimator, 1/s^2. Chosen by the sweep in
    #: ``scripts/tune_gains.py``: large enough to clear the drag offset within a
    #: couple of seconds, small enough not to add a lightly damped mode of its
    #: own to the position loop. See :class:`MpcTracker`.
    disturbance_gain: float = 1.5
    #: Clamp on the estimated disturbance, m/s^2. Sized to cover aerodynamic
    #: drag at full speed plus a strong wind, and no more: a larger limit lets a
    #: modelling error masquerade as a disturbance.
    disturbance_limit: float = 4.0


class _AxisProblem:
    """Condensed QP for one axis: builds the constant Hessian and the gradient."""

    def __init__(self, weights: MpcWeights, axis: int):
        n = weights.horizon
        dt = weights.step_s
        alpha = float(np.exp(-dt / weights.lag_tau_s))

        a = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, alpha]])
        b = np.array([0.0, 0.0, 1.0 - alpha])

        # Prediction matrices: X = phi @ x0 + gamma @ U, with X stacking
        # states 1..N (state 0 is known and contributes no cost we can affect).
        phi = np.zeros((3 * n, 3))
        gamma = np.zeros((3 * n, n))
        power = np.eye(3)
        for k in range(n):
            power = power @ a if k > 0 else a.copy()
            phi[3 * k: 3 * k + 3, :] = power
        for k in range(n):
            for j in range(k + 1):
                # A^(k-j) @ B, i.e. the effect of u_j on state k+1.
                block = np.linalg.matrix_power(a, k - j) @ b
                gamma[3 * k: 3 * k + 3, j] = block

        stage = np.diag(
            [
                weights.position_weight[axis],
                weights.velocity_weight[axis],
                weights.acceleration_weight[axis],
            ]
        )
        q_bar = np.zeros((3 * n, 3 * n))
        for k in range(n):
            block = stage * (weights.terminal_scale if k == n - 1 else 1.0)
            q_bar[3 * k: 3 * k + 3, 3 * k: 3 * k + 3] = block

        r_bar = np.eye(n) * weights.input_weight[axis]

        # First-difference matrix, so that (D @ U - d0) is the vector of input
        # increments with d0 carrying the previously applied command.
        d = np.eye(n) - np.eye(n, k=-1)
        s_bar = np.eye(n) * weights.input_rate_weight[axis]

        # Effect of a constant unmodelled acceleration on the predicted states.
        # It acts on position and velocity directly, bypassing the actuator lag,
        # because that is what a disturbance does: the wind does not have to
        # wait for the vehicle to tilt.
        disturbance_step = np.array([0.5 * dt * dt, dt, 0.0])
        gamma_disturbance = np.zeros(3 * n)
        state = np.zeros(3)
        for k in range(n):
            state = a @ state + disturbance_step
            gamma_disturbance[3 * k: 3 * k + 3] = state

        self.n = n
        self.phi = phi
        self.gamma = gamma
        self.gamma_disturbance = gamma_disturbance
        self.q_bar = q_bar
        self.d = d
        self.s_bar = s_bar
        self._gamma_q = gamma.T @ q_bar
        self._d_s = d.T @ s_bar

        hessian = 2.0 * (self._gamma_q @ gamma + r_bar + d.T @ s_bar @ d)
        hessian = 0.5 * (hessian + hessian.T)  # kill asymmetry from round-off
        self.solver = BoxQpSolver(
            hessian,
            max_iterations=weights.solver_iterations,
            tolerance=weights.solver_tolerance,
        )

    def gradient(self, x0: np.ndarray, reference: np.ndarray, previous_input: float,
                 disturbance: float = 0.0) -> np.ndarray:
        """Linear cost term for the current state, reference window and disturbance."""
        d0 = np.zeros(self.n)
        d0[0] = previous_input
        free_response = self.phi @ x0 + self.gamma_disturbance * disturbance
        return 2.0 * (self._gamma_q @ (free_response - reference) - self._d_s @ d0)


class MpcTracker(TrajectoryTracker):
    """Condensed linear MPC with preview, box constraints and warm starting."""

    name = "mpc"

    def __init__(self, params, weights: MpcWeights = None, max_tilt: float = None):
        super().__init__(params)
        self.weights = weights or MpcWeights()
        self.max_tilt = control_max_tilt(params) if max_tilt is None else max_tilt

        self._problems = [_AxisProblem(self.weights, axis) for axis in range(3)]
        self._bounds = self._build_bounds()
        self._solution = [np.zeros(self.weights.horizon) for _ in range(3)]
        self._previous_input = np.zeros(3)
        #: Internal estimate of the acceleration the vehicle is currently
        #: achieving, propagated through the same lag the model assumes. The
        #: estimator does not provide acceleration, and differentiating its
        #: velocity would inject GNSS noise straight into the demand.
        self._acceleration_state = np.zeros(3)
        #: Estimated constant acceleration the model does not account for.
        self._disturbance = np.zeros(3)
        #: Whether the previous step's demand hit a limit. Used to freeze the
        #: disturbance estimate, exactly as the PID freezes its integrator.
        self._was_saturated = False
        self._initialised = False

    def reset(self, state: VehicleState) -> None:
        for solution in self._solution:
            solution[:] = 0.0
        self._previous_input[:] = 0.0
        self._acceleration_state[:] = 0.0
        self._disturbance[:] = 0.0
        self._was_saturated = False
        self._initialised = True

    def _build_bounds(self) -> list:
        """Per-axis input bounds. Constant, so built once."""
        w = self.weights
        n = w.horizon
        horizontal = (np.full(n, -w.max_acceleration_xy), np.full(n, w.max_acceleration_xy))
        # NED: negative is up. Climb authority is limited by thrust, descent by
        # how fast we are willing to drop, which is a comfort limit not a
        # physical one.
        vertical = (np.full(n, -w.max_acceleration_up), np.full(n, w.max_acceleration_down))
        return [horizontal, horizontal, vertical]

    def update(self, state: VehicleState, reference: ReferenceProvider,
               time_s: float, dt: float) -> AttitudeSetpoint:
        started = _time.perf_counter()
        if not self._initialised:
            self.reset(state)
        w = self.weights

        # Preview: the reference across the whole horizon, not just now.
        horizon = reference.sample_horizon(time_s, w.step_s, w.horizon)
        current = reference.sample(time_s)

        # --- disturbance estimate ------------------------------------------
        # An MPC has no integrator. Left alone it settles wherever the position
        # cost balances the input cost, which against a steady force — drag at
        # cruise, or a headwind — is a standing offset: 0.73 m on a 5 m/s
        # straight leg before this was added. Rather than bolt a correction onto
        # the output, the estimate is fed into the *model*, so the optimiser
        # plans the whole horizon knowing the force is there.
        #
        # Position error is the quantity that must go to zero, so it is the one
        # integrated. Note the **sign**: falling behind the reference means the
        # position error ``ref - p`` is positive, and the disturbance
        # responsible is pushing the *other* way, so the estimate must decrease.
        # Getting this backwards is not a subtle degradation — the estimate runs
        # to its clamp and stays there, and the tracking error settles at two
        # metres instead of two centimetres.
        #
        # An earlier attempt compared achieved against model-predicted
        # acceleration instead. That form cannot distinguish a real disturbance
        # from a mismatch between the model's actuator lag and the vehicle's,
        # and feeds that confusion back through the optimiser.
        #
        # Frozen while saturated, exactly as the PID freezes its integrator, so
        # neither controller winds up against a limit it cannot push past.
        position_error = current.position - np.asarray(state.position, dtype=float)
        if not self._was_saturated:
            self._disturbance = np.clip(
                self._disturbance - w.disturbance_gain * position_error * dt,
                -w.disturbance_limit,
                w.disturbance_limit,
            )

        # Flatten the horizon into a (3, 3N) block once, rather than rebuilding
        # a 3N vector inside the per-axis loop. The reference window is the same
        # object for all three axes; only the column read out of it differs.
        window = np.empty((w.horizon, 3, 3))
        for k, setpoint in enumerate(horizon):
            window[k, 0] = setpoint.position
            window[k, 1] = setpoint.velocity
            window[k, 2] = setpoint.acceleration
        # (horizon, state, axis) -> (axis, horizon * state), matching the
        # condensed prediction's stacking of [p, v, a] per step.
        reference_vectors = np.ascontiguousarray(window.transpose(2, 0, 1)).reshape(3, -1)

        command = np.zeros(3)
        iterations = 0
        solve_time = 0.0
        for axis in range(3):
            problem = self._problems[axis]
            x0 = np.array(
                [state.position[axis], state.velocity[axis], self._acceleration_state[axis]]
            )
            gradient = problem.gradient(
                x0, reference_vectors[axis], self._previous_input[axis],
                self._disturbance[axis],
            )
            lower, upper = self._bounds[axis]

            # Warm start from the previous solution shifted one step, with the
            # tail held. On a smooth trajectory this is close to optimal already.
            warm = np.empty(w.horizon)
            warm[:-1] = self._solution[axis][1:]
            warm[-1] = self._solution[axis][-1]

            solve_started = _time.perf_counter()
            solution = problem.solver.solve(gradient, lower, upper, warm_start=warm)
            solve_time += _time.perf_counter() - solve_started
            self._solution[axis] = solution
            iterations = max(iterations, problem.solver.last_iterations)
            command[axis] = solution[0]

        self._previous_input = command.copy()

        acceleration, tilt_saturated = tilt_limit_acceleration(command, self.max_tilt)
        at_bound = bool(
            np.any(np.abs(command[0:2]) >= w.max_acceleration_xy - 1e-6)
            or command[2] <= -w.max_acceleration_up + 1e-6
            or command[2] >= w.max_acceleration_down - 1e-6
        )

        # Propagate the internal acceleration state with the control step, not
        # the prediction step, so it stays consistent with how often we re-solve.
        decay = float(np.exp(-max(dt, 1e-6) / w.lag_tau_s))
        self._acceleration_state = (
            decay * self._acceleration_state + (1.0 - decay) * acceleration
        )

        self._was_saturated = bool(tilt_saturated or at_bound)
        self.diagnostics = TrackerDiagnostics(
            update_time_s=_time.perf_counter() - started,
            solve_time_s=solve_time,
            iterations=iterations,
            saturated=self._was_saturated,
            acceleration_demand=acceleration.copy(),
            velocity_setpoint=current.velocity.copy(),
        )
        return acceleration_to_attitude(acceleration, current.yaw, self.params,
                                        yaw_rate=current.yaw_rate)

    def predicted_trajectory(self, state: VehicleState) -> np.ndarray:
        """Positions the solver expects to fly, ``(horizon, 3)``.

        Not used by the control loop. It exists so the notebooks can draw what
        the MPC was planning at a given instant next to what actually happened,
        which is the only way to tell a tracking failure from a planning one.
        """
        w = self.weights
        out = np.zeros((w.horizon, 3))
        for axis in range(3):
            problem = self._problems[axis]
            x0 = np.array(
                [state.position[axis], state.velocity[axis], self._acceleration_state[axis]]
            )
            # The disturbance term must be included, or this draws a plan the
            # optimiser never made: the solution was chosen knowing about the
            # disturbance, so predicting without it shows the wrong trajectory.
            predicted = (
                problem.phi @ x0
                + problem.gamma @ self._solution[axis]
                + problem.gamma_disturbance * self._disturbance[axis]
            )
            out[:, axis] = predicted[0::3]
        return out

    def describe(self) -> dict:
        w = self.weights
        return {
            "name": self.name,
            "horizon": w.horizon,
            "step_s": w.step_s,
            "horizon_s": w.horizon * w.step_s,
            "lag_tau_s": w.lag_tau_s,
            "position_weight": w.position_weight.tolist(),
            "velocity_weight": w.velocity_weight.tolist(),
            "input_weight": w.input_weight.tolist(),
            "input_rate_weight": w.input_rate_weight.tolist(),
            "terminal_scale": w.terminal_scale,
            "disturbance_gain": w.disturbance_gain,
            "max_acceleration_xy": w.max_acceleration_xy,
            "max_tilt_deg": float(np.degrees(self.max_tilt)),
            "solver": "projected-Newton box QP",
        }
