#!/usr/bin/env python3
"""Measure and verify the MPC's QP solver.

Two things at once, because they belong together: the solver's *accuracy*,
checked against the KKT conditions and against ``scipy.optimize`` on random
problems, and its *cost* on the actual problem the MPC solves. A solver that is
fast and slightly wrong is worse than useless in a control loop, because the
error looks like a tuning problem.

The timing figure is the one number in this repository that is genuinely
machine-dependent — it measures the computer as much as the code. It is quoted
anyway, because the relative cost of MPC against PID is part of the finding, and
it is produced by this script rather than typed into a document.

Run on an otherwise idle machine:  python scripts/benchmark_qp.py
"""

from __future__ import annotations

import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control.parallel import limit_blas_threads  # noqa: E402

limit_blas_threads()

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from mission_control.control.mpc_tracker import _AxisProblem  # noqa: E402
from mission_control.control.qp import BoxQpSolver  # noqa: E402
from mission_control.tuning import mpc_weights  # noqa: E402

RNG = np.random.default_rng(20260810)


def check_accuracy(trials: int = 40) -> tuple:
    """Random problems, checked two independent ways."""
    worst_kkt = 0.0
    worst_gap = 0.0
    for _ in range(trials):
        size = int(RNG.integers(3, 30))
        root = RNG.normal(size=(size, size))
        hessian = root @ root.T + 0.5 * np.eye(size)
        hessian = 0.5 * (hessian + hessian.T)
        gradient = RNG.normal(size=size)
        lower = -np.abs(RNG.normal(size=size))
        upper = np.abs(RNG.normal(size=size))

        solver = BoxQpSolver(hessian, max_iterations=200, tolerance=1e-12)
        solution = solver.solve(gradient, lower, upper)

        reference = minimize(
            lambda x: 0.5 * x @ hessian @ x + gradient @ x,
            np.zeros(size),
            jac=lambda x: hessian @ x + gradient,
            bounds=list(zip(lower, upper)),
            method="L-BFGS-B",
            options=dict(ftol=1e-16, gtol=1e-14, maxiter=20000),
        )
        worst_kkt = max(worst_kkt, solver.kkt_residual(solution, gradient, lower, upper))
        worst_gap = max(
            worst_gap,
            abs(solver.objective(solution, gradient)
                - (0.5 * reference.x @ hessian @ reference.x + gradient @ reference.x)),
        )
    return worst_kkt, worst_gap


def benchmark(repeats: int = 5000) -> dict:
    """Cost of one axis solve on the MPC's real problem, warm and cold."""
    weights = mpc_weights()
    problem = _AxisProblem(weights, 0)

    x0 = np.array([1.0, 0.5, 0.0])
    reference = np.tile([2.0, 0.0, 0.0], weights.horizon)
    gradient = problem.gradient(x0, reference, 0.0, 0.0)
    lower = np.full(weights.horizon, -weights.max_acceleration_xy)
    upper = np.full(weights.horizon, weights.max_acceleration_xy)

    warm = problem.solver.solve(gradient, lower, upper)
    residual = problem.solver.kkt_residual(warm, gradient, lower, upper)

    started = time.perf_counter()
    for _ in range(repeats):
        problem.solver.solve(gradient, lower, upper, warm_start=warm)
    warm_us = (time.perf_counter() - started) / repeats * 1e6
    warm_iterations = problem.solver.last_iterations

    started = time.perf_counter()
    for _ in range(repeats):
        problem.solver.solve(gradient, lower, upper)
    cold_us = (time.perf_counter() - started) / repeats * 1e6
    cold_iterations = problem.solver.last_iterations

    return {
        "horizon": weights.horizon,
        "warm_us": warm_us,
        "warm_iterations": warm_iterations,
        "cold_us": cold_us,
        "cold_iterations": cold_iterations,
        "kkt_residual": residual,
        "condition_number": float(np.linalg.cond(problem.solver.hessian)),
        # The scale of the linear term is what forces the relative convergence
        # tolerance, and it is much larger than the Hessian's conditioning would
        # suggest: the weights are in the hundreds and the reference is in tens
        # of metres, so the product runs to six figures.
        "gradient_scale": float(np.max(np.abs(gradient))),
    }


def main() -> int:
    print("QP solver: projected Newton with an Armijo line search on the projected arc")
    print(f"  machine: {platform.processor() or platform.machine()}, "
          f"Python {platform.python_version()}, numpy {np.__version__}")
    print()

    worst_kkt, worst_gap = check_accuracy()
    print("Accuracy on 40 random box-constrained QPs, sizes 3 to 30:")
    print(f"  worst KKT residual                 {worst_kkt:.3e}")
    print(f"  worst objective gap vs L-BFGS-B    {worst_gap:.3e}")
    print()

    result = benchmark()
    print(f"Cost on the MPC's own problem (horizon {result['horizon']}, "
          f"one axis of three):")
    print(f"  Hessian condition number           {result['condition_number']:.3e}")
    print(f"  gradient magnitude                 {result['gradient_scale']:.3e}")
    print(f"  warm-started  {result['warm_us']:8.1f} us  "
          f"({result['warm_iterations']} iteration(s))")
    print(f"  cold          {result['cold_us']:8.1f} us  "
          f"({result['cold_iterations']} iteration(s))")
    print(f"  KKT residual at the solution       {result['kkt_residual']:.3e}")
    print()
    print("  Three axes are solved per control step, at 50 Hz, so the control")
    print(f"  period budget is 20 000 us and the optimisation uses "
          f"{3 * result['warm_us']:.0f} us of it. Warm starting is worth "
          f"{result['cold_us'] / max(result['warm_us'], 1e-9):.0f}x here.")
    print()
    print("  The gradient magnitude, not the conditioning, is what forces the")
    print("  convergence tolerance to be relative rather than absolute: an")
    print("  absolute 1e-8 on a gradient of this size is below what float64 can")
    print("  resolve, and the solver would grind to its iteration cap every step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
