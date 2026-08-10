"""The box-constrained QP solver the MPC depends on.

The solver is checked two independent ways — against the KKT conditions, and
against ``scipy.optimize`` — because it is the one component whose failure would
be silent. A slightly wrong QP solution still produces a plausible-looking
acceleration, and the MPC would just track a little worse for a reason nobody
could find.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from mission_control.control.qp import BoxQpSolver

RNG = np.random.default_rng(4242)


def random_problem(size, conditioning=1.0):
    root = RNG.normal(size=(size, size))
    hessian = root @ root.T + conditioning * np.eye(size)
    hessian = 0.5 * (hessian + hessian.T)
    gradient = RNG.normal(size=size) * 10.0
    lower = -np.abs(RNG.normal(size=size)) - 0.1
    upper = np.abs(RNG.normal(size=size)) + 0.1
    return hessian, gradient, lower, upper


def reference_solution(hessian, gradient, lower, upper):
    result = minimize(
        lambda x: 0.5 * x @ hessian @ x + gradient @ x,
        np.zeros(hessian.shape[0]),
        jac=lambda x: hessian @ x + gradient,
        bounds=list(zip(lower, upper)),
        method="L-BFGS-B",
        options=dict(ftol=1e-16, gtol=1e-14, maxiter=20000),
    )
    return result.x


@pytest.mark.parametrize("size", [1, 2, 5, 12, 25, 40])
def test_satisfies_kkt_conditions(size):
    for _ in range(10):
        hessian, gradient, lower, upper = random_problem(size)
        solver = BoxQpSolver(hessian)
        solution = solver.solve(gradient, lower, upper)
        assert np.all(solution >= lower - 1e-12)
        assert np.all(solution <= upper + 1e-12)
        scale = max(1.0, float(np.max(np.abs(gradient))))
        assert solver.kkt_residual(solution, gradient, lower, upper) < 1e-6 * scale


@pytest.mark.parametrize("size", [3, 10, 25])
def test_matches_scipy(size):
    for _ in range(10):
        hessian, gradient, lower, upper = random_problem(size)
        solver = BoxQpSolver(hessian)
        solution = solver.solve(gradient, lower, upper)
        reference = reference_solution(hessian, gradient, lower, upper)
        ours = solver.objective(solution, gradient)
        theirs = 0.5 * reference @ hessian @ reference + gradient @ reference
        # Compare objectives, not arguments: on a flat problem two very
        # different points can both be optimal.
        assert ours <= theirs + 1e-8 * max(1.0, abs(theirs))


def test_unconstrained_problem_matches_direct_solve():
    hessian, gradient, _, _ = random_problem(15)
    solver = BoxQpSolver(hessian)
    huge = np.full(15, 1e6)
    solution = solver.solve(gradient, -huge, huge)
    assert np.allclose(solution, np.linalg.solve(hessian, -gradient), atol=1e-8)


def test_all_variables_can_be_active():
    """Every variable pinned to a bound is a valid and awkward case."""
    hessian = np.eye(4)
    gradient = np.array([10.0, 10.0, -10.0, -10.0])
    lower = np.array([-1.0, -1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])
    solver = BoxQpSolver(hessian)
    solution = solver.solve(gradient, lower, upper)
    assert np.allclose(solution, [-1.0, -1.0, 1.0, 1.0])


def test_warm_start_reduces_iterations():
    hessian, gradient, lower, upper = random_problem(25)
    solver = BoxQpSolver(hessian)
    cold = solver.solve(gradient, lower, upper)
    cold_iterations = solver.last_iterations
    warm = solver.solve(gradient, lower, upper, warm_start=cold)
    assert solver.last_iterations <= cold_iterations
    assert np.allclose(cold, warm, atol=1e-8)


def test_warm_start_outside_bounds_is_projected():
    hessian, gradient, lower, upper = random_problem(8)
    solver = BoxQpSolver(hessian)
    wild = np.full(8, 1e6)
    solution = solver.solve(gradient, lower, upper, warm_start=wild)
    assert np.all(solution >= lower - 1e-12)
    assert np.all(solution <= upper + 1e-12)
    assert solver.kkt_residual(solution, gradient, lower, upper) < 1e-6 * max(
        1.0, float(np.max(np.abs(gradient)))
    )


def test_rejects_asymmetric_hessian():
    hessian = np.array([[2.0, 1.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        BoxQpSolver(hessian)


def test_converges_on_a_badly_conditioned_hessian():
    """The condensed MPC Hessian has a condition number in the millions.

    The relative convergence tolerance exists for exactly this case: an absolute
    threshold is unreachable at this scale, and the solver would silently grind
    to its iteration cap on every control step.
    """
    size = 25
    scales = np.logspace(0, 6, size)
    basis = np.linalg.qr(RNG.normal(size=(size, size)))[0]
    hessian = basis @ np.diag(scales) @ basis.T
    hessian = 0.5 * (hessian + hessian.T)
    gradient = RNG.normal(size=size) * 1e4
    lower, upper = np.full(size, -8.0), np.full(size, 8.0)

    solver = BoxQpSolver(hessian)
    solution = solver.solve(gradient, lower, upper)
    assert solver.last_iterations < solver.max_iterations
    residual = solver.kkt_residual(solution, gradient, lower, upper)
    assert residual < 1e-6 * float(np.max(np.abs(gradient)))
