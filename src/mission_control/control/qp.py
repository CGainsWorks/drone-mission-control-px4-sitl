"""Box-constrained QP solver for the MPC.

Solves

.. math::

    \\min_u \\; \\tfrac{1}{2} u^\\top H u + g^\\top u
    \\quad \\text{subject to} \\quad l \\le u \\le b

Why not a library
-----------------

``acados`` and ``osqp`` are the right answer on a real vehicle. Both are
compiled dependencies, and a portfolio result that cannot be reproduced without
a working toolchain is not much of a result. The problem here is small (a few
dozen dense variables) with only box constraints, which is exactly the case a
short, exact solver handles well.

Why projected Newton rather than ADMM
-------------------------------------

ADMM was tried first and is a poor fit here. A condensed MPC Hessian is
ill-conditioned — position is integrated twice from acceleration, so the cheap
and expensive directions differ by orders of magnitude — and a single scalar
penalty parameter cannot serve both ends of that spread. It was still at 4e-4
KKT residual after 60 iterations and about a millisecond per axis.

Bertsekas' projected Newton method (1982) suits it far better. It identifies
which constraints are active, takes an exact Newton step in the remaining free
variables, and projects back onto the box with an Armijo line search. The active
set is usually identified within two or three iterations and the method then
terminates finitely at the exact solution, so there is no penalty parameter and
no accuracy/iteration trade-off to tune.

The factorisation is of the *free* block, which changes as constraints activate,
so it cannot be cached across iterations the way a fixed-Hessian method could.
At this size that costs microseconds and buys exactness. Warm starting from the
previous control step usually reproduces the same active set immediately.

``tests/test_qp.py`` checks the result against the KKT conditions and against
``scipy.optimize`` on random problems, so the solver is not taken on trust.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class BoxQpSolver:
    """Projected-Newton active-set solver for a box-constrained QP.

    Parameters
    ----------
    hessian:
        Symmetric positive-definite ``H``, constant across solves.
    max_iterations:
        Cap on outer iterations. Reaching it means the active set is still
        churning, which is worth knowing about; the caller sees it through
        :attr:`last_iterations`.
    tolerance:
        Termination threshold on the infinity norm of the projected gradient,
        the natural optimality measure for a box-constrained problem. Applied
        **relative to the problem scale**, not absolutely.

        That matters because of the magnitudes involved rather than the
        conditioning: the MPC's position weights are in the hundreds and its
        reference is in tens of metres, so the linear term runs to six figures.
        An absolute 1e-8 on a gradient that size is below what float64 can
        resolve, and the solver grinds to its iteration cap on every step.
        ``scripts/benchmark_qp.py`` prints both numbers for the shipped
        configuration.
    """

    def __init__(self, hessian: np.ndarray, max_iterations: int = 50,
                 tolerance: float = 1e-8):
        self.hessian = np.asarray(hessian, dtype=float)
        self.size = self.hessian.shape[0]
        if self.hessian.shape != (self.size, self.size):
            raise ValueError("hessian must be square")
        if not np.allclose(self.hessian, self.hessian.T, atol=1e-9, rtol=0.0):
            raise ValueError("hessian must be symmetric")

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.last_iterations = 0
        #: Diagonal, used as a fallback search direction if the free block turns
        #: out to be numerically indefinite.
        self._diagonal = np.clip(np.diag(self.hessian), 1e-12, None)

    # ------------------------------------------------------------------
    def _projected_gradient(self, x: np.ndarray, gradient: np.ndarray,
                            lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Gradient with components zeroed where the box blocks that direction."""
        blocked_low = (x <= lower + 1e-12) & (gradient > 0.0)
        blocked_high = (x >= upper - 1e-12) & (gradient < 0.0)
        return np.where(blocked_low | blocked_high, 0.0, gradient)

    def solve(self, gradient: np.ndarray, lower: np.ndarray, upper: np.ndarray,
              warm_start: np.ndarray = None) -> np.ndarray:
        """Solve for the given linear term and bounds.

        ``warm_start`` should be the previous solution shifted by one step. On a
        smoothly varying trajectory it reproduces the previous active set, so
        the solve finishes in one or two iterations.
        """
        gradient = np.asarray(gradient, dtype=float).reshape(self.size)
        lower = np.asarray(lower, dtype=float).reshape(self.size)
        upper = np.asarray(upper, dtype=float).reshape(self.size)

        x = (
            np.clip(np.asarray(warm_start, dtype=float).reshape(self.size), lower, upper)
            if warm_start is not None
            else np.clip(np.zeros(self.size), lower, upper)
        )

        objective = 0.5 * x @ self.hessian @ x + gradient @ x
        self.last_iterations = 0
        threshold = self.tolerance * max(1.0, float(np.max(np.abs(gradient))))

        for iteration in range(1, self.max_iterations + 1):
            self.last_iterations = iteration
            grad = self.hessian @ x + gradient
            projected = self._projected_gradient(x, grad, lower, upper)
            if np.max(np.abs(projected)) <= threshold:
                return x

            # Bertsekas' epsilon-active set: a variable counts as binding only
            # if it sits at a bound *and* the gradient pushes it further out.
            # The tolerance is tied to how far from optimality we are, so the
            # set cannot thrash near the solution.
            epsilon = min(1e-4, float(np.max(np.abs(projected))) * 1e-3)
            binding = ((x <= lower + epsilon) & (grad > 0.0)) | (
                (x >= upper - epsilon) & (grad < 0.0)
            )
            free = ~binding

            direction = np.zeros(self.size)
            # Binding variables get a scaled steepest-descent step; the
            # projection will hold them at the bound unless the gradient has
            # genuinely reversed.
            direction[binding] = -grad[binding] / self._diagonal[binding]
            if np.any(free):
                block = self.hessian[np.ix_(free, free)]
                try:
                    direction[free] = cho_solve(cho_factor(block), -grad[free])
                except np.linalg.LinAlgError:
                    direction[free] = -grad[free] / self._diagonal[free]

            # Armijo backtracking along the *projected* arc, not the ray: the
            # step is only a descent direction after projection.
            step = 1.0
            improved = False
            for _ in range(30):
                candidate = np.clip(x + step * direction, lower, upper)
                candidate_objective = (
                    0.5 * candidate @ self.hessian @ candidate + gradient @ candidate
                )
                if candidate_objective <= objective - 1e-4 * float(grad @ (x - candidate)):
                    x = candidate
                    objective = candidate_objective
                    improved = True
                    break
                step *= 0.5

            if not improved:
                # No decrease available along the arc: we are at the solution to
                # within round-off, and shrinking the step further just burns
                # iterations.
                return x

        return x

    # ------------------------------------------------------------------
    def objective(self, u: np.ndarray, gradient: np.ndarray) -> float:
        u = np.asarray(u, dtype=float).reshape(self.size)
        return float(0.5 * u @ self.hessian @ u + gradient @ u)

    def kkt_residual(self, u: np.ndarray, gradient: np.ndarray, lower: np.ndarray,
                     upper: np.ndarray) -> float:
        """Max violation of the first-order optimality conditions.

        For a box-constrained QP the optimum has zero gradient in the interior, a
        non-negative gradient at a lower bound and a non-positive one at an upper
        bound. Used by the tests as an independent check rather than trusting the
        solver's own termination criterion.
        """
        u = np.asarray(u, dtype=float).reshape(self.size)
        grad = self.hessian @ u + gradient
        at_lower = u <= lower + 1e-9
        at_upper = u >= upper - 1e-9
        interior = ~(at_lower | at_upper)
        violation = 0.0
        if np.any(interior):
            violation = max(violation, float(np.max(np.abs(grad[interior]))))
        if np.any(at_lower):
            violation = max(violation, float(np.max(np.maximum(-grad[at_lower], 0.0))))
        if np.any(at_upper):
            violation = max(violation, float(np.max(np.maximum(grad[at_upper], 0.0))))
        return violation
