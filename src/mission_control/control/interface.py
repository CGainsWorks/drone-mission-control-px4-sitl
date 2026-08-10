"""The contract every trajectory tracker obeys."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from ..types import AttitudeSetpoint, TrajectorySetpoint, VehicleState


class ReferenceProvider(abc.ABC):
    """A trajectory that can be sampled at any time, including the future.

    Both trackers are handed the same provider. A PID controller chooses to
    sample only the present; an MPC samples its whole horizon. Giving them the
    same object rather than pre-slicing the reference for each is what keeps the
    preview advantage an honest property of the controller instead of a
    difference in what the harness fed it.
    """

    @abc.abstractmethod
    def sample(self, time_s: float) -> TrajectorySetpoint:
        """Reference at ``time_s``. Must be defined outside the trajectory's
        span too, by holding the endpoints, so an MPC horizon can run past the
        end of a leg without special-casing."""

    def sample_horizon(self, start_s: float, step_s: float, count: int) -> list:
        return [self.sample(start_s + (i + 1) * step_s) for i in range(count)]


@dataclass
class TrackerDiagnostics:
    """Per-step internals, logged for analysis and used by the metrics.

    Two separate times, because reporting one would misstate the result in
    opposite directions. ``update_time_s`` is the whole control step, which is
    what a flight computer's scheduler cares about. ``solve_time_s`` is the
    optimisation alone. For the MPC most of the difference is sampling the
    reference across the horizon — real work, but work an implementation could
    cache, whereas the QP is irreducible. Quoting only the total would make the
    optimiser look an order of magnitude more expensive than it is; quoting only
    the solve would hide the cost of preview.

    ``saturated`` counts steps where the acceleration demand hit the tilt limit,
    which is the honest way to notice a controller is being flattered by a
    scenario it cannot actually fly.
    """

    update_time_s: float = 0.0
    solve_time_s: float = 0.0
    iterations: int = 0
    saturated: bool = False
    acceleration_demand: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_setpoint: np.ndarray = field(default_factory=lambda: np.zeros(3))


class TrajectoryTracker(abc.ABC):
    """Position/velocity reference in, attitude and thrust command out."""

    #: Short name used in logs, filenames and results tables.
    name = "base"

    def __init__(self, params):
        self.params = params
        self.diagnostics = TrackerDiagnostics()

    @abc.abstractmethod
    def reset(self, state: VehicleState) -> None:
        """Clear internal state. Called on mode entry so a tracker never carries
        an integrator or a warm start across a mode change."""

    @abc.abstractmethod
    def update(self, state: VehicleState, reference: ReferenceProvider,
               time_s: float, dt: float) -> AttitudeSetpoint:
        """Produce one attitude setpoint."""

    def describe(self) -> dict:
        """Tuning summary written into the results so a number in the README can
        be traced to the gains that produced it."""
        return {"name": self.name}
