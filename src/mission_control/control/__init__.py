"""Trajectory tracking controllers.

Both trackers implement :class:`TrajectoryTracker` and emit the same
:class:`~mission_control.types.AttitudeSetpoint`, so the comparison harness can
swap them without touching anything else. That constraint is the point: it is
what makes "MPC beats PID by X" a statement about the controllers rather than
about two different test rigs.
"""

from .common import acceleration_to_attitude, tilt_limit_acceleration
from .interface import ReferenceProvider, TrackerDiagnostics, TrajectoryTracker
from .mpc_tracker import MpcTracker, MpcWeights
from .pid_tracker import PidGains, PidTracker
from .qp import BoxQpSolver

__all__ = [
    "ReferenceProvider",
    "TrackerDiagnostics",
    "TrajectoryTracker",
    "acceleration_to_attitude",
    "tilt_limit_acceleration",
    "PidTracker",
    "PidGains",
    "MpcTracker",
    "MpcWeights",
    "BoxQpSolver",
    "make_tracker",
    "TRACKERS",
]


#: Registry used by the comparison harness and the CLI so a controller can be
#: selected by name from a mission file or an argument.
TRACKERS = {"pid": PidTracker, "mpc": MpcTracker}


def make_tracker(name: str, params, tuned: bool = True, **kwargs) -> TrajectoryTracker:
    """Construct a tracker by name.

    Raises on an unknown name rather than falling back to a default, so a typo
    in a scenario file fails loudly.

    Unless ``tuned`` is ``False`` or explicit gains are passed, the gains come
    from ``config/tuned_gains.yaml``. Routing every construction through here is
    what guarantees both controllers in a comparison are running the gains the
    tuning script actually chose, rather than whichever defaults happen to be in
    the source.
    """
    key = name.lower()
    if key not in TRACKERS:
        raise KeyError(f"unknown tracker {name!r}; available: {sorted(TRACKERS)}")

    cls = TRACKERS[key]
    if tuned and not kwargs:
        from ..tuning import mpc_weights, pid_gains

        if cls is PidTracker:
            return cls(params, gains=pid_gains())
        if cls is MpcTracker:
            return cls(params, weights=mpc_weights())
    return cls(params, **kwargs)
