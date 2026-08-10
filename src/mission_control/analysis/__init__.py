"""Metrics, the scenario grid, the comparison harness and the figures.

This package is where measurement lives, kept apart from the things being
measured. Nothing under ``analysis`` is imported by the flight software, so a
change to a metric or a plot cannot alter a controller's behaviour.
"""

from .metrics import estimator_metrics, settling_metrics, summarise, tracking_metrics
from .scenarios import ScenarioSpec, build_grid, load_mission_for, make_config

__all__ = [
    "tracking_metrics",
    "settling_metrics",
    "estimator_metrics",
    "summarise",
    "ScenarioSpec",
    "build_grid",
    "load_mission_for",
    "make_config",
]
