"""Battery model.

Deliberately simple, because the failsafe layer only needs two things from a
battery and both are cheap to model: a state of charge that falls faster when
the vehicle works harder, and a terminal voltage that **sags under load and
recovers when the load drops**.

That second property is the whole reason the low-battery failsafe is debounced.
A voltage-based estimate dips during a climb and recovers in the cruise that
follows, and a naive threshold on it triggers a return that the vehicle did not
need. Modelling the sag is what makes the debounce testable instead of
decorative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatteryParams:
    """A 6S 5000 mAh lithium-polymer pack, roughly."""

    capacity_ah: float = 5.0
    cells: int = 6
    #: Open-circuit voltage per cell at full and empty.
    cell_full_v: float = 4.20
    cell_empty_v: float = 3.30
    internal_resistance_ohm: float = 0.045
    #: Hover current, A. Everything else is scaled from thrust against this.
    hover_current_a: float = 12.0
    #: Constant avionics and payload draw, A.
    baseline_current_a: float = 1.2
    initial_remaining: float = 1.0


class BatteryModel:
    """Tracks charge and terminal voltage from the commanded thrust."""

    def __init__(self, params: BatteryParams = None):
        self.params = params or BatteryParams()
        self.remaining = float(self.params.initial_remaining)
        self.current_a = self.params.baseline_current_a
        self.voltage_v = self._open_circuit_voltage()

    def _open_circuit_voltage(self) -> float:
        p = self.params
        cell = p.cell_empty_v + (p.cell_full_v - p.cell_empty_v) * self.remaining
        return cell * p.cells

    def step(self, normalised_thrust: float, hover_thrust: float, dt: float) -> None:
        """Advance by ``dt``, drawing current implied by the thrust command.

        Electrical power goes roughly as thrust to the 3/2 power for a propeller
        in hover, so current is scaled that way rather than linearly. It matters:
        a climb at 1.4x hover thrust draws about 1.65x hover current, not 1.4x.
        """
        p = self.params
        ratio = max(float(normalised_thrust) / max(hover_thrust, 1e-6), 0.0)
        self.current_a = p.baseline_current_a + p.hover_current_a * ratio**1.5

        drawn_ah = self.current_a * dt / 3600.0
        self.remaining = float(np.clip(self.remaining - drawn_ah / p.capacity_ah, 0.0, 1.0))
        self.voltage_v = self._open_circuit_voltage() - self.current_a * p.internal_resistance_ohm

    def state(self):
        from ..types import BatteryState

        return BatteryState(
            voltage_v=self.voltage_v, remaining=self.remaining, current_a=self.current_a
        )
