"""Wind and gust model.

A steady wind vector plus a first-order Gauss-Markov (Ornstein-Uhlenbeck) gust
on each axis. This is the standard low-order stand-in for the Dryden turbulence
spectrum: it has the right variance and the right correlation time, which is
what determines whether a controller's preview horizon helps, and it is cheap
and exactly reproducible from a seed.

Wind is the disturbance that separates the two trackers, so it is generated from
its own ``numpy`` generator. Changing the controller cannot change the wind
sequence, which is what makes the comparison paired rather than merely averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindParams:
    """Wind field configuration.

    ``gust_sigma`` is the stationary standard deviation per axis and
    ``gust_tau`` the correlation time. Defaults are a still day; the comparison
    scenarios override them.
    """

    mean_ned: tuple = (0.0, 0.0, 0.0)
    gust_sigma: float = 0.0
    gust_tau: float = 2.0
    #: Vertical gusts are weaker than horizontal ones in the atmospheric
    #: boundary layer; this scales sigma on the down axis.
    vertical_scale: float = 0.5

    @property
    def is_still(self) -> bool:
        return self.gust_sigma <= 0.0 and not np.any(np.asarray(self.mean_ned))


class WindField:
    """Time-varying wind in NED, in m/s."""

    def __init__(self, params: WindParams = None, seed: int = 0):
        self.params = params or WindParams()
        self._rng = np.random.default_rng(seed)
        self._gust = np.zeros(3)
        self._sigma = np.array(
            [
                self.params.gust_sigma,
                self.params.gust_sigma,
                self.params.gust_sigma * self.params.vertical_scale,
            ]
        )
        # Start from the stationary distribution rather than zero, so a short
        # run is not dominated by the spin-up transient.
        if self.params.gust_sigma > 0.0:
            self._gust = self._rng.normal(0.0, 1.0, 3) * self._sigma

    def step(self, dt: float) -> np.ndarray:
        """Advance the gust state and return the total wind vector."""
        if self.params.gust_sigma > 0.0:
            tau = max(self.params.gust_tau, 1e-3)
            decay = np.exp(-dt / tau)
            # Exact discretisation of the OU process: preserves the stationary
            # variance for any dt, unlike an Euler update.
            diffusion = self._sigma * np.sqrt(1.0 - decay**2)
            self._gust = decay * self._gust + diffusion * self._rng.normal(0.0, 1.0, 3)
        return np.asarray(self.params.mean_ned, dtype=float) + self._gust

    @property
    def value(self) -> np.ndarray:
        return np.asarray(self.params.mean_ned, dtype=float) + self._gust
