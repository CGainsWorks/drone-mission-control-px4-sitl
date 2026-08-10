"""The interface between the mission-control layer and a vehicle.

Everything above this line — mission state machine, trajectory generation,
estimation, both trackers, geofence, failsafe matrix — is vehicle-agnostic. It
consumes a state estimate and emits an attitude setpoint. What sits underneath
is either the reference simulation in :mod:`mission_control.sim` or a real PX4
autopilot over MAVLink.

Keeping that boundary sharp is the point of the architecture, and it is what
makes the headline comparison meaningful: the same controller objects, driven by
the same mission executive, fly both. Nothing about a controller changes when
the backend does.

The interface is deliberately small. Anything a backend cannot express is
something the layer above should not be relying on.
"""

from __future__ import annotations

import abc

from ..types import AttitudeSetpoint, BatteryState, VehicleState


class BackendError(RuntimeError):
    """Raised when a backend cannot do what was asked of it."""


class VehicleBackend(abc.ABC):
    """What the mission-control layer needs from a vehicle."""

    #: Short name used in logs and results.
    name = "backend"

    # -- lifecycle ------------------------------------------------------
    @abc.abstractmethod
    def connect(self, timeout_s: float = 30.0) -> None:
        """Establish the link and block until the vehicle is describable."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the link. Must be safe to call twice."""

    # -- commands -------------------------------------------------------
    @abc.abstractmethod
    def arm(self) -> None:
        """Arm the vehicle. Raises :class:`BackendError` if refused."""

    @abc.abstractmethod
    def disarm(self) -> None: ...

    @abc.abstractmethod
    def start_offboard(self) -> None:
        """Hand trajectory control to this software.

        PX4 requires a stream of setpoints to be flowing *before* offboard mode
        will engage, and drops out of it if the stream stalls. Implementations
        must handle that; callers should not have to know.
        """

    @abc.abstractmethod
    def stop_offboard(self) -> None: ...

    @abc.abstractmethod
    def send_attitude_setpoint(self, setpoint: AttitudeSetpoint) -> None:
        """Send one attitude and thrust command."""

    # -- telemetry ------------------------------------------------------
    @abc.abstractmethod
    def state(self) -> VehicleState:
        """Latest state estimate, in the local NED frame."""

    @abc.abstractmethod
    def battery(self) -> BatteryState: ...

    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    @property
    def link_silence_s(self) -> float:
        """Seconds since the last telemetry. Feeds the link-loss failsafe."""
        return 0.0

    def describe(self) -> dict:
        return {"backend": self.name}

    # -- context manager ------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
