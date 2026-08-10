"""Vehicle backends: the reference simulation and PX4 over MAVLink.

``Px4SitlBackend`` is imported lazily by :func:`make_backend` so that MAVSDK
stays an optional dependency. Importing this package must never fail because
MAVSDK is absent.
"""

from .backend import BackendError, VehicleBackend

__all__ = ["VehicleBackend", "BackendError", "make_backend"]


def make_backend(name: str, **kwargs) -> VehicleBackend:
    """Construct a backend by name.

    ``px4`` needs MAVSDK; the error if it is missing says so explicitly rather
    than surfacing as an ImportError from three frames down.
    """
    key = name.lower()
    if key in ("px4", "px4-sitl", "sitl"):
        from .px4 import Px4SitlBackend

        return Px4SitlBackend(**kwargs)
    raise KeyError(f"unknown backend {name!r}; available: px4")
