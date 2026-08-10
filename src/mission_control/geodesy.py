"""Geodetic conversions.

Missions are written in latitude/longitude/altitude because that is what a
surveyor, a CAA flight notification and a PX4 mission item all use. Everything
downstream of the loader works in a local north-east-down (NED) frame anchored
at a mission origin, because control and estimation in degrees is miserable.

The projection is the standard local-tangent-plane linearisation of the WGS-84
ellipsoid. It is exact at the origin and accumulates curvature error with
distance; :func:`projection_error_bound` quantifies that so callers can assert
on it rather than assume. For inspection missions (hundreds of metres) the error
is well under a centimetre, which is below RTK noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS-84 defining parameters.
WGS84_A = 6378137.0  # semi-major axis, m
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared

#: Standard gravity used throughout. The sim, the EKF and the controllers must
#: all use the same value or the estimator will fight a constant bias.
GRAVITY = 9.80665


def meridional_radius(lat_rad: float) -> float:
    """Radius of curvature in the meridian (north-south) at ``lat_rad``."""
    s = math.sin(lat_rad)
    return WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s * s) ** 1.5


def transverse_radius(lat_rad: float) -> float:
    """Radius of curvature in the prime vertical (east-west) at ``lat_rad``."""
    s = math.sin(lat_rad)
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)


@dataclass(frozen=True)
class GeodeticOrigin:
    """Anchor for the local NED frame.

    ``altitude_m`` is height above the WGS-84 ellipsoid, matching what GNSS
    reports natively. Mission altitudes are relative to this origin, so a
    mission is portable to another site by changing only the origin.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(f"latitude out of range: {self.latitude_deg}")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError(f"longitude out of range: {self.longitude_deg}")

    def to_ned(self, latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
        """Project a geodetic point into the local NED frame, in metres."""
        lat0 = math.radians(self.latitude_deg)
        dlat = math.radians(latitude_deg - self.latitude_deg)
        dlon = math.radians(longitude_deg - self.longitude_deg)
        north = dlat * (meridional_radius(lat0) + self.altitude_m)
        east = dlon * (transverse_radius(lat0) + self.altitude_m) * math.cos(lat0)
        down = -(altitude_m - self.altitude_m)
        return np.array([north, east, down], dtype=float)

    def to_geodetic(self, ned: np.ndarray) -> tuple:
        """Inverse of :meth:`to_ned`. Returns ``(lat_deg, lon_deg, alt_m)``."""
        ned = np.asarray(ned, dtype=float)
        lat0 = math.radians(self.latitude_deg)
        lat = self.latitude_deg + math.degrees(ned[0] / (meridional_radius(lat0) + self.altitude_m))
        lon = self.longitude_deg + math.degrees(
            ned[1] / ((transverse_radius(lat0) + self.altitude_m) * math.cos(lat0))
        )
        alt = self.altitude_m - ned[2]
        return lat, lon, alt


def projection_error_bound(range_m: float, latitude_deg: float = 51.5) -> float:
    """Worst-case horizontal error of the flat-earth projection, in metres.

    Second-order term of the spherical-to-plane expansion: a point ``d`` metres
    from the origin is misplaced by roughly ``d**3 / (6 R**2)``. Used by the
    tests to justify the linearisation for the mission sizes in this repository
    instead of asserting it is "small enough".
    """
    radius = transverse_radius(math.radians(latitude_deg))
    return range_m**3 / (6.0 * radius**2)
