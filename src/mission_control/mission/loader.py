"""Loading and resolving mission files.

Validation happens first and completely: nothing is resolved into a
:class:`~mission_control.mission.schema.Mission` until the whole file has passed,
so a partially-valid file cannot produce a partially-built mission that fails
somewhere less obvious later.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..geodesy import GeodeticOrigin
from .schema import (
    FailsafeSpec,
    GeofenceSpec,
    Mission,
    MissionDefaults,
    MissionItem,
    MissionValidationError,
    validate_mission_dict,
)


def _read_structured(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyYAML is required to read .yaml mission files; "
                "install it or use the .json form"
            ) from error
        return yaml.safe_load(text)
    return json.loads(text)


def load_mission(path: str) -> Mission:
    """Read and validate a mission file. Raises :class:`MissionValidationError`."""
    data = _read_structured(path)
    mission = mission_from_dict(data)
    mission.source_path = os.path.abspath(path)
    return mission


def mission_from_dict(data: dict) -> Mission:
    """Validate a mission mapping and resolve it into a :class:`Mission`."""
    problems = validate_mission_dict(data)
    if problems:
        raise MissionValidationError(problems)

    origin_data = data["origin"]
    origin = GeodeticOrigin(
        latitude_deg=float(origin_data["latitude"]),
        longitude_deg=float(origin_data["longitude"]),
        altitude_m=float(origin_data.get("altitude", 0.0)),
    )

    defaults = _resolve_defaults(data.get("defaults") or {})
    items = [_resolve_item(entry, origin, defaults) for entry in data["items"]]
    geofence = _resolve_geofence(data.get("geofence") or {}, origin)
    failsafe = _resolve_failsafe(data.get("failsafe") or {})

    home = np.zeros(3)
    if "home" in data and isinstance(data["home"], dict):
        home_data = data["home"]
        if "latitude" in home_data:
            home = origin.to_ned(
                float(home_data["latitude"]),
                float(home_data["longitude"]),
                float(home_data.get("altitude", origin.altitude_m)),
            )
        else:
            home = np.array(
                [float(home_data.get("north", 0.0)), float(home_data.get("east", 0.0)), 0.0]
            )

    mission = Mission(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        origin=origin,
        items=items,
        defaults=defaults,
        geofence=geofence,
        failsafe=failsafe,
        home_ned=home,
    )
    _check_geometry(mission)
    return mission


def _resolve_defaults(data: dict) -> MissionDefaults:
    defaults = MissionDefaults()
    for key, value in data.items():
        setattr(defaults, key, value if isinstance(value, str) else float(value))
    return defaults


def _resolve_position(entry: dict, origin: GeodeticOrigin, fallback_altitude: float) -> np.ndarray:
    altitude = float(entry.get("altitude", fallback_altitude))
    if "latitude" in entry:
        return origin.to_ned(
            float(entry["latitude"]), float(entry["longitude"]), origin.altitude_m + altitude
        )
    return np.array([float(entry.get("north", 0.0)), float(entry.get("east", 0.0)), -altitude])


def _resolve_item(entry: dict, origin: GeodeticOrigin, defaults: MissionDefaults) -> MissionItem:
    item_type = entry["type"]
    altitude = float(entry.get("altitude", 0.0))
    item = MissionItem(
        type=item_type,
        altitude=altitude,
        speed=float(entry["speed"]) if "speed" in entry else None,
        hold_s=float(entry.get("hold_s", 0.0)),
        acceptance_radius=(
            float(entry["acceptance_radius"]) if "acceptance_radius" in entry else None
        ),
        yaw_mode=entry.get("yaw_mode"),
        yaw_deg=float(entry["yaw_deg"]) if "yaw_deg" in entry else None,
        radius=float(entry.get("radius", 0.0)),
        sweep_deg=float(entry.get("sweep_deg", 360.0)),
        clockwise=bool(entry.get("clockwise", True)),
        label=str(entry.get("label", "")),
    )
    if item.is_positional:
        item.position_ned = _resolve_position(entry, origin, altitude)
        item.altitude = -float(item.position_ned[2])
    return item


def _resolve_geofence(data: dict, origin: GeodeticOrigin) -> GeofenceSpec:
    polygon = np.zeros((0, 2))
    if data.get("polygon"):
        # Geodetic vertices, projected into the local frame once at load time.
        polygon = np.array(
            [origin.to_ned(float(lat), float(lon), origin.altitude_m)[0:2]
             for lat, lon in data["polygon"]]
        )
    elif data.get("polygon_ned"):
        polygon = np.array([[float(n), float(e)] for n, e in data["polygon_ned"]])

    return GeofenceSpec(
        polygon_ned=polygon,
        min_altitude=float(data.get("min_altitude", 2.0)),
        max_altitude=float(data.get("max_altitude", 120.0)),
        action=str(data.get("action", "rtl")),
        warning_margin=float(data.get("warning_margin", 5.0)),
    )


def _resolve_failsafe(data: dict) -> FailsafeSpec:
    spec = FailsafeSpec()
    for key, value in data.items():
        setattr(spec, key, value if isinstance(value, str) else float(value))
    return spec


def _check_geometry(mission: Mission) -> None:
    """Cross-field checks that need the resolved geometry.

    These are the ones worth catching on the ground: a mission that starts
    outside its own geofence, or one that flies above its own altitude ceiling,
    will otherwise arm happily and trip a failsafe seconds after takeoff.
    """
    from ..safety.geofence import Geofence

    problems = []
    fence = Geofence.from_spec(mission.geofence)

    for index, item in enumerate(mission.items):
        if item.type == "takeoff" and item.altitude > mission.geofence.max_altitude:
            problems.append(
                f"items[{index}].altitude: takeoff to {item.altitude} m is above the "
                f"geofence ceiling of {mission.geofence.max_altitude} m"
            )
        if not item.is_positional:
            continue
        altitude = -float(item.position_ned[2])
        if altitude > mission.geofence.max_altitude:
            problems.append(
                f"items[{index}]: altitude {altitude:.1f} m is above the geofence "
                f"ceiling of {mission.geofence.max_altitude} m"
            )
        if altitude < mission.geofence.min_altitude:
            problems.append(
                f"items[{index}]: altitude {altitude:.1f} m is below the geofence "
                f"floor of {mission.geofence.min_altitude} m"
            )
        if fence.is_defined and not fence.contains_horizontal(item.position_ned):
            problems.append(f"items[{index}]: position lies outside the geofence polygon")

    if fence.is_defined and not fence.contains_horizontal(mission.home_ned):
        problems.append("home: lies outside the geofence polygon")

    if problems:
        raise MissionValidationError(problems)
