"""Mission definition schema and validation.

A mission is data, not code. It is written in geodetic coordinates because that
is what a site survey, a CAA flight notification and a PX4 mission item all use,
and it is validated strictly before anything is flown.

Strictly means: unknown keys are an error, not a warning. A mission file with
``altitiude: 30`` in it should fail on the ground rather than fly at the default
height, and a typo in a geofence key should never silently disable the geofence.
Every problem carries the path to the offending field, and validation collects
all of them rather than stopping at the first, so one edit-run cycle fixes the
whole file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geodesy import GeodeticOrigin


class MissionValidationError(ValueError):
    """Raised when a mission file is not valid. Carries every problem found."""

    def __init__(self, problems):
        self.problems = list(problems)
        joined = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{len(self.problems)} problem(s) in mission definition:\n{joined}")


class _Validator:
    """Collects problems with a field path, so messages point at the mistake."""

    def __init__(self):
        self.problems = []

    def error(self, path: str, message: str) -> None:
        self.problems.append(f"{path}: {message}")

    def require_keys(self, path: str, data: dict, required, optional) -> None:
        if not isinstance(data, dict):
            self.error(path, f"expected a mapping, got {type(data).__name__}")
            return
        allowed = set(required) | set(optional)
        for key in required:
            if key not in data:
                self.error(path, f"missing required key {key!r}")
        for key in data:
            if key not in allowed:
                closest = _closest(key, allowed)
                hint = f"; did you mean {closest!r}?" if closest else ""
                self.error(f"{path}.{key}", f"unknown key{hint}")

    def number(self, path: str, value, minimum=None, maximum=None, default=None):
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.error(path, f"expected a number, got {type(value).__name__}")
            return default
        value = float(value)
        if not math.isfinite(value):
            self.error(path, "must be finite")
            return default
        if minimum is not None and value < minimum:
            self.error(path, f"must be >= {minimum}, got {value}")
        if maximum is not None and value > maximum:
            self.error(path, f"must be <= {maximum}, got {value}")
        return value

    def choice(self, path: str, value, options, default=None):
        if value is None:
            return default
        if value not in options:
            self.error(path, f"must be one of {sorted(options)}, got {value!r}")
            return default
        return value


def _closest(key: str, candidates) -> str:
    """Cheap nearest-neighbour suggestion for a misspelled key."""
    best, best_score = None, 0.0
    for candidate in candidates:
        common = len(set(key.lower()) & set(candidate.lower()))
        score = common / max(len(set(key.lower()) | set(candidate.lower())), 1)
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= 0.6 else ""


# ----------------------------------------------------------------------
# mission items
# ----------------------------------------------------------------------
ITEM_TYPES = ("takeoff", "waypoint", "orbit", "hold", "rtl", "land")


@dataclass
class MissionItem:
    """One step of a mission.

    Positions may be given either geodetically (``latitude``/``longitude``/
    ``altitude``) or in local metres relative to the origin (``north``/``east``/
    ``altitude``). The loader resolves both into ``position_ned`` so nothing
    downstream has to care which was written.
    """

    type: str
    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    altitude: float = 0.0
    speed: float = None
    hold_s: float = 0.0
    acceptance_radius: float = None
    yaw_mode: str = None
    yaw_deg: float = None
    #: Orbit only.
    radius: float = 0.0
    sweep_deg: float = 360.0
    clockwise: bool = True
    #: Free-text label, carried into the logs and results tables.
    label: str = ""

    @property
    def is_positional(self) -> bool:
        return self.type in ("waypoint", "orbit", "hold")


@dataclass
class GeofenceSpec:
    """Geofence as written in the mission file."""

    polygon_ned: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    min_altitude: float = 2.0
    max_altitude: float = 120.0
    #: What to do on breach. Resolved against the failsafe matrix at runtime.
    action: str = "rtl"
    #: How far inside the boundary the predictive check starts warning, m.
    warning_margin: float = 5.0

    @property
    def is_defined(self) -> bool:
        return self.polygon_ned.shape[0] >= 3


@dataclass
class FailsafeSpec:
    """Thresholds for the failsafe matrix, overridable per mission."""

    low_battery_remaining: float = 0.25
    critical_battery_remaining: float = 0.12
    link_loss_timeout_s: float = 3.0
    link_loss_critical_s: float = 15.0
    gnss_loss_timeout_s: float = 2.0
    rtl_altitude: float = 30.0
    #: Actions, matched to the names in :mod:`mission_control.safety.failsafe`.
    low_battery_action: str = "rtl"
    critical_battery_action: str = "land"
    link_loss_action: str = "hold"
    link_loss_critical_action: str = "rtl"
    gnss_loss_action: str = "land"
    geofence_action: str = "rtl"


@dataclass
class MissionDefaults:
    """Values applied to any item that does not override them."""

    cruise_speed: float = 4.0
    acceptance_radius: float = 1.5
    yaw_mode: str = "path"
    max_tangential_acceleration: float = 2.5
    max_lateral_acceleration: float = 3.0
    max_climb_rate: float = 2.5
    max_descent_rate: float = 1.5
    #: ``round`` fits an arc into each corner, giving a reference the vehicle can
    #: actually fly. ``stop`` comes to rest at every waypoint. ``sharp`` flies
    #: the corner at full speed with a velocity step, which no vehicle can
    #: follow — used deliberately in the comparison to measure what each
    #: controller does with an infeasible reference.
    corner_mode: str = "round"
    corner_radius: float = 4.0


@dataclass
class Mission:
    """A validated mission, ready to execute."""

    name: str
    origin: GeodeticOrigin
    items: list
    defaults: MissionDefaults = field(default_factory=MissionDefaults)
    geofence: GeofenceSpec = field(default_factory=GeofenceSpec)
    failsafe: FailsafeSpec = field(default_factory=FailsafeSpec)
    description: str = ""
    #: Home is where RTL returns to: the origin unless the file says otherwise.
    home_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def takeoff_altitude(self) -> float:
        for item in self.items:
            if item.type == "takeoff":
                return item.altitude
        return 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "items": len(self.items),
            "item_types": [item.type for item in self.items],
            "cruise_speed": self.defaults.cruise_speed,
            "corner_mode": self.defaults.corner_mode,
            "geofence_vertices": int(self.geofence.polygon_ned.shape[0]),
            "altitude_band": [self.geofence.min_altitude, self.geofence.max_altitude],
        }


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
MISSION_KEYS_REQUIRED = ("name", "origin", "items")
MISSION_KEYS_OPTIONAL = ("description", "defaults", "geofence", "failsafe", "home")

ITEM_KEYS = {
    "takeoff": (("altitude",), ("speed", "yaw_deg", "label")),
    "waypoint": (
        (),
        ("latitude", "longitude", "altitude", "north", "east", "speed", "hold_s",
         "acceptance_radius", "yaw_mode", "yaw_deg", "label"),
    ),
    "orbit": (
        ("radius",),
        ("latitude", "longitude", "altitude", "north", "east", "sweep_deg",
         "clockwise", "speed", "yaw_mode", "yaw_deg", "label"),
    ),
    "hold": ((), ("hold_s", "latitude", "longitude", "altitude", "north", "east",
                  "yaw_mode", "yaw_deg", "label")),
    "rtl": ((), ("altitude", "speed", "label")),
    "land": ((), ("speed", "label")),
}


def validate_mission_dict(data: dict) -> list:
    """Check a raw mission mapping. Returns the list of problems, empty if valid."""
    validator = _Validator()
    if not isinstance(data, dict):
        return [f"<root>: expected a mapping, got {type(data).__name__}"]

    validator.require_keys("<root>", data, MISSION_KEYS_REQUIRED, MISSION_KEYS_OPTIONAL)

    name = data.get("name")
    if name is not None and not isinstance(name, str):
        validator.error("name", "expected a string")

    origin = data.get("origin")
    if origin is not None:
        validator.require_keys("origin", origin, ("latitude", "longitude"), ("altitude",))
        if isinstance(origin, dict):
            validator.number("origin.latitude", origin.get("latitude"), -90.0, 90.0)
            validator.number("origin.longitude", origin.get("longitude"), -180.0, 180.0)
            validator.number("origin.altitude", origin.get("altitude"), -500.0, 9000.0, 0.0)

    items = data.get("items")
    if items is None:
        pass  # already reported as missing
    elif not isinstance(items, list) or not items:
        validator.error("items", "expected a non-empty list")
    else:
        _validate_items(validator, items)

    if "defaults" in data:
        _validate_defaults(validator, data["defaults"])
    if "geofence" in data:
        _validate_geofence(validator, data["geofence"])
    if "failsafe" in data:
        _validate_failsafe(validator, data["failsafe"])

    return validator.problems


def _validate_items(validator: _Validator, items: list) -> None:
    for index, item in enumerate(items):
        path = f"items[{index}]"
        if not isinstance(item, dict):
            validator.error(path, f"expected a mapping, got {type(item).__name__}")
            continue
        item_type = item.get("type")
        if item_type not in ITEM_TYPES:
            validator.error(f"{path}.type", f"must be one of {list(ITEM_TYPES)}, got {item_type!r}")
            continue

        required, optional = ITEM_KEYS[item_type]
        validator.require_keys(path, item, ("type",) + required, optional)

        validator.number(f"{path}.altitude", item.get("altitude"), 0.0, 500.0)
        validator.number(f"{path}.speed", item.get("speed"), 0.1, 30.0)
        validator.number(f"{path}.hold_s", item.get("hold_s"), 0.0, 3600.0)
        validator.number(f"{path}.acceptance_radius", item.get("acceptance_radius"), 0.1, 100.0)
        validator.number(f"{path}.yaw_deg", item.get("yaw_deg"), -360.0, 360.0)
        validator.number(f"{path}.radius", item.get("radius"), 1.0, 500.0)
        validator.number(f"{path}.sweep_deg", item.get("sweep_deg"), 1.0, 3600.0)
        validator.choice(f"{path}.yaw_mode", item.get("yaw_mode"), ("path", "fixed", "poi"))

        has_geodetic = "latitude" in item or "longitude" in item
        has_local = "north" in item or "east" in item
        if has_geodetic and has_local:
            validator.error(path, "give either latitude/longitude or north/east, not both")
        if has_geodetic:
            if "latitude" not in item or "longitude" not in item:
                validator.error(path, "latitude and longitude must be given together")
            validator.number(f"{path}.latitude", item.get("latitude"), -90.0, 90.0)
            validator.number(f"{path}.longitude", item.get("longitude"), -180.0, 180.0)
        if has_local:
            validator.number(f"{path}.north", item.get("north"), -50000.0, 50000.0)
            validator.number(f"{path}.east", item.get("east"), -50000.0, 50000.0)
        if item_type in ("waypoint", "orbit") and not (has_geodetic or has_local):
            validator.error(path, f"a {item_type} needs a position")

    if items and isinstance(items[0], dict) and items[0].get("type") != "takeoff":
        validator.error("items[0].type", "a mission must begin with a takeoff item")


def _validate_defaults(validator: _Validator, defaults) -> None:
    validator.require_keys(
        "defaults",
        defaults,
        (),
        ("cruise_speed", "acceptance_radius", "yaw_mode", "max_tangential_acceleration",
         "max_lateral_acceleration", "max_climb_rate", "max_descent_rate",
         "corner_mode", "corner_radius"),
    )
    if not isinstance(defaults, dict):
        return
    validator.number("defaults.cruise_speed", defaults.get("cruise_speed"), 0.1, 30.0)
    validator.number("defaults.acceptance_radius", defaults.get("acceptance_radius"), 0.1, 100.0)
    validator.number("defaults.corner_radius", defaults.get("corner_radius"), 0.5, 200.0)
    validator.number("defaults.max_tangential_acceleration",
                     defaults.get("max_tangential_acceleration"), 0.1, 20.0)
    validator.number("defaults.max_lateral_acceleration",
                     defaults.get("max_lateral_acceleration"), 0.1, 20.0)
    validator.number("defaults.max_climb_rate", defaults.get("max_climb_rate"), 0.1, 20.0)
    validator.number("defaults.max_descent_rate", defaults.get("max_descent_rate"), 0.1, 20.0)
    validator.choice("defaults.yaw_mode", defaults.get("yaw_mode"), ("path", "fixed", "poi"))
    validator.choice("defaults.corner_mode", defaults.get("corner_mode"),
                     ("round", "stop", "sharp"))


def _validate_geofence(validator: _Validator, geofence) -> None:
    validator.require_keys(
        "geofence", geofence, (),
        ("polygon", "polygon_ned", "min_altitude", "max_altitude", "action", "warning_margin"),
    )
    if not isinstance(geofence, dict):
        return
    validator.number("geofence.min_altitude", geofence.get("min_altitude"), 0.0, 500.0)
    validator.number("geofence.max_altitude", geofence.get("max_altitude"), 1.0, 500.0)
    validator.number("geofence.warning_margin", geofence.get("warning_margin"), 0.0, 100.0)
    validator.choice("geofence.action", geofence.get("action"),
                     ("hold", "rtl", "land", "terminate"))

    minimum = geofence.get("min_altitude")
    maximum = geofence.get("max_altitude")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        if float(maximum) <= float(minimum):
            validator.error("geofence.max_altitude", "must be above min_altitude")

    for key in ("polygon", "polygon_ned"):
        polygon = geofence.get(key)
        if polygon is None:
            continue
        if not isinstance(polygon, list) or len(polygon) < 3:
            validator.error(f"geofence.{key}", "expected at least 3 vertices")
            continue
        for index, vertex in enumerate(polygon):
            if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
                validator.error(f"geofence.{key}[{index}]", "expected a pair of numbers")


def _validate_failsafe(validator: _Validator, failsafe) -> None:
    numeric = {
        "low_battery_remaining": (0.0, 1.0),
        "critical_battery_remaining": (0.0, 1.0),
        "link_loss_timeout_s": (0.1, 600.0),
        "link_loss_critical_s": (0.1, 3600.0),
        "gnss_loss_timeout_s": (0.1, 600.0),
        "rtl_altitude": (1.0, 400.0),
    }
    actions = ("none", "warn", "hold", "rtl", "land", "terminate")
    action_keys = (
        "low_battery_action", "critical_battery_action", "link_loss_action",
        "link_loss_critical_action", "gnss_loss_action", "geofence_action",
    )
    validator.require_keys("failsafe", failsafe, (), tuple(numeric) + action_keys)
    if not isinstance(failsafe, dict):
        return
    for key, (low, high) in numeric.items():
        validator.number(f"failsafe.{key}", failsafe.get(key), low, high)
    for key in action_keys:
        validator.choice(f"failsafe.{key}", failsafe.get(key), actions)

    low = failsafe.get("low_battery_remaining")
    critical = failsafe.get("critical_battery_remaining")
    if isinstance(low, (int, float)) and isinstance(critical, (int, float)):
        if float(critical) >= float(low):
            validator.error(
                "failsafe.critical_battery_remaining",
                "must be below low_battery_remaining, or the low-battery action never fires",
            )
