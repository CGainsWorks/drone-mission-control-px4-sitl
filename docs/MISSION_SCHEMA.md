# Mission schema

A mission is a YAML (or JSON) file describing where to fly, how fast, inside
what boundary, and what to do when something goes wrong. It is the one input
that arrives from outside the software, so it is validated strictly: **unknown
keys are an error, not a warning**, and every problem in a file is reported at
once rather than one per run.

Validate a file before flying it:

```bash
python scripts/validate_missions.py missions/facade_inspection.yaml
```

## Top level

| Key | Required | Meaning |
| --- | :---: | --- |
| `name` | yes | Identifier, used in results and log filenames. |
| `origin` | yes | Anchor for the local NED frame. |
| `items` | yes | The flight plan. Must start with a `takeoff`. |
| `description` | no | Free text. |
| `defaults` | no | Values applied to items that do not override them. |
| `geofence` | no | Containment boundary. |
| `failsafe` | no | Thresholds and actions for the failsafe matrix. |
| `home` | no | Where RTL returns to. Defaults to the origin. |

## `origin`

```yaml
origin:
  latitude: 51.5074      # degrees, WGS-84
  longitude: -0.1278
  altitude: 12.0         # metres above the WGS-84 ellipsoid
```

Everything downstream works in metres north/east/down from here. The projection
is a local tangent plane; its error is under a millimetre at 500 m and is
checked in `tests/test_mission_schema.py`.

## `defaults`

| Key | Default | Meaning |
| --- | ---: | --- |
| `cruise_speed` | 4.0 | Target ground speed, m/s. |
| `acceptance_radius` | 1.5 | How close counts as arrived, m. |
| `yaw_mode` | `path` | `path`, `fixed` or `poi`. |
| `max_tangential_acceleration` | 2.5 | Along-track acceleration limit, m/s². |
| `max_lateral_acceleration` | 3.0 | Cornering limit, m/s². Bounds speed in a turn to `sqrt(a·R)`. |
| `max_climb_rate` | 2.5 | m/s. |
| `max_descent_rate` | 1.5 | m/s. |
| `corner_mode` | `round` | `round`, `stop` or `sharp`. |
| `corner_radius` | 4.0 | Radius fitted into a corner in `round` mode, m. |

### `yaw_mode`

- `path` — face the direction of travel. Turns through 180° at a lawnmower
  turnaround.
- `fixed` — hold a constant heading, from `yaw_deg`.
- `poi` — face the first orbit centre in the mission. This is the inspection
  case: the aircraft crabs sideways down a facade with the camera on the wall.

### `corner_mode`

This is the setting that most changes what the comparison measures.

- **`round`** fits an arc of `corner_radius` into each corner, capped so it never
  consumes more than 45% of either leg. The reference stays curvature-limited
  and therefore flyable. This is the realistic case.
- **`stop`** keeps the corner sharp and pins the speed to zero there. Also
  flyable, and slow.
- **`sharp`** keeps the corner sharp and does *not* pin the speed, so the
  reference contains a velocity step no vehicle can follow. Deliberately
  infeasible; the comparison uses it to see what each controller does when asked
  for the impossible.

## `items`

Every item has a `type`. Positions may be given either geodetically
(`latitude`/`longitude`/`altitude`) or in local metres (`north`/`east`/
`altitude`) — but not both in the same item.

### `takeoff`

```yaml
- type: takeoff
  altitude: 18.0         # metres above the origin
```

Must be the first item.

### `waypoint`

```yaml
- type: waypoint
  north: 25.0
  east: -25.0
  altitude: 18.0
  speed: 3.0             # optional override
  hold_s: 3.0            # optional: stop here for this long
  yaw_mode: poi          # optional override
  yaw_deg: 90.0          # required if yaw_mode is fixed
  label: north-west corner
```

### `orbit`

```yaml
- type: orbit
  north: 0.0             # centre
  east: 0.0
  altitude: 20.0
  radius: 18.0
  sweep_deg: 360.0
  clockwise: true
```

The circle is joined **tangentially**: the transit leg runs to the tangent point
rather than to the nearest point on the circle, because arriving radially would
be a 90° velocity step.

### `hold`

```yaml
- type: hold
  hold_s: 10.0
  north: 0.0             # optional; defaults to where the vehicle is
  east: 0.0
```

### `rtl` and `land`

```yaml
- type: rtl              # climb to failsafe.rtl_altitude, transit home, descend
- type: land             # descend where the vehicle is
```

## `geofence`

```yaml
geofence:
  polygon_ned:           # local metres, at least 3 vertices
    - [-80.0, -80.0]
    - [-80.0, 80.0]
    - [80.0, 80.0]
    - [80.0, -80.0]
  # or polygon: [[lat, lon], ...] in degrees
  min_altitude: 3.0
  max_altitude: 60.0
  warning_margin: 6.0    # start warning this far inside the boundary
  action: rtl            # hold | rtl | land | terminate
```

The check is **predictive**: it evaluates where the vehicle will be over a
lookahead, using its current velocity, not only where it is. A vehicle at 4 m/s
is already outside by the time a positional check notices.

The altitude floor is suspended during takeoff and landing, because a vehicle on
the ground is below any floor worth setting. The polygon and ceiling stay armed
for the whole flight.

## `failsafe`

```yaml
failsafe:
  low_battery_remaining: 0.25
  critical_battery_remaining: 0.12
  link_loss_timeout_s: 3.0
  link_loss_critical_s: 15.0
  gnss_loss_timeout_s: 2.0
  rtl_altitude: 35.0
  low_battery_action: rtl
  critical_battery_action: land
  link_loss_action: hold
  link_loss_critical_action: rtl
  gnss_loss_action: land
  geofence_action: rtl
```

Actions are `none`, `warn`, `hold`, `rtl`, `land` or `terminate`. See
[`FAILSAFE_MATRIX.md`](FAILSAFE_MATRIX.md) for the full table, which is
generated from the rules themselves.

`critical_battery_remaining` must be below `low_battery_remaining`, or the
low-battery action never fires — validation rejects it.

## What validation catches

Beyond types and ranges, these cross-field checks run at load time and all of
them are cases that would otherwise arm happily and fail seconds after takeoff:

- A waypoint above the geofence ceiling or below its floor.
- A waypoint, or home, outside the geofence polygon.
- A takeoff altitude above the ceiling.
- A misspelled key, with a suggested correction.
- Geodetic and local coordinates in the same item.
- A geofence whose ceiling is below its floor.

The test suite additionally checks the *generated path* stays inside the fence,
which is a stronger condition than the waypoints doing so: corner rounding cuts
inside a turn, but an orbit bulges outward from its centre.
