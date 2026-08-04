# drone-mission-control-px4-sitl

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

> ### Clean-room reimplementation
> This repository is an **independent reimplementation of a problem class**, built from
> scratch on personal time using public or synthetic data and open-source simulation.
> It is **not** derived from, and contains no material belonging to, any employer or
> client: no employer source code, schematics, calibration data, client data or internal
> documentation. It exists to demonstrate the architecture and engineering approach only.

A mission-control layer for an autonomous inspection multirotor, developed against PX4 software-in-the-loop. Compares MPC and PID trajectory tracking on identical missions, with an RTK-GPS and IMU EKF state estimator, geofencing and a tested failsafe matrix.

---

## What this covers

CV entries this repository is the evidence for:

- Autonomous inspection drone mission control (Convex Technology)

---

## Requirements

### Hardware

- None required. PX4 SITL only
- Optional physical target: multirotor with RTK-GPS, IMU, PX4 or ArduPilot flight controller and a companion computer

### Software

- PX4 SITL with Gazebo
- MAVSDK or MAVROS
- RTK-GPS and IMU EKF state estimator
- MPC trajectory tracker (acados, do-mpc or a custom QP)
- PID trajectory tracker for comparison
- Mission state machine
- Geofence and failsafe logic
- PlotJuggler and pyulog for log analysis

### Report and documentation deliverables

- Mission planning schema
- Tracking error results comparing MPC against PID on identical missions
- Failsafe test matrix
- Geofence validation
- Notes on UK CAA drone operating rules for any real-flight extension

---

## Inputs

- Waypoint mission definition
- RTK corrections
- GNSS and IMU measurements
- Geofence polygons
- Battery state

## Outputs

- Attitude and position setpoints over MAVLink
- Mission progress and state
- Tracking error telemetry
- Failsafe triggers with cause
- ULog flight logs

---

## Intended structure

```
src/
  mission/                mission schema, state machine
  estimation/             RTK and IMU EKF
  control/                mpc_tracker, pid_tracker, shared interface
  safety/                 geofence, failsafe matrix
  mavlink_bridge/
missions/                 example mission definitions
analysis/                 ULog parsing, tracking error notebooks
sitl/                     PX4 launch configs, Gazebo worlds
docs/
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] PX4 SITL running with a Gazebo world
- [ ] MAVLink connection, arm, takeoff and land under script control
- [ ] Mission definition schema and loader
- [ ] Mission state machine executing a waypoint sequence
- [ ] EKF state estimator with RTK and IMU inputs
- [ ] PID trajectory tracker
- [ ] MPC trajectory tracker
- [ ] Identical-mission comparison run with error metrics
- [ ] Geofence enforcement
- [ ] Failsafe matrix: link loss, low battery, GNSS loss, geofence breach
- [ ] ULog analysis notebooks

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: drone-mission-control-px4-sitl
A mission-control layer for an autonomous inspection multirotor, developed against PX4 software-in-the-loop. Compares MPC and PID trajectory tracking on identical missions, with an RTK-GPS and IMU EKF state estimator, geofencing and a tested failsafe matrix.

TARGET STACK
- PX4 SITL with Gazebo
- MAVSDK or MAVROS
- RTK-GPS and IMU EKF state estimator
- MPC trajectory tracker (acados, do-mpc or a custom QP)
- PID trajectory tracker for comparison
- Mission state machine
- Geofence and failsafe logic
- PlotJuggler and pyulog for log analysis

HARDWARE CONTEXT
- None required. PX4 SITL only
- Optional physical target: multirotor with RTK-GPS, IMU, PX4 or ArduPilot flight controller and a companion computer

THE SYSTEM MUST TAKE THESE INPUTS
- Waypoint mission definition
- RTK corrections
- GNSS and IMU measurements
- Geofence polygons
- Battery state

AND PRODUCE THESE OUTPUTS
- Attitude and position setpoints over MAVLink
- Mission progress and state
- Tracking error telemetry
- Failsafe triggers with cause
- ULog flight logs

INTENDED REPOSITORY LAYOUT
src/
  mission/                mission schema, state machine
  estimation/             RTK and IMU EKF
  control/                mpc_tracker, pid_tracker, shared interface
  safety/                 geofence, failsafe matrix
  mavlink_bridge/
missions/                 example mission definitions
analysis/                 ULog parsing, tracking error notebooks
sitl/                     PX4 launch configs, Gazebo worlds
docs/

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.
- This is a clean-room reimplementation. Do not reproduce any proprietary design, protocol detail or dataset. Build the equivalent capability independently.

START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
