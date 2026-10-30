#!/usr/bin/env python3
"""Tune both controllers, on the training scenarios only, and record the result.

The rule this script exists to enforce: **both controllers are tuned by the same
procedure, on the same scenarios, against the same objective.** A comparison
where one side was tuned carefully and the other was left at its defaults
measures the tuning, not the controllers.

What is tuned
-------------

*PID*: the position and velocity loop gains are not searched. They are placed —
a natural frequency and a damping ratio per axis give ``vel_p = 2 zeta wn`` and
``pos_p = wn^2 / vel_p`` — and the *search* is over that pair, which is two
interpretable numbers rather than six opaque ones.

*MPC*: horizon length, prediction step, the position/velocity weight ratio, the
input-rate penalty and the disturbance gain.

The objective
-------------

RMS position error over the mission phase, **subject to two constraints**. Both
constraints exist because minimising tracking error alone is not a well-posed
objective — it drives any controller to the edge of stability, and this search
demonstrated that twice before the constraints were added.

*A jerk budget on the acceleration demand.* Physical rather than chosen to make
a number look good: the attitude loop's rate limit is 220 deg/s, so the fastest
a tilt can change is about 3.8 rad/s, which at hover is a horizontal
acceleration slew of roughly ``g * 3.8 = 37 m/s^3``. A demand whose RMS jerk
approaches that is asking the inner loop to live at its rate limit, with no
margin left for a gust. The budget is two thirds of it.

*A step-settling requirement.* The training missions are smooth and feasible by
construction, so a candidate can score beautifully on them and be unflyable on
anything discontinuous. The first tuning to meet the jerk budget did exactly
that: excellent on the survey, and on a 6 m step it settled into a permanent
bang-bang limit cycle. Operators retask aircraft mid-flight, so a discontinuous
reference is not hypothetical.

Both controllers are held to both constraints, which is the part that matters:
neither is allowed to buy accuracy with roughness or instability the other is
denied.

Training set
------------

``facade_inspection`` at nominal speed in calm air, only. Everything else in the
grid — the corridor survey, the tower orbit, the fast settings, the gusty
settings, the infeasible-reference case — is held out and never seen here.

Output goes to ``config/tuned_gains.yaml``, which the comparison harness loads.
Committing that file is what makes a number in the README traceable to the gains
that produced it.

Run:  python scripts/tune_gains.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Before numpy, and before anything that imports it: one BLAS thread per
# process, with multiprocessing supplying the parallelism. See
# mission_control.parallel for why.
from mission_control.parallel import limit_blas_threads, worker_count  # noqa: E402

limit_blas_threads()

import numpy as np  # noqa: E402

from mission_control.analysis.scenarios import (  # noqa: E402
    WIND_CONDITIONS,
    ScenarioSpec,
    load_mission_for,
)
from mission_control.control import MpcTracker, MpcWeights, PidGains, PidTracker  # noqa: E402
from mission_control.runner import ScenarioConfig, run_scenario  # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "config", "tuned_gains.yaml")

#: The plant tuned against. ``reference`` is the comparison's own airframe and
#: attitude loop. ``sih`` is a surrogate for PX4 SITL: PX4 SIH's airframe
#: (mission_control.sitl.sih_quadx_params) with the command delay measured
#: against PX4 by scripts/identify_px4_lag.py, used to place gains for the SITL
#: campaign. Each candidate carries its plant, so worker processes need no
#: shared state.
REFERENCE_PLANT = {"airframe": "reference", "command_delay_s": 0.0, "lag_tau_s": None}

#: Jerk budget on the acceleration demand, m/s^3 RMS. See the module docstring:
#: two thirds of the slew rate the attitude loop can actually deliver at its
#: 220 deg/s rate limit. Applied identically to both controllers.
JERK_BUDGET_MS3 = 25.0

#: Hard cap on a single training flight, s. The tuning mission takes about 140 s
#: to fly, so this is generous — but it has to exist, because the interesting
#: failure mode in a gain sweep is not divergence. A badly tuned candidate that
#: oscillates around the path at a couple of metres never trips the divergence
#: guard, never reaches its final waypoint, and runs until something stops it.
#: One such candidate can take longer than the rest of the sweep combined.
TRAINING_TIMEOUT_S = 260.0

#: Position error above which a training run is abandoned. Much tighter than the
#: runner's own divergence guard, which is sized to catch a vehicle that has been
#: lost rather than one that is merely tracking badly. A candidate this far off
#: is not going to win, so there is nothing to learn by flying the rest of it.
TRAINING_ABANDON_M = 8.0

#: A candidate must also settle after a step in the reference: mean absolute
#: position error below this over the final seconds of a 6 m step, m.
#:
#: This is the second constraint the objective needs, and it exists for the same
#: reason as the jerk budget. The training missions are smooth and feasible by
#: construction, so a candidate can score extremely well on them while being
#: unflyable on anything discontinuous. The first tuning that met the jerk budget
#: did exactly that: excellent on the survey, and on a step it settled into a
#: permanent bang-bang limit cycle, oscillating +-0.28 m with the acceleration
#: demand slamming between its bounds. A real mission produces discontinuities
#: whenever an operator retasks the aircraft, so this is not a hypothetical.
STEP_SETTLE_M = 0.10
#: Size of the step, m, and how long to fly it.
STEP_SIZE_M = 6.0
STEP_DURATION_S = 18.0

#: The training scenarios. Two seeds, so a tuning is not selected by one lucky
#: gust sequence, but only the tuning mission and only calm air.
TRAINING = [
    ScenarioSpec("facade_inspection", "calm", "nominal", "round", seed)
    for seed in (1, 2)
]


def _build_tracker(candidate: dict):
    """Construct a tracker from a plain dict.

    Candidates are dicts rather than closures so the search can be handed to a
    process pool. Tuning is by far the most expensive thing in the repository —
    over a hundred full missions — and running it on one core takes the better
    part of an hour.
    """
    if candidate["tracker"] == "pid":
        gains = PidGains(
            position_p=np.asarray(candidate["position_p"], dtype=float),
            velocity_p=np.asarray(candidate["velocity_p"], dtype=float),
        )
        return lambda p, **kw: PidTracker(p, gains)

    weights = MpcWeights(horizon=candidate["horizon"], step_s=candidate["step_s"])
    if candidate["plant"]["lag_tau_s"] is not None:
        weights.lag_tau_s = float(candidate["plant"]["lag_tau_s"])
    weights.position_weight = np.asarray(candidate["position_weight"], dtype=float)
    weights.velocity_weight = np.asarray(candidate["velocity_weight"], dtype=float)
    weights.input_rate_weight = np.asarray(candidate["input_rate_weight"], dtype=float)
    weights.disturbance_gain = candidate["disturbance_gain"]
    return lambda p, **kw: MpcTracker(p, weights)


class _StepReference:
    """A position step, for the settling check. Deliberately not smooth."""

    def __init__(self, start, target, at_s=1.0):
        import numpy as _np

        self.start = _np.asarray(start, dtype=float)
        self.target = _np.asarray(target, dtype=float)
        self.at_s = at_s

    def sample(self, time_s):
        from mission_control.types import TrajectorySetpoint

        position = self.target if time_s >= self.at_s else self.start
        return TrajectorySetpoint(time_s=time_s, position=position.copy())

    def sample_horizon(self, start_s, step_s, count):
        return [self.sample(start_s + (i + 1) * step_s) for i in range(count)]


def _vehicle(plant: dict):
    from mission_control import sitl
    from mission_control.sim import QuadrotorParams

    if plant["airframe"] == "reference":
        return QuadrotorParams()
    return sitl.AIRFRAMES[plant["airframe"]]()


def step_settling(candidate: dict) -> float:
    """Mean absolute position error over the tail of a step response, m.

    Cheap next to a mission run — eighteen seconds of flight against a hundred
    and forty — and it is the check that rejects a candidate which tracks a
    smooth trajectory beautifully and limit-cycles on a discontinuous one.
    """
    from mission_control.sim import InnerLoop, QuadrotorPlant
    from mission_control.types import VehicleState

    params = _vehicle(candidate["plant"])
    delay_steps = int(round(candidate["plant"]["command_delay_s"] / 0.02))
    in_flight = []
    tracker = _build_tracker(candidate)(params)
    start = np.array([0.0, 0.0, -15.0])
    reference = _StepReference(start, start + np.array([STEP_SIZE_M, 0.0, 0.0]))

    plant = QuadrotorPlant(params, VehicleState(position=start.copy()))
    loop = InnerLoop(params)
    tracker.reset(plant.vehicle_state())

    sim_dt, control_dt = 0.002, 0.02
    setpoint = None
    next_control = 0.0
    time_s = 0.0
    tail_errors = []
    for _ in range(int(STEP_DURATION_S / sim_dt)):
        if time_s + 1e-9 >= next_control:
            next_control += control_dt
            in_flight.append(tracker.update(plant.vehicle_state(time_s), reference,
                                            time_s, control_dt))
            # The command the attitude loop acts on is the one issued
            # delay_steps control periods ago: the surrogate's transport delay.
            setpoint = in_flight.pop(0) if len(in_flight) > delay_steps else in_flight[0]
            if time_s > STEP_DURATION_S - 5.0:
                tail_errors.append(
                    float(np.linalg.norm(plant.state[0:3] - reference.sample(time_s).position))
                )
        command = loop.update(plant.state[6:10], plant.state[10:13], setpoint, sim_dt)
        plant.step(command, sim_dt)
        time_s += sim_dt

    return float(np.mean(tail_errors)) if tail_errors else float("inf")


def score(candidate: dict) -> tuple:
    """Score one candidate. Returns ``(rms, peak, jerk, settle)``.

    A diverged run scores infinity rather than a large number, so the search can
    never trade a divergence against a slightly better average elsewhere.
    """
    import mission_control.control as control_registry
    from mission_control.analysis.metrics import tracking_metrics

    tracker_name = candidate["tracker"]
    factory = _build_tracker(candidate)
    errors, peaks, jerks = [], [], []

    for spec in candidate["specs"]:
        mission = load_mission_for(spec)
        config = ScenarioConfig(
            mission=mission,
            tracker=tracker_name,
            seed=spec.seed,
            wind=WIND_CONDITIONS[spec.wind],
            max_duration_s=TRAINING_TIMEOUT_S,
            vehicle=_vehicle(candidate["plant"]),
            command_delay_s=candidate["plant"]["command_delay_s"],
            divergence_limit_m=TRAINING_ABANDON_M,
        )
        original = control_registry.TRACKERS[tracker_name]
        control_registry.TRACKERS[tracker_name] = factory
        try:
            result = run_scenario(config)
        finally:
            control_registry.TRACKERS[tracker_name] = original

        mask = result.log.mask_state("mission")
        if result.diverged or not np.any(mask):
            return float("inf"), float("inf"), float("inf"), float("inf")
        metrics = tracking_metrics(result.log, "mission")
        errors.append(metrics["rms_error_m"])
        peaks.append(metrics["max_error_m"])
        jerks.append(metrics["rms_jerk_ms3"])

    return (float(np.mean(errors)), float(np.max(peaks)), float(np.max(jerks)),
            step_settling(candidate))


def feasible(jerk: float, settle: float) -> bool:
    """Whether a candidate meets both constraints."""
    return jerk <= JERK_BUDGET_MS3 and settle <= STEP_SETTLE_M


def select_best(candidates: list, scores: list):
    """Lowest RMS among candidates meeting both constraints.

    If nothing qualifies the search does not silently pick the least-bad
    violator: it says so, because a grid where everything is infeasible is a
    grid that was set up wrong.
    """
    usable = [
        (rms, peak, jerk, settle, candidate)
        for candidate, (rms, peak, jerk, settle) in zip(candidates, scores)
        if np.isfinite(rms)
    ]
    qualified = [entry for entry in usable if feasible(entry[2], entry[3])]
    if qualified:
        return min(qualified, key=lambda entry: entry[0]), True

    print(f"  WARNING: no candidate met both the {JERK_BUDGET_MS3:.0f} m/s^3 jerk "
          f"budget and the {STEP_SETTLE_M:.2f} m step-settling requirement; "
          "falling back to the best-settling candidate")
    return min(usable, key=lambda entry: entry[3]), False


def evaluate(candidates: list, workers: int = None, progress=None) -> list:
    """Score every candidate, in parallel where possible.

    Uses ``imap`` rather than ``map`` so results can be reported as they land.
    ``map`` blocks until the whole batch finishes, which on a sweep this size is
    several minutes of a script that looks like it has hung.
    """
    workers = worker_count(workers)
    if workers <= 1 or len(candidates) == 1:
        results = []
        for index, candidate in enumerate(candidates):
            results.append(score(candidate))
            if progress:
                progress(index, results[-1])
        return results

    import multiprocessing

    context = multiprocessing.get_context("spawn")
    results = []
    with context.Pool(workers) as pool:
        # imap preserves order, so results line up with candidates.
        for index, result in enumerate(pool.imap(score, candidates)):
            results.append(result)
            if progress:
                progress(index, result)
    return results


def tune_pid(quick=False, workers=None, plant=REFERENCE_PLANT) -> dict:
    """Search over closed-loop natural frequency and damping."""
    print("Tuning PID (pole placement over natural frequency and damping)")

    # Ranges chosen to *bracket* the optimum rather than stop at it. A first
    # pass topped out at the edge of the grid (wn_xy = 2.4, wn_z = 3.5), which
    # means the search had not converged, only run out of room — and a
    # comparison against an under-tuned baseline is worth nothing. These extend
    # far enough that the best value has worse neighbours on both sides.
    horizontal = [(2.4, 0.8), (3.6, 0.8), (4.8, 0.8), (6.0, 0.8), (4.8, 1.0)]
    vertical = [3.5, 5.0, 6.5]
    if plant["airframe"] != "reference":
        # A slower plant: the same shape of grid, moved down so it still
        # brackets the optimum rather than stopping at its upper edge.
        # A first pass over 0.8 to 2.4 rad/s ran to its upper edge, so this
        # one extends far enough to have worse neighbours on both sides.
        horizontal = [(1.2, 0.8), (1.8, 0.8), (2.4, 0.8), (3.0, 0.8), (3.6, 0.8),
                      (2.4, 1.0)]
        vertical = [2.5, 3.5]
    specs = TRAINING[:1] if quick else TRAINING
    if quick:
        horizontal, vertical = horizontal[:2], vertical[:1]

    candidates = []
    for (wn_xy, zeta_xy), wn_z in itertools.product(horizontal, vertical):
        zeta_z = 0.9
        velocity_xy = 2.0 * zeta_xy * wn_xy
        velocity_z = 2.0 * zeta_z * wn_z
        candidates.append({
            "tracker": "pid",
            "plant": plant,
            "specs": specs,
            "wn_xy": wn_xy, "zeta_xy": zeta_xy, "wn_z": wn_z, "zeta_z": zeta_z,
            "position_p": [wn_xy**2 / velocity_xy, wn_xy**2 / velocity_xy,
                           wn_z**2 / velocity_z],
            "velocity_p": [velocity_xy, velocity_xy, velocity_z],
        })

    print(f"  {len(candidates)} candidates x {len(specs)} scenario(s), "
          f"jerk budget {JERK_BUDGET_MS3:.0f} m/s^3, "
          f"step settling {STEP_SETTLE_M:.2f} m")
    print(f"  {'wn_xy':>6} {'zeta_xy':>8} {'wn_z':>6} {'RMS (m)':>9} {'max (m)':>9} "
          f"{'jerk':>8} {'settle':>8}")

    def report(index, result):
        candidate = candidates[index]
        rms, peak, jerk, settle = result
        mark = "  " if feasible(jerk, settle) else (
            " J" if jerk > JERK_BUDGET_MS3 else " S")
        print(f"  {candidate['wn_xy']:6.2f} {candidate['zeta_xy']:8.2f} "
              f"{candidate['wn_z']:6.2f} {rms:9.4f} {peak:9.4f} {jerk:8.2f} "
              f"{settle:8.4f}{mark}", flush=True)

    scores = evaluate(candidates, workers, progress=report)
    print("  (J = over the jerk budget, S = does not settle after a step)")

    (rms, peak, jerk, settle, chosen), _ = select_best(candidates, scores)
    wn_xy, zeta_xy = chosen["wn_xy"], chosen["zeta_xy"]
    wn_z, zeta_z = chosen["wn_z"], chosen["zeta_z"]
    gains = PidGains(
        position_p=np.asarray(chosen["position_p"], dtype=float),
        velocity_p=np.asarray(chosen["velocity_p"], dtype=float),
    )
    print(f"  best: wn_xy={wn_xy}, zeta_xy={zeta_xy}, wn_z={wn_z} -> RMS {rms:.4f} m\n")
    return {
        "natural_frequency_xy": wn_xy,
        "damping_xy": zeta_xy,
        "natural_frequency_z": wn_z,
        "damping_z": zeta_z,
        "position_p": [round(float(v), 4) for v in gains.position_p],
        "velocity_p": [round(float(v), 4) for v in gains.velocity_p],
        "velocity_i": [round(float(v), 4) for v in gains.velocity_i],
        "velocity_d": [round(float(v), 4) for v in gains.velocity_d],
        "training_rms_m": round(rms, 5),
        "training_max_m": round(peak, 5),
        "training_jerk_ms3": round(jerk, 3),
        "training_step_settle_m": round(settle, 4),
    }


def tune_mpc(quick=False, workers=None, plant=REFERENCE_PLANT) -> dict:
    """Search over horizon, weights and the disturbance gain."""
    print("Tuning MPC (horizon, weights, disturbance gain)")

    # As with the PID, these bracket the optimum rather than stopping at it.
    #
    # Horizon length is in the search but barely moves the result: 1.6 s and
    # 2.0 s of preview score identically to four decimal places on these
    # missions, because the vehicle's settling time is well under a second and
    # nothing useful happens beyond that. It is kept in the search so that
    # statement is measured rather than assumed.
    horizons = [(20, 0.08), (25, 0.08)]
    position_weights = [30.0, 100.0, 300.0]
    rate_weights = [0.02, 0.10, 0.50]
    disturbance_gains = [4.0, 8.0]
    if plant["airframe"] != "reference":
        # Against a plant with a command delay the optimum moves to lighter
        # position weights and heavier input-rate penalties: a first pass on
        # the grid above ran to its lowest q_pos and highest r_du, and every
        # candidate there with r_du of 0.10 or less broke the jerk budget at
        # q_pos 30. This grid is centred on that region instead. One horizon
        # only, since the reference search showed horizon length does not
        # move the result.
        horizons = [(20, 0.08)]
        position_weights = [10.0, 30.0, 60.0, 100.0]
        rate_weights = [0.2, 0.5, 2.0]
    specs = TRAINING[:1] if quick else TRAINING
    if quick:
        horizons, position_weights = horizons[:1], position_weights[:1]
        rate_weights, disturbance_gains = rate_weights[:1], disturbance_gains[:1]

    candidates = []
    for (horizon, step), q_pos, r_du, k_d in itertools.product(
        horizons, position_weights, rate_weights, disturbance_gains
    ):
        candidates.append({
            "tracker": "mpc",
            "plant": plant,
            "specs": specs,
            "horizon": horizon, "step_s": step, "q_pos": q_pos,
            "r_du": r_du, "disturbance_gain": k_d,
            "position_weight": [q_pos, q_pos, q_pos * 1.5],
            "velocity_weight": [q_pos * 0.1, q_pos * 0.1, q_pos * 0.15],
            "input_rate_weight": [r_du, r_du, r_du],
        })

    print(f"  {len(candidates)} candidates x {len(specs)} scenario(s), "
          f"jerk budget {JERK_BUDGET_MS3:.0f} m/s^3, "
          f"step settling {STEP_SETTLE_M:.2f} m")
    print(f"  {'N':>3} {'dt':>5} {'span':>5} {'q_pos':>7} {'r_du':>6} {'k_d':>5} "
          f"{'RMS (m)':>9} {'max (m)':>9} {'jerk':>8} {'settle':>8}")

    def report(index, result):
        candidate = candidates[index]
        rms, peak, jerk, settle = result
        mark = "  " if feasible(jerk, settle) else (
            " J" if jerk > JERK_BUDGET_MS3 else " S")
        print(f"  {candidate['horizon']:3d} {candidate['step_s']:5.2f} "
              f"{candidate['horizon'] * candidate['step_s']:5.2f} {candidate['q_pos']:7.1f} "
              f"{candidate['r_du']:6.2f} {candidate['disturbance_gain']:5.1f} "
              f"{rms:9.4f} {peak:9.4f} {jerk:8.2f} {settle:8.4f}{mark}", flush=True)

    scores = evaluate(candidates, workers, progress=report)
    print("  (J = over the jerk budget, S = does not settle after a step)")

    (rms, peak, jerk, settle, chosen), _ = select_best(candidates, scores)
    horizon, step = chosen["horizon"], chosen["step_s"]
    q_pos, r_du, k_d = chosen["q_pos"], chosen["r_du"], chosen["disturbance_gain"]
    weights = MpcWeights(horizon=horizon, step_s=step)
    if plant["lag_tau_s"] is not None:
        weights.lag_tau_s = float(plant["lag_tau_s"])
    weights.position_weight = np.asarray(chosen["position_weight"], dtype=float)
    weights.velocity_weight = np.asarray(chosen["velocity_weight"], dtype=float)
    weights.input_rate_weight = np.asarray(chosen["input_rate_weight"], dtype=float)
    weights.disturbance_gain = k_d
    print(f"  best: N={horizon}, dt={step}, q_pos={q_pos}, r_du={r_du}, k_d={k_d} "
          f"-> RMS {rms:.4f} m\n")
    return {
        "horizon": horizon,
        "step_s": step,
        "horizon_s": round(horizon * step, 3),
        "lag_tau_s": weights.lag_tau_s,
        "position_weight": [round(float(v), 4) for v in weights.position_weight],
        "velocity_weight": [round(float(v), 4) for v in weights.velocity_weight],
        "input_weight": [round(float(v), 4) for v in weights.input_weight],
        "input_rate_weight": [round(float(v), 4) for v in weights.input_rate_weight],
        "terminal_scale": weights.terminal_scale,
        "disturbance_gain": k_d,
        "disturbance_limit": weights.disturbance_limit,
        "training_rms_m": round(rms, 5),
        "training_max_m": round(peak, 5),
        "training_jerk_ms3": round(jerk, 3),
        "training_step_settle_m": round(settle, 4),
    }


#: Prose describing the objective, written into the generated file. Kept as a
#: block scalar there: it contains colons, and an unquoted YAML scalar
#: containing a colon is a syntax error rather than a string — which is exactly
#: how the first version of this file turned out to be unreadable by the loader
#: that consumes it.
OBJECTIVE_TEXT = (
    "Minimise mean RMS position error over the mission phase, measured against "
    "ground truth, subject to the acceleration demand's RMS jerk staying within "
    f"{JERK_BUDGET_MS3:.0f} m/s^3. The constraint is what makes the problem well "
    "posed: minimising error alone drives any controller to the edge of "
    "stability. Both controllers are held to the same budget."
)


def write_config(pid: dict, mpc: dict, path: str = CONFIG_PATH,
                 plant: dict = REFERENCE_PLANT) -> str:
    """Write ``config/tuned_gains.yaml``, then read it back to prove it parses.

    The read-back is not paranoia. This file is generated prose and numbers, and
    the thing that consumes it is a YAML parser; a generator that emits something
    almost-valid produces a failure a long way from its cause.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Generated by scripts/tune_gains.py. Do not edit by hand.",
        "#",
        "# Both controllers were tuned by the same procedure, on the same training",
        "# scenarios, against the same constrained objective, spelled out below.",
        "# Every other scenario in the comparison grid is held out and never seen",
        "# by the tuner.",
        "",
        "training:",
        "  mission: facade_inspection",
        "  speed: nominal",
        "  wind: calm",
        "  seeds: [1, 2]",
        f"  jerk_budget_ms3: {JERK_BUDGET_MS3}",
        f"  step_settle_budget_m: {STEP_SETTLE_M}",
        "  objective: >-",
    ]
    lines.extend(f"    {line}" for line in _wrap(OBJECTIVE_TEXT, 72))
    if plant["airframe"] != "reference":
        lines.append(f"  airframe: {plant['airframe']}")
        lines.append(f"  command_delay_s: {plant['command_delay_s']}")
        lines.append(f"  mpc_lag_tau_s: {plant['lag_tau_s']}")
    lines.append("")

    for section, values in (("pid", pid), ("mpc", mpc)):
        lines.append(f"{section}:")
        for key, value in values.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # newline="\n" so the file is byte-identical whichever platform generated
    # it: CI regenerates and compares, and a CRLF/LF mismatch would read as a
    # change that nobody made.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except ImportError:
        return path
    for section in ("training", "pid", "mpc"):
        if section not in parsed:
            raise RuntimeError(
                f"{path} was written but parses without a {section!r} "
                "section; the generator is emitting invalid YAML"
            )
    return path


def _wrap(text: str, width: int) -> list:
    import textwrap

    return textwrap.wrap(text, width=width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="a much smaller search, for smoke-testing the script")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="parallel workers (default: cores - 1)")
    parser.add_argument("--airframe", default="reference", choices=["reference", "sih"],
                        help="sih: tune for the PX4 SITL campaign, on a surrogate of "
                             "PX4 SIH with its measured command delay")
    args = parser.parse_args()

    if args.airframe == "reference":
        plant, output = REFERENCE_PLANT, CONFIG_PATH
    else:
        from mission_control import sitl

        plant = {"airframe": args.airframe,
                 "command_delay_s": sitl.PX4_SIH_COMMAND_DELAY_S,
                 "lag_tau_s": sitl.PX4_SIH_COMMAND_DELAY_S + sitl.PX4_SIH_LAG_TAU_S}
        output = sitl.SIH_GAINS_PATH

    started = time.perf_counter()
    pid = tune_pid(args.quick, args.workers, plant)
    mpc = tune_mpc(args.quick, args.workers, plant)
    path = write_config(pid, mpc, output, plant)

    print(f"wrote {os.path.relpath(path, ROOT)} in {time.perf_counter() - started:.1f} s")
    print()
    print(f"  PID training RMS  {pid['training_rms_m']:.4f} m  "
          f"(jerk {pid['training_jerk_ms3']:.1f} of {JERK_BUDGET_MS3:.0f} m/s^3)")
    print(f"  MPC training RMS  {mpc['training_rms_m']:.4f} m  "
          f"(jerk {mpc['training_jerk_ms3']:.1f} of {JERK_BUDGET_MS3:.0f} m/s^3)")
    print()
    print("These are training numbers. They are not the result; the result is what")
    print("happens on the held-out scenarios, which analysis/run_comparison.py measures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
