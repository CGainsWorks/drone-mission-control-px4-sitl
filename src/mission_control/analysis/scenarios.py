"""The scenario grid the comparison runs over.

Design of the experiment, which matters more than the numbers it produces:

**Paired.** Every scenario is flown by both controllers with the same seed, so
the wind gusts, the IMU noise and the GNSS errors are identical sample for
sample. The difference between two runs of a pair is the controller and nothing
else.

**Held out.** Gains are tuned on ``facade_inspection`` at nominal speed only.
``corridor_survey`` and ``tower_orbit``, the aggressive speeds and the
infeasible-reference case are never seen by the tuner. A controller that only
wins where it was tuned has not won.

**Spanning, not cherry-picked.** The grid deliberately includes the cases where
each controller should do well. Straight legs with gentle corners favour neither.
Continuous curvature favours preview. An infeasible reference punishes whichever
controller is more willing to chase it. Reporting only the flattering half of
that would be worthless.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from ..mission.loader import load_mission
from ..runner import ScenarioConfig
from ..sim.sensors import GnssParams
from ..sim.wind import WindParams
from ..types import GnssFixType

MISSIONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "missions"
)

#: Missions used for tuning. Everything else is held out.
TUNING_MISSIONS = ("facade_inspection",)

#: Wind conditions. "gusty" is roughly a 15 knot day with the turbulence an
#: inspection aircraft actually meets near a structure.
WIND_CONDITIONS = {
    "calm": WindParams(),
    "gusty": WindParams(mean_ned=(4.0, 2.5, 0.0), gust_sigma=1.5, gust_tau=1.8),
}

#: Speed settings, as a multiplier on each mission's own cruise speed.
SPEED_SETTINGS = {"nominal": 1.0, "fast": 1.5}

#: Seeds. Three is enough to see whether a difference survives the noise without
#: making the whole grid take an hour.
SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class ScenarioSpec:
    """One cell of the grid, before a controller is chosen."""

    mission_name: str
    wind: str
    speed: str
    corner_mode: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.mission_name}|{self.speed}|{self.wind}|{self.corner_mode}|s{self.seed}"

    @property
    def label(self) -> str:
        return f"{self.mission_name} {self.speed}/{self.wind}/{self.corner_mode}"

    @property
    def is_held_out(self) -> bool:
        return (
            self.mission_name not in TUNING_MISSIONS
            or self.speed != "nominal"
            or self.corner_mode != "round"
            or self.wind != "calm"
        )


def build_grid(quick: bool = False) -> list:
    """The full scenario grid.

    ``quick`` cuts it to one seed and drops the aggressive settings, for CI
    smoke runs and for anyone who wants an answer in a minute rather than ten.
    """
    missions = ["facade_inspection", "corridor_survey", "tower_orbit"]
    speeds = ["nominal"] if quick else list(SPEED_SETTINGS)
    winds = ["calm"] if quick else list(WIND_CONDITIONS)
    seeds = SEEDS[:1] if quick else SEEDS

    specs = []
    for mission_name in missions:
        for speed in speeds:
            for wind in winds:
                for seed in seeds:
                    specs.append(
                        ScenarioSpec(mission_name, wind, speed, "round", seed)
                    )

    if not quick:
        # The infeasible-reference case. Only on the corridor survey, whose
        # 180-degree turnarounds make a velocity step the most brutal: a
        # reference that reverses direction instantaneously at full speed.
        for wind in WIND_CONDITIONS:
            for seed in SEEDS:
                specs.append(
                    ScenarioSpec("corridor_survey", wind, "nominal", "sharp", seed)
                )
    return specs


def load_mission_for(spec: ScenarioSpec):
    """Load the mission and apply the scenario's speed and corner settings."""
    path = os.path.join(MISSIONS_DIR, f"{spec.mission_name}.yaml")
    mission = load_mission(path)
    mission = copy.deepcopy(mission)
    mission.defaults.cruise_speed *= SPEED_SETTINGS[spec.speed]
    mission.defaults.corner_mode = spec.corner_mode
    return mission


def make_config(spec: ScenarioSpec, tracker: str, use_truth_state: bool = False,
                mission=None) -> ScenarioConfig:
    """Build the runner configuration for one scenario and one controller."""
    mission = mission if mission is not None else load_mission_for(spec)
    return ScenarioConfig(
        mission=mission,
        tracker=tracker,
        seed=spec.seed,
        label=f"{spec.key}|{tracker}",
        wind=WIND_CONDITIONS[spec.wind],
        gnss=GnssParams(
            rate_hz=5.0,
            latency_s=0.08,
            fix_schedule=[(0.0, GnssFixType.RTK_FIXED)],
        ),
        use_truth_state=use_truth_state,
        max_duration_s=400.0,
    )


#: Failsafe scenarios. Separate from the tracking grid because they are not
#: about tracking: each one drives a single row of the failsafe matrix and
#: checks the vehicle does the right thing.
FAILSAFE_SCENARIOS = {
    "link_loss_brief": {
        "description": "telemetry silent for 6 s, long enough to hold, not to return",
        "link_dropouts": ((25.0, 31.0),),
        "expect_action": "hold",
    },
    "link_loss_permanent": {
        "description": "telemetry lost at 25 s and never recovers",
        "link_dropouts": ((25.0, 1e6),),
        "expect_action": "rtl",
    },
    "low_battery": {
        # The drain rate matters and was wrong the first time. At 60x the pack
        # crossed the low threshold and the critical one within a second of each
        # other, so the vehicle landed rather than returning — correct behaviour,
        # but it tested the escalation path rather than the low-battery one. At
        # 3x there is time for the return to actually happen.
        "description": "pack starts at 30% and drains until the low threshold trips",
        "initial_battery": 0.30,
        "battery_drain_scale": 3.0,
        "expect_action": "rtl",
    },
    "battery_critical": {
        "description": "pack starts at 14% and reaches the critical threshold quickly",
        "initial_battery": 0.14,
        "battery_drain_scale": 3.0,
        "expect_action": "land",
    },
    "gnss_degraded": {
        "description": "RTK corrections stop, solution falls back to float",
        "rtk_dropouts": ((25.0, 1e6),),
        "expect_action": "none",
    },
    "gnss_loss": {
        "description": "GNSS drops to no fix at 30 s",
        "gnss_schedule": [(0.0, GnssFixType.RTK_FIXED), (30.0, GnssFixType.NO_FIX)],
        "expect_action": "land",
    },
    "geofence_predicted": {
        # Fence tightened around a survey that still fits inside it. The
        # predictive check should stop the vehicle before it crosses, so the
        # *breach* rule never fires at all. That is the fence working, and it is
        # why this expects a hold rather than a return.
        "description": "fence tightened until the survey grazes it",
        "shrink_geofence": 22.0,
        "shrink_waypoints": 0.6,
        "expect_action": "hold",
    },
    "geofence_breach": {
        # Forcing a *real* breach is harder than it looks, and the reason is a
        # good sign: the predictive check stops the vehicle before the boundary
        # every time, even when the planned path leaves the fence entirely. It
        # fires a hold, the vehicle decelerates in about five metres, and the
        # breach rule never gets to fire.
        #
        # So the fault injected here is a wind beyond what the airframe can hold
        # against. At 15 m/s the drag exceeds the maximum horizontal thrust
        # available at the tilt limit, so the vehicle is pushed downwind whatever
        # it commands, and the crossing genuinely cannot be prevented.
        "description": "fence tightened, then a wind stronger than the aircraft can hold",
        "shrink_geofence": 22.0,
        "shrink_waypoints": 0.6,
        "wind": WindParams(mean_ned=(15.0, 9.0, 0.0), gust_sigma=2.0),
        "expect_action": "rtl",
    },
}
