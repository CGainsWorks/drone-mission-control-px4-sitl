"""Failsafe matrix.

The design decision worth defending: the failsafe logic is a **table**, not a
tree of ``if`` statements. Every rule is a row with a condition, an action, a
priority and a debounce time. That has three consequences that matter.

* The matrix in ``docs/FAILSAFE_MATRIX.md`` is *generated from this table*, so
  the documentation cannot drift away from the behaviour.
* ``tests/test_failsafe.py`` iterates the table and drives each row
  independently, so adding a rule without a test that fires it is visible.
* Priority is explicit. When the battery is critical *and* the link is down
  *and* the vehicle is outside the fence, the outcome is defined by a number in
  a row rather than by which branch happened to be checked first.

Two behaviours are deliberate and both are the conservative choice:

* **Actions latch and escalate but never de-escalate.** A vehicle that has begun
  returning home because the battery is low does not resume the mission because
  the voltage recovered under lighter load — that voltage recovery is a lie the
  battery tells you every time you reduce current. Escalation is allowed:
  a low-battery RTL becomes a critical-battery LAND.
* **Conditions must persist.** Every rule carries a debounce time, so a single
  dropped telemetry packet or one noisy GNSS epoch cannot trigger a return.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FailsafeAction(enum.IntEnum):
    """What to do, ordered by severity. Higher always wins."""

    NONE = 0
    WARN = 1
    HOLD = 2
    RTL = 3
    LAND = 4
    TERMINATE = 5

    @property
    def label(self) -> str:
        return self.name.lower()


ACTION_BY_NAME = {action.label: action for action in FailsafeAction}


@dataclass
class SystemStatus:
    """Everything the failsafe rules are allowed to look at.

    Passing one snapshot rather than the live subsystems keeps the rules pure
    functions of observable state, which is what makes them testable row by row.
    """

    time_s: float = 0.0
    battery_remaining: float = 1.0
    battery_voltage_v: float = 25.2
    link_silence_s: float = 0.0
    link_connected: bool = True
    gnss_usable: bool = True
    gnss_is_rtk: bool = True
    time_since_gnss_update_s: float = 0.0
    position_uncertainty_m: float = 0.0
    estimator_healthy: bool = True
    geofence_breached: bool = False
    geofence_predicted_breach: bool = False
    geofence_margin_m: float = float("inf")
    in_air: bool = False
    armed: bool = False


@dataclass
class FailsafeRule:
    """One row of the matrix."""

    name: str
    #: Human-readable trigger, used in the generated documentation.
    condition_text: str
    #: ``SystemStatus -> bool``.
    condition: object
    action: FailsafeAction
    #: Higher wins when several rules fire. Ties resolve by action severity.
    priority: int
    #: How long the condition must hold before the rule fires, s.
    debounce_s: float = 0.0
    #: Rationale, carried into the generated documentation.
    rationale: str = ""
    #: Rules that only make sense airborne do not fire on the ground.
    airborne_only: bool = True


@dataclass
class FailsafeEvent:
    """A rule firing, recorded for the log and the results table."""

    time_s: float
    rule: str
    action: FailsafeAction
    detail: str = ""


def build_default_rules(spec) -> list:
    """The failsafe matrix, with thresholds taken from a mission's spec.

    ``spec`` is a :class:`~mission_control.mission.schema.FailsafeSpec`.
    """
    resolve = ACTION_BY_NAME.__getitem__

    return [
        FailsafeRule(
            name="battery_critical",
            condition_text=f"battery remaining < {spec.critical_battery_remaining:.0%}",
            condition=lambda s: s.battery_remaining < spec.critical_battery_remaining,
            action=resolve(spec.critical_battery_action),
            priority=100,
            debounce_s=1.0,
            rationale=(
                "Below this there is not enough energy to reach home, so descending "
                "where it is beats running the pack flat en route."
            ),
        ),
        FailsafeRule(
            name="battery_low",
            condition_text=f"battery remaining < {spec.low_battery_remaining:.0%}",
            condition=lambda s: s.battery_remaining < spec.low_battery_remaining,
            action=resolve(spec.low_battery_action),
            priority=50,
            debounce_s=2.0,
            rationale=(
                "Enough left to fly home with reserve. Debounced because voltage sag "
                "under a gust reads as a lower state of charge than is real."
            ),
        ),
        FailsafeRule(
            name="gnss_loss",
            condition_text=f"no usable GNSS for {spec.gnss_loss_timeout_s:.1f} s",
            condition=lambda s: (not s.gnss_usable)
            or s.time_since_gnss_update_s > spec.gnss_loss_timeout_s,
            action=resolve(spec.gnss_loss_action),
            priority=90,
            debounce_s=0.5,
            rationale=(
                "This aircraft has no vision or optical-flow fallback, so without GNSS "
                "the position estimate dead-reckons on IMU alone and diverges within "
                "seconds. RTL is not available because RTL needs a position."
            ),
        ),
        FailsafeRule(
            name="estimator_uncertain",
            condition_text="EKF horizontal position 1-sigma above its limit",
            condition=lambda s: not s.estimator_healthy,
            action=FailsafeAction.LAND,
            priority=85,
            debounce_s=1.0,
            rationale=(
                "The fix can be present and the estimate still untrustworthy — after a "
                "long dropout, or with the innovation gate rejecting everything. "
                "Covariance is the honest test, not fix type."
            ),
        ),
        FailsafeRule(
            name="geofence_breach",
            condition_text="outside the geofence polygon or altitude band",
            condition=lambda s: s.geofence_breached,
            action=resolve(spec.geofence_action),
            priority=80,
            debounce_s=0.0,
            rationale=(
                "No debounce: the position estimate is already filtered, and the fence "
                "has a warning margin inside it, so by the time this fires the vehicle "
                "has ignored several metres of warning."
            ),
        ),
        FailsafeRule(
            name="geofence_predicted",
            condition_text="current velocity leads outside the fence within the lookahead",
            condition=lambda s: s.geofence_predicted_breach,
            action=FailsafeAction.HOLD,
            priority=60,
            debounce_s=0.3,
            rationale=(
                "Stopping before the boundary rather than after it. HOLD rather than "
                "RTL so a mission grazing its own fence is not abandoned outright."
            ),
        ),
        FailsafeRule(
            name="link_loss_critical",
            condition_text=f"no telemetry for {spec.link_loss_critical_s:.0f} s",
            condition=lambda s: s.link_silence_s > spec.link_loss_critical_s,
            action=resolve(spec.link_loss_critical_action),
            priority=70,
            debounce_s=0.0,
            rationale=(
                "The operator is not coming back. Return home, where the aircraft is "
                "recoverable and back in radio range."
            ),
        ),
        FailsafeRule(
            name="link_loss",
            condition_text=f"no telemetry for {spec.link_loss_timeout_s:.0f} s",
            condition=lambda s: s.link_silence_s > spec.link_loss_timeout_s,
            action=resolve(spec.link_loss_action),
            priority=40,
            debounce_s=0.0,
            rationale=(
                "Hold position and wait. Most link losses are brief — an antenna null "
                "on a turn — and abandoning a mission for one is expensive."
            ),
        ),
    ]


class FailsafeMatrix:
    """Evaluates the rules each cycle and resolves them to one action."""

    def __init__(self, rules=None, spec=None):
        if rules is None:
            if spec is None:
                from ..mission.schema import FailsafeSpec

                spec = FailsafeSpec()
            rules = build_default_rules(spec)
        self.rules = list(rules)
        self.action = FailsafeAction.NONE
        self.active_rule = ""
        self.events = []
        self._since = {rule.name: None for rule in self.rules}
        self._fired = set()

    def reset(self) -> None:
        self.action = FailsafeAction.NONE
        self.active_rule = ""
        self.events = []
        self._since = {rule.name: None for rule in self.rules}
        self._fired = set()

    def triggered_rules(self, status: SystemStatus) -> list:
        """Rules whose condition has held for long enough. Ordered by priority."""
        ready = []
        for rule in self.rules:
            if rule.airborne_only and not status.in_air:
                self._since[rule.name] = None
                continue
            try:
                holds = bool(rule.condition(status))
            except Exception:  # a broken rule must not take the vehicle with it
                holds = False
            if not holds:
                self._since[rule.name] = None
                continue
            first_seen = self._since[rule.name]
            if first_seen is None:
                self._since[rule.name] = status.time_s
                first_seen = status.time_s
            if status.time_s - first_seen >= rule.debounce_s:
                ready.append(rule)
        ready.sort(key=lambda rule: (rule.priority, int(rule.action)), reverse=True)
        return ready

    def update(self, status: SystemStatus) -> FailsafeAction:
        """Evaluate the matrix and return the action in force."""
        for rule in self.triggered_rules(status):
            if rule.action > self.action:
                self.action = rule.action
                self.active_rule = rule.name
                self.events.append(
                    FailsafeEvent(
                        time_s=status.time_s,
                        rule=rule.name,
                        action=rule.action,
                        detail=rule.condition_text,
                    )
                )
                self._fired.add(rule.name)
            elif rule.name not in self._fired and rule.action == self.action:
                # Same severity from a different cause: worth recording, since
                # "why did it return" is the first question after a flight.
                self.events.append(
                    FailsafeEvent(
                        time_s=status.time_s,
                        rule=rule.name,
                        action=rule.action,
                        detail=rule.condition_text,
                    )
                )
                self._fired.add(rule.name)
            break  # only the highest-priority rule sets the action
        return self.action

    def to_table(self) -> list:
        """The matrix as rows, for generating documentation."""
        return [
            {
                "rule": rule.name,
                "trigger": rule.condition_text,
                "action": rule.action.label,
                "priority": rule.priority,
                "debounce_s": rule.debounce_s,
                "airborne_only": rule.airborne_only,
                "rationale": rule.rationale,
            }
            for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True)
        ]
