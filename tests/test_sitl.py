"""The PX4 SITL campaign's definitions, its link-loss relay and its analysis.

None of this needs PX4 or MAVSDK. The flights themselves run in CI's SITL jobs;
these tests pin down everything around them, including that the committed
summary is what the committed flight logs actually say.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time

import numpy as np
import pytest

from mission_control import sitl
from mission_control.analysis.metrics import tracking_metrics
from mission_control.logs.flight_log import STATE_CODES
from mission_control.mavlink_bridge.link_proxy import UdpLinkProxy
from mission_control.mission import load_mission
from mission_control.mission.builder import build_mission_path
from mission_control.safety import Geofence
from mission_control.safety.failsafe import build_default_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISSIONS = os.path.join(ROOT, "missions")
SITL_RESULTS = os.path.join(ROOT, "sitl", "results")


# ---------------------------------------------------------------------------
# Campaign definition
# ---------------------------------------------------------------------------
class TestCampaign:
    def test_full_campaign_flies_every_mission_with_both_trackers(self):
        flights = sitl.campaign_flights("full")
        tracking = {(f["mission"], f["tracker"]) for f in flights if f["scenario"] == "none"}
        assert tracking == {(m, t) for m in sitl.TRACKING_MISSIONS for t in sitl.TRACKERS}

    def test_full_campaign_covers_every_failsafe_row(self):
        flights = sitl.campaign_flights("full")
        scenarios = {f["scenario"] for f in flights if f["scenario"] != "none"}
        assert scenarios == set(sitl.SITL_FAILSAFE_SCENARIOS)
        assert len({f["name"] for f in flights}) == len(flights)

    def test_smoke_is_one_short_flight(self):
        (flight,) = sitl.campaign_flights("smoke")
        mission = load_mission(os.path.join(MISSIONS, f"{flight['mission']}.yaml"))
        trajectory, _ = build_mission_path(np.array([0.0, 0.0, -10.0]), mission)
        assert trajectory.duration < 30.0

    def test_only_the_transfer_flights_use_the_reference_gains(self):
        transfer = sitl.campaign_flights("transfer")
        assert {f["tracker"] for f in transfer} == set(sitl.TRACKERS)
        assert all(f["gains"] == "reference" for f in transfer)
        for kind in ("full", "smoke"):
            assert all(f["gains"] == "airframe" for f in sitl.campaign_flights(kind))

    def test_a_flight_without_a_ulog_reports_no_simulator_faults(self, tmp_path):
        assert sitl.simulator_faults(str(tmp_path)) == []

    def test_unknown_campaign_is_an_error(self):
        with pytest.raises(KeyError):
            sitl.campaign_flights("everything")

    def test_expectations_agree_with_the_matrix(self):
        """Each scenario's expected action is what its rule does on this mission."""
        mission = load_mission(os.path.join(MISSIONS, "facade_inspection.yaml"))
        rules = {rule.name: rule for rule in build_default_rules(mission.failsafe)}
        for name, spec in sitl.SITL_FAILSAFE_SCENARIOS.items():
            assert spec["expect_rule"] in rules, name
            assert rules[spec["expect_rule"]].action.label == spec["expect_action"], name

    def test_tightened_fence_is_a_real_breach_for_the_survey(self):
        """The facade survey must genuinely leave the tightened fence, while
        home stays inside it so a return is possible."""
        mission = load_mission(os.path.join(MISSIONS, "facade_inspection.yaml"))
        spec = sitl.SITL_FAILSAFE_SCENARIOS["geofence_breach"]
        fence = Geofence.from_spec(sitl.tightened_fence(mission.geofence, spec["half_width_m"]))
        trajectory, _ = build_mission_path(np.array([0.0, 0.0, -18.0]), mission)
        outside = [not fence.contains_horizontal(p) for p in trajectory.path.sample(400)]
        assert any(outside)
        assert fence.contains_horizontal(mission.home_ned)
        # The original is untouched.
        assert Geofence.from_spec(mission.geofence).contains_horizontal(
            np.array([25.0, 25.0, -18.0]))

    def test_sih_airframe_matches_the_pinned_px4_defaults(self):
        params = sitl.sih_quadx_params()
        assert params.mass == pytest.approx(1.0)
        assert params.max_thrust == pytest.approx(20.0)
        assert params.thrust_to_weight == pytest.approx(20.0 / 9.80665, rel=1e-3)

    def test_px4_version_matches_the_image(self):
        with open(os.path.join(ROOT, "sitl", "sih", "Dockerfile"), encoding="utf-8") as handle:
            assert f"ARG PX4_VERSION={sitl.PX4_VERSION}" in handle.read()


# ---------------------------------------------------------------------------
# Link-loss relay
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestLinkProxy:
    @pytest.fixture
    def rig(self):
        listen, client_port = _free_port(), _free_port()
        autopilot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        autopilot.bind(("127.0.0.1", 0))
        autopilot.settimeout(1.0)
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.bind(("127.0.0.1", client_port))
        client.settimeout(1.0)
        proxy = UdpLinkProxy(listen_port=listen, client_port=client_port).start()
        yield proxy, autopilot, client, ("127.0.0.1", listen)
        proxy.close()
        autopilot.close()
        client.close()

    def test_forwards_both_directions(self, rig):
        proxy, autopilot, client, relay_addr = rig
        autopilot.sendto(b"heartbeat", relay_addr)
        data, back = client.recvfrom(100)
        assert data == b"heartbeat"
        client.sendto(b"setpoint", back)
        data, _ = autopilot.recvfrom(100)
        assert data == b"setpoint"
        assert proxy.counters["down"] == 1 and proxy.counters["up"] == 1

    def test_drop_cuts_both_directions_then_restores(self, rig):
        proxy, autopilot, client, relay_addr = rig
        autopilot.sendto(b"first", relay_addr)
        _, back = client.recvfrom(100)

        proxy.drop_for(0.3)
        assert proxy.dropping
        autopilot.sendto(b"lost-down", relay_addr)
        client.sendto(b"lost-up", back)
        client.settimeout(0.15)
        with pytest.raises(socket.timeout):
            client.recvfrom(100)
        autopilot.settimeout(0.15)
        with pytest.raises(socket.timeout):
            autopilot.recvfrom(100)

        time.sleep(0.35)
        assert not proxy.dropping
        client.settimeout(1.0)
        autopilot.sendto(b"back", relay_addr)
        assert client.recvfrom(100)[0] == b"back"
        assert proxy.counters["down_dropped"] >= 1 and proxy.counters["up_dropped"] >= 1


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _synthetic_loop(n=1500, rate=50.0):
    t = np.arange(n) / rate
    loop = {name: np.zeros(n) for name in sitl.LOOP_COLUMNS}
    loop["time_s"] = t
    # A circle at 4 m/s, flown perfectly.
    omega = 4.0 / 15.0
    loop["ref_north"] = 15.0 * np.cos(omega * t)
    loop["ref_east"] = 15.0 * np.sin(omega * t)
    loop["ref_down"] = np.full(n, -18.0)
    loop["ref_vn"] = -4.0 * np.sin(omega * t)
    loop["ref_ve"] = 4.0 * np.cos(omega * t)
    for axis in ("north", "east", "down"):
        loop[f"est_{axis}"] = loop[f"ref_{axis}"].copy()
    loop["mission_state"] = np.full(n, float(STATE_CODES["mission"]))
    return loop


class TestAnalysis:
    def test_loop_log_round_trip_is_byte_stable(self, tmp_path):
        rows = [[float(i + j) for j in range(len(sitl.LOOP_COLUMNS))] for i in range(5)]
        first = sitl.write_loop_log(str(tmp_path / "a.csv.gz"), rows)
        second = sitl.write_loop_log(str(tmp_path / "b.csv.gz"), rows)
        assert open(first, "rb").read() == open(second, "rb").read()
        loop = sitl.read_loop_log(first)
        assert list(loop) == list(sitl.LOOP_COLUMNS)
        assert loop["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_clock_alignment_recovers_offset_and_rate(self):
        loop = _synthetic_loop()
        est = np.column_stack([loop["est_north"], loop["est_east"], loop["est_down"]])
        offset, scale = 7.3, 0.95
        # PX4 time for each loop sample, then resampled at 5 Hz like the ULog.
        ulog_t = np.arange(5.0, offset + scale * loop["time_s"][-1] + 2.0, 0.2)
        loop_time_at_ulog = (ulog_t - offset) / scale
        ulog_est = np.column_stack([
            np.interp(loop_time_at_ulog, loop["time_s"], est[:, axis]) for axis in range(3)
        ])
        found_offset, found_scale, residual = sitl.align_clocks(
            loop["time_s"], est, ulog_t, ulog_est)
        assert found_scale == pytest.approx(scale, abs=2e-3)
        assert found_offset == pytest.approx(offset, abs=0.05)
        assert residual < 0.05

    def test_truth_from_the_ulog_feeds_the_shared_metrics(self):
        loop = _synthetic_loop()
        est = np.column_stack([loop["est_north"], loop["est_east"], loop["est_down"]])
        ulog = {
            "truth_t": loop["time_s"] + 3.0,
            "truth_ned": est + np.array([0.0, 0.0, 0.5]),  # truth 0.5 m low
        }
        log = sitl.build_flight_log(loop, ulog, 3.0)
        metrics = tracking_metrics(log, "mission")
        assert metrics["rms_error_m"] == pytest.approx(0.5, abs=1e-6)
        assert metrics["rms_cross_track_m"] == pytest.approx(0.5, abs=1e-3)

    def test_failsafe_match_needs_the_right_rule_and_action(self):
        base = {"scenario": "low_battery", "result": "failsafe"}
        assert sitl.failsafe_matches({**base, "observed_action": "rtl",
                                      "rules_fired": ["battery_low"]})
        assert not sitl.failsafe_matches({**base, "observed_action": "rtl",
                                          "rules_fired": ["geofence_breach"]})
        assert not sitl.failsafe_matches({**base, "observed_action": "land",
                                          "rules_fired": ["battery_low"]})

    def test_readme_block_replaces_cleanly(self):
        summary = {"comparison": [{"mission": "tower_orbit", "pid_rms_m": 1.0, "mpc_rms_m": 0.5,
                                   "pid_cross_track_m": 0.4, "mpc_cross_track_m": 0.3}],
                   "failsafes": []}
        block = sitl.readme_sitl_block(summary)
        text = f"before\n{sitl.README_BEGIN}\nstale\n{sitl.README_END}\nafter\n"
        updated = sitl.replace_block(text, block)
        assert sitl.extract_block(updated) == block
        assert updated.startswith("before\n") and updated.endswith("\nafter\n")
        assert "`tower orbit` | 1.00 m | 0.50 m" in block


# ---------------------------------------------------------------------------
# The committed results
# ---------------------------------------------------------------------------
def _committed_summary():
    path = os.path.join(SITL_RESULTS, "summary.json")
    if not os.path.exists(path):
        pytest.skip("no committed SITL summary")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class TestCommittedResults:
    def test_every_committed_flight_behaved_as_specified(self):
        summary = _committed_summary()
        assert summary["px4_version"] == sitl.PX4_VERSION
        assert "Simulated" in summary["note"]
        for flight in summary["flights"]:
            if flight["scenario"] == "none":
                assert flight["result"] == "completed", flight["flight"]
            else:
                assert flight["matches_expectation"], flight["flight"]

    def test_summary_is_what_the_committed_logs_say(self):
        """Recompute every flight from its committed logs and compare."""
        pytest.importorskip("pyulog")
        summary = _committed_summary()
        for committed in summary["flights"]:
            directory = os.path.join(SITL_RESULTS, "flights", committed["flight"])
            mission = load_mission(os.path.join(MISSIONS, f"{committed['mission']}.yaml"))
            recomputed = sitl.summarise_flight(directory, mission)
            for key in ("rms_error_m", "rms_cross_track_m", "max_error_m", "observed_action",
                        "rules_fired", "landed", "result"):
                expected = committed.get(key)
                actual = recomputed.get(key)
                if isinstance(expected, float):
                    assert actual == pytest.approx(expected, abs=2e-3), (committed["flight"], key)
                else:
                    assert actual == expected, (committed["flight"], key)

    def test_readme_numbers_match_their_sources(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import check_readme

        problems = check_readme.check(
            check_readme.read("README.md"), check_readme.read("docs/RESULTS.md"),
            check_readme.read("docs/ESTIMATOR.md"), check_readme.read("docs/CONTROLLERS.md"),
            _committed_summary(), check_readme.read("sitl/README.md"),
        )
        assert problems == []
