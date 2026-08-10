"""The ULog writer and reader.

A log format is only useful if something else can read it, so the round trip is
checked against this module's own reader always, and against ``pyulog`` when it
happens to be installed. The point of writing ULog rather than CSV is that
PlotJuggler, Flight Review and pyulog can all open the result; a writer that
only this repository can read would have missed the point entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from mission_control.logs.ulog import ULOG_MAGIC, UlogError, UlogReader, UlogWriter

FIELDS = [
    ("timestamp", "uint64_t"),
    ("x", "float"),
    ("y", "float"),
    ("z", "float"),
    ("valid", "bool"),
    ("count", "int32_t"),
]


def write_sample(path, rows=50):
    with UlogWriter(path, start_timestamp_us=1_000_000) as writer:
        writer.write_info("sys_name", "mission_control")
        writer.write_info("run_id", "test-run")
        writer.write_parameter("MPC_XY_P", 1.25)
        writer.write_parameter("horizon", 25)
        writer.add_topic("vehicle_local_position", FIELDS)
        for index in range(rows):
            writer.write(
                "vehicle_local_position",
                timestamp_us=index * 20_000,
                x=index * 0.5, y=-index * 0.25, z=-15.0,
                valid=index % 2 == 0, count=index,
            )
        writer.log_message("mission started", timestamp_us=0, level=6)
    return path


class TestRoundTrip:
    def test_header_and_magic(self, tmp_path):
        path = write_sample(str(tmp_path / "a.ulg"))
        with open(path, "rb") as handle:
            assert handle.read(len(ULOG_MAGIC)) == ULOG_MAGIC

    def test_values_survive(self, tmp_path):
        path = write_sample(str(tmp_path / "b.ulg"), rows=40)
        reader = UlogReader(path)
        data = reader.topics["vehicle_local_position"]
        assert len(data) == 40
        assert data["timestamp"][0] == 0
        assert data["timestamp"][-1] == 39 * 20_000
        assert data["x"][10] == pytest.approx(5.0)
        assert data["y"][10] == pytest.approx(-2.5)
        assert data["z"][0] == pytest.approx(-15.0)
        assert bool(data["valid"][0]) is True
        assert bool(data["valid"][1]) is False
        assert data["count"][7] == 7

    def test_info_and_parameters(self, tmp_path):
        path = write_sample(str(tmp_path / "c.ulg"))
        reader = UlogReader(path)
        assert reader.info["sys_name"] == "mission_control"
        assert reader.info["run_id"] == "test-run"
        assert reader.parameters["MPC_XY_P"] == pytest.approx(1.25)
        assert reader.parameters["horizon"] == 25

    def test_logged_messages(self, tmp_path):
        path = write_sample(str(tmp_path / "d.ulg"))
        reader = UlogReader(path)
        assert reader.messages
        assert reader.messages[0]["text"] == "mission started"
        assert reader.messages[0]["level"] == 6

    def test_batch_matches_single_writes(self, tmp_path):
        single = str(tmp_path / "single.ulg")
        batch = str(tmp_path / "batch.ulg")
        timestamps = np.arange(30, dtype=np.uint64) * 20_000
        columns = {
            "x": np.arange(30) * 0.5,
            "y": -np.arange(30) * 0.25,
            "z": np.full(30, -15.0),
            "valid": np.arange(30) % 2 == 0,
            "count": np.arange(30),
        }
        with UlogWriter(single) as writer:
            writer.add_topic("t", FIELDS)
            for index in range(30):
                writer.write("t", int(timestamps[index]),
                             **{key: value[index] for key, value in columns.items()})
        with UlogWriter(batch) as writer:
            writer.add_topic("t", FIELDS)
            writer.write_batch("t", timestamps, columns)

        first = UlogReader(single).topics["t"]
        second = UlogReader(batch).topics["t"]
        for name in first.dtype.names:
            assert np.array_equal(first[name], second[name])


class TestFieldLayout:
    def test_fields_are_ordered_largest_first(self, tmp_path):
        """ULog packs without padding, so field order is part of the format."""
        path = str(tmp_path / "order.ulg")
        with UlogWriter(path) as writer:
            writer.add_topic(
                "mixed",
                [("timestamp", "uint64_t"), ("small", "uint8_t"), ("big", "double"),
                 ("medium", "float")],
            )
            writer.write("mixed", 0, small=3, big=1.5, medium=2.5)
        reader = UlogReader(path)
        names = [name for name, _ in reader.formats["mixed"]]
        assert names == ["timestamp", "big", "medium", "small"]
        row = reader.topics["mixed"][0]
        assert row["big"] == pytest.approx(1.5)
        assert row["medium"] == pytest.approx(2.5)
        assert row["small"] == 3

    def test_first_field_must_be_timestamp(self, tmp_path):
        with UlogWriter(str(tmp_path / "bad.ulg")) as writer:
            with pytest.raises(UlogError, match="timestamp"):
                writer.add_topic("t", [("x", "float"), ("timestamp", "uint64_t")])

    def test_timestamp_must_be_uint64(self, tmp_path):
        with UlogWriter(str(tmp_path / "bad2.ulg")) as writer:
            with pytest.raises(UlogError, match="uint64_t"):
                writer.add_topic("t", [("timestamp", "float"), ("x", "float")])

    def test_unknown_type_is_rejected(self, tmp_path):
        with UlogWriter(str(tmp_path / "bad3.ulg")) as writer:
            with pytest.raises(UlogError, match="unsupported"):
                writer.add_topic("t", [("timestamp", "uint64_t"), ("x", "float128")])

    def test_missing_field_is_rejected(self, tmp_path):
        with UlogWriter(str(tmp_path / "bad4.ulg")) as writer:
            writer.add_topic("t", FIELDS)
            with pytest.raises(UlogError, match="missing field"):
                writer.write("t", 0, x=1.0)


class TestErrors:
    def test_not_a_ulog_file(self, tmp_path):
        path = str(tmp_path / "junk.ulg")
        with open(path, "wb") as handle:
            handle.write(b"not a ulog at all")
        with pytest.raises(UlogError, match="magic"):
            UlogReader(path)

    def test_topics_cannot_be_added_after_data(self, tmp_path):
        with UlogWriter(str(tmp_path / "late.ulg")) as writer:
            writer.add_topic("t", FIELDS)
            writer.write("t", 0, x=0.0, y=0.0, z=0.0, valid=True, count=0)
            with pytest.raises(UlogError, match="before any data"):
                writer.add_topic("u", FIELDS)

    def test_duplicate_topic_is_rejected(self, tmp_path):
        with UlogWriter(str(tmp_path / "dup.ulg")) as writer:
            writer.add_topic("t", FIELDS)
            with pytest.raises(UlogError, match="already declared"):
                writer.add_topic("t", FIELDS)


class TestFlightLogExport:
    def test_flight_log_writes_a_readable_ulog(self, tmp_path, simple_mission):
        """The real export path, end to end on a short flight."""
        from mission_control.runner import ScenarioConfig, run_scenario

        result = run_scenario(
            ScenarioConfig(mission=simple_mission, tracker="pid", seed=1,
                           max_duration_s=25.0)
        )
        path = str(tmp_path / "flight.ulg")
        result.log.to_ulog(path, info={"tracker": "pid"}, parameters={"gain": 1.25})

        reader = UlogReader(path)
        assert reader.info["tracker"] == "pid"
        assert reader.parameters["gain"] == pytest.approx(1.25)
        for topic in ("vehicle_local_position", "vehicle_local_position_groundtruth",
                      "vehicle_local_position_setpoint", "vehicle_attitude_setpoint",
                      "vehicle_status", "battery_status", "estimator_status"):
            assert topic in reader.topics, f"missing topic {topic}"
            assert len(reader.topics[topic]) == len(result.log)

        truth = reader.topics["vehicle_local_position_groundtruth"]
        assert np.allclose(truth["x"], result.log["true_north"], atol=1e-5)

    def test_csv_export_has_every_column(self, tmp_path, simple_mission):
        import csv

        from mission_control.logs.flight_log import COLUMNS
        from mission_control.runner import ScenarioConfig, run_scenario

        result = run_scenario(
            ScenarioConfig(mission=simple_mission, tracker="pid", seed=1,
                           max_duration_s=25.0)
        )
        path = str(tmp_path / "flight.csv")
        result.log.to_csv(path)
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == list(COLUMNS)
        assert len(rows) == len(result.log) + 1


def test_pyulog_can_read_our_files(tmp_path):
    """Cross-check with the reference implementation, when it is available.

    ``pyulog`` is an optional dependency. When it is absent the round trip is
    still covered by this module's own reader; when it is present this is the
    check that the file really is ULog rather than merely self-consistent.
    """
    pyulog = pytest.importorskip("pyulog")
    path = write_sample(str(tmp_path / "cross.ulg"), rows=25)
    parsed = pyulog.ULog(path)
    names = {dataset.name for dataset in parsed.data_list}
    assert "vehicle_local_position" in names
    dataset = next(d for d in parsed.data_list if d.name == "vehicle_local_position")
    assert len(dataset.data["timestamp"]) == 25
    assert dataset.data["x"][10] == pytest.approx(5.0)
