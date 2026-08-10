"""ULog v1 reader and writer.

ULog is PX4's flight log format. Writing it, rather than a pile of CSV, means
the logs from this repository open in PlotJuggler, ``pyulog``, Flight Review and
every other tool an autopilot engineer already has — which is most of the value
of using a standard format at all.

There is no dependency on ``pyulog``: the format is short enough to implement
directly, and a self-contained implementation is the difference between a
notebook that runs on a clean machine and one that does not. ``pyulog`` is used
in the tests as an *optional* cross-check when it happens to be installed;
otherwise the round-trip test uses this module's own reader.

Format summary
--------------

A header, then a stream of length-prefixed messages::

    header:  b'ULog\\x01\\x12\\x35'  version:uint8  timestamp:uint64
    message: size:uint16  type:uint8  payload[size]

The message types implemented here are the ones a flight log actually needs:

======  ==========================================================
type    meaning
======  ==========================================================
``B``   flag bits; must be the first message
``F``   format definition: ``name:type field;type field;``
``I``   information, key/value
``P``   parameter, key/value
``A``   subscribe a format to a numeric message id
``D``   one data sample for a subscribed id
``L``   a logged text message with a level and a timestamp
======  ==========================================================

Two details the specification is strict about and this implementation honours:

* Every logged message's **first field must be** ``uint64_t timestamp``, in
  microseconds. Readers rely on it.
* Fields are laid out with **no padding**, so they are ordered by descending
  size. The writer sorts them, rather than trusting the caller to.
"""

from __future__ import annotations

import struct
from collections import OrderedDict

import numpy as np

ULOG_MAGIC = b"ULog\x01\x12\x35"
ULOG_VERSION = 1

MESSAGE_FLAG_BITS = b"B"
MESSAGE_FORMAT = b"F"
MESSAGE_INFO = b"I"
MESSAGE_PARAMETER = b"P"
MESSAGE_ADD_LOGGED = b"A"
MESSAGE_DATA = b"D"
MESSAGE_LOGGING = b"L"

#: ULog type name to (struct code, size in bytes, numpy dtype).
TYPES = OrderedDict(
    [
        ("uint64_t", ("Q", 8, "u8")),
        ("int64_t", ("q", 8, "i8")),
        ("double", ("d", 8, "f8")),
        ("float", ("f", 4, "f4")),
        ("uint32_t", ("I", 4, "u4")),
        ("int32_t", ("i", 4, "i4")),
        ("uint16_t", ("H", 2, "u2")),
        ("int16_t", ("h", 2, "i2")),
        ("uint8_t", ("B", 1, "u1")),
        ("int8_t", ("b", 1, "i1")),
        ("bool", ("?", 1, "?")),
    ]
)


class UlogError(ValueError):
    """Raised when a ULog file is malformed."""


def _sorted_fields(fields):
    """Order fields largest-first so the packed layout needs no padding.

    ``timestamp`` is forced to the front regardless; the format requires it
    there and it is a ``uint64_t`` so it sorts there anyway, but being explicit
    means a caller who renames it gets an error rather than a subtly bad file.
    """
    for name, type_name in fields:
        if type_name not in TYPES:
            raise UlogError(f"unsupported ULog field type {type_name!r} for field {name!r}")
    if not fields or fields[0][0] != "timestamp":
        raise UlogError("the first field of a logged message must be 'timestamp'")
    if fields[0][1] != "uint64_t":
        raise UlogError("'timestamp' must be uint64_t")
    rest = sorted(fields[1:], key=lambda field: -TYPES[field[1]][1])
    return [fields[0]] + rest


class UlogWriter:
    """Writes a ULog v1 file."""

    def __init__(self, path: str, start_timestamp_us: int = 0):
        self._handle = open(path, "wb")
        self.path = path
        self._formats = OrderedDict()
        self._message_ids = {}
        self._struct = {}
        self._next_id = 0
        self._definitions_closed = False

        self._handle.write(ULOG_MAGIC)
        self._handle.write(struct.pack("<BQ", ULOG_VERSION, int(start_timestamp_us)))
        # Flag bits: no compatibility flags set, no appended data.
        self._write_message(MESSAGE_FLAG_BITS, bytes(8) + bytes(8) + struct.pack("<3Q", 0, 0, 0))

    # ------------------------------------------------------------------
    def _write_message(self, message_type: bytes, payload: bytes) -> None:
        if len(payload) > 0xFFFF:
            raise UlogError("ULog message payload exceeds 65535 bytes")
        self._handle.write(struct.pack("<HB", len(payload), ord(message_type)) + payload)

    def add_topic(self, name: str, fields) -> None:
        """Declare a message format and subscribe it to a message id."""
        if self._definitions_closed:
            raise UlogError("topics must be added before any data is written")
        if name in self._formats:
            raise UlogError(f"topic {name!r} already declared")

        ordered = _sorted_fields(list(fields))
        definition = name + ":" + "".join(
            f"{type_name} {field};" for field, type_name in ordered
        )
        self._write_message(MESSAGE_FORMAT, definition.encode("utf-8"))

        self._formats[name] = ordered
        self._struct[name] = struct.Struct(
            "<" + "".join(TYPES[type_name][0] for _, type_name in ordered)
        )

    def write_info(self, key: str, value) -> None:
        """Write an information key/value pair (strings and floats)."""
        self._write_key_value(MESSAGE_INFO, key, value)

    def write_parameter(self, key: str, value) -> None:
        """Write a parameter. Used to record controller gains inside the log."""
        self._write_key_value(MESSAGE_PARAMETER, key, value)

    def _write_key_value(self, message_type: bytes, key: str, value) -> None:
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            type_name = f"char[{len(encoded)}]"
            payload_value = encoded
        elif isinstance(value, (bool, np.bool_)):
            type_name, payload_value = "bool", struct.pack("<?", bool(value))
        elif isinstance(value, (int, np.integer)):
            type_name, payload_value = "int32_t", struct.pack("<i", int(value))
        else:
            type_name, payload_value = "float", struct.pack("<f", float(value))

        key_field = f"{type_name} {key}".encode("utf-8")
        self._write_message(
            message_type, struct.pack("<B", len(key_field)) + key_field + payload_value
        )

    def _close_definitions(self) -> None:
        """Emit the subscriptions that mark the start of the data section."""
        if self._definitions_closed:
            return
        for name in self._formats:
            message_id = self._next_id
            self._next_id += 1
            self._message_ids[name] = message_id
            self._write_message(
                MESSAGE_ADD_LOGGED,
                struct.pack("<BH", 0, message_id) + name.encode("utf-8"),
            )
        self._definitions_closed = True

    # ------------------------------------------------------------------
    def write(self, topic: str, timestamp_us: int, **values) -> None:
        """Write one sample of ``topic``. Every declared field must be given."""
        self._close_definitions()
        if topic not in self._formats:
            raise UlogError(f"unknown topic {topic!r}")

        packed = [int(timestamp_us)]
        for field, type_name in self._formats[topic][1:]:
            if field not in values:
                raise UlogError(f"missing field {field!r} for topic {topic!r}")
            value = values[field]
            if type_name == "bool":
                packed.append(bool(value))
            elif TYPES[type_name][0] in "QqIiHhBb":
                packed.append(int(value))
            else:
                packed.append(float(value))

        payload = struct.pack("<H", self._message_ids[topic]) + self._struct[topic].pack(*packed)
        self._write_message(MESSAGE_DATA, payload)

    def write_batch(self, topic: str, timestamps_us, columns: dict) -> None:
        """Write many samples at once from arrays. Much faster than a loop."""
        self._close_definitions()
        if topic not in self._formats:
            raise UlogError(f"unknown topic {topic!r}")

        timestamps = np.asarray(timestamps_us, dtype=np.uint64)
        fields = self._formats[topic][1:]
        arrays = []
        for field, type_name in fields:
            if field not in columns:
                raise UlogError(f"missing column {field!r} for topic {topic!r}")
            arrays.append(np.asarray(columns[field]).reshape(-1))
            if arrays[-1].size != timestamps.size:
                raise UlogError(f"column {field!r} length does not match timestamps")

        packer = self._struct[topic]
        header = struct.pack("<HB", 2 + packer.size, ord(MESSAGE_DATA)) + struct.pack(
            "<H", self._message_ids[topic]
        )
        chunks = []
        for index in range(timestamps.size):
            row = [int(timestamps[index])]
            for (field, type_name), array in zip(fields, arrays):
                value = array[index]
                if type_name == "bool":
                    row.append(bool(value))
                elif TYPES[type_name][0] in "QqIiHhBb":
                    row.append(int(value))
                else:
                    row.append(float(value))
            chunks.append(header + packer.pack(*row))
        self._handle.write(b"".join(chunks))

    def log_message(self, text: str, timestamp_us: int, level: int = 6) -> None:
        """Write a text log entry. ``level`` follows syslog: 6 is info, 4 warning."""
        self._close_definitions()
        self._write_message(
            MESSAGE_LOGGING,
            struct.pack("<BQ", int(level), int(timestamp_us)) + text.encode("utf-8"),
        )

    def close(self) -> None:
        self._close_definitions()
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class UlogReader:
    """Reads a ULog v1 file written by :class:`UlogWriter` or by PX4.

    Data comes back as numpy structured arrays keyed by topic name, which is
    what the analysis notebooks want. Unknown message types are skipped rather
    than raising, so a log written by a newer PX4 still opens.
    """

    def __init__(self, path: str):
        with open(path, "rb") as handle:
            self._data = handle.read()
        self.path = path
        self.formats = OrderedDict()
        self.info = {}
        self.parameters = {}
        self.messages = []
        self.topics = {}
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self) -> None:
        data = self._data
        if not data.startswith(ULOG_MAGIC):
            raise UlogError("not a ULog file: bad magic")
        offset = len(ULOG_MAGIC)
        self.version, self.start_timestamp_us = struct.unpack_from("<BQ", data, offset)
        offset += 9

        subscriptions = {}
        raw = {}

        while offset + 3 <= len(data):
            size, message_type = struct.unpack_from("<HB", data, offset)
            offset += 3
            if offset + size > len(data):
                raise UlogError("truncated ULog message")
            payload = data[offset: offset + size]
            offset += size
            kind = bytes([message_type])

            if kind == MESSAGE_FORMAT:
                name, fields = self._parse_format(payload.decode("utf-8"))
                self.formats[name] = fields
            elif kind == MESSAGE_INFO:
                key, value = self._parse_key_value(payload)
                self.info[key] = value
            elif kind == MESSAGE_PARAMETER:
                key, value = self._parse_key_value(payload)
                self.parameters[key] = value
            elif kind == MESSAGE_ADD_LOGGED:
                _, message_id = struct.unpack_from("<BH", payload, 0)
                name = payload[3:].decode("utf-8")
                subscriptions[message_id] = name
                raw.setdefault(name, [])
            elif kind == MESSAGE_DATA:
                message_id = struct.unpack_from("<H", payload, 0)[0]
                name = subscriptions.get(message_id)
                if name is not None:
                    raw[name].append(payload[2:])
            elif kind == MESSAGE_LOGGING:
                level, timestamp = struct.unpack_from("<BQ", payload, 0)
                self.messages.append(
                    {
                        "level": level,
                        "timestamp_us": timestamp,
                        "text": payload[9:].decode("utf-8", errors="replace"),
                    }
                )
            # Anything else is a message type this reader does not need.

        for name, rows in raw.items():
            self.topics[name] = self._decode(name, rows)

    @staticmethod
    def _parse_format(definition: str) -> tuple:
        name, _, body = definition.partition(":")
        fields = []
        for entry in body.split(";"):
            if not entry.strip():
                continue
            type_name, _, field = entry.strip().partition(" ")
            fields.append((field, type_name))
        return name, fields

    @staticmethod
    def _parse_key_value(payload: bytes) -> tuple:
        key_length = payload[0]
        key_field = payload[1: 1 + key_length].decode("utf-8")
        type_name, _, key = key_field.partition(" ")
        raw = payload[1 + key_length:]
        if type_name.startswith("char["):
            return key, raw.decode("utf-8", errors="replace")
        code = TYPES[type_name][0]
        return key, struct.unpack("<" + code, raw)[0]

    def _decode(self, name: str, rows) -> np.ndarray:
        fields = self.formats.get(name)
        if fields is None:
            raise UlogError(f"data for topic {name!r} with no format definition")
        dtype = np.dtype(
            {
                "names": [field for field, _ in fields],
                "formats": [TYPES[type_name][2] for _, type_name in fields],
            }
        )
        if not rows:
            return np.zeros(0, dtype=dtype)
        return np.frombuffer(b"".join(rows), dtype=dtype)

    # ------------------------------------------------------------------
    def to_dataframe(self, topic: str):
        """A topic as a pandas DataFrame, with a seconds column added."""
        import pandas as pd

        frame = pd.DataFrame(self.topics[topic])
        if "timestamp" in frame:
            frame["time_s"] = (frame["timestamp"] - frame["timestamp"].iloc[0]) * 1e-6
        return frame

    def summary(self) -> dict:
        return {
            "path": self.path,
            "topics": {name: int(len(rows)) for name, rows in self.topics.items()},
            "parameters": len(self.parameters),
            "messages": len(self.messages),
        }
