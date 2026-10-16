"""A UDP relay in the MAVLink path that can cut the link on demand.

Link loss is the one failsafe that cannot be injected from either end. Telling
PX4 to stop talking, or telling MAVSDK to stop listening, tests a code path
that a real radio dropout never takes. So the link-loss scenarios put this relay
between the two and have it drop every datagram, in both directions, for the
length of the fault:

    PX4 :14580  <->  relay :14540  |  relay :ephemeral  <->  MAVSDK :14541

PX4's onboard MAVLink instance sends to 127.0.0.1:14540 by default, so the
relay sits on that port and PX4 needs no reconfiguration. The relay forwards to
MAVSDK on ``client_port`` and returns MAVSDK's replies to whichever address PX4
last sent from. While a drop is in force, neither side receives anything:
telemetry stops, setpoints stop, heartbeats stop. PX4's offboard-loss and
data-link-loss failsafes see exactly what they would see on a real dropout, and
so does the mission layer's link-loss rule.

Pure standard library and one thread, so it is testable without PX4.
"""

from __future__ import annotations

import selectors
import socket
import threading
import time


class UdpLinkProxy:
    """Bidirectional UDP relay with a switchable drop."""

    def __init__(self, listen_host: str = "127.0.0.1", listen_port: int = 14540,
                 client_host: str = "127.0.0.1", client_port: int = 14541):
        self.listen = (listen_host, int(listen_port))
        self.client = (client_host, int(client_port))
        self._autopilot_addr = None
        self._drop_until = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._autopilot_sock = None
        self._client_sock = None
        #: Datagrams forwarded and dropped, per direction. Logged with each
        #: flight so a link-loss run can show the relay really did cut it.
        self.counters = {"up": 0, "down": 0, "up_dropped": 0, "down_dropped": 0}

    # ------------------------------------------------------------------
    def start(self) -> "UdpLinkProxy":
        self._autopilot_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._autopilot_sock.bind(self.listen)
        self._client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._client_sock.bind((self.listen[0], 0))
        self._thread = threading.Thread(target=self._run, name="mavlink-relay", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for sock in (self._autopilot_sock, self._client_sock):
            if sock is not None:
                sock.close()
        self._autopilot_sock = self._client_sock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.close()
        return False

    # ------------------------------------------------------------------
    def drop_for(self, duration_s: float) -> None:
        """Cut the link, both directions, for ``duration_s`` seconds from now."""
        with self._lock:
            self._drop_until = time.monotonic() + float(duration_s)

    def restore(self) -> None:
        with self._lock:
            self._drop_until = 0.0

    @property
    def dropping(self) -> bool:
        with self._lock:
            return time.monotonic() < self._drop_until

    # ------------------------------------------------------------------
    def _run(self) -> None:
        selector = selectors.DefaultSelector()
        selector.register(self._autopilot_sock, selectors.EVENT_READ, "down")
        selector.register(self._client_sock, selectors.EVENT_READ, "up")
        try:
            while not self._stop.is_set():
                for key, _ in selector.select(timeout=0.1):
                    try:
                        data, addr = key.fileobj.recvfrom(65535)
                    except OSError:
                        continue
                    dropping = self.dropping
                    if key.data == "down":
                        # Autopilot to client. Remember where the autopilot
                        # sends from, so replies go back to the same socket.
                        self._autopilot_addr = addr
                        if dropping:
                            self.counters["down_dropped"] += 1
                            continue
                        self._send(self._client_sock, data, self.client)
                        self.counters["down"] += 1
                    else:
                        if dropping or self._autopilot_addr is None:
                            self.counters["up_dropped"] += 1
                            continue
                        self._send(self._autopilot_sock, data, self._autopilot_addr)
                        self.counters["up"] += 1
        finally:
            selector.close()

    @staticmethod
    def _send(sock, data: bytes, addr) -> None:
        try:
            sock.sendto(data, addr)
        except OSError:
            # Nobody listening yet (ICMP port unreachable surfaces here on
            # Linux). Losing a datagram is what UDP does; carry on.
            pass
