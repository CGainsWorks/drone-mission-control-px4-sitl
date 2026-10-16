#!/usr/bin/env python3
"""Fly the PX4 SITL campaign: one fresh PX4 container per flight.

For each flight this starts PX4 SIH (``sitl/sih/Dockerfile``) with its home set
to the mission's origin, runs ``scripts/run_px4_sitl.py`` in a second container
sharing PX4's network namespace, then copies out the flight script's log and
PX4's own ULog and removes both containers. A fresh autopilot per flight means
no flight inherits another's parameters, battery state or a latched failure
detector.

Nothing is bind-mounted: outputs are copied out with ``docker cp``. A writable
bind mount owned by the container's user works on Docker Desktop and fails on
a Linux CI runner, so it is not relied on.

Usage:
    python scripts/run_sitl_campaign.py --campaign smoke --output results/sitl-smoke
    python scripts/run_sitl_campaign.py --campaign full   # writes sitl/results/flights

Then ``python analysis/make_sitl_results.py`` turns the flights into the
committed summary. Every number from this campaign is simulated (PX4 SIH).
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mission_control import sitl  # noqa: E402
from mission_control.mission import load_mission  # noqa: E402

PX4_CONTAINER = "mc-sitl-px4"
FLIGHT_CONTAINER = "mc-sitl-flight"


def docker(*args, check=True, capture=False, timeout=None):
    return subprocess.run(["docker", *args], check=check, timeout=timeout,
                          capture_output=capture, text=True)


def remove_containers() -> None:
    docker("rm", "-f", PX4_CONTAINER, FLIGHT_CONTAINER, check=False, capture=True)


def fly(flight: dict, px4_image: str, flight_image: str, output: str,
        flight_timeout_s: float) -> int:
    mission = load_mission(os.path.join(ROOT, "missions", f"{flight['mission']}.yaml"))
    destination = os.path.join(output, flight["name"])
    if os.path.isdir(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)

    remove_containers()
    origin = mission.origin
    # --network none: PX4 gets a loopback interface and nothing else, and the
    # flight container joins that namespace. MAVLink's fixed UDP ports then
    # cannot meet any other simulator on the same Docker host, and nothing
    # outside can reach them. It is also why the flight script and PX4 find
    # each other on 127.0.0.1 with no configuration.
    docker("run", "-d", "--name", PX4_CONTAINER, "--network", "none",
           "-e", f"PX4_HOME_LAT={origin.latitude_deg}",
           "-e", f"PX4_HOME_LON={origin.longitude_deg}",
           "-e", f"PX4_HOME_ALT={origin.altitude_m}",
           px4_image, capture=True)

    command = ["python", "scripts/run_px4_sitl.py",
               "--mission", flight["mission"], "--tracker", flight["tracker"],
               "--scenario", flight["scenario"], "--name", flight["name"],
               "--output", "/tmp/flight", "--timeout", str(int(flight_timeout_s))]
    started = time.monotonic()
    try:
        result = docker("run", "--name", FLIGHT_CONTAINER,
                        "--network", f"container:{PX4_CONTAINER}",
                        flight_image, *command, check=False,
                        timeout=flight_timeout_s + 300)
        code = result.returncode
    except subprocess.TimeoutExpired:
        print(f"  {flight['name']}: flight container did not finish; stopping it")
        code = 124
    print(f"  {flight['name']}: exit {code} after {time.monotonic() - started:.0f} s")

    docker("cp", f"{FLIGHT_CONTAINER}:/tmp/flight/.", destination, check=False)

    # PX4 closes the ULog on disarm. One flight per container, so there is
    # exactly one log; take the newest anyway.
    listing = docker("exec", PX4_CONTAINER, "sh", "-c",
                     "ls -t /px4/rootfs/log/*/*.ulg 2>/dev/null | head -n 1",
                     check=False, capture=True)
    ulog = listing.stdout.strip()
    if ulog:
        raw = os.path.join(destination, "px4.ulg")
        docker("cp", f"{PX4_CONTAINER}:{ulog}", raw, check=False)
        if os.path.exists(raw):
            # Compressed for committing: ULog is a binary format that gzip
            # shrinks about threefold. mtime=0 and no file name in the header,
            # so the bytes depend only on the log.
            with open(raw, "rb") as source, open(raw + ".gz", "wb") as target:
                with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as gz:
                    gz.write(source.read())
            os.remove(raw)
    else:
        print(f"  {flight['name']}: PX4 wrote no ULog")
        code = code or 1
    remove_containers()
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default="full", choices=["full", "smoke"])
    parser.add_argument("--only", default="", help="comma-separated flight names")
    parser.add_argument("--px4-image", default=os.environ.get("PX4_IMAGE",
                                                              "mission-control-px4-sih:latest"))
    parser.add_argument("--flight-image", default=os.environ.get("FLIGHT_IMAGE",
                                                                 "mission-control-sitl:latest"))
    parser.add_argument("--output", default=os.path.join(ROOT, "sitl", "results", "flights"))
    parser.add_argument("--flight-timeout", type=float, default=600.0)
    args = parser.parse_args()

    flights = sitl.campaign_flights(args.campaign)
    if args.only:
        wanted = set(args.only.split(","))
        flights = [flight for flight in flights if flight["name"] in wanted]
    os.makedirs(args.output, exist_ok=True)

    print(f"{len(flights)} flight(s) against {args.px4_image}")
    failures = []
    try:
        for flight in flights:
            print(f"\n=== {flight['name']}")
            if fly(flight, args.px4_image, args.flight_image, args.output,
                   args.flight_timeout) != 0:
                failures.append(flight["name"])
    finally:
        remove_containers()

    print()
    if failures:
        print(f"{len(failures)} flight(s) did not finish cleanly: {', '.join(failures)}")
        return 1
    print(f"all {len(flights)} flight(s) finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
