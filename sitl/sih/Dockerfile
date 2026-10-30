# PX4 SITL with the SIH physics model, headless, built from a pinned release.
#
# SIH ("simulation in hardware") is PX4's own rigid-body model compiled into the
# autopilot binary. It needs no Gazebo, no GPU and no display, starts in a few
# seconds and costs one core, which is what makes it possible to fly the stack
# against a real PX4 on every push. The Gazebo image (../gazebo.Dockerfile) is
# kept for anyone who wants to watch a flight; it is not what produces the
# committed SITL results.
#
# Two stages. The first is the PX4 project's own build container at the tag
# matching the release, so the toolchain is the one PX4 builds that release
# with. The second carries only the binary and its startup files on a plain
# Ubuntu base: about 100 MB instead of several gigabytes.

ARG PX4_VERSION=v1.17.0

FROM px4io/px4-dev:${PX4_VERSION} AS build
ARG PX4_VERSION

# --shallow-submodules matters: --depth 1 alone applies only to the
# superproject and fetching the full history of every submodule is the
# difference between about two minutes and several hours.
RUN git clone --depth 1 --shallow-submodules --recurse-submodules \
        --branch ${PX4_VERSION} \
        https://github.com/PX4/PX4-Autopilot.git /src/px4

WORKDIR /src/px4
# The nolockstep variant and build only.
#
# In the default (lockstep) build PX4's clock is simulation time, which SIH
# advances in fixed steps and then sleeps for the remainder of each step. The
# sleeps overshoot, so simulation time runs slow: measured here at 0.944 of
# wall-clock time. The trackers run on wall-clock time in a separate process,
# so in a lockstep build the vehicle's physics and the controller disagree
# about how long a second is by over 5% and the ULog's clock drifts away from
# the flight script's. The nolockstep build runs PX4 and SIH on the real
# monotonic clock, which is what an external real-time controller needs.
#
# `make px4_sitl_nolockstep sihsim_quadx` would also start the simulator
# inside the build layer and never return, so this builds the target only.
RUN make px4_sitl_nolockstep


FROM ubuntu:24.04
ARG PX4_VERSION

# bc is used by PX4's startup script; nothing else is needed at run time
# because the binary links only libc, libm and libstdc++.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/px4/build/px4_sitl_nolockstep/bin /opt/px4/bin
COPY --from=build /src/px4/build/px4_sitl_nolockstep/etc /opt/px4/etc

# A short topic list instead of PX4's default logging profile, so a whole
# flight's ULog is a few hundred kilobytes and can be committed. PX4 reads
# this file in place of SDLOG_PROFILE when it exists.
COPY logger_topics.txt /opt/px4/etc/logging/logger_topics.txt

# The parameters this project flies with, applied after PX4's own defaults.
# See the file for why each one is there and why they are not set through
# PX4's environment hook.
COPY 10040_sihsim_quadx.post /opt/px4/etc/init.d-posix/airframes/10040_sihsim_quadx.post

ENV PX4_VERSION=${PX4_VERSION} \
    PX4_SIM_MODEL=sihsim_quadx

WORKDIR /px4
EXPOSE 14540/udp 14550/udp

# -d: no interactive pxh console. With the console attached to a redirected
# stdout it repaints the prompt continuously and the log grows without limit.
# The positional argument is where etc/ lives; -w is the working directory,
# which is where PX4 writes log/ and its parameter file.
CMD ["/opt/px4/bin/px4", "-d", "-w", "/px4/rootfs", "/opt/px4/etc"]
