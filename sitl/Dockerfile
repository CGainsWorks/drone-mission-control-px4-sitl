# PX4 SITL with Gazebo, headless, exposing MAVLink on UDP 14540.
#
# Separate from the repository's main image on purpose: this is a large
# toolchain that the reproducible result does not need. Keeping it behind a
# compose profile means `docker compose up` stays a two-minute operation.
FROM px4io/px4-dev-simulation-jammy:latest

ARG PX4_VERSION=v1.15.0

RUN git clone --depth 1 --branch ${PX4_VERSION} --recursive \
        https://github.com/PX4/PX4-Autopilot.git /px4 \
    && cd /px4 \
    && DONT_RUN=1 make px4_sitl gz_x500

WORKDIR /px4

# A Gazebo world with a structure to inspect, matching the facade mission.
COPY worlds/ /px4/Tools/simulation/gz/worlds/
COPY params/ /px4/build/px4_sitl_default/etc/init.d-posix/

# Headless: no GUI, and MAVLink offered on the standard offboard port so the
# mission-control container can reach it by service name.
ENV HEADLESS=1 \
    PX4_SIM_MODEL=gz_x500 \
    PX4_GZ_STANDALONE=0

EXPOSE 14540/udp 14550/udp

CMD ["make", "px4_sitl", "gz_x500"]
