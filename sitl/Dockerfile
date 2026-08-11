# PX4 SITL with Gazebo, headless, exposing MAVLink on UDP 14540.
#
# Separate from the repository's main image on purpose: this is a large
# toolchain that the reproducible result does not need. Keeping it behind a
# compose profile means `docker compose up` stays a two-minute operation.
FROM px4io/px4-dev-simulation-jammy:latest

ARG PX4_VERSION=v1.15.0

# --shallow-submodules is not a micro-optimisation, it is the difference
# between a usable image and an unusable one. `--depth 1` applies only to the
# superproject; without --shallow-submodules git fetches the *full history* of
# all 36 submodules, several of which are large and old. Measured on the same
# machine and connection:
#
#   --depth 1 --recursive                        still running after 9 hours
#   --depth 1 --shallow-submodules --recursive    98 seconds, 1.1 GB
#
# The top-level clone alone is 2 seconds, so this is not a network problem and
# no amount of waiting fixes it.
RUN git clone --depth 1 --shallow-submodules --recurse-submodules \
        --branch ${PX4_VERSION} \
        https://github.com/PX4/PX4-Autopilot.git /px4

# Shallow submodules come with no tags, and PX4's version-header generator does
# `re.findall(r'nuttx-[0-9]+\.[0-9]+\.[0-9]+', ...)[-1]` — which raises
# IndexError on an empty list and stops the build at step 52 of 985, with a
# traceback that says nothing about submodule depth.
#
# Fetching only the tag refs, still at depth 1, costs 25 seconds and is enough:
# the generator wants a tag *name*, not the history behind it. NuttX is not even
# used by a SITL build; it is needed here solely to satisfy that one lookup.
RUN git -C /px4/platforms/nuttx/NuttX/nuttx \
        fetch --depth 1 origin "refs/tags/*:refs/tags/*"

# `px4_sitl_default` is the build; `px4_sitl gz_x500` is the *run*. The second
# form is what PX4's own documentation shows, and DONT_RUN=1 is documented to
# suppress the launch — but it does not, for the Gazebo targets in v1.15. Used
# here it builds, then starts PX4 and Gazebo inside the build layer and sits
# there forever repainting an interactive `pxh>` prompt into the build log. It
# does not fail, it does not finish, and the log grows without limit; the first
# attempt at this reached 549 MB before it was killed.
RUN cd /px4 && make px4_sitl_default

# Where `make px4_sitl gz_x500` runs the binary from, and where the generated
# gz_env.sh that points Gazebo at PX4's models and worlds lives. Starting px4
# from anywhere else leaves it unable to find either.
WORKDIR /px4/build/px4_sitl_default/src/modules/simulation/gz_bridge

# A Gazebo world with a structure to inspect, matching the facade mission.
COPY worlds/ /px4/Tools/simulation/gz/worlds/

# Named px4-rc.params, not offboard.params. PX4's rcS sources exactly one
# user parameter file and it is this name (`. px4-rc.params`); a file with any
# other name in the same directory is never read, and nothing warns about it.
# The stock file is comments only, so replacing it loses nothing.
COPY params/offboard.params /px4/build/px4_sitl_default/etc/init.d-posix/px4-rc.params

# Headless, with the world and the launch point both chosen deliberately.
#
# PX4_GZ_WORLD selects the world. Without it PX4 loads `default`, the world
# file above is never opened, and the vehicle flies a facade survey around
# nothing.
#
# PX4_GZ_MODEL_POSE is not cosmetic either. The vehicle spawns at the world
# origin by default, and the world origin is the centre of the structure the
# mission inspects — a 24 m box — so the vehicle starts *inside* the building,
# entombed. It arms, tilts towards its first waypoint and grinds along the
# floor at full throttle, which looks like a controller fault and is not one.
# 35 m south-west puts it on open ground with the survey still centred on the
# structure, because the mission frame is anchored to the mission's declared
# origin rather than to wherever the vehicle happened to launch.
#
# PX4_GZ_STANDALONE is deliberately *absent*, not set to 0. PX4 tests it with
# `[ -z "${PX4_GZ_STANDALONE}" ]` — is it empty — and starts Gazebo only if it
# is. Setting it to 0 to mean "not standalone" therefore selects standalone,
# and PX4 waits for a Gazebo that nothing in this image ever launches:
#
#   INFO  [init] Standalone PX4 launch, waiting for Gazebo
#   ERROR [gz_bridge] Service call timed out. Check GZ_SIM_RESOURCE_PATH ...
#
# The error names the one thing that is not wrong.
ENV HEADLESS=1 \
    PX4_SIM_MODEL=gz_x500 \
    PX4_GZ_WORLD=inspection \
    PX4_GZ_MODEL_POSE="-35,-35,0.2,0,0,0"

EXPOSE 14540/udp 14550/udp

# `-d` runs PX4 without its interactive `pxh>` console. With the console
# attached to a redirected stdout it repaints the prompt continuously: an
# otherwise idle container wrote a 549 MB log file, and the startup messages
# worth reading were buried at the top of it.
CMD ["/px4/build/px4_sitl_default/bin/px4", "-d"]
