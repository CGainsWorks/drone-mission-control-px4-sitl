# Everything needed to reproduce the headline result: Python, numpy, scipy,
# matplotlib. No ROS, no Gazebo, no GPU, no network at run time.
#
# PX4 SITL lives in a separate image (see sitl/), because it is a 2 GB
# toolchain that the comparison does not need and that most readers will never
# run. Keeping it out of here is the difference between "docker compose up" and
# "docker compose up, twenty minutes later".
FROM python:3.11-slim-bookworm AS base

# Matplotlib needs a writable cache directory, and a non-root user is the
# right default for something that mounts the working tree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first, so editing source does not invalidate the layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY analysis/ ./analysis/
COPY missions/ ./missions/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY docs/ ./docs/
COPY pytest.ini ./

CMD ["bash", "scripts/run_all.sh"]


# The PX4 backend needs MAVSDK, which the comparison does not. It is a separate
# stage rather than an extra line in the base image because it pulls in a
# gRPC stack and a bundled mavsdk_server binary, and none of that belongs in the
# image whose whole point is being small enough to run anywhere.
#
# `docker compose --profile sitl run --rm sitl-mission` builds this target.
FROM base AS sitl
# Just the two the backend needs, at the versions requirements-optional.txt
# pins. Jupyter is in that file too and has no business in a flight container.
RUN pip install --no-cache-dir "mavsdk>=1.4" "pyulog>=1.0"
