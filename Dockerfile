# Everything needed to reproduce the headline result: Python, numpy, scipy,
# matplotlib. No ROS, no Gazebo, no GPU, no network at run time.
#
# PX4 SITL lives in a separate image (see sitl/), because it is a 2 GB
# toolchain that the comparison does not need and that most readers will never
# run. Keeping it out of here is the difference between "docker compose up" and
# "docker compose up, twenty minutes later".
FROM python:3.11-slim-bookworm

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
