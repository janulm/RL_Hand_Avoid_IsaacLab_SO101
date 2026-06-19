#!/bin/bash
# Run the SO-ARM101 Reach-Avoid container with GUI (X11) + USB (motor bus /
# cameras) passthrough, and persistent Isaac Sim shader/asset caches.
#
# Usage:
#   ./docker/run.sh                  # interactive bash inside the container
#   ./docker/run.sh <cmd> [args...]  # run a command, e.g.:
#   ./docker/run.sh python scripts/skrl/train.py --task Isaac-SO-ARM101-ReachAvoid-Direct-v0 --headless
#
# The repo is mounted at /workspace/RL_Hand_Avoid_IsaacLab_SO101 so host edits
# are live inside the container.
set -e

IMAGE_NAME="${IMAGE_NAME:-so101-avoid}"
CONTAINER_NAME="${CONTAINER_NAME:-so101-avoid}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Allow local X11 clients (the container) to talk to the host display server.
xhost +local:root >/dev/null 2>&1 || xhost + >/dev/null 2>&1 || true

# Persistent Isaac Sim caches on the host (huge speedup on 2nd+ launch).
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache} \
         ~/docker/isaac-sim/{logs,data,documents}

# Use a TTY only when one is available (so the script also works non-interactively).
if [ -t 0 ]; then TTY_FLAGS="-it"; else TTY_FLAGS="-i"; fi

docker run --name "${CONTAINER_NAME}" ${TTY_FLAGS} --rm \
    --privileged --gpus all --network=host \
    -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" -e "OMNI_KIT_ACCEPT_EULA=YES" \
    -e "PYTHONUNBUFFERED=1" \
    -v /etc/localtime:/etc/localtime:ro \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
    -v "$REPO_ROOT:/workspace/RL_Hand_Avoid_IsaacLab_SO101" \
    "${IMAGE_NAME}:latest" "$@"
