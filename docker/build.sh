#!/bin/bash
# Build the SO-ARM101 Reach-Avoid Docker image.
# Usage:  ./docker/build.sh   (run from the repo root)
#
# NOTE: the base image (nvcr.io/nvidia/isaac-lab:2.3.2) is large (~20 GB) and
# is pulled from NVIDIA NGC on first build. nvcr.io isaac-lab images are public
# (no NGC login required to pull).
set -e

IMAGE_NAME="${IMAGE_NAME:-so101-avoid}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
docker build -t "${IMAGE_NAME}:latest" -f docker/Dockerfile .
echo ""
echo "Built ${IMAGE_NAME}:latest  — run it with ./docker/run.sh"
