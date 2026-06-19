# SO-ARM101 Reach-Avoid — Isaac Lab training/sim image.
#
# Built on NVIDIA's official Isaac Lab image (Isaac Sim 5.x + Isaac Lab 2.3 +
# rsl_rl/skrl/rl_games already inside). This is the same base the NVIDIA
# Sim-to-Real SO-101 workshop uses on RTX 5090 (Blackwell), so the GPU works
# out of the box. CUDA/torch are pre-baked — we only add what we need.
FROM nvcr.io/nvidia/isaac-lab:2.3.2

# Isaac Sim's bundled python (NOT the system python). Everything must run
# through this interpreter so it can find omni.* / isaaclab.*.
ENV PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ENV PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:$PATH"
# Accept the Omniverse EULA non-interactively (otherwise Kit hangs on stdin).
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

# Runtime libs needed for the Isaac Sim GUI / X11 display passthrough.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Extra Python deps for our task (perception/eval helpers). Installed into the
# Isaac Sim python. Pinned loosely; opencv is headless to avoid pulling a
# conflicting Qt into the Kit environment.
RUN $PYTHON -m pip install --no-cache-dir \
    "opencv-python-headless" \
    "matplotlib" \
    "tensorboard"

# Project lives here; mounted at runtime so source edits are picked up live.
WORKDIR /workspace/RL_Hand_Avoid_IsaacLab_SO101

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Keep __pycache__ out of the (read-only-ish) mounted source tree.
RUN mkdir -p /tmp/pycache && chmod 1777 /tmp/pycache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
