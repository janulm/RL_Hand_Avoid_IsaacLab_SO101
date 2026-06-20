"""Shared deploy configuration (pure stdlib -- safe to import in BOTH the Isaac
docker container and the host `lerobot` conda env).

The real -> sim mirror bridge has two processes:
  * ``real_client.py``  runs in the host `lerobot` conda env, owns the serial bus.
  * ``sim_bridge.py``   runs inside the Isaac docker container, owns the GUI twin
    and the policy.

They talk over localhost UDP (the container uses ``--network=host``). The real
robot's joint state drives the sim; the sim never commands the real arm. Edit the
values below to match your setup.
"""

from __future__ import annotations

import json
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = DEPLOY_DIR / "contract.json"
MAPPING_PATH = DEPLOY_DIR / "mapping.json"

# --- LeRobot follower -------------------------------------------------------
FOLLOWER_PORT = "/dev/lerobot_follower"
FOLLOWER_ID = "my_awesome_follower_arm"
USE_DEGREES = True  # body joints in degrees (gripper is always 0..100)

# Motor order as exposed by lerobot's SO101Follower (insertion order of the bus).
LEROBOT_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# --- control / timing -------------------------------------------------------
CONTROL_HZ = 30.0  # must match the trained env control rate (decimation/dt)
CONTROL_DT = 1.0 / CONTROL_HZ

# --- IPC (localhost UDP) ----------------------------------------------------
# Client -> bridge: real joint state. Bridge -> client: arm joint targets.
IPC_HOST = "127.0.0.1"
BRIDGE_PORT = 5601  # bridge binds here, client sends here
CLIENT_PORT = 5602  # client binds here, bridge replies here

# --- safety -----------------------------------------------------------------
# Per-step clamp on |goal - present| in lerobot units (degrees for body joints).
# This is lerobot's own `max_relative_target`; keep it small for first bring-up.
MAX_RELATIVE_TARGET = 4.0
# Exponential moving average on the policy action (0 = no smoothing, ->1 = heavy).
ACTION_EMA = 0.5
# Extra global scale on the policy action for cautious first runs (1.0 = trained).
ACTION_SCALE_DEPLOY = 1.0
# Seconds to linearly ramp from the arm's current pose to the home pose at start.
RAMP_S = 2.5

# --- gripper ----------------------------------------------------------------
# The policy controls 5 arm joints only; the gripper is held at a fixed value
# (lerobot units, 0..100). Adjust if "open" is at the other end on your arm.
GRIPPER_HOLD_POS = 50.0

# --- task: cycling hardcoded targets (robot ROOT frame, metres) -------------
# Axes (confirmed via the GUI triad): +X = forward, +Y = left, +Z = up.
TARGETS = (
    (0.20, 0.00, 0.15),
    (0.18, 0.12, 0.10),
    (0.18, -0.12, 0.12),
    (0.10, 0.00, 0.22),
)
TARGET_HOLD_S = 4.0  # seconds to dwell on each target before advancing

# --- velocity estimation ----------------------------------------------------
# The follower reports positions only; joint velocity is finite-differenced and
# low-pass filtered (EMA) before going into the observation.
VEL_EMA = 0.5


def load_contract() -> dict:
    """Load the env contract dumped by ``export_contract.py`` (run in docker)."""
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"{CONTRACT_PATH} not found. Run `./docker/run.sh python deploy/export_contract.py` first."
        )
    with open(CONTRACT_PATH) as f:
        return json.load(f)


def load_mapping() -> dict:
    """Load the lerobot<->Isaac joint mapping written by ``calibrate_mapping.py``."""
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"{MAPPING_PATH} not found. Run `python deploy/calibrate_mapping.py` (lerobot env) first."
        )
    with open(MAPPING_PATH) as f:
        return json.load(f)
