"""Joint mapping between lerobot follower units and Isaac (URDF) units.

LeRobot reports body joints in DEGREES with 0 at the mechanical mid-range, while
Isaac uses RADIANS with 0 at the URDF zero -- so each joint differs by a SIGN and
an OFFSET. We avoid solving the absolute offset by anchoring on a captured "home"
pose: the lerobot degrees recorded when the arm is physically at the Isaac default
pose. Then, per body joint ``j`` (matched by name):

    q_isaac[j]  = default_rad[j] + SIGN[j] * deg2rad(q_lr_deg[j] - home_lr_deg[j])
    q_lr_deg[j] = home_lr_deg[j] + SIGN[j] * rad2deg(q_target_rad[j] - default_rad[j])

The gripper is not policy-controlled and is held fixed, so its observation
contribution is reported as "at default" (relative 0).

Pure stdlib -- safe to import in the lerobot conda env.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

GRIPPER = "gripper"


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def default_mapping(joint_names) -> dict:
    """Identity-ish starting guess: sign +1, home 0 for every joint."""
    return {
        "sign": {n: 1 for n in joint_names},
        "home_lr_deg": {n: 0.0 for n in joint_names},
    }


def save_mapping(path: Path, mapping: dict) -> None:
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def lr_to_isaac_q(q_lr_by_name: dict, mapping: dict, contract: dict) -> list:
    """LeRobot reading (dict name->units) -> Isaac joint positions (rad), ordered
    like ``contract['joint_names']``."""
    default = contract["default_rad"]
    sign = mapping["sign"]
    home = mapping["home_lr_deg"]
    out = []
    for name in contract["joint_names"]:
        if name == GRIPPER:
            # held fixed -> report at default (relative 0 in the observation)
            out.append(default[name])
            continue
        q_lr = q_lr_by_name[name]
        out.append(default[name] + sign[name] * deg2rad(q_lr - home[name]))
    return out


def isaac_targets_to_lr(target_rad_by_name: dict, mapping: dict, contract: dict) -> dict:
    """Isaac arm joint targets (dict name->rad) -> lerobot goal degrees (dict name->deg)."""
    default = contract["default_rad"]
    sign = mapping["sign"]
    home = mapping["home_lr_deg"]
    out = {}
    for name, q_t in target_rad_by_name.items():
        out[name] = home[name] + sign[name] * rad2deg(q_t - default[name])
    return out


def isaac_q_to_lr(q_rad_by_name: dict, mapping: dict, contract: dict) -> dict:
    """Isaac joint positions (dict name->rad) -> lerobot degrees (for the ramp/home)."""
    return isaac_targets_to_lr(q_rad_by_name, mapping, contract)
