"""Export the sim contract the deploy bridge/client need (RUN INSIDE DOCKER).

    ./docker/run.sh python deploy/export_contract.py

Builds the ReachAvoid env headless (proprio, 1 env), reads Isaac's *actual*
joint ordering, default pose, joint limits, action scale and control rate, and
dumps them to ``deploy/contract.json``. Everything downstream reads this file so
no convention is hardcoded in two places.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the sim->real contract.")
parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # never need a window for this

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from so_arm101_avoid.tasks.reach_avoid.reach_avoid_env import (  # noqa: E402
    SoArm101ReachAvoidEnv,
    SoArm101ReachAvoidEnvCfg,
)


def _as_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def main():
    cfg = SoArm101ReachAvoidEnvCfg()
    cfg.use_vision = False
    cfg.hand_spawn_prob = 0.0
    cfg.domain_randomization = False
    cfg.scene.num_envs = 1
    cfg.__post_init__()

    env = SoArm101ReachAvoidEnv(cfg)
    env.reset()
    robot = env.robot

    joint_names = list(robot.data.joint_names)
    default = _as_list(robot.data.default_joint_pos[0])

    # joint position limits (rad); attribute name varies across Isaac Lab versions
    limits = None
    for attr in ("soft_joint_pos_limits", "joint_pos_limits"):
        if hasattr(robot.data, attr):
            limits = _as_list(getattr(robot.data, attr)[0])
            break

    arm_names = list(cfg.arm_joint_names)
    contract = {
        "joint_names": joint_names,
        "default_rad": {n: float(default[i]) for i, n in enumerate(joint_names)},
        "joint_limits_rad": (
            {n: [float(limits[i][0]), float(limits[i][1])] for i, n in enumerate(joint_names)}
            if limits is not None
            else None
        ),
        "arm_joint_names": arm_names,
        "arm_joint_ids": _as_list(env._arm_joint_ids),
        "action_scale": float(cfg.action_scale),
        "decimation": int(cfg.decimation),
        "sim_dt": float(cfg.sim.dt),
        "control_dt": float(cfg.sim.dt * cfg.decimation),
        "ee_offset": list(cfg.ee_offset),
        "ee_body": cfg.ee_body_name,
        # proprio layout: joint_pos_rel(J) + joint_vel(J) + target(3) + last_action(A)
        "n_joints": len(joint_names),
        "n_actions": len(arm_names),
        "proprio_dim": int(2 * len(joint_names) + 3 + len(arm_names)),
    }

    out = Path(args.output) if args.output else Path(__file__).resolve().parent / "contract.json"
    out.write_text(json.dumps(contract, indent=2))

    print("\n[export_contract] wrote", out)
    print(json.dumps(contract, indent=2))
    print("\nIsaac joint order:", joint_names)
    print("Arm action joints:", arm_names, "-> ids", contract["arm_joint_ids"])

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
