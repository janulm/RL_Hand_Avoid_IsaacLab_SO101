"""Quick sanity check for the SO-ARM101 Reach-Avoid Direct env.

Instantiates the env with a few sub-envs, resets, and steps random actions,
printing observation shapes, reward stats, and mask coverage. Run headless:

    python scripts/smoke_test.py --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# camera rendering must be enabled for the image observation
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import so_arm101_avoid  # noqa: E402, F401  (registers the gym envs)
from so_arm101_avoid.tasks.reach_avoid.reach_avoid_env import (  # noqa: E402
    SoArm101ReachAvoidEnv,
    SoArm101ReachAvoidEnvCfg,
)


def main():
    cfg = SoArm101ReachAvoidEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = SoArm101ReachAvoidEnv(cfg)

    obs, _ = env.reset()
    cam = obs["policy"]["camera"]
    proprio = obs["policy"]["proprio"]
    print(f"[smoke] camera obs : shape={tuple(cam.shape)} dtype={cam.dtype} "
          f"min={cam.min().item():.3f} max={cam.max().item():.3f}")
    print(f"[smoke] proprio obs: shape={tuple(proprio.shape)}")
    if cfg.obs_mode.endswith("mask") or "mask" in cfg.obs_mode:
        mask = cam[..., -1]
        print(f"[smoke] mask coverage (frac on): {mask.mean().item():.4f}")

    rew_hist = []
    mask_hist = []
    for i in range(args_cli.steps):
        action = torch.rand(env.num_envs, 5, device=env.device) * 2.0 - 1.0
        obs, rew, terminated, truncated, info = env.step(action)
        rew_hist.append(rew.mean().item())
        if "mask" in cfg.obs_mode:
            mask_hist.append(obs["policy"]["camera"][..., -1].mean().item())

    print(f"[smoke] stepped {args_cli.steps} steps OK")
    if mask_hist:
        import numpy as np
        print(f"[smoke] mask coverage over steps: mean={np.mean(mask_hist):.4f} "
              f"max={np.max(mask_hist):.4f} (>0 on {np.mean([m > 0 for m in mask_hist]) * 100:.0f}% of steps)")
    print(f"[smoke] mean reward per step: {sum(rew_hist) / len(rew_hist):.4f}")
    print(f"[smoke] last reward: min={rew.min().item():.3f} max={rew.max().item():.3f}")
    print(f"[smoke] terminated={terminated.sum().item()} truncated={truncated.sum().item()}")
    print("[smoke] SUCCESS")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
