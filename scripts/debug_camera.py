"""Visualize the exact camera observation the policy receives.

Runs the Reach-Avoid env and, for each step, saves a panel showing, per camera:
    [ RGB | hand mask | RGB+mask overlay ]
so you can confirm the analytic hand-mask channel lines up with the rendered
hand. Also writes an animated GIF.

Run (headless is fine — this just dumps images):

    python scripts/debug_camera.py --headless --steps 60 --num_envs 2

Output: verification_output/camera_debug/
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--out", type=str, default="verification_output/camera_debug")
parser.add_argument("--upscale", type=int, default=4, help="Integer upscale factor for the tiny images.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional skrl checkpoint to drive the arm.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so_arm101_avoid  # noqa: E402, F401
from so_arm101_avoid.tasks.reach_avoid.reach_avoid_env import (  # noqa: E402
    SoArm101ReachAvoidEnv,
    SoArm101ReachAvoidEnvCfg,
)


def split_channels(cfg):
    """Return the channel layout: list of (cam_name, has_rgb, has_mask)."""
    cams = []
    order = []
    if cfg.camera_view in ("overhead", "both"):
        order.append("overhead")
    if cfg.camera_view in ("wrist", "both"):
        order.append("wrist")
    has_rgb = "rgb" in cfg.obs_mode
    has_mask = "mask" in cfg.obs_mode
    for name in order:
        cams.append((name, has_rgb, has_mask))
    return cams


def make_panel(cam_img: np.ndarray, layout, upscale: int) -> np.ndarray:
    """cam_img: (H, W, C) float in [0,1]. Build a labelled RGB|mask|overlay panel
    per camera, stacked vertically."""
    H, W, _ = cam_img.shape
    c = 0
    rows = []
    for name, has_rgb, has_mask in layout:
        if has_rgb:
            rgb = (cam_img[..., c : c + 3] * 255).astype(np.uint8)
            c += 3
        else:
            rgb = np.zeros((H, W, 3), np.uint8)
        if has_mask:
            mask = (cam_img[..., c] > 0.5).astype(np.uint8)
            c += 1
        else:
            mask = np.zeros((H, W), np.uint8)

        mask_vis = np.repeat((mask * 255)[..., None], 3, axis=2)
        overlay = rgb.copy()
        overlay[mask > 0] = (0.4 * overlay[mask > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)

        panels = [("RGB", rgb), ("mask", mask_vis), ("overlay", overlay)]
        scaled = []
        for label, img in panels:
            big = cv2.resize(img, (W * upscale, H * upscale), interpolation=cv2.INTER_NEAREST)
            # cv2 uses BGR; our data is RGB -> convert for correct colors on disk
            big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
            cv2.putText(big, f"{name}:{label}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            scaled.append(big)
        rows.append(np.concatenate(scaled, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    cfg = SoArm101ReachAvoidEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.domain_randomization = False
    env = SoArm101ReachAvoidEnv(cfg)
    layout = split_channels(cfg)

    os.makedirs(args_cli.out, exist_ok=True)

    policy = None
    if args_cli.checkpoint:
        policy = torch.load(args_cli.checkpoint, map_location=env.device)
        print(f"[debug_camera] loaded checkpoint {args_cli.checkpoint}")

    obs, _ = env.reset()
    frames_env0 = []
    for step in range(args_cli.steps):
        action = torch.rand(env.num_envs, 5, device=env.device) * 2.0 - 1.0
        obs, rew, term, trunc, info = env.step(action)
        cam = obs["policy"]["camera"]  # (N, H, W, C)
        panel = make_panel(cam[0].detach().float().cpu().numpy(), layout, args_cli.upscale)
        frames_env0.append(panel)
        cv2.imwrite(os.path.join(args_cli.out, f"env0_step{step:03d}.png"), panel)

    print(f"[debug_camera] wrote {len(frames_env0)} PNG frames to {args_cli.out}")

    # write a GIF for easy scrubbing
    try:
        import imageio.v2 as imageio

        gif_path = os.path.join(args_cli.out, "env0.gif")
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_env0]
        imageio.mimsave(gif_path, rgb_frames, fps=10)
        print(f"[debug_camera] wrote {gif_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[debug_camera] (gif skipped: {e})")

    print("[debug_camera] DONE")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
