# RL Hand-Avoid — SO-ARM101 (Isaac Lab)

Vision-based **human-hand avoidance** reinforcement learning for the
[SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) low-cost arm, trained
in **Isaac Lab 2.3 / Isaac Sim 5.x** and intended for sim-to-real deployment on
a real SO-101 follower via [LeRobot](https://github.com/huggingface/lerobot).

The arm reaches a target while a moving "hand" obstacle sweeps through its
workspace; the policy must steer clear. The policy sees a camera image (RGB +
a hand-segmentation channel) plus proprioception, and outputs joint targets.

> Built for the HUD / RSI **RL Environments** hackathon. The scored substance is
> the *environment + verification + a measured improvement*; the real arm
> dodging a hand is the wow factor.

---

## Why the Direct workflow (not manager-based)

Isaac Lab's **manager-based** workflow only supports a flat `Box` observation
and cannot expose composite image+state observations to the policy — in
practice the camera term is silently ignored (confirmed by the Isaac Lab
maintainers in
[#2613](https://github.com/isaac-sim/IsaacLab/issues/2613) and
[#2743](https://github.com/isaac-sim/IsaacLab/discussions/2743)). Vision RL must
therefore use the **Direct** workflow with a `gym.spaces.Dict` observation,
which is what this repo does.

---

## Hardware (deploy target)

- SO-101 leader + follower arms (LeRobot teleop / motor bus).
- RGB cameras only: a wrist camera and an overhead camera (no depth).
- Tested GPU: RTX 5090 Laptop (Blackwell), Ubuntu 24.04.

---

## Repository layout

```
docker/                     # Dockerized Isaac Lab (build.sh, run.sh, Dockerfile)
scripts/
  skrl/train.py, play.py    # skrl PPO training / playback
  list_envs.py              # list registered SO-ARM101 tasks
  smoke_test.py             # quick env sanity check (reset + step)
  debug_camera.py           # dump RGB|mask|overlay frames of the policy input
  create_charuco.py         # make a ChArUco board (camera calibration)
  align_cameras.py          # align/calibrate real cameras (deploy)
source/so_arm101_avoid/     # the Isaac Lab extension (our env lives here)
  so_arm101_avoid/
    robots/trs_so101/       # vendored SO-101 URDF + ArticulationCfg
    tasks/reach_avoid/      # Direct ReachAvoid env + cfg + skrl agent yaml
```

Registered tasks:

- `Isaac-SO-ARM101-ReachAvoid-Direct-v0` — training.
- `Isaac-SO-ARM101-ReachAvoid-Direct-Play-v0` — few-env playback/eval.

---

## Quick start (Docker)

Everything runs inside a container built on NVIDIA's official
`nvcr.io/nvidia/isaac-lab:2.3.2` image (Blackwell-ready). The image is large
(~20 GB) on first build.

```bash
# 1. Build the image
./docker/build.sh

# 2. Run a command in the container (GUI + USB passthrough handled for you)
./docker/run.sh <command>
```

`docker/run.sh` mounts the repo, forwards the X11 display (`xhost`, `DISPLAY`,
`.Xauthority`), passes through `/dev` (motor bus + cameras), and keeps Isaac Sim
shader/asset caches under `~/docker/isaac-sim/` so the 2nd+ launch is fast.

### Common commands

```bash
# List our registered tasks
./docker/run.sh python scripts/list_envs.py --headless

# Sanity-check the env (reset + random steps, prints obs shapes & reward)
./docker/run.sh python scripts/smoke_test.py --headless

# Visualize EXACTLY what the policy sees: RGB | hand-mask | overlay (+ a gif)
./docker/run.sh python scripts/debug_camera.py --headless --steps 60
#   -> verification_output/camera_debug/

# Train (headless). --enable_cameras is required for the image observation.
./docker/run.sh python scripts/skrl/train.py \
    --task Isaac-SO-ARM101-ReachAvoid-Direct-v0 --headless --enable_cameras

# Watch a trained policy in the GUI
./docker/run.sh python scripts/skrl/play.py \
    --task Isaac-SO-ARM101-ReachAvoid-Direct-Play-v0 --enable_cameras
```

> Tip: `--max_iterations N` and `--num_envs N` are handy for short runs.

---

## Observation / action contract

Keep sim, eval, and deploy in agreement on these.

**Observation** (`gym.spaces.Dict`):

| Key | Shape | Contents |
|-----|-------|----------|
| `camera` | `(H, W, C)` | per active camera: RGB (3) and/or hand mask (1). Default 4ch = RGB + mask. |
| `proprio` | `(20,)` | `joint_pos_rel` (6) + `joint_vel` (6) + `target_pos` (3) + `last_action` (5) |

**Action**: 5 arm joint-position targets (gripper unused), applied as
`target = default_joint_pos + action_scale * action`.

The obstacle is a **realistic human-arm mesh** (tagged `class:hand`). By default
the hand mask is the **true silhouette** taken from the camera's semantic
segmentation (`mask_source="seg"`) — the same hand/not-hand signal a MediaPipe
mask gives at deploy time. A cheaper analytic blob projected from the hand's 3D
position is available as `mask_source="projected"` (no segmentation render).

A green **goal marker** is drawn at the sampled target so you can see, in the GUI
and in the camera RGB, where the end-effector should reach.

The arm mesh ships pre-decimated (`arm_lowpoly.usd`, ~8k tris) since the original
is ~1M tris of pure geometry — invisible at the policy's render resolution but far
heavier. Regenerate at any budget with `scripts/decimate_arm.py` (e.g.
`./docker/run.sh bash -c "python -m pip install -q fast-simplification && python scripts/decimate_arm.py --target_tris 3000"`).

---

## Configurable knobs (`SoArm101ReachAvoidEnvCfg`)

| Field | Default | Options / notes |
|-------|---------|-----------------|
| `camera_view` | `"overhead"` | `"overhead"`, `"wrist"`, `"both"` |
| `obs_mode` | `"rgb+mask"` | `"rgb+mask"`, `"rgb"`, `"mask"` |
| `mask_source` | `"seg"` | `"seg"` (true silhouette) or `"projected"` (analytic blob) |
| `show_goal_marker` | `True` | draw the green target marker (GUI/RGB) |
| `domain_randomization` | `True` | **stub** — not yet implemented |
| `image_height/width` | `100` | policy image resolution |
| reward weights | — | `w_track`, `w_clearance`, `clearance_std`, `collision_distance`, ... |

These flags are the levers for the planned ablations (modality, DR, reward
shaping).

---

## Status / roadmap

- [x] Dockerized Isaac Lab (shareable, GUI + USB passthrough).
- [x] Direct-workflow `ReachAvoid` env (SO-101 + camera + moving hand).
- [x] RGB + true-silhouette hand-mask Dict observation (verified aligned via `debug_camera.py`).
- [x] Realistic human-arm obstacle (semantic segmentation) + goal marker.
- [x] skrl PPO (CNN + MLP) pipeline validated (camera is actually used).
- [ ] Domain randomization + reward tuning.
- [ ] Eval harness (success %, collision %, min-clearance) + plots.
- [ ] LeRobot deploy bridge (camera + joints → policy → joint targets, safety clamp).
- [ ] Real-robot avoidance video. ArUco-glove perception as a fallback.

---

## Credits

- **SO-101 URDF + `ArticulationCfg`**: vendored from
  [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (BSD-3-Clause).
- **Base image / sim**: NVIDIA Isaac Lab + the
  [Sim-to-Real SO-101 workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
  Docker patterns.
- **Reach-avoid design reference**: the vision-RL UR5 obstacle-avoidance repo
  [aparame/RL_UR5_IsaacLab](https://github.com/aparame/RL_UR5_IsaacLab).
