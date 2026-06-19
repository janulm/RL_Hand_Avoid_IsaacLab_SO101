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
configs/experiments/        # self-contained experiment files (env + full agent), one per run
scripts/
  skrl/train.py, play.py    # skrl PPO training / playback (take --exp_config)
  list_envs.py              # list registered SO-ARM101 tasks
  smoke_test.py             # quick env sanity check (reset + step)
  debug_camera.py           # dump RGB|mask|overlay frames of the policy input
  create_charuco.py         # make a ChArUco board (camera calibration)
  align_cameras.py          # align/calibrate real cameras (deploy)
source/so_arm101_avoid/     # the Isaac Lab extension (our env lives here)
  so_arm101_avoid/
    robots/trs_so101/       # vendored SO-101 URDF + ArticulationCfg
    tasks/reach_avoid/      # Direct ReachAvoid env + cfg + default agent yaml (fallback)
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

# Train via an experiment config (recommended -- see "Experiment configs" below).
# Each config is self-contained (env + full agent); cameras auto-enable unless
# the config sets use_vision: false.
./docker/run.sh python scripts/skrl/train.py --headless \
    --exp_config configs/experiments/reach_only_proprio.yaml

# Or train directly from the CLI (uses the default agent; --enable_cameras for vision).
./docker/run.sh python scripts/skrl/train.py \
    --task Isaac-SO-ARM101-ReachAvoid-Direct-v0 --headless --enable_cameras

# Watch a trained policy in the GUI (same experiment -> same env + network)
./docker/run.sh python scripts/skrl/play.py \
    --exp_config configs/experiments/reach_only_proprio.yaml
```

> Tip: `--max_iterations N` and `--num_envs N` are handy for short runs, and
> override the values from an `--exp_config` when passed.

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

**Reward** (logged per-term to TensorBoard under `Info / ...`): the main term is
`-w_track * dist`, where `dist` is the Euclidean distance from the **gripper tip**
(`gripper_link` + `ee_offset`) to the target point, plus a fine `tanh` shaping
bonus, a hand-clearance penalty, an action-rate penalty, and a collision penalty.
View live curves with:

```bash
CONTAINER_NAME=so101-tb ./docker/run.sh python -m tensorboard.main --logdir logs/skrl --bind_all --port 6006
# then open http://localhost:6006  (look for reward/total, dist_tip_to_target, ...)
```

The obstacle is a **realistic human-arm mesh** (tagged `class:hand`). By default
the hand mask is the **true silhouette** taken from the camera's semantic
segmentation (`mask_source="seg"`) — the same hand/not-hand signal a MediaPipe
mask gives at deploy time. A cheaper analytic blob projected from the hand's 3D
position is available as `mask_source="projected"` (no segmentation render).

A green **goal marker** is drawn at the sampled target so you can see where the
end-effector should reach. It is a **debug-draw overlay**, so it shows up only in
the **GUI viewport (non-headless)** and is **never captured by the policy
cameras** — the policy never "sees" the answer. Targets are sampled in a box
(`target_*_range`) and clamped onto a reachable sphere (`reach_center` /
`reach_radius`) so a large box never yields an out-of-reach corner (the SO-101
has only ~0.30 m of reach).

The arm mesh ships pre-decimated (`arm_lowpoly.usd`, ~8k tris) since the original
is ~1M tris of pure geometry — invisible at the policy's render resolution but far
heavier. Regenerate at any budget with `scripts/decimate_arm.py` (e.g.
`./docker/run.sh bash -c "python -m pip install -q fast-simplification && python scripts/decimate_arm.py --target_tris 3000"`).

---

## Configurable knobs (`SoArm101ReachAvoidEnvCfg`)

| Field | Default | Options / notes |
|-------|---------|-----------------|
| `use_vision` | `True` | `False` → proprio-only obs, no cameras (fast reach smoke test; pair with the proprio agent cfg) |
| `camera_view` | `"overhead"` | `"overhead"`, `"wrist"`, `"both"` |
| `obs_mode` | `"rgb+mask"` | `"rgb+mask"`, `"rgb"`, `"mask"` |
| `hand_spawn_prob` | `0.0` | per-env probability the hand obstacle is present (curriculum: 0 = pure reach) |
| `target_*_range` | ±0.30 / ±0.30 / 0.02..0.40 | target sampling box, symmetric around the root (m) |
| `clamp_targets_to_reach` | `False` | `True` → clamp targets onto the reachable sphere; `False` → allow out-of-reach targets (minimise distance) |
| `target_resample_time_s` | `4.0` | resample the target mid-episode every N s (≈2 reaches/episode); `0` → fixed per episode |
| `noise_joint_pos` / `noise_joint_vel` | `0.01` / `0.01` | sim2real obs noise on joints (only when `domain_randomization=True`) |
| `reach_center` / `reach_radius` | `(0,0,0.12)` / `0.30` | the reachable sphere used when clamping is on |
| `ee_offset` | `(-0.008,0,-0.098)` | gripper-tip offset (m, in `gripper_link` frame); the reward distance is measured to this point |
| `mask_source` | `"seg"` | `"seg"` (true silhouette) or `"projected"` (analytic blob) |
| `show_goal_marker` | `True` | green target overlay — **GUI only, never in the policy cameras** |
| `domain_randomization` | `True` | **stub** — not yet implemented |
| `image_height/width` | `144`/`256` | policy image resolution |
| reward weights | — | `w_track`, `w_clearance`, `clearance_std`, `collision_distance`, ... |

These flags are the levers for the planned ablations (modality, DR, reward
shaping).

### Experiment configs (recommended way to train)

Each file in `configs/experiments/*.yaml` is **one self-contained experiment**:
run name, task, env overrides (hand spawn, target sampling, vision on/off, ...)
**and the full agent** (network + PPO hyperparameters). Launch with `--exp_config`;
checkpoints land in `logs/skrl/so101_reachavoid/<name>_<timestamp>/`.

```bash
# 1) fastest sanity check: proprio-only reach (no cameras)
./docker/run.sh python scripts/skrl/train.py --headless \
    --exp_config configs/experiments/reach_only_proprio.yaml

# 2) reach with the camera/CNN path
./docker/run.sh python scripts/skrl/train.py --headless \
    --exp_config configs/experiments/reach_only_vision.yaml

# 3) full reach-avoid
./docker/run.sh python scripts/skrl/train.py --headless \
    --exp_config configs/experiments/reach_avoid.yaml

# play the latest run of an experiment (same env + network, auto-finds checkpoint)
./docker/run.sh python scripts/skrl/play.py \
    --exp_config configs/experiments/reach_only_proprio.yaml
```

See `configs/experiments/README.md` for the schema. (`agents/skrl_ppo_cfg.yaml`
is only the framework default for plain `--task` runs without `--exp_config`.)

---

## Status / roadmap

- [x] Dockerized Isaac Lab (shareable, GUI + USB passthrough).
- [x] Direct-workflow `ReachAvoid` env (SO-101 + camera + moving hand).
- [x] RGB + true-silhouette hand-mask Dict observation (verified aligned via `debug_camera.py`).
- [x] Realistic human-arm obstacle (semantic segmentation) + goal marker.
- [x] skrl PPO (CNN + MLP) pipeline validated (camera is actually used).
- [x] Proprio-only mode (`use_vision: false`) + experiment-config training (`--exp_config`, named/timestamped checkpoints).
- [x] Mid-episode target resampling + joint observation noise (first DR piece).
- [ ] More domain randomization (lighting, hand appearance, dynamics) + reward tuning.
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
