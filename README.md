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

> Tip: `--max_iterations N` and `--num_envs N` are handy for short runs and
> override the values from an `--exp_config`. For quick sweeps, `train.py` also
> takes `--run_name NAME` and repeatable `--set KEY=VALUE` overrides (YAML-parsed,
> sci-notation safe) on top of a config, e.g.
> `--run_name try_hi --set env.action_scale=1.0 --set agent.agent.entropy_loss_scale=0.002`.
> Use `env.` for env-cfg fields and `agent.` for the skrl agent dict. Each run's
> fully-resolved config is dumped to `<run_dir>/params/{env,agent}.yaml`.

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

A green **goal marker** (and an XYZ **base triad**: X=red/Y=green/Z=blue) is drawn
in the GUI. Both are **debug-draw overlays**, so they show up only in the **GUI
viewport (non-headless)** and are **never captured by the policy cameras** — the
policy never "sees" the answer. Targets are sampled in a box (`target_*_range`) in
front of the arm (**+X = forward**) and clamped onto a reachable sphere
(`reach_center` / `reach_radius`) so every goal is physically reachable (the
SO-101 has only ~0.28 m of reach).

The arm mesh ships pre-decimated (`arm_lowpoly.usd`, ~8k tris) since the original
is ~1M tris of pure geometry — invisible at the policy's render resolution but far
heavier. Regenerate at any budget with `scripts/decimate_arm.py` (e.g.
`./docker/run.sh bash -c "python -m pip install -q fast-simplification && python scripts/decimate_arm.py --target_tris 3000"`).

---

## Configurable knobs (`SoArm101ReachAvoidEnvCfg`)

| Field | Default | Options / notes |
|-------|---------|-----------------|
| `use_vision` | `True` | `False` → proprio-only obs, no cameras (fast reach smoke test; see `reach_only_proprio.yaml`) |
| `camera_view` | `"overhead"` | `"overhead"`, `"wrist"`, `"both"` |
| `obs_mode` | `"rgb+mask"` | `"rgb+mask"`, `"rgb"`, `"mask"` |
| `hand_spawn_prob` | `0.0` | per-env probability the hand obstacle is present (curriculum: 0 = pure reach) |
| `action_scale` | `1.0` | joint authority: `target = default + action_scale·action`. A sweep showed `0.5` capped reach error at ~8 cm; `1.0` ≈ 3.7 cm; higher gave no gain but jerkier motion |
| `target_*_range` | x 0.05..0.30 / y ±0.18 / z 0.05..0.32 | target sampling box in the root frame (**+X = forward, +Y = left, +Z = up**); sampled in front of the arm |
| `clamp_targets_to_reach` | `True` | `True` → clamp targets onto the reachable sphere (every goal reachable); `False` + a symmetric box → allow out-of-reach targets (just minimise distance) |
| `target_resample_time_s` | `4.0` | resample the target mid-episode every N s (≈2 reaches/episode); `0` → fixed per episode |
| `noise_joint_pos` / `noise_joint_vel` | `0.01` / `0.01` | sim2real obs noise on joints (only when `domain_randomization=True`) |
| `reach_center` / `reach_radius` | `(0,0,0.12)` / `0.28` | the reachable sphere used when clamping is on |
| `ee_offset` | `(-0.008,0,-0.098)` | gripper-tip offset (m, in `gripper_link` frame); the reward distance is measured to this point |
| `mask_source` | `"seg"` | `"seg"` (true silhouette) or `"projected"` (analytic blob) |
| `show_goal_marker` | `True` | green target overlay — **GUI only, never in the policy cameras** |
| `show_frame_axes` | `True` | XYZ triad (X=red/Y=green/Z=blue) at each base — **GUI only**; shows the frame the target box is sampled in |
| `domain_randomization` | `True` | currently enables joint observation noise; more DR (lighting/appearance/dynamics) planned |
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

# play the latest run of an experiment (auto-loads best_agent.pt of the newest run)
./docker/run.sh python scripts/skrl/play.py \
    --exp_config configs/experiments/reach_only_proprio.yaml --num_envs 16

# ...or play a specific checkpoint (e.g. the final snapshot)
./docker/run.sh python scripts/skrl/play.py \
    --exp_config configs/experiments/reach_only_proprio.yaml --num_envs 16 \
    --checkpoint logs/skrl/so101_reachavoid/reach_only_proprio_<TS>/checkpoints/agent_9600.pt
```

Each run saves periodic `agent_<step>.pt` snapshots plus `best_agent.pt` (skrl's
highest-mean-reward checkpoint) under `<run_dir>/checkpoints/`. With no
`--checkpoint`, play auto-loads `best_agent.pt` from the most recent matching run.

In the GUI viewport, **right-mouse + WASD** flies the camera (Q/E = down/up),
**middle-mouse** pans, scroll zooms, and **F** focuses the selected prim.

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
- [x] Proprio reach tuned: forward+reachable targets and an `action_scale` sweep took tip→target error from ~8.7 cm to ~3.7 cm.

**Deployment-first plan.** Each stage must **verify a working deploy on the real
SO-101 before moving on**. Perception is added bottom-up: **binary arm/no-arm mask
first, RGB only once that works.**

- [ ] **Deploy proprio reach** on the physical LeRobot follower: drive the trained reach policy with hard-coded target points; verify physics, joint mapping, and that the motion is safe/sane.
- [ ] **Deploy vision reach (mask-only)**: train a policy with visual input — the **binary arm/no-arm mask only** — whose task is still to reach a hard-coded target; verify it deploys.
- [ ] **Fix arm integration / orientation** in sim (the arm is currently visually mis-oriented) so sim and real agree.
- [ ] **Train with arm detection in sim**: feed the segmentation mask of the arm/hand obstacle (still mask-only); verify deploy.
- [ ] **Real-camera segmentation**: produce the arm/no-arm mask from the physical camera at deploy time, matching the mask the policy trained on.

> **Hackathon baseline** = everything above working end-to-end (mask-only, overhead camera).

Then, building on the baseline (re-verify deploy after each step):

- [ ] Integrate **RGB** on top of the mask (4-channel), only once mask-only is solid.
- [ ] **Sim realism for RGB**: white table/floor, real-matching (color-block) robot colors, and randomized lighting / shadows / brightness so the real camera isn't confused.
- [ ] Match **camera intrinsics / extrinsics** to the real overhead camera.
- [ ] **Multiple cameras**: add the wrist camera alongside the overhead (overhead-only until here).
- [ ] **Stronger domain randomization** (lighting, hand appearance, dynamics) + RL env / reward tuning.
- [ ] **Eval harness** (success %, collision %, min-clearance) + plots, and a **real-robot avoidance video**. ArUco-glove perception as a fallback.

---

## Credits

- **SO-101 URDF + `ArticulationCfg`**: vendored from
  [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (BSD-3-Clause).
- **Base image / sim**: NVIDIA Isaac Lab + the
  [Sim-to-Real SO-101 workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
  Docker patterns.
- **Reach-avoid design reference**: the vision-RL UR5 obstacle-avoidance repo
  [aparame/RL_UR5_IsaacLab](https://github.com/aparame/RL_UR5_IsaacLab).
