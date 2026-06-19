# Experiment configs

Each YAML here is **one self-contained experiment** — it fully defines what runs:
the run name, the task, how many envs / how long to train, the environment
overrides (hand spawn probability, hand motion, target sampling, vision on/off,
...) **and the full agent** (network architecture + PPO hyperparameters). There is
nothing else to look at: open one file and you see the entire experiment.

Train with `--exp_config`; checkpoints and logs land in
`logs/skrl/so101_reachavoid/<name>_<timestamp>/`.

```bash
# Train
./docker/run.sh python scripts/skrl/train.py --headless \
    --exp_config configs/experiments/reach_only_proprio.yaml

# Play the latest run of that experiment (same env + network, auto-finds checkpoint)
./docker/run.sh python scripts/skrl/play.py \
    --exp_config configs/experiments/reach_only_proprio.yaml

# Full reach-avoid, resuming from a reach checkpoint (curriculum)
./docker/run.sh python scripts/skrl/train.py --headless \
    --exp_config configs/experiments/reach_avoid.yaml \
    --checkpoint logs/skrl/so101_reachavoid/reach_only_vision_<ts>/checkpoints/best_agent.pt
```

## Available experiments

| file | vision | hand | envs | what it's for |
| ---- | ------ | ---- | ---- | ------------- |
| `reach_only_proprio.yaml` | off | none | 2048 | fastest reach sanity check (MLP, no cameras) |
| `reach_only_vision.yaml`  | on  | none | 128  | reach with the CNN/camera path |
| `reach_avoid.yaml`        | on  | 80%  | 96   | full reach-avoid task |

## Schema

Top level:

| key              | meaning                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `name`           | run name; checkpoints go in `<name>_<timestamp>/`                |
| `task`           | gym id (default `Isaac-SO-ARM101-ReachAvoid-Direct-v0`)         |
| `num_envs`       | parallel environments                                            |
| `max_iterations` | PPO policy-update iterations (sets `trainer.timesteps`)          |
| `seed`           | RNG seed                                                         |
| `env`            | overrides applied to `SoArm101ReachAvoidEnvCfg` (dotted keys ok) |
| `agent`          | full skrl runner cfg: `models`, `memory`, `agent`, `trainer`     |

Notes:
- The `agent` block is the complete skrl config (same format the skrl `Runner`
  consumes). Proprio experiments use an MLP-only network; vision experiments use
  the CNN(image)+MLP(proprio) network.
- CLI flags override the YAML when passed (e.g. `--num_envs 4096`,
  `--max_iterations 300`). `enable_cameras` is auto-set unless `env.use_vision: false`.
- `agents/skrl_ppo_cfg.yaml` in the extension is only the framework default for
  plain `--task` runs (no `--exp_config`); experiments here override it entirely.
