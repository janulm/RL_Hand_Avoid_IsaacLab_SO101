# so_arm101_avoid

Vision-based human-hand **Reach-Avoid** RL environment for the SO-ARM101, built
on the Isaac Lab **Direct** workflow (the manager-based workflow cannot expose
composite image+state observations, so vision RL must use Direct).

Registered tasks:

- `Isaac-SO-ARM101-ReachAvoid-Direct-v0` — training.
- `Isaac-SO-ARM101-ReachAvoid-Direct-Play-v0` — few-env eval/visualization.

See the repo root README for the full plan, Docker usage, and the
observation/action contract.
