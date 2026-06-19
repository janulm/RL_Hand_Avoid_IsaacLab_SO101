import gymnasium as gym

from . import agents
from .reach_avoid_env import (
    SoArm101ReachAvoidEnv,
    SoArm101ReachAvoidEnvCfg,
    SoArm101ReachAvoidEnvCfg_PLAY,
)

gym.register(
    id="Isaac-SO-ARM101-ReachAvoid-Direct-v0",
    entry_point=f"{__name__}.reach_avoid_env:SoArm101ReachAvoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SoArm101ReachAvoidEnvCfg,
        # Framework default agent (used for plain `--task` runs without an
        # --exp_config). Experiment files in configs/experiments/ are
        # self-contained and override this entirely.
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SO-ARM101-ReachAvoid-Direct-Play-v0",
    entry_point=f"{__name__}.reach_avoid_env:SoArm101ReachAvoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SoArm101ReachAvoidEnvCfg_PLAY,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
