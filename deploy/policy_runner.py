"""Standalone deterministic policy for proprio reach (used by the bridge in docker).

skrl saves the PPO policy as a plain ``state_dict`` inside the checkpoint. The
proprio policy is just an MLP trunk (``net_container``) + a mean head
(``policy_layer``); the state preprocessor is disabled for Dict observations, so
no input normalization is needed. We rebuild that MLP from the agent config and
load the matching weights, then return the (clipped) mean action for deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class ProprioPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        in_dim: int,
        n_actions: int,
        layers=(256, 128, 64),
        activation: str = "elu",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        act = _ACT[activation.lower()]
        mods, d = [], in_dim
        for h in layers:
            mods += [nn.Linear(d, h), act()]
            d = h
        self.net = nn.Sequential(*mods)
        self.policy_layer = nn.Linear(d, n_actions)
        self._load(checkpoint_path)
        self.net.to(self.device).eval()
        self.policy_layer.to(self.device).eval()

    def _load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected checkpoint format in {path}")
        net_sd, pol_sd = {}, {}
        for k, v in sd.items():
            if k.startswith("net_container."):
                net_sd[k[len("net_container."):]] = v
            elif k.startswith("policy_layer."):
                pol_sd[k[len("policy_layer."):]] = v
        missing_n, _ = self.net.load_state_dict(net_sd, strict=False)
        missing_p, _ = self.policy_layer.load_state_dict(pol_sd, strict=False)
        if missing_n or missing_p:
            raise RuntimeError(
                "Policy weights did not match the rebuilt MLP "
                f"(missing net={missing_n}, policy={missing_p}). "
                "Check the layers/activation passed from the agent config."
            )

    @torch.no_grad()
    def act(self, proprio) -> "list[float]":
        x = torch.as_tensor(proprio, dtype=torch.float32, device=self.device).reshape(1, -1)
        a = self.policy_layer(self.net(x)).clamp(-1.0, 1.0)
        return a.squeeze(0).cpu().tolist()
