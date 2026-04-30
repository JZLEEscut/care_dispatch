"""Policy wrapper for trained PPO checkpoints."""
from __future__ import annotations

from typing import Any

import torch

from src.baselines.base_policy import BasePolicy
from src.env.dispatch_env import CareDispatchEnv
from src.env.state_builder import build_flat_state
from src.graph.graph_builder import build_graph_state
from src.models.action_mask import apply_action_mask


class PPOPolicy(BasePolicy):
    def __init__(self, model, model_type: str, env_config: dict[str, Any], model_config: dict[str, Any] | None = None, method_name: str | None = None, device: str | torch.device = "cpu"):
        self.model = model
        self.model_type = model_type
        self.env_config = env_config
        self.model_config = model_config or {}
        self.method_name = method_name or f"{model_type}_ppo"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        mask_cfg = self.model_config.get("action_mask", self.env_config.get("action_mask", {}))
        self.action_mask_enabled = bool(mask_cfg.get("enabled", True)) if isinstance(mask_cfg, dict) else bool(mask_cfg)

    @torch.no_grad()
    def select_action(self, state: dict[str, Any], env: CareDispatchEnv, deterministic: bool = True) -> int:
        mask_np = env.get_action_mask()
        if self.model_type == "gat":
            obs = build_graph_state(state, mask_np, self.env_config, device=self.device)
            logits, _ = self.model(obs)
        else:
            flat = build_flat_state(state, self.env_config)
            obs = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
            logits, _ = self.model(obs)
        mask = torch.as_tensor(mask_np, dtype=torch.float32, device=self.device)
        logits = logits.reshape(-1)
        if self.action_mask_enabled:
            logits = apply_action_mask(logits, mask).reshape(-1)
        if deterministic:
            return int(torch.argmax(logits).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
