"""Helpers to train/load PPO policies for experiment runners."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from src.common.paths import resolve_project_path
from src.common.seed import set_global_seed
from src.env.dispatch_env import CareDispatchEnv
from src.models.factory import build_actor_critic
from src.rl.ppo_trainer import PPOTrainer
from src.scripts._script_utils import deep_update, load_yaml


def merge_model_switches_into_env(env_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(env_cfg)
    for key in ("action_mask", "preemption"):
        if key in model_cfg:
            out[key] = deepcopy(model_cfg[key])
    return out


def load_or_train_ppo(
    model_type: str,
    model_config_path: str,
    base_env_config: dict[str, Any],
    seed: int,
    checkpoint_path_template: str,
    train_if_no_checkpoint: bool = True,
    force_train: bool = False,
    total_timesteps: int | None = None,
    rollout_steps: int | None = None,
    device: str | None = None,
    checkpoint_name: str | None = None,
):
    set_global_seed(seed)
    model_cfg = load_yaml(model_config_path)
    model_cfg["model_type"] = model_type
    if total_timesteps is not None:
        model_cfg.setdefault("ppo", {})["total_timesteps"] = int(total_timesteps)
    if rollout_steps is not None:
        model_cfg.setdefault("ppo", {})["rollout_steps"] = int(rollout_steps)
    env_cfg = merge_model_switches_into_env(base_env_config, model_cfg)

    env = CareDispatchEnv(env_cfg)
    env.reset(seed=seed)
    model = build_actor_critic(model_cfg, len(env.workers))
    ckpt_path = resolve_project_path(checkpoint_path_template.format(seed=seed))
    if ckpt_path.exists() and not force_train:
        ckpt = torch.load(ckpt_path, map_location=device or "cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return model, model_cfg, env_cfg, ckpt_path
    if ckpt_path.exists() and force_train:
        print(f"force_train=true, overwriting_checkpoint={ckpt_path}", flush=True)
    if not train_if_no_checkpoint and not force_train:
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    trainer = PPOTrainer(env, model, env_cfg, model_cfg, model_type, seed=seed, device=device)
    ckpt_name = checkpoint_name or Path(checkpoint_path_template.format(seed=seed)).name
    result = trainer.train(checkpoint_name=ckpt_name)
    return model, model_cfg, env_cfg, result.checkpoint_path
