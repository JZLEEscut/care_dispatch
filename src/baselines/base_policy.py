"""Unified policy interface for benchmark and ablation experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.env.dispatch_env import CareDispatchEnv


class BasePolicy(ABC):
    """All benchmark methods return an action for env.step(action)."""

    method_name: str = "base_policy"

    @abstractmethod
    def select_action(self, state: dict[str, Any], env: CareDispatchEnv, deterministic: bool = True) -> int:
        raise NotImplementedError
