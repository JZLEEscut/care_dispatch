"""Pure heuristic dispatch baseline.

The upper-level decision still uses the same M+1 action interface: choose one
worker or reject.  Route feasibility and cost are delegated to the existing
bottom-level insertion heuristic.
"""
from __future__ import annotations

from typing import Any

from src.baselines.base_policy import BasePolicy
from src.env.dispatch_env import CareDispatchEnv
from src.heuristics.cost import get_cost_value
from src.heuristics.insertion import evaluate_worker_insertion


class PureHeuristicPolicy(BasePolicy):
    method_name = "pure_heuristic"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @staticmethod
    def _reject_penalty(env: CareDispatchEnv) -> float:
        task = env.active_task
        if task is None:
            return 0.0
        if task.priority == 1:
            return get_cost_value(env.config, "reject_penalty_emergency", 1500.0)
        return get_cost_value(env.config, "reject_penalty_regular", 500.0)

    def select_action(self, state: dict[str, Any], env: CareDispatchEnv, deterministic: bool = True) -> int:
        task = env.active_task
        n_workers = len(env.workers)
        reject_action = n_workers
        if task is None:
            return reject_action

        mask = env.get_action_mask()
        best_worker: int | None = None
        best_cost = float("inf")
        for k, worker in enumerate(env.workers):
            if k >= len(mask) or mask[k] <= 0:
                continue
            result = evaluate_worker_insertion(worker, task, env.tasks, env.now, env.config)
            if result.feasible and result.delta_total_cost < best_cost:
                best_cost = float(result.delta_total_cost)
                best_worker = k

        if best_worker is None:
            return reject_action

        # If serving the task is more expensive than rejecting it, reject.
        if best_cost > self._reject_penalty(env):
            return reject_action
        return int(best_worker)
