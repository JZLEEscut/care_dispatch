"""Rolling OR-Tools baseline.

This implementation solves a small assignment-style snapshot with OR-Tools
CP-SAT when available.  It deliberately returns only the worker action for the
current active task and never rewrites the environment route plan directly.
If OR-Tools is unavailable, times out, or fails, the policy falls back to the
PureHeuristicPolicy.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.baselines.base_policy import BasePolicy
from src.baselines.pure_heuristic_policy import PureHeuristicPolicy
from src.env.dispatch_env import CareDispatchEnv
from src.env.entities import Task
from src.heuristics.cost import get_cost_value
from src.heuristics.insertion import evaluate_worker_insertion


class RollingORToolsPolicy(BasePolicy):
    method_name = "rolling_ortools"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.config = cfg
        local = cfg.get("rolling_ortools", cfg)
        self.time_limit_seconds = float(local.get("time_limit_seconds", 2.0))
        self.lookahead_tasks = int(local.get("lookahead_tasks", 8))
        self.fallback_policy = PureHeuristicPolicy(cfg.get("pure_heuristic", {}))

    def _snapshot_tasks(self, env: CareDispatchEnv) -> list[Task]:
        active = env.active_task
        if active is None:
            return []
        waiting = [
            t for t in env.tasks.values()
            if t.status == "waiting" and t.id != active.id and t.release_time <= env.now + 1e-9
        ]
        waiting.sort(key=lambda t: (-t.priority, t.release_time, t.id))
        return [active] + waiting[: self.lookahead_tasks]

    def _reject_penalty(self, task: Task, env: CareDispatchEnv) -> float:
        if task.priority == 1:
            return get_cost_value(env.config, "reject_penalty_emergency", 1500.0)
        return get_cost_value(env.config, "reject_penalty_regular", 500.0)

    def _solve_snapshot_and_get_active_task_worker(self, state: dict[str, Any], env: CareDispatchEnv) -> int | None:
        try:
            from ortools.sat.python import cp_model
        except Exception:
            return None

        tasks = self._snapshot_tasks(env)
        if not tasks:
            return None
        active_id = tasks[0].id
        n_workers = len(env.workers)
        reject_idx = n_workers

        # Independent insertion-cost approximation. Costs are scaled to ints
        # for CP-SAT.  Infeasible worker-task pairs are not added.
        scale = 100
        model = cp_model.CpModel()
        x: dict[tuple[int, int], Any] = {}
        costs: dict[tuple[int, int], int] = {}
        for ti, task in enumerate(tasks):
            # Reject is always feasible.
            rkey = (ti, reject_idx)
            x[rkey] = model.NewBoolVar(f"x_{ti}_reject")
            costs[rkey] = int(round(self._reject_penalty(task, env) * scale))
            for k, worker in enumerate(env.workers):
                if worker.skill < task.skill_req:
                    continue
                result = evaluate_worker_insertion(worker, task, env.tasks, env.now, env.config)
                if not result.feasible:
                    continue
                key = (ti, k)
                x[key] = model.NewBoolVar(f"x_{ti}_{k}")
                costs[key] = int(round(max(0.0, result.delta_total_cost) * scale))
            model.Add(sum(var for (tti, _), var in x.items() if tti == ti) == 1)

        # Softly discourage overloading one worker within the snapshot.
        for k in range(n_workers):
            vars_k = [var for (ti, kk), var in x.items() if kk == k]
            if vars_k:
                model.Add(sum(vars_k) <= max(1, len(tasks)))

        model.Minimize(sum(costs[key] * var for key, var in x.items()))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        for (ti, k), var in x.items():
            if tasks[ti].id == active_id and solver.Value(var) == 1:
                if k == reject_idx:
                    return reject_idx
                return int(k)
        return None

    def select_action(self, state: dict[str, Any], env: CareDispatchEnv, deterministic: bool = True) -> int:
        try:
            action = self._solve_snapshot_and_get_active_task_worker(state, env)
            if action is None:
                return self.fallback_policy.select_action(state, env, deterministic=True)
            if 0 <= int(action) <= len(env.workers):
                return int(action)
            return self.fallback_policy.select_action(state, env, deterministic=True)
        except Exception:
            return self.fallback_policy.select_action(state, env, deterministic=True)
