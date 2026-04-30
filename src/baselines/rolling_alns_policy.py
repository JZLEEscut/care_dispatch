"""Lightweight rolling ALNS baseline.

The policy works on a short snapshot and returns only the worker assigned to the
current active task. It does not directly change env.routes; env.step(action)
remains the single execution interface.
"""
from __future__ import annotations

import random
from copy import deepcopy
from time import perf_counter
from typing import Any

from src.baselines.base_policy import BasePolicy
from src.baselines.pure_heuristic_policy import PureHeuristicPolicy
from src.env.dispatch_env import CareDispatchEnv
from src.env.entities import Task, Worker
from src.heuristics.cost import get_cost_value
from src.heuristics.insertion import evaluate_worker_insertion


class RollingALNSPolicy(BasePolicy):
    method_name = "rolling_alns"

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.config = cfg
        local = cfg.get("rolling_alns", cfg)
        self.iterations = int(local.get("iterations", 100))
        self.time_limit_seconds = float(local.get("time_limit_seconds", 2.0))
        self.lookahead_tasks = int(local.get("lookahead_tasks", 10))
        self.destroy_ratio = float(local.get("destroy_ratio", 0.25))
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

    def _greedy_construct(self, tasks: list[Task], env: CareDispatchEnv, order: list[int] | None = None) -> tuple[dict[int, int], float]:
        # Use copied worker routes so that assignments within the snapshot can
        # affect later insertion costs without touching the real environment.
        workers = deepcopy(env.workers)
        assignments: dict[int, int] = {}
        total_cost = 0.0
        task_indices = order if order is not None else list(range(len(tasks)))

        for idx in task_indices:
            task = tasks[idx]
            best_k = len(workers)
            best_result = None
            best_cost = self._reject_penalty(task, env)
            for k, worker in enumerate(workers):
                if worker.skill < task.skill_req:
                    continue
                result = evaluate_worker_insertion(worker, task, env.tasks, env.now, env.config)
                if result.feasible and result.delta_total_cost < best_cost:
                    best_cost = float(result.delta_total_cost)
                    best_result = result
                    best_k = k
            assignments[task.id] = int(best_k)
            total_cost += float(best_cost)
            if best_result is not None and best_k < len(workers):
                workers[best_k].route = list(best_result.new_route)
        return assignments, total_cost

    def _worst_order(self, tasks: list[Task], env: CareDispatchEnv) -> list[int]:
        scores = []
        for i, task in enumerate(tasks):
            best = self._reject_penalty(task, env)
            for worker in env.workers:
                result = evaluate_worker_insertion(worker, task, env.tasks, env.now, env.config)
                if result.feasible:
                    best = min(best, float(result.delta_total_cost))
            scores.append((best, i))
        scores.sort(reverse=True)
        return [i for _, i in scores]

    def select_action(self, state: dict[str, Any], env: CareDispatchEnv, deterministic: bool = True) -> int:
        start = perf_counter()
        try:
            tasks = self._snapshot_tasks(env)
            if not tasks:
                return len(env.workers)
            active_id = tasks[0].id

            best_assign, best_cost = self._greedy_construct(tasks, env)
            rng = random.Random(int(env.now * 1000) + active_id + len(tasks))

            for it in range(max(1, self.iterations)):
                if perf_counter() - start > self.time_limit_seconds:
                    break
                order = list(range(len(tasks)))
                if it % 3 == 0:
                    rng.shuffle(order)  # random_remove + greedy repair effect
                elif it % 3 == 1:
                    order = self._worst_order(tasks, env)  # worst_remove first
                else:
                    # relocate-like perturbation: active first, then shuffle rest
                    rest = order[1:]
                    rng.shuffle(rest)
                    order = [0] + rest
                assign, cost = self._greedy_construct(tasks, env, order=order)
                if cost < best_cost:
                    best_assign, best_cost = assign, cost

            action = int(best_assign.get(active_id, len(env.workers)))
            if 0 <= action <= len(env.workers):
                return action
            return self.fallback_policy.select_action(state, env, deterministic=True)
        except Exception:
            return self.fallback_policy.select_action(state, env, deterministic=True)
