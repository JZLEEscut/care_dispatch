"""Event-driven dynamic care dispatch environment."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from src.data.scenario_builder import build_scenario
from src.env.entities import Task, Worker
from src.env.metrics import EpisodeMetrics
from src.env.state_builder import build_state_dict
from src.heuristics.cost import distance, get_cost_value
from src.heuristics.insertion import evaluate_worker_insertion
from src.heuristics.delay_propagation import simulate_route
from src.heuristics.legality import action_is_worker


class CareDispatchEnv:
    def __init__(self, config: dict | str | Path):
        if isinstance(config, (str, Path)):
            with Path(config).open("r", encoding="utf-8") as f:
                self.config: dict[str, Any] = yaml.safe_load(f) or {}
        else:
            self.config = deepcopy(config)

        self.tasks: dict[int, Task] = {}
        self.workers: list[Worker] = []
        self.now = 0.0
        self.active_task_id: int | None = None
        self.step_count = 0
        self.metrics = EpisodeMetrics()
        self.rng = np.random.default_rng(None)

    @property
    def day_start(self) -> float:
        return float(self.config.get("time", {}).get("day_start", 0.0))

    @property
    def day_end(self) -> float:
        return float(self.config.get("time", {}).get("day_end", 480.0))

    @property
    def max_steps_per_episode(self) -> int:
        return int(self.config.get("time", {}).get("max_steps_per_episode", 200))

    @property
    def active_task(self) -> Task | None:
        if self.active_task_id is None:
            return None
        return self.tasks.get(self.active_task_id)

    def reset(self, seed: int | None = None) -> dict:
        self.rng = np.random.default_rng(seed)
        scenario = build_scenario(self.config, seed=seed)
        self.tasks = {t.id: t for t in scenario["tasks"]}
        self.workers = scenario["workers"]
        self.now = self.day_start
        self.active_task_id = None
        self.step_count = 0
        self.metrics = EpisodeMetrics()
        self._select_or_advance_to_next_decision()
        return self.build_state()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        start_clock = perf_counter()
        self.step_count += 1
        task = self.active_task
        if task is None:
            return self.build_state(), 0.0, self.is_done(), {"no_active_task": True}

        info: dict[str, Any] = {
            "task_id": task.id,
            "action": int(action),
            "delta_distance_cost": 0.0,
            "delta_tardiness_cost": 0.0,
            "delta_preemption_cost": 0.0,
            "reject_cost": 0.0,
            "delta_total_cost": 0.0,
            "illegal_action": False,
        }

        n_workers = len(self.workers)
        if int(action) == n_workers:
            reject_cost = self._reject_penalty(task)
            task.status = "rejected"
            self.active_task_id = None
            info.update({"reject_cost": reject_cost, "delta_total_cost": reject_cost})
            self.metrics.reject_count += 1
            if task.priority == 1:
                self.metrics.emergency_reject_count += 1
            reward = -float(reject_cost)
            self._accumulate_costs(info, reward)
            self._select_or_advance_to_next_decision()
            self._record_decision_time(start_clock)
            return self.build_state(), reward, self.is_done(), info

        if not action_is_worker(action, n_workers):
            return self._illegal_step(start_clock, info, reason="action_out_of_range")

        worker = self.workers[int(action)]
        result = evaluate_worker_insertion(worker, task, self.tasks, self.now, self.config)
        if not result.feasible:
            return self._illegal_step(start_clock, info, reason=result.reason)

        # Estimate response time at assignment using the same route simulator
        # that drives cost calculation.  A dynamic emergency is not actually
        # responded to at its release instant; the meaningful response time is
        # the planned service start time minus release_time after insertion.
        # This fixes the previous all-zero response metric while keeping the
        # reward/cost definition unchanged.
        response_time = None
        if task.priority == 1:
            records = simulate_route(worker, list(result.new_route), self.tasks, start_time=self.now, start_location=worker.location)
            for rec in records:
                if rec.task_id == task.id:
                    response_time = max(0.0, float(rec.start_time) - float(task.release_time))
                    break

        worker.route = list(result.new_route)
        task.status = "assigned"
        if response_time is not None:
            self.metrics.emergency_response_time_total += response_time
            self.metrics.emergency_response_count += 1
            info["emergency_response_time"] = response_time
        self.active_task_id = None

        info.update(
            {
                "delta_distance_cost": result.delta_distance_cost,
                "delta_tardiness_cost": result.delta_tardiness_cost,
                "delta_preemption_cost": result.delta_preemption_cost,
                "delta_total_cost": result.delta_total_cost,
                "preempted": result.preempted,
                "reason": result.reason,
            }
        )
        if result.preempted:
            self.metrics.preemption_count += 1
        reward = -float(result.delta_total_cost)
        self._accumulate_costs(info, reward)
        self._select_or_advance_to_next_decision()
        self._record_decision_time(start_clock)
        return self.build_state(), reward, self.is_done(), info

    def get_action_mask(self) -> np.ndarray:
        n_workers = len(self.workers)
        mask = np.ones(n_workers + 1, dtype=np.float32)
        task = self.active_task
        if task is None:
            mask[:n_workers] = 0.0
            mask[n_workers] = 1.0
            return mask

        # Basic hard constraints are always masked. In addition, run a
        # light-weight feasibility check through the same bottom-level
        # insertion heuristic used by step(). This prevents PPO from being
        # punished for choosing workers whose current route can no longer
        # accept the active task within the shift, while keeping reject always
        # legal as required by the modelling manual.
        feasibility_check = bool(self.config.get("action_mask", {}).get("feasibility_check", True))
        for i, worker in enumerate(self.workers):
            if worker.skill < task.skill_req:
                mask[i] = 0.0
                continue
            if worker.ready_time > self.day_end or self.now > worker.shift_end:
                mask[i] = 0.0
                continue
            if feasibility_check:
                result = evaluate_worker_insertion(worker, task, self.tasks, self.now, self.config)
                if not result.feasible:
                    mask[i] = 0.0
        mask[n_workers] = 1.0
        if float(mask.sum()) <= 0:
            self.metrics.action_mask_all_zero_count += 1
        return mask

    def build_state(self) -> dict:
        return build_state_dict(self.now, self.active_task, self.workers, self.tasks)

    def is_done(self) -> bool:
        if self.now >= self.day_end:
            return True
        if self.step_count >= self.max_steps_per_episode:
            return True
        return all(t.status in {"done", "rejected", "rejected_illegal"} for t in self.tasks.values())

    def _reject_penalty(self, task: Task) -> float:
        if task.priority == 1:
            return get_cost_value(self.config, "reject_penalty_emergency", 200.0)
        return get_cost_value(self.config, "reject_penalty_regular", 80.0)

    def _illegal_step(self, start_clock: float, info: dict[str, Any], reason: str) -> tuple[dict, float, bool, dict]:
        task = self.active_task
        penalty = float(self.config.get("cost", {}).get("illegal_penalty", 1000.0))
        reward = -penalty
        if task is not None:
            task.status = "rejected_illegal"
            self.active_task_id = None
        info.update({
            "illegal_action": True,
            "illegal_reason": reason,
            "reason": reason,
            "delta_total_cost": penalty,
        })
        self.metrics.illegal_action_count += 1
        if "skill" in str(reason):
            self.metrics.illegal_skill_mismatch_count += 1
        if "shift" in str(reason) or "overtime" in str(reason) or "finish_after" in str(reason):
            self.metrics.illegal_overtime_count += 1
        if "preempt" in str(reason):
            self.metrics.illegal_preemption_count += 1
        self.metrics.episode_reward += reward
        self.metrics.total_cost += penalty
        self._select_or_advance_to_next_decision()
        self._record_decision_time(start_clock)
        return self.build_state(), reward, self.is_done(), info

    def _accumulate_costs(self, info: dict[str, Any], reward: float) -> None:
        self.metrics.episode_reward += float(reward)
        self.metrics.total_cost += float(info.get("delta_total_cost", 0.0))
        self.metrics.distance_cost += float(info.get("delta_distance_cost", 0.0))
        self.metrics.tardiness_cost += float(info.get("delta_tardiness_cost", 0.0))
        self.metrics.reject_cost += float(info.get("reject_cost", 0.0))
        preempt_ind = float(info.get("delta_preemption_cost", 0.0))
        self.metrics.preemption_cost += get_cost_value(self.config, "mu_preemption", 30.0) * preempt_ind

    def _record_decision_time(self, start_clock: float) -> None:
        self.metrics.decision_time_total += perf_counter() - start_clock
        self.metrics.decision_count += 1

    def _select_or_advance_to_next_decision(self) -> None:
        if self.is_done():
            return

        while True:
            available = [
                t for t in self.tasks.values()
                if t.status == "waiting" and t.release_time <= self.now + 1e-9
            ]
            if available:
                # Earliest release first, then emergency first, then ID.
                available.sort(key=lambda t: (t.release_time, -t.priority, t.id))
                self.active_task_id = available[0].id
                self.now = max(self.now, available[0].release_time)
                return

            future_releases = [
                t.release_time for t in self.tasks.values()
                if t.status == "waiting" and t.release_time > self.now + 1e-9
            ]
            if not future_releases:
                self._advance_workers_to(self.day_end)
                self.active_task_id = None
                return

            next_time = min(future_releases)
            self._advance_workers_to(next_time)
            self.now = next_time
            if self.is_done():
                self.active_task_id = None
                return

    def _advance_workers_to(self, target_time: float) -> None:
        target_time = float(min(target_time, self.day_end))
        for worker in self.workers:
            self._advance_one_worker(worker, target_time)
        self.now = max(self.now, target_time)

    def _advance_one_worker(self, worker: Worker, target_time: float) -> None:
        if not worker.route:
            worker.current_status = "idle"
            worker.current_task_id = None
            worker.current_task_interruptible = False
            worker.ready_time = max(worker.ready_time, min(target_time, worker.shift_end))
            return

        loc = worker.location
        current_time = max(self.now, worker.ready_time)
        remaining = list(worker.route)

        for idx, task_id in enumerate(worker.route):
            task = self.tasks[task_id]
            arrival = current_time + distance(loc, task.location)
            start = max(arrival, task.earliest)
            finish = start + task.service_time

            if finish <= target_time + 1e-9:
                task.status = "done"
                worker.workload += task.service_time
                loc = task.location
                worker.set_location(*loc)
                worker.ready_time = finish
                worker.current_status = "idle"
                worker.current_task_id = None
                remaining = worker.route[idx + 1 :]
                current_time = finish
                continue

            remaining = worker.route[idx:]
            task.status = "in_service" if start <= target_time else task.status
            worker.current_task_id = task_id
            worker.current_task_interruptible = task.interruptible

            if target_time < arrival:
                # Linear interpolation along the trip for a better current location.
                travel = max(arrival - current_time, 1e-9)
                ratio = max(0.0, min(1.0, (target_time - current_time) / travel))
                x = loc[0] + ratio * (task.x - loc[0])
                y = loc[1] + ratio * (task.y - loc[1])
                worker.set_location(x, y)
                worker.current_status = "traveling"
                worker.ready_time = target_time
            else:
                worker.set_location(task.x, task.y)
                worker.current_status = "serving"
                worker.ready_time = target_time
            worker.route = list(remaining)
            return

        worker.route = list(remaining)
        if not worker.route:
            worker.current_status = "idle"
            worker.current_task_id = None
            worker.current_task_interruptible = False
