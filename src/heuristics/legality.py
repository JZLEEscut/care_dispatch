"""Hard feasibility checks for dispatch actions."""
from __future__ import annotations

from src.env.entities import Task, Worker


def skill_feasible(worker: Worker, task: Task) -> bool:
    return int(worker.skill) >= int(task.skill_req)


def shift_available(worker: Worker, now: float) -> bool:
    return float(worker.ready_time) <= float(worker.shift_end) and float(now) <= float(worker.shift_end)


def route_within_shift(finish_time: float, worker: Worker) -> bool:
    return float(finish_time) <= float(worker.shift_end)


def action_is_worker(action: int, num_workers: int) -> bool:
    return 0 <= int(action) < int(num_workers)
