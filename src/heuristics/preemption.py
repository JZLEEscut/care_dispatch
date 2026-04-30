"""Finite preemption candidate generation.

Preemption is not an upper-level PPO action. It is considered only as an
internal insertion candidate for emergency tasks.
"""
from __future__ import annotations

from src.env.entities import Task, Worker


def preemption_enabled(config) -> bool:
    if isinstance(config, dict):
        return bool(config.get("preemption", {}).get("enabled", True))
    p = getattr(config, "preemption", None)
    if p is None:
        return True
    if isinstance(p, dict):
        return bool(p.get("enabled", True))
    return bool(getattr(p, "enabled", True))


def can_consider_preemption(worker: Worker, active_task: Task, all_tasks: dict[int, Task], config) -> tuple[bool, str]:
    if not preemption_enabled(config):
        return False, "preemption_disabled"
    if active_task.priority != 1:
        return False, "active_task_not_emergency"
    if worker.current_status not in {"traveling", "serving"}:
        return False, "worker_not_busy"
    if not worker.route:
        return False, "empty_route"

    current_id = worker.current_task_id if worker.current_task_id is not None else worker.route[0]
    current = all_tasks.get(current_id)
    if current is None:
        return False, "current_task_missing"
    if current.priority != 0:
        return False, "current_task_not_regular"
    if worker.current_status == "serving" and not worker.current_task_interruptible:
        return False, "current_task_not_interruptible"
    return True, "ok"


def preemptive_route(worker: Worker, active_task: Task) -> list[int]:
    """Place emergency task before the current committed route.

    If the worker is serving an interruptible regular task, that task remains in
    the route after the emergency task, which implements reset-and-restart.
    """
    return [active_task.id] + list(worker.route)
