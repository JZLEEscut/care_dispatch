"""Distance and route-cost utilities."""
from __future__ import annotations

import math
from typing import Iterable

from src.env.entities import RouteRecord, Task, Worker


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance, also used as travel time at speed=1."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def travel_time(a: tuple[float, float], b: tuple[float, float]) -> float:
    return distance(a, b)


def route_distance(
    worker: Worker,
    route: list[int],
    all_tasks: dict[int, Task],
    start_location: tuple[float, float] | None = None,
) -> float:
    loc = start_location if start_location is not None else worker.location
    total = 0.0
    for task_id in route:
        task = all_tasks[task_id]
        nxt = task.location
        total += distance(loc, nxt)
        loc = nxt
    return total


def total_tardiness(records: Iterable[RouteRecord]) -> float:
    return float(sum(r.tardiness for r in records))


def get_cost_value(config, key: str, default: float) -> float:
    if isinstance(config, dict):
        return float(config.get("cost", {}).get(key, default))
    cost_cfg = getattr(config, "cost", None)
    if cost_cfg is None:
        return float(default)
    if isinstance(cost_cfg, dict):
        return float(cost_cfg.get(key, default))
    return float(getattr(cost_cfg, key, default))


def tardiness_weight_for_task(task: Task, config) -> float:
    if task.priority == 1:
        return get_cost_value(config, "lambda_tardiness_emergency", 6.0)
    return get_cost_value(config, "lambda_tardiness_regular", 2.0)
