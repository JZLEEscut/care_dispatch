"""Generate care tasks from Solomon customers."""
from __future__ import annotations

import numpy as np

from src.data.solomon_loader import SolomonCustomer
from src.env.entities import Task


def _sample_skill(rng: np.random.Generator, distribution: list[float]) -> int:
    probs = np.asarray(distribution, dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice([1, 2, 3], p=probs))


def generate_tasks_from_customers(customers: list[SolomonCustomer], config: dict, seed: int | None = None) -> list[Task]:
    rng = np.random.default_rng(seed)
    data_cfg = config.get("data", {})
    dyn_cfg = config.get("dynamic_arrival", {})
    skill_cfg = config.get("skills", {})
    time_cfg = config.get("time", {})

    num_tasks = int(data_cfg.get("num_tasks", max(0, len(customers) - 1)))
    task_skill_distribution = skill_cfg.get("task_skill_distribution", [0.50, 0.35, 0.15])
    day_start = float(time_cfg.get("day_start", 0))
    day_end = float(time_cfg.get("day_end", 480))
    dynamic_enabled = bool(dyn_cfg.get("enabled", True))
    dynamic_ratio = float(dyn_cfg.get("dynamic_ratio", 0.20)) if dynamic_enabled else 0.0
    emergency_ratio = float(dyn_cfg.get("emergency_ratio", 0.20)) if dynamic_enabled else 0.0

    # Solomon row 0 is normally depot; task IDs remain compact 0..N-1.
    customer_rows = [c for c in customers if c.customer_id != 0][:num_tasks]
    tasks: list[Task] = []
    for idx, c in enumerate(customer_rows):
        is_dynamic = bool(dynamic_enabled and rng.random() < dynamic_ratio)
        is_emergency = bool(dynamic_enabled and rng.random() < emergency_ratio)
        priority = 1 if is_emergency else 0

        if priority == 1 or is_dynamic:
            release_time = float(rng.uniform(day_start, day_end * 0.8))
            is_dynamic = True
        else:
            release_time = 0.0

        tasks.append(
            Task(
                id=idx,
                x=float(c.x),
                y=float(c.y),
                service_time=float(c.service_time),
                earliest=float(c.ready_time),
                latest=float(c.due_time),
                skill_req=_sample_skill(rng, task_skill_distribution),
                priority=priority,
                interruptible=bool(priority == 0 and rng.random() < 0.70),
                release_time=release_time,
                is_dynamic=is_dynamic,
                status="waiting",
            )
        )
    return tasks


def generate_synthetic_customers(num_tasks: int, seed: int | None = None) -> list[SolomonCustomer]:
    """Small fallback for tests and first-run smoke checks when RC101 is absent."""
    rng = np.random.default_rng(seed)
    rows = [SolomonCustomer(0, 50.0, 50.0, 0.0, 0.0, 480.0, 0.0)]
    for i in range(1, num_tasks + 1):
        x = float(rng.uniform(0, 100))
        y = float(rng.uniform(0, 100))
        ready = float(rng.uniform(0, 240))
        due = ready + float(rng.uniform(60, 180))
        service = float(rng.uniform(5, 20))
        rows.append(SolomonCustomer(i, x, y, 0.0, ready, due, service))
    return rows
