"""Generate care workers."""
from __future__ import annotations

import numpy as np

from src.env.entities import Worker


def generate_workers(config: dict, depot: tuple[float, float] = (0.0, 0.0), seed: int | None = None) -> list[Worker]:
    rng = np.random.default_rng(seed)
    data_cfg = config.get("data", {})
    skill_cfg = config.get("skills", {})
    time_cfg = config.get("time", {})

    num_workers = int(data_cfg.get("num_workers", 5))
    probs = np.asarray(skill_cfg.get("worker_skill_distribution", [0.40, 0.40, 0.20]), dtype=float)
    probs = probs / probs.sum()
    skills = rng.choice([1, 2, 3], size=num_workers, p=probs)
    shift_start = float(time_cfg.get("day_start", 0))
    shift_end = float(time_cfg.get("day_end", 480))

    return [
        Worker(
            id=i,
            skill=int(skills[i]),
            x=float(depot[0]),
            y=float(depot[1]),
            shift_start=shift_start,
            shift_end=shift_end,
            ready_time=shift_start,
        )
        for i in range(num_workers)
    ]
