"""Build one episode scenario from config."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.common.paths import resolve_project_path
from src.data.solomon_loader import SolomonCustomer, load_solomon_file
from src.data.task_generator import generate_synthetic_customers, generate_tasks_from_customers
from src.data.worker_generator import generate_workers


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, updates: dict) -> dict:
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def build_scenario(config: dict, seed: int | None = None) -> dict:
    data_cfg = config.get("data", {})
    solomon_file = data_cfg.get("solomon_file", "data/solomon/RC101.txt")
    num_tasks = int(data_cfg.get("num_tasks", 50))

    try:
        customers = load_solomon_file(resolve_project_path(solomon_file))
    except FileNotFoundError:
        if not bool(data_cfg.get("allow_synthetic_if_missing", True)):
            raise
        customers = generate_synthetic_customers(num_tasks=num_tasks, seed=seed)

    depot_row = next((c for c in customers if c.customer_id == 0), customers[0])
    depot = (float(depot_row.x), float(depot_row.y))
    tasks = generate_tasks_from_customers(customers, config, seed=seed)
    workers = generate_workers(config, depot=depot, seed=None if seed is None else seed + 10_000)
    return {"tasks": tasks, "workers": workers, "depot": depot}
