"""Environment state construction."""
from __future__ import annotations

from src.env.entities import Task, Worker


def build_state_dict(now: float, active_task: Task | None, workers: list[Worker], all_tasks: dict[int, Task]) -> dict:
    unfinished = [t for t in all_tasks.values() if t.status not in {"done", "rejected", "rejected_illegal"}]
    return {
        "now": float(now),
        "active_task": active_task,
        "workers": workers,
        "worker_routes": [list(w.route) for w in workers],
        "unfinished_tasks": unfinished,
    }


def build_flat_state(state: dict, config: dict):
    """Build the MLP-PPO vector from the same node features as GAT-PPO."""
    from src.graph.graph_builder import build_node_feature_matrix
    return build_node_feature_matrix(state, config).reshape(-1).astype('float32')


def flat_state_dim(num_workers: int, node_dim: int = 13) -> int:
    return int((num_workers + 1) * node_dim)
