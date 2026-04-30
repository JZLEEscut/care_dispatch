from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from src.env.entities import Task, Worker
from src.heuristics.cost import distance, travel_time

NODE_FEATURE_DIM = 13
EDGE_FEATURE_DIM = 2

@dataclass
class GraphState:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    action_mask: torch.Tensor
    meta: dict[str, Any]

def _day_end(cfg: dict) -> float:
    return max(float(cfg.get('time', {}).get('day_end', 480.0)), 1.0)

def _coord_norm(cfg: dict) -> float:
    return max(float(cfg.get('state', {}).get('coord_norm', 100.0)), 1.0)

def _route_norm(cfg: dict, workers: list[Worker]) -> float:
    v = cfg.get('state', {}).get('max_route_len')
    return max(float(v), 1.0) if v is not None else max(1.0, float(max((len(w.route) for w in workers), default=0) + 1))

def task_node_features(task: Task | None, now: float, cfg: dict) -> list[float]:
    de = _day_end(cfg); sn = max(float(cfg.get('state', {}).get('service_time_norm', 60.0)), 1.0)
    if task is None:
        return [1.0, 0.0] + [0.0] * 11
    return [1.0, 0.0, 0.0, 0.0, now / de, task.latest / de, task.service_time / sn,
            task.skill_req / 3.0, 0.0, float(task.priority), float(task.interruptible), 0.0, 0.0]

def worker_node_features(worker: Worker, task: Task | None, now: float, workers: list[Worker], cfg: dict) -> list[float]:
    de = _day_end(cfg); cn = _coord_norm(cfg); rn = _route_norm(cfg, workers)
    sn = max(float(cfg.get('state', {}).get('service_time_norm', 60.0)), 1.0)
    if task is None:
        rel_x = rel_y = skill_gap = latest = priority = interruptible = service = 0.0
    else:
        rel_x = (worker.x - task.x) / cn; rel_y = (worker.y - task.y) / cn
        skill_gap = (worker.skill - task.skill_req) / 3.0; latest = task.latest / de
        priority = float(task.priority); interruptible = float(task.interruptible); service = task.service_time / sn
    return [0.0, 1.0, rel_x, rel_y, worker.ready_time / de, latest, service, worker.skill / 3.0,
            skill_gap, priority, interruptible, worker.workload / de, len(worker.route) / rn]

def build_node_feature_matrix(state: dict, cfg: dict) -> np.ndarray:
    now = float(state.get('now', 0.0)); task = state.get('active_task'); workers = list(state.get('workers', []))
    rows = [task_node_features(task, now, cfg)] + [worker_node_features(w, task, now, workers, cfg) for w in workers]
    return np.asarray(rows, dtype=np.float32)

def build_fully_connected_edges(state: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    task = state.get('active_task'); workers = list(state.get('workers', [])); cn = _coord_norm(cfg)
    coords = [task.location if task is not None else (0.0, 0.0)] + [w.location for w in workers]
    pairs, attrs = [], []
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            if i == j: continue
            d = distance(a, b); pairs.append((i, j)); attrs.append([d / cn, travel_time(a, b) / cn])
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64).T, np.asarray(attrs, dtype=np.float32)

def build_graph_state(state: dict, action_mask, cfg: dict, device='cpu') -> GraphState:
    x = build_node_feature_matrix(state, cfg); ei, ea = build_fully_connected_edges(state, cfg)
    return GraphState(torch.as_tensor(x, dtype=torch.float32, device=device),
                      torch.as_tensor(ei, dtype=torch.long, device=device),
                      torch.as_tensor(ea, dtype=torch.float32, device=device),
                      torch.as_tensor(action_mask, dtype=torch.float32, device=device),
                      {'now': float(state.get('now', 0.0)), 'num_workers': len(state.get('workers', [])),
                       'active_task_id': None if state.get('active_task') is None else state['active_task'].id})
