"""Unified evaluation loop for experiments 2, 3 and 4."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from src.common.utils import ensure_dir
from src.env.dispatch_env import CareDispatchEnv


def _episode_row(method: str, seed: int, episode: int, env: CareDispatchEnv, runtime: float, scenario_id: str = "", solomon_file: str = "") -> dict[str, Any]:
    metrics = env.metrics.as_dict()
    total_tasks = len(env.tasks)
    emergency_count = sum(1 for t in env.tasks.values() if t.priority == 1)
    completed_count = sum(1 for t in env.tasks.values() if t.status == "done")
    reject_count = int(metrics.get("reject_count", 0))
    emergency_reject_count = int(metrics.get("emergency_reject_count", 0))
    illegal_count = int(metrics.get("illegal_action_count", 0))
    decision_count = max(1, int(metrics.get("decision_count", 0)))

    return {
        "method": method,
        "seed": int(seed),
        "episode": int(episode),
        "scenario_id": scenario_id,
        "solomon_file": solomon_file,
        "num_tasks": int(total_tasks),
        "num_workers": int(len(env.workers)),
        "total_cost": float(metrics.get("total_cost", 0.0)),
        "episode_reward": float(metrics.get("episode_reward", 0.0)),
        "distance_cost": float(metrics.get("distance_cost", 0.0)),
        "tardiness_cost": float(metrics.get("tardiness_cost", 0.0)),
        "reject_cost": float(metrics.get("reject_cost", 0.0)),
        "preemption_cost": float(metrics.get("preemption_cost", 0.0)),
        "reject_count": reject_count,
        "emergency_reject_count": emergency_reject_count,
        "completed_count": int(completed_count),
        "total_tasks": int(total_tasks),
        "emergency_count": int(emergency_count),
        "reject_rate": reject_count / max(1, total_tasks),
        "emergency_reject_rate": emergency_reject_count / max(1, emergency_count),
        "completion_rate": completed_count / max(1, total_tasks),
        "mean_emergency_response_time": float(metrics.get("mean_emergency_response_time", 0.0)),
        "tardiness_total": float(metrics.get("tardiness_cost", 0.0)),
        "preemption_count": float(metrics.get("preemption_count", 0.0)),
        "illegal_action_count": illegal_count,
        "illegal_action_rate": illegal_count / decision_count,
        "illegal_skill_mismatch_count": int(metrics.get("illegal_skill_mismatch_count", 0)),
        "illegal_overtime_count": int(metrics.get("illegal_overtime_count", 0)),
        "illegal_preemption_count": int(metrics.get("illegal_preemption_count", 0)),
        "mean_decision_time": float(metrics.get("mean_decision_time", 0.0)),
        "runtime_episode": float(runtime),
    }


def evaluate_policy(
    policy,
    env_config: dict[str, Any],
    seeds: list[int],
    eval_episodes: int,
    output_raw_path: str | Path | None = None,
    method_name: str | None = None,
    scenario_id: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    method = method_name or getattr(policy, "method_name", policy.__class__.__name__)
    solomon_file = str(env_config.get("data", {}).get("solomon_file", ""))
    for seed in seeds:
        for episode in range(int(eval_episodes)):
            env = CareDispatchEnv(deepcopy(env_config))
            ep_seed = int(seed) * 100_000 + int(episode)
            state = env.reset(seed=ep_seed)
            start = perf_counter()
            steps = 0
            while not env.is_done():
                action = int(policy.select_action(state, env, deterministic=True))
                state, _reward, done, _info = env.step(action)
                steps += 1
                if done or steps > int(env_config.get("time", {}).get("max_steps_per_episode", 200)) + 10:
                    break
            rows.append(_episode_row(method, seed, episode, env, perf_counter() - start, scenario_id, solomon_file))
    df = pd.DataFrame(rows)
    if output_raw_path is not None:
        path = Path(output_raw_path)
        ensure_dir(path.parent)
        df.to_csv(path, index=False)
    return df
