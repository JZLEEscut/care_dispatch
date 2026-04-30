from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.dispatch_env import CareDispatchEnv


def test_emergency_response_metric_uses_planned_service_start():
    cfg = {
        "data": {"solomon_file": "data/solomon/RC101.txt", "num_tasks": 5, "num_workers": 2, "allow_synthetic_if_missing": True},
        "time": {"day_start": 0, "day_end": 480, "max_steps_per_episode": 50},
        "dynamic_arrival": {"enabled": True, "dynamic_ratio": 1.0, "emergency_ratio": 1.0},
        "skills": {"task_skill_distribution": [1.0, 0.0, 0.0], "worker_skill_distribution": [1.0, 0.0, 0.0]},
        "cost": {"distance_weight": 1.0, "lambda_tardiness_regular": 2.0, "lambda_tardiness_emergency": 6.0, "mu_preemption": 30.0, "reject_penalty_regular": 80.0, "reject_penalty_emergency": 200.0},
        "preemption": {"enabled": True},
        "action_mask": {"feasibility_check": True},
    }
    env = CareDispatchEnv(cfg)
    state = env.reset(seed=123)
    task = env.active_task
    assert task is not None and task.priority == 1
    mask = env.get_action_mask()
    worker_action = int(next(i for i, v in enumerate(mask[:-1]) if v == 1.0))
    _, _, _, info = env.step(worker_action)
    assert env.metrics.emergency_response_count == 1
    assert "emergency_response_time" in info
    assert env.metrics.mean_emergency_response_time >= 0.0
