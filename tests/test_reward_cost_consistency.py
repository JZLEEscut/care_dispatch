from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.dispatch_env import CareDispatchEnv


def test_reward_equals_negative_delta_total_cost_for_reject():
    cfg = {
        "data": {"num_tasks": 2, "num_workers": 1, "allow_synthetic_if_missing": True},
        "time": {"day_start": 0, "day_end": 480, "max_steps_per_episode": 20},
        "dynamic_arrival": {"enabled": False},
        "skills": {
            "task_skill_distribution": [1.0, 0.0, 0.0],
            "worker_skill_distribution": [1.0, 0.0, 0.0],
        },
        "cost": {"reject_penalty_regular": 80.0, "reject_penalty_emergency": 200.0},
    }
    env = CareDispatchEnv(cfg)
    env.reset(seed=2)
    reject_action = len(env.workers)
    _, reward, _, info = env.step(reject_action)
    assert reward == -info["delta_total_cost"]
    assert info["reject_cost"] > 0
