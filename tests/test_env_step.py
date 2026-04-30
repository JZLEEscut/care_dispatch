from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.dispatch_env import CareDispatchEnv


def test_env_random_legal_steps_complete_or_progress():
    cfg = {
        "data": {"num_tasks": 5, "num_workers": 3, "allow_synthetic_if_missing": True},
        "time": {"day_start": 0, "day_end": 480, "max_steps_per_episode": 50},
        "dynamic_arrival": {"enabled": False},
        "skills": {
            "task_skill_distribution": [1.0, 0.0, 0.0],
            "worker_skill_distribution": [0.0, 0.0, 1.0],
        },
        "cost": {"reject_penalty_regular": 80.0, "reject_penalty_emergency": 200.0},
    }
    env = CareDispatchEnv(cfg)
    env.reset(seed=1)
    for _ in range(20):
        mask = env.get_action_mask()
        action = int([i for i, v in enumerate(mask) if v == 1.0][0])
        _, _, done, info = env.step(action)
        assert "illegal_action" in info
        if done:
            break
    assert env.step_count > 0
